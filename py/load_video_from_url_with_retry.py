import hashlib
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import cv2
import folder_paths
import numpy as np
import torch
from comfy.utils import ProgressBar, common_upscale

# =========================================================
# Constants
# =========================================================

BIGMAX = 2**53 - 1
DIMMAX = 8192

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".mpeg",
    ".mpg",
}


# =========================================================
# ComfyUI compatible flexible types
# =========================================================


class MultiInput(str):
    def __new__(cls, string, allowed_types="*"):
        obj = super().__new__(cls, string)
        obj.allowed_types = allowed_types
        return obj

    def __ne__(self, other):
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types


imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])

floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])


# =========================================================
# Basic helpers
# =========================================================


def is_url(value):
    if not isinstance(value, str):
        return False

    value = value.strip().lower()

    return value.startswith("http://") or value.startswith("https://")


def strip_path(path):
    path = path.strip()

    if path.startswith('"'):
        path = path[1:]

    if path.endswith('"'):
        path = path[:-1]

    return path


def calculate_file_hash(path):
    h = hashlib.sha256()

    h.update(
        os.path.abspath(path).encode(
            "utf-8",
            errors="ignore",
        )
    )

    try:
        h.update(str(os.path.getmtime(path)).encode())

        h.update(str(os.path.getsize(path)).encode())

    except OSError:
        pass

    return h.hexdigest()


def hash_path(path):
    if path is None:
        return "input"

    if is_url(path):
        # URL 本身作为缓存判断依据
        return hashlib.sha256(path.encode("utf-8")).hexdigest()

    path = strip_path(path)

    if not os.path.isfile(path):
        return "DNE"

    return calculate_file_hash(path)


def validate_path(path, allow_none=False):
    if path is None:
        return allow_none

    if not isinstance(path, str):
        return False

    path = strip_path(path)

    if is_url(path):
        return True

    if not os.path.isfile(path):
        return f"Invalid video file: {path}"

    return True


# =========================================================
# FFmpeg
# =========================================================


def find_ffmpeg():
    """
    Completely independent from VHS.

    Priority:
    1. BIMO_FFMPEG_PATH environment variable
    2. system ffmpeg
    3. imageio_ffmpeg if available
    """

    env_path = os.environ.get("BIMO_FFMPEG_PATH")

    if env_path and os.path.isfile(env_path):
        return env_path

    system_ffmpeg = shutil.which("ffmpeg")

    if system_ffmpeg:
        return system_ffmpeg

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        path = get_ffmpeg_exe()

        if path and os.path.isfile(path):
            return path

    except Exception:
        pass

    return None


FFMPEG_PATH = find_ffmpeg()


# =========================================================
# AUDIO
# =========================================================


def get_audio(
    file,
    start_time=0,
    duration=0,
):
    """
    Output format is compatible with ComfyUI AUDIO:

    {
        "waveform": Tensor [batch, channels, samples],
        "sample_rate": int
    }
    """

    if FFMPEG_PATH is None:
        raise RuntimeError("FFmpeg was not found. Install FFmpeg or set BIMO_FFMPEG_PATH.")

    args = [
        FFMPEG_PATH,
        "-hide_banner",
        "-i",
        file,
    ]

    if start_time > 0:
        args += [
            "-ss",
            str(start_time),
        ]

    if duration > 0:
        args += [
            "-t",
            str(duration),
        ]

    args += [
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]

    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    except subprocess.CalledProcessError as e:
        error = e.stderr.decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(f"Failed to extract audio:\n{error}")

    stderr = result.stderr.decode(
        "utf-8",
        errors="replace",
    )

    # Try to preserve original sample rate/channels
    match = re.search(
        r",\s*(\d+)\s*Hz,\s*(mono|stereo)",
        stderr,
        re.IGNORECASE,
    )

    if match:
        sample_rate = int(match.group(1))

        channel_name = match.group(2).lower()

        channels = 1 if channel_name == "mono" else 2

    else:
        sample_rate = 44100
        channels = 2

    raw = result.stdout

    if not raw:
        # Empty AUDIO tensor
        waveform = torch.zeros(
            (1, channels, 0),
            dtype=torch.float32,
        )

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

    audio = torch.frombuffer(
        bytearray(raw),
        dtype=torch.float32,
    )

    # Protect against malformed audio size
    usable = len(audio) // channels * channels

    audio = audio[:usable]

    audio = audio.reshape(-1, channels).transpose(0, 1).unsqueeze(0)

    return {
        "waveform": audio,
        "sample_rate": sample_rate,
    }


class LazyAudioMap(Mapping):
    """
    Same idea as VHS:
    audio isn't decoded until something actually accesses it.
    """

    def __init__(
        self,
        file,
        start_time,
        duration,
    ):
        self.file = file
        self.start_time = start_time
        self.duration = duration
        self._data = None

    def _load(self):
        if self._data is None:
            self._data = get_audio(
                self.file,
                self.start_time,
                self.duration,
            )

    def __getitem__(self, key):
        self._load()
        return self._data[key]

    def __iter__(self):
        self._load()
        return iter(self._data)

    def __len__(self):
        self._load()
        return len(self._data)


def lazy_get_audio(
    file,
    start_time=0,
    duration=0,
):
    return LazyAudioMap(
        file,
        start_time,
        duration,
    )


# =========================================================
# URL download with retry
# =========================================================


def guess_extension(url):
    try:
        path = unquote(urlparse(url).path)

        ext = os.path.splitext(path)[1].lower()

        if ext in VIDEO_EXTENSIONS:
            return ext

    except Exception:
        pass

    return ".mp4"


def build_temp_filename(url):
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]

    ext = guess_extension(url)

    return f"bimo_video_{url_hash}{ext}"


def download_video_with_retry(
    url,
    retry_count,
    retry_interval,
    timeout,
):
    """
    retry_count:
        Number of retries AFTER the first request.

        retry_count = 3
        -> maximum total attempts = 4

    timeout:
        Socket/network timeout.
        It is NOT the maximum total download duration.
    """

    temp_dir = folder_paths.get_temp_directory()

    os.makedirs(
        temp_dir,
        exist_ok=True,
    )

    filename = build_temp_filename(url)

    final_path = os.path.join(
        temp_dir,
        filename,
    )

    part_path = final_path + ".part"

    total_attempts = int(retry_count) + 1

    last_error = None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/152 Safari/537.36"
        ),
        "Accept": "*/*",
    }

    for attempt in range(
        1,
        total_attempts + 1,
    ):
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass

        try:
            print(f"[BIMO LoadVideoRetry] Download attempt {attempt}/{total_attempts}")

            print(f"[BIMO LoadVideoRetry] URL: {url}")

            request = Request(
                url,
                headers=headers,
                method="GET",
            )

            with urlopen(
                request,
                timeout=float(timeout),
            ) as response:
                status = getattr(
                    response,
                    "status",
                    200,
                )

                if status is not None and status >= 400:
                    raise RuntimeError(f"HTTP {status}")

                content_length = response.headers.get("Content-Length")

                expected_size = None

                if content_length:
                    try:
                        expected_size = int(content_length)
                    except ValueError:
                        pass

                downloaded = 0

                with open(
                    part_path,
                    "wb",
                ) as f:
                    while True:
                        chunk = response.read(1024 * 1024)

                        if not chunk:
                            break

                        f.write(chunk)

                        downloaded += len(chunk)

            if downloaded <= 0:
                raise RuntimeError("Downloaded file is empty.")

            if expected_size is not None and downloaded != expected_size:
                raise RuntimeError(
                    "Incomplete download. "
                    f"Expected "
                    f"{expected_size} bytes, "
                    f"received "
                    f"{downloaded} bytes."
                )

            os.replace(
                part_path,
                final_path,
            )

            print(f"[BIMO LoadVideoRetry] Download completed: {final_path}")

            return final_path

        except (
            HTTPError,
            URLError,
            TimeoutError,
            ConnectionError,
            OSError,
            RuntimeError,
        ) as e:
            last_error = e

            print(f"[BIMO LoadVideoRetry] Attempt {attempt} failed: {type(e).__name__}: {e}")

            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except OSError:
                    pass

            if attempt >= total_attempts:
                break

            if retry_interval > 0:
                print(f"[BIMO LoadVideoRetry] Retrying in {retry_interval} seconds...")

                time.sleep(float(retry_interval))

    raise RuntimeError(
        "Video download failed.\n"
        f"Attempts: {total_attempts}\n"
        f"URL: {url}\n"
        f"Last error: "
        f"{type(last_error).__name__}: "
        f"{last_error}"
    )


# =========================================================
# Image resize
# =========================================================


def target_size(
    width,
    height,
    custom_width,
    custom_height,
    downscale_ratio=8,
):
    if custom_width == 0 and custom_height == 0:
        new_width = width
        new_height = height

    elif custom_height == 0:
        new_width = custom_width

        new_height = height * custom_width / width

    elif custom_width == 0:
        new_height = custom_height

        new_width = width * custom_height / height

    else:
        new_width = custom_width
        new_height = custom_height

    ratio = downscale_ratio if downscale_ratio else 1

    new_width = int(new_width / ratio + 0.5) * ratio

    new_height = int(new_height / ratio + 0.5) * ratio

    new_width = max(
        ratio,
        new_width,
    )

    new_height = max(
        ratio,
        new_height,
    )

    return (
        int(new_width),
        int(new_height),
    )


# =========================================================
# Video decoder
# =========================================================


def read_video_frames(
    video,
    force_rate,
    custom_width,
    custom_height,
    frame_load_cap,
    skip_first_frames,
    select_every_nth,
):
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))

        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if source_fps <= 0:
            source_fps = 30.0

        if source_width <= 0 or source_height <= 0:
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError("Unable to read first video frame.")

            source_height, source_width = frame.shape[:2]

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0,
            )

        if source_frame_count > 0:
            source_duration = source_frame_count / source_fps
        else:
            source_duration = 0.0

        target_fps = float(force_rate) if force_rate > 0 else source_fps

        new_width, new_height = target_size(
            source_width,
            source_height,
            int(custom_width),
            int(custom_height),
            8,
        )

        # Estimate progress
        if source_duration > 0:
            estimated = int(source_duration * target_fps)
        else:
            estimated = source_frame_count if source_frame_count > 0 else 0

        estimated = max(
            0,
            estimated - int(skip_first_frames),
        )

        estimated = int(
            np.ceil(
                estimated
                / max(
                    1,
                    int(select_every_nth),
                )
            )
        )

        if frame_load_cap > 0:
            estimated = min(
                estimated,
                int(frame_load_cap),
            )

        pbar = ProgressBar(estimated)

        frames = []

        source_index = 0
        resampled_index = 0
        selected_index = 0

        # Time at which next target frame
        # should be accepted
        next_target_time = 0.0

        target_interval = 1.0 / target_fps

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_time = source_index / source_fps

            source_index += 1

            # Resampling for force_rate
            if force_rate > 0 and frame_time + 1e-9 < next_target_time:
                continue

            if force_rate > 0:
                while next_target_time <= frame_time + 1e-9:
                    next_target_time += target_interval

            # Skip after FPS conversion
            if resampled_index < skip_first_frames:
                resampled_index += 1
                continue

            relative_index = resampled_index - skip_first_frames

            resampled_index += 1

            if relative_index % select_every_nth != 0:
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            tensor = torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255.0)

            # [H,W,C] -> [1,C,H,W]
            if new_width != source_width or new_height != source_height:
                tensor = tensor.unsqueeze(0).movedim(-1, 1)

                tensor = common_upscale(
                    tensor,
                    new_width,
                    new_height,
                    "lanczos",
                    "center",
                )

                tensor = tensor.movedim(1, -1).squeeze(0)

            frames.append(tensor)

            selected_index += 1

            pbar.update_absolute(
                selected_index,
                estimated,
            )

            if frame_load_cap > 0 and selected_index >= frame_load_cap:
                break

        if not frames:
            raise RuntimeError("No frames generated.")

        images = torch.stack(
            frames,
            dim=0,
        )

        loaded_fps = target_fps / select_every_nth

        loaded_frame_count = len(images)

        loaded_duration = loaded_frame_count / loaded_fps

        video_info = {
            "source_fps": source_fps,
            "source_frame_count": source_frame_count,
            "source_duration": source_duration,
            "source_width": source_width,
            "source_height": source_height,
            "loaded_fps": loaded_fps,
            "loaded_frame_count": loaded_frame_count,
            "loaded_duration": loaded_duration,
            "loaded_width": new_width,
            "loaded_height": new_height,
        }

        # Audio start/duration follows loaded
        # portion of the video
        effective_target_fps = target_fps

        start_time = skip_first_frames / effective_target_fps

        audio_duration = loaded_duration if frame_load_cap > 0 else 0

        audio = lazy_get_audio(
            video,
            start_time=start_time,
            duration=audio_duration,
        )

        return (
            images,
            loaded_frame_count,
            audio,
            video_info,
        )

    finally:
        cap.release()


# =========================================================
# Node
# =========================================================


class LoadVideoPathWithRetry:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Local path or https://.../video.mp4",
                    },
                ),
                "force_rate": (
                    floatOrInt,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 60,
                        "step": 1,
                        "disable": 0,
                    },
                ),
                "custom_width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": DIMMAX,
                        "step": 1,
                        "disable": 0,
                    },
                ),
                "custom_height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": DIMMAX,
                        "step": 1,
                        "disable": 0,
                    },
                ),
                "frame_load_cap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": BIGMAX,
                        "step": 1,
                        "disable": 0,
                    },
                ),
                "skip_first_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": BIGMAX,
                        "step": 1,
                    },
                ),
                "select_every_nth": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": BIGMAX,
                        "step": 1,
                    },
                ),
                # -------------------------
                # Download retry controls
                # -------------------------
                "retry_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 20,
                        "step": 1,
                    },
                ),
                "retry_interval": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 300.0,
                        "step": 0.5,
                    },
                ),
                "timeout": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 1.0,
                        "max": 600.0,
                        "step": 1.0,
                    },
                ),
            }
        }

    CATEGORY = "BIMO/Video"

    FUNCTION = "load_video"

    # Keep VHS-compatible outputs
    RETURN_TYPES = (
        imageOrLatent,
        "INT",
        "AUDIO",
        "VHS_VIDEOINFO",
    )

    RETURN_NAMES = (
        "IMAGE",
        "frame_count",
        "audio",
        "video_info",
    )

    def load_video(
        self,
        video,
        force_rate,
        custom_width,
        custom_height,
        frame_load_cap,
        skip_first_frames,
        select_every_nth,
        retry_count,
        retry_interval,
        timeout,
    ):

        video = strip_path(video)

        validation = validate_path(video)

        if validation is not True:
            raise ValueError(str(validation))

        # URL -> retry downloader
        if is_url(video):
            local_path = download_video_with_retry(
                url=video,
                retry_count=retry_count,
                retry_interval=retry_interval,
                timeout=timeout,
            )

        # Local path
        else:
            local_path = video

        return read_video_frames(
            video=local_path,
            force_rate=float(force_rate),
            custom_width=int(custom_width),
            custom_height=int(custom_height),
            frame_load_cap=int(frame_load_cap),
            skip_first_frames=int(skip_first_frames),
            select_every_nth=int(select_every_nth),
        )

    @classmethod
    def IS_CHANGED(
        cls,
        video,
        **kwargs,
    ):
        return hash_path(video)

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        video,
        **kwargs,
    ):
        return validate_path(
            video,
            allow_none=True,
        )


NODE_CLASS_MAPPINGS = {
    "BIMOLoadVideoPathWithRetry": LoadVideoPathWithRetry,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "BIMOLoadVideoPathWithRetry": "Load Video Path With Retry",
}
