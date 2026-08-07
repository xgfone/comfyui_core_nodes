from . import any


class SwitchCaseNodePro:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "switch_condition": ("STRING", {"default": "", "multiline": False}),
                "case_1": ("STRING", {"default": "1", "multiline": False}),
                "case_2": ("STRING", {"default": "2", "multiline": False}),
                "case_3": ("STRING", {"default": "3", "multiline": False}),
                "case_4": ("STRING", {"default": "4", "multiline": False}),
                "case_5": ("STRING", {"default": "5", "multiline": False}),
                "case_6": ("STRING", {"default": "6", "multiline": False}),
                "case_7": ("STRING", {"default": "7", "multiline": False}),
                "case_8": ("STRING", {"default": "8", "multiline": False}),
                "case_9": ("STRING", {"default": "9", "multiline": False}),
                "input_default": (any,),
            },
            "optional": {
                "input_1": (any,),
                "input_2": (any,),
                "input_3": (any,),
                "input_4": (any,),
                "input_5": (any,),
                "input_6": (any,),
                "input_7": (any,),
                "input_8": (any,),
                "input_9": (any,),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("?",)
    FUNCTION = "switch_case"
    CATEGORY = "ConchShellCopy/Utility"

    def switch_case(
        self,
        switch_condition,
        case_1,
        case_2,
        case_3,
        case_4,
        case_5,
        case_6,
        case_7,
        case_8,
        case_9,
        input_default,
        input_1=None,
        input_2=None,
        input_3=None,
        input_4=None,
        input_5=None,
        input_6=None,
        input_7=None,
        input_8=None,
        input_9=None,
    ):

        output = input_default
        if switch_condition == case_1 and input_1 is not None:
            output = input_1
        elif switch_condition == case_2 and input_2 is not None:
            output = input_2
        elif switch_condition == case_3 and input_3 is not None:
            output = input_3
        elif switch_condition == case_4 and input_4 is not None:
            output = input_4
        elif switch_condition == case_5 and input_5 is not None:
            output = input_5
        elif switch_condition == case_6 and input_6 is not None:
            output = input_6
        elif switch_condition == case_7 and input_7 is not None:
            output = input_7
        elif switch_condition == case_8 and input_8 is not None:
            output = input_8
        elif switch_condition == case_9 and input_9 is not None:
            output = input_9

        return (output,)


NODE_CLASS_MAPPINGS = {"SwitchCaseNodePro": SwitchCaseNodePro}
NODE_DISPLAY_NAME_MAPPINGS = {"SwitchCaseNodePro": "Switch Case Node Pro"}
