import { app } from "../../scripts/app.js";

const NODE_NAME = "BimoImageListMerge";
const MAX_INPUTS = 64;

function updateInputs(node, value) {
    const count = Math.max(1, Math.min(MAX_INPUTS, Math.trunc(Number(value) || 2)));
    // Remove from the end so LiteGraph can correctly disconnect removed slots.
    for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--) {
        const match = /^image_(\d+)$/.exec(node.inputs[i].name);
        if (match && Number(match[1]) > count) node.removeInput(i);
    }
    for (let i = 1; i <= count; i++) {
        if (!node.inputs?.some((input) => input.name === `image_${i}`)) {
            node.addInput(`image_${i}`, "IMAGE");
        }
    }
    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "bimo.imageListMerge",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            const widget = this.widgets?.find((item) => item.name === "input_count");
            if (widget) {
                const callback = widget.callback;
                widget.callback = (...args) => {
                    callback?.apply(widget, args);
                    updateInputs(this, widget.value);
                };
                updateInputs(this, widget.value);
            }
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            const widget = this.widgets?.find((item) => item.name === "input_count");
            if (widget) updateInputs(this, widget.value);
            return result;
        };
    },
});
