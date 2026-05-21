from .nodes.timbre_encode import AudioTimbreEncode
from .nodes.conditioning_inject import TimbreConditioningInject
from .nodes.kv_capture import SelfAttentionCapture
from .nodes.kv_inject import SelfAttentionInject
from .nodes.per_step_inject import PerStepSelfAttentionInject
from .nodes.per_step_inject_per_layer import PerStepSelfAttentionInjectPerLayer
from .nodes.inspector import ReferenceInspector

NODE_CLASS_MAPPINGS = {
    "AudioTimbreEncode": AudioTimbreEncode,
    "TimbreConditioningInject": TimbreConditioningInject,
    "SelfAttentionCapture": SelfAttentionCapture,
    "SelfAttentionInject": SelfAttentionInject,
    "PerStepSelfAttentionInject": PerStepSelfAttentionInject,
    "PerStepSelfAttentionInjectPerLayer": PerStepSelfAttentionInjectPerLayer,
    "ReferenceInspector": ReferenceInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioTimbreEncode": "Audio Timbre Encode",
    "TimbreConditioningInject": "Timbre Conditioning Inject",
    "SelfAttentionCapture": "Self-Attention Capture",
    "SelfAttentionInject": "Self-Attention Inject",
    "PerStepSelfAttentionInject": "Per-Step Self-Attention Inject",
    "PerStepSelfAttentionInjectPerLayer": "Per-Step Self-Attention Inject (Per Layer)",
    "ReferenceInspector": "Reference Inspector",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

