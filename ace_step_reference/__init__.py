from .nodes.timbre_encode import AudioTimbreEncode
from .nodes.conditioning_inject import TimbreConditioningInject
from .nodes.kv_capture import SelfAttentionCapture
from .nodes.kv_inject import SelfAttentionInject
from .nodes.per_step_inject import PerStepSelfAttentionInject
from .nodes.per_step_inject_per_layer import PerStepSelfAttentionInjectPerLayer
from .nodes.per_step_inject_per_layer_step_limited import PerStepSelfAttentionInjectPerLayerStepLimited
from .nodes.per_step_cross_attn_inject_per_layer import PerStepCrossAttentionInjectPerLayer
from .nodes.inspector import ReferenceInspector
from .nodes.debug_inject import DebugPerStepSAInject
from .nodes.per_step_self_attn_inject_legacy import PerStepSelfAttentionInjectLegacy
from .nodes.per_step_self_attn_inject_legacy2 import PerStepSelfAttentionInjectLegacy2
from .nodes.per_step_self_attn_inject_legacy2_step_limited import PerStepSelfAttentionInjectLegacy2StepLimited


NODE_CLASS_MAPPINGS = {
    "AudioTimbreEncode": AudioTimbreEncode,
    "TimbreConditioningInject": TimbreConditioningInject,
    "SelfAttentionCapture": SelfAttentionCapture,
    "SelfAttentionInject": SelfAttentionInject,
    "PerStepSelfAttentionInject": PerStepSelfAttentionInject,
    "PerStepSelfAttentionInjectPerLayer": PerStepSelfAttentionInjectPerLayer,
    "PerStepSelfAttentionInjectPerLayerStepLimited": PerStepSelfAttentionInjectPerLayerStepLimited,
    "PerStepCrossAttentionInjectPerLayer": PerStepCrossAttentionInjectPerLayer,
    "ReferenceInspector": ReferenceInspector,
    "DebugPerStepSAInject": DebugPerStepSAInject,
    "PerStepSelfAttentionInjectLegacy": PerStepSelfAttentionInjectLegacy,
    "PerStepSelfAttentionInjectLegacy2": PerStepSelfAttentionInjectLegacy2,
    "PerStepSelfAttentionInjectLegacy2StepLimited": PerStepSelfAttentionInjectLegacy2StepLimited,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioTimbreEncode": "Audio Timbre Encode",
    "TimbreConditioningInject": "Timbre Conditioning Inject",
    "SelfAttentionCapture": "Self-Attention Capture",
    "SelfAttentionInject": "Self-Attention Inject",
    "PerStepSelfAttentionInject": "Per-Step Self-Attention Inject",
    "PerStepSelfAttentionInjectPerLayer": "Per-Step Self-Attention Inject (Per Layer)",
    "PerStepSelfAttentionInjectPerLayerStepLimited": "Per-Step Self-Attention Inject (Per Layer, Step Limited)",
    "PerStepCrossAttentionInjectPerLayer": "Per-Step Cross-Attention Inject (Per Layer)",
    "ReferenceInspector": "Reference Inspector",
    "PerStepSelfAttentionInjectLegacy": "Per Step Self Attention Inject (Legacy - pre per-layer)",
    "PerStepSelfAttentionInjectLegacy2": "Per Step Self Attention Inject (Legacy 2 - step+time taper)",
    "PerStepSelfAttentionInjectLegacy2StepLimited": "Per Step Self Attention Inject (step+time taper, step limited)",
}



__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

