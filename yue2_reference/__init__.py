from .nodes.kv_inject import YuE2SelfAttentionCapture, YuE2SelfAttentionInject
from .nodes.per_step_inject_per_layer import YuE2PerStepSAInjectPerLayer
from .nodes.per_step_inject_step_limited import YuE2PerStepSAInjectStepLimited


NODE_CLASS_MAPPINGS = {
    "YuE2SelfAttentionCapture": YuE2SelfAttentionCapture,
    "YuE2SelfAttentionInject": YuE2SelfAttentionInject,
    "YuE2PerStepSAInjectPerLayer": YuE2PerStepSAInjectPerLayer,
    "YuE2PerStepSAInjectStepLimited": YuE2PerStepSAInjectStepLimited,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2SelfAttentionCapture": "YuE2 Self-Attention Capture",
    "YuE2SelfAttentionInject": "YuE2 Self-Attention Inject",
    "YuE2PerStepSAInjectPerLayer": "YuE2 Per-Step SA Inject (Per Layer)",
    "YuE2PerStepSAInjectStepLimited": "YuE2 Per-Step SA Inject (step+time taper, step limited)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
