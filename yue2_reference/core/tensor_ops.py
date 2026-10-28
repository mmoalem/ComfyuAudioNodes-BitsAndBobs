import math
import torch
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Layer / step / time tapers  (mirrors ace_step_reference.core.tensor_ops)
# ──────────────────────────────────────────────────────────────────────────────

def compute_layer_strength(
    layer_idx: int, start_layer: int, end_layer: int, base_strength: float, taper: str
) -> float:
    if layer_idx < start_layer or layer_idx > end_layer:
        return 0.0
    if taper == "none":
        return base_strength
    total = end_layer - start_layer + 1
    if total <= 1:
        return base_strength
    pos = layer_idx - start_layer
    normalized = pos / (total - 1)
    if taper == "linear":
        factor = normalized
    elif taper == "cosine":
        factor = (1 - math.cos(normalized * math.pi)) / 2
    else:
        factor = normalized
    return base_strength * factor


def compute_step_multiplier(t: float, mode: str) -> float:
    """Scale by diffusion timestep (1.0 at start, 0.0 at end of sampling)."""
    t = max(0.0, min(1.0, float(t)))
    progress = 1.0 - t  # 0 at start, 1 at end
    if mode == "none":
        return 1.0
    elif mode == "fade_out":
        return 1.0 - progress
    elif mode == "fade_in":
        return progress
    elif mode == "cosine_fade_out":
        return (1.0 + math.cos(progress * math.pi)) / 2.0
    elif mode == "cosine_fade_in":
        return (1.0 - math.cos(progress * math.pi)) / 2.0
    elif mode == "cosine_bell":
        return math.sin(progress * math.pi)
    return 1.0


def compute_time_multiplier(seq_len: int, mode: str, device, dtype) -> Tensor:
    """Temporal multiplier of shape [1, 1, seq_len, 1] for K/V scaling."""
    if mode == "none":
        return torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
    t = torch.linspace(0.0, 1.0, seq_len, device=device, dtype=dtype)
    if mode == "fade_out":
        mult = 1.0 - t
    elif mode == "fade_in":
        mult = t
    elif mode == "cosine_fade_out":
        mult = (1.0 + torch.cos(t * math.pi)) / 2.0
    elif mode == "cosine_fade_in":
        mult = (1.0 - torch.cos(t * math.pi)) / 2.0
    elif mode == "cosine_bell":
        mult = torch.sin(t * math.pi)
    else:
        mult = torch.ones_like(t)
    return mult.view(1, 1, seq_len, 1)
