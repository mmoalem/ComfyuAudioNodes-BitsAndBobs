import torch
import torch.nn.functional as F
from torch import Tensor


def merge(primary: Tensor, reference: Tensor, strength: float, mode: str) -> Tensor:
    reference = safe_cast(reference, primary.dtype, primary.device)
    if mode == "concat":
        return merge_concat(primary, reference, strength)
    elif mode == "lerp":
        return merge_lerp(primary, reference, strength)
    elif mode == "add":
        return merge_add(primary, reference, strength)
    elif mode == "replace":
        return merge_replace(primary, reference, strength)
    else:
        raise ValueError(f"Unknown merge mode: {mode}")


def merge_concat(primary: Tensor, reference: Tensor, strength: float) -> Tensor:
    ref_scaled = reference * strength
    return torch.cat([primary, ref_scaled], dim=1)


def merge_lerp(primary: Tensor, reference: Tensor, strength: float) -> Tensor:
    ref_aligned = align_sequence(reference, primary.shape[1], "interpolate")
    return primary * (1 - strength) + ref_aligned * strength


def merge_add(primary: Tensor, reference: Tensor, strength: float) -> Tensor:
    ref_aligned = align_sequence(reference, primary.shape[1], "interpolate")
    return primary + ref_aligned * strength


def merge_replace(primary: Tensor, reference: Tensor, strength: float) -> Tensor:
    """Silences the primary tensor by the strength factor, and adds the raw aligned reference scaled by strength."""
    ref_aligned = align_sequence(reference, primary.shape[1], "interpolate")
    return primary * (1.0 - strength) + ref_aligned * strength


def align_sequence(tensor: Tensor, target_length: int, mode: str) -> Tensor:
    current_length = tensor.shape[1]
    if current_length == target_length:
        return tensor
    if mode == "interpolate":
        if tensor.ndim == 3:
            t = tensor.transpose(1, 2)
            aligned = F.interpolate(
                t, size=target_length, mode="linear", align_corners=False
            )
            return aligned.transpose(1, 2)
        return F.interpolate(
            tensor, size=target_length, mode="linear", align_corners=False
        )
    elif mode == "pad":
        if current_length < target_length:
            pad_size = target_length - current_length
            padding = [0, 0, 0, pad_size] if tensor.ndim == 3 else [0, pad_size]
            return torch.nn.functional.pad(tensor, padding, value=0.0)
        return tensor
    elif mode == "trim":
        return (
            tensor[:, :target_length, :]
            if tensor.ndim == 3
            else tensor[:, :target_length]
        )
    else:
        raise ValueError(f"Unknown align mode: {mode}")


def compute_layer_strength(
    layer_idx: int, start_layer: int, end_layer: int, base_strength: float, taper: str
) -> float:
    if layer_idx < start_layer or layer_idx > end_layer:
        return 0.0
    if taper == "none":
        return base_strength
    total_layers = end_layer - start_layer + 1
    if total_layers <= 1:
        return base_strength
    position = layer_idx - start_layer
    normalized = position / (total_layers - 1)
    if taper == "linear":
        factor = normalized
    elif taper == "cosine":
        import math

        factor = (1 - math.cos(normalized * math.pi)) / 2
    else:
        factor = normalized
    return base_strength * factor


def safe_cast(tensor: Tensor, dtype, device) -> Tensor:
    if tensor.dtype == dtype and tensor.device == device:
        return tensor
    return tensor.to(dtype=dtype, device=device)


def compute_step_multiplier(t: float, mode: str) -> float:
    """Compute a strength multiplier from the current diffusion timestep.

    In ACE-Step's rectified-flow sampling the timestep runs from ~1.0
    (pure noise, first step) down to ~0.0 (clean output, last step).

    Args:
        t:    Current normalised timestep, clamped internally to [0, 1].
        mode: One of:
              "none"             — always 1.0 (no step-based scaling).
              "fade_out"         — linear 1.0 → 0.0  (strong early, gone by end).
              "fade_in"          — linear 0.0 → 1.0  (absent early, full by end).
              "cosine_fade_out"  — smooth cosine 1.0 → 0.0.
              "cosine_fade_in"   — smooth cosine 0.0 → 1.0.
              "cosine_bell"      — 0 → peak at midpoint → 0  (sin envelope).

    Returns:
        Scalar in [0, 1] to multiply against the base injection strength.
    """
    import math
    t = max(0.0, min(1.0, float(t)))   # clamp — schedulers can slightly overshoot
    progress = 1.0 - t                  # 0.0 at start of sampling, 1.0 at end

    if mode == "none":
        return 1.0
    elif mode == "fade_out":
        return t                                            # 1 → 0
    elif mode == "fade_in":
        return progress                                     # 0 → 1
    elif mode == "cosine_fade_out":
        return (1.0 + math.cos(progress * math.pi)) / 2.0  # smooth 1 → 0
    elif mode == "cosine_fade_in":
        return (1.0 - math.cos(progress * math.pi)) / 2.0  # smooth 0 → 1
    elif mode == "cosine_bell":
        return math.sin(progress * math.pi)                 # 0 → 1 → 0
    else:
        return 1.0


def compute_time_multiplier(seq_len: int, mode: str, device, dtype) -> torch.Tensor:
    """Compute a temporal strength multiplier mask across the sequence length.
    
    Returns a tensor of shape [1, 1, seq_len, 1] which broadcasts correctly
    against the [batch, num_heads, seq_len, head_dim] K/V tensors.
    """
    import math
    if mode == "none":
        return torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
    
    # Create normalized timeline from 0.0 (start of audio) to 1.0 (end of audio)
    t = torch.linspace(0.0, 1.0, seq_len, device=device, dtype=dtype)
    
    if mode == "fade_out":
        mult = 1.0 - t                                    # 1 → 0
    elif mode == "fade_in":
        mult = t                                          # 0 → 1
    elif mode == "cosine_fade_out":
        mult = (1.0 + torch.cos(t * math.pi)) / 2.0       # smooth 1 → 0
    elif mode == "cosine_fade_in":
        mult = (1.0 - torch.cos(t * math.pi)) / 2.0       # smooth 0 → 1
    elif mode == "cosine_bell":
        mult = torch.sin(t * math.pi)                     # 0 → 1 → 0
    else:
        mult = torch.ones_like(t)
        
    return mult.view(1, 1, seq_len, 1)
