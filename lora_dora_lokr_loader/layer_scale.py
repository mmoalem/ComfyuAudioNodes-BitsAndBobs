"""
layer_scale.py — Per-layer-category strength scaling for ACE-Step adapters.

RULE 3: Layer scaling bakes into patch *strength* (not tensor data) to keep the
downstream ComfyUI application path (add_patches) unchanged.

Classification runs on the MAPPED model key (final weight key), not the raw
LoRA file key. Order matters: cross_attn must be checked before self_attn.
"""

import logging
from typing import Dict, Any, Tuple

_LOG = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# ACE-Step layer pattern definitions
# ──────────────────────────────────────────────────────────────────────────────

# Checked in order. cross_attn MUST come before self_attn because cross-attention
# keys often contain "attn" which would otherwise match self_attn patterns first.
_ACESTEOP_LAYER_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "cross_attn": (
        "cross_attn",
        "cross_attention",
        "attn2",
    ),
    "self_attn": (
        "self_attn",
        "self_attention",
        "attn1",
    ),
    "ffn": (
        ".ff.",
        ".ffn.",
        ".mlp.",
        "mlp_gate_proj",
        "mlp_up_proj",
        "mlp_down_proj",
        "fc1",
        "fc2",
        "linear_in",
        "linear_out",
        "net.0",
        "net.2",
    ),
}


def classify_key(key: str) -> str:
    """
    Classify a patch key into 'cross_attn', 'self_attn', 'ffn', or 'other'.

    Args:
        key: The mapped model weight key (e.g. "diffusion_model.layers.0.cross_attn_k_proj.weight")

    Returns:
        One of "cross_attn", "self_attn", "ffn", "other"
    """
    k = key.lower()
    for label, patterns in _ACESTEOP_LAYER_PATTERNS.items():
        if any(p in k for p in patterns):
            return label
    return "other"


def scale_patches(
    patches: Dict[str, Any],
    self_attn_scale: float,
    cross_attn_scale: float,
    ffn_scale: float,
    other_scale: float,
    lora_name: str = "",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Walk every key in the patches dict, classify it by layer category, and
    multiply the patch strength by the corresponding scale factor.

    ComfyUI patches dict structure (from comfy.lora.load_lora):
      key -> (strength, *adapter_tensors)

    We scale the `strength` scalar (first element of the tuple) rather than
    modifying any tensor data.  This preserves all downstream math in
    ComfyUI's weight_adapter modules unchanged.

    If a scale factor is 1.0, the patch is passed through unchanged.

    Args:
        patches:          The patches dict returned by comfy.lora.load_lora()
        self_attn_scale:  Multiplier for self-attention layers
        cross_attn_scale: Multiplier for cross-attention layers
        ffn_scale:        Multiplier for FFN/MLP layers
        other_scale:      Multiplier for all other layers
        lora_name:        Used in verbose logging only
        verbose:          Print per-category summary when True

    Returns:
        A new patches dict with scaled strengths.
    """
    scale_map = {
        "cross_attn": float(cross_attn_scale),
        "self_attn":  float(self_attn_scale),
        "ffn":        float(ffn_scale),
        "other":      float(other_scale),
    }

    # Fast path: all scales are 1.0, nothing to do
    if all(v == 1.0 for v in scale_map.values()):
        return patches

    scaled: Dict[str, Any] = {}
    counts  = {"cross_attn": 0, "self_attn": 0, "ffn": 0, "other": 0}

    for key, patch in patches.items():
        category = classify_key(str(key))
        factor   = scale_map[category]
        counts[category] += 1

        if factor == 1.0:
            scaled[key] = patch
        elif factor == 0.0:
            # Drop this patch entirely — zeroes its contribution.
            continue
        else:
            # Object or Tuple patch logic...
            try:
                strength = patch[0]
                rest     = patch[1:]
                try:
                    new_strength = float(strength) * factor
                except Exception:
                    import torch
                    new_strength = strength * factor
                scaled[key] = (new_strength,) + rest
            except (TypeError, KeyError):
                import copy, torch
                new_adapter = copy.copy(patch)
                v = list(patch.weights)
                for i, w in enumerate(v):
                    if isinstance(w, torch.Tensor):
                        v[i] = w * factor
                        break
                new_adapter.weights = tuple(v)
                scaled[key] = new_adapter

    # Always log the application summary if we're actually changing anything
    tag = f" [{lora_name}]" if lora_name else ""
    _LOG.info("[ACEStep LayerScale]%s", tag)
    for cat, num in counts.items():
        if num > 0 or scale_map[cat] != 1.0:
            _LOG.info(
                "  %-12s : %3d patches × %.2f%s",
                cat, num, scale_map[cat],
                " (DROPPED)" if scale_map[cat] == 0 else ""
            )

    return scaled

    return scaled
