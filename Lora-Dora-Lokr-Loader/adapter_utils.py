"""
adapter_utils.py — Adapter type detection and base name extraction.

RULE: detect_adapter_type() is called ONCE per file, before any transform runs.
All subsequent functions check adapter_type before doing anything.
"""

from typing import Dict, Set, Any

# Adapter suffixes that denote each type
_LOKR_KEYS  = {"lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b"}
_LOHA_KEYS  = {"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"}
_DORA_KEYS  = {"dora_scale", "lora_magnitude_vector"}
_LORA_KEYS  = {"lora_up", "lora_down", "lora_A", "lora_B"}

# All known adapter suffixes used for base-name stripping
_ALL_ADAPTER_SUFFIXES = [
    # LoKr
    ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b",
    # LoHa
    ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
    # LoRA / DoRA
    ".lora_up.weight", ".lora_down.weight",
    ".lora_A.weight", ".lora_B.weight",
    ".lora_A.default.weight", ".lora_B.default.weight",
    ".lora_A", ".lora_B",
    "_lora.up.weight", "_lora.down.weight",
    ".lora.up.weight", ".lora.down.weight",
    ".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight",
    # Magnitude / scale
    ".dora_scale", ".lora_magnitude_vector",
    ".w_norm", ".b_norm",
    # Misc
    ".alpha", ".diff", ".diff_b", ".set_weight", ".reshape_weight",
]


def detect_adapter_type(state_dict: Dict[str, Any], sample_n: int = 20) -> str:
    """
    Inspect a sample of keys from a loaded safetensors state_dict and return one of:
      "lokr"    — has lokr_w1 / lokr_w2 / lokr_w1_a / etc.
      "loha"    — has hada_w1_a / hada_w1_b / etc.
      "dora"    — has lora_up/down AND dora_scale or lora_magnitude_vector
      "lora"    — has lora_up/down / lora_A/B only
      "unknown" — fallback

    Runs on a sample of at most `sample_n` keys for speed.
    Detection priority: lokr > loha > dora > lora > unknown
    """
    keys = list(state_dict.keys())

    # Sample: take keys spread evenly across the dict
    if len(keys) > sample_n:
        step = max(1, len(keys) // sample_n)
        sample = [keys[i] for i in range(0, len(keys), step)][:sample_n]
    else:
        sample = keys

    seen_suffixes: Set[str] = set()
    for k in sample:
        # Extract the portion after the last '.'
        parts = str(k).rsplit(".", 1)
        if len(parts) == 2:
            seen_suffixes.add(parts[1])
        # Also check multi-part suffixes like lora_w2_a
        parts2 = str(k).split(".")
        for i in range(len(parts2)):
            seen_suffixes.add(".".join(parts2[i:]))

    # Priority order: lokr > loha > dora > lora
    if seen_suffixes & _LOKR_KEYS:
        return "lokr"
    if seen_suffixes & _LOHA_KEYS:
        return "loha"

    has_lora = bool(seen_suffixes & _LORA_KEYS)
    has_dora = bool(seen_suffixes & _DORA_KEYS)

    # Fall back to scanning all keys for dora_scale when sample missed it
    if has_lora and not has_dora:
        has_dora = any(
            str(k).endswith(".dora_scale") or str(k).endswith(".lora_magnitude_vector")
            for k in keys
        )

    if has_lora and has_dora:
        return "dora"
    if has_lora:
        return "lora"

    # Last resort: scan all keys for lokr/loha markers we might have missed in sample
    for k in keys:
        kl = str(k).lower()
        if any(m in kl for m in ("lokr_w", "hada_w")):
            if "hada" in kl:
                return "loha"
            return "lokr"

    return "unknown"


def get_base_names(state_dict: Dict[str, Any]) -> Set[str]:
    """
    Return the set of unique base module names by stripping known adapter suffixes.

    Example:
      "lycoris_layers_0_cross_attn_k_proj.lokr_w1" -> "lycoris_layers_0_cross_attn_k_proj"
    """
    bases: Set[str] = set()
    for key in state_dict.keys():
        ks = str(key)
        for suffix in _ALL_ADAPTER_SUFFIXES:
            if ks.endswith(suffix):
                bases.add(ks[: -len(suffix)])
                break
    return bases
