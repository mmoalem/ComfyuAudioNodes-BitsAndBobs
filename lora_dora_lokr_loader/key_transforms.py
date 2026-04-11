"""
key_transforms.py — All key-remapping functions for the ACEStep Universal Adapter Loader.

CRITICAL RULE: Every function MUST be gated on adapter_type.
Never run a LoRA/DoRA-specific fix on a LoKr or LoHa file, and vice versa.
The adapter_type is determined ONCE at load time before any transform runs.

ACE-Step LoKr/LoHa key format (confirmed from safetensors inspection):
  lycoris_condition_embedder.lokr_w1
  lycoris_layers_N_cross_attn_k_proj.lokr_w1
  lycoris_layers_N_self_attn_q_proj.lokr_w2
  lycoris_layers_N_mlp_gate_proj.lokr_w2_a
  lycoris_layers_N_mlp_gate_proj.lokr_w2_b
  lycoris_proj_in_1.lokr_w1
  lycoris_time_embed_linear_1.lokr_w1
  lycoris_time_embed_time_proj.lokr_w1

Transform rule:
  lycoris_layers_N_<rest>  ->  diffusion_model.layers.N.<rest>
  lycoris_<other>          ->  diffusion_model.<other>
  (underscores in <rest> kept — they are part of the module's own name)
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

_LOG = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# ACE-Step lycoris prefix normalisation
# ──────────────────────────────────────────────────────────────────────────────

# Matches: lycoris_layers_<N>_<rest>
_LYCORIS_LAYER_RE = re.compile(r"^lycoris_layers_(\d+)_(.+?)(\..+)$")
# Matches: lycoris_<anything_not_layers>
_LYCORIS_OTHER_RE = re.compile(r"^lycoris_(.+?)(\..+)$")

# Known ACEStep DiT nested sub-modules inside each DiT layer.
# Ordered longest-first to prevent "self_attn" matching "self_attn_norm".
_ACESTEP_LAYER_SUBMODULES = [
    "cross_attn_norm",
    "self_attn_norm",
    "mlp_norm",
    "cross_attn",
    "self_attn",
    "mlp",
]


def _acesteop_lycoris_base_to_diffusion(base: str) -> Optional[str]:
    """
    Convert a lycoris_ base name to its diffusion_model.* equivalent.

    ACEStep15 DiT layer structure uses nested sub-modules:
      layers.N.cross_attn   (AceStepAttention with .q_proj, .k_proj, .v_proj, .o_proj)
      layers.N.self_attn    (AceStepAttention with .q_proj, .k_proj, .v_proj, .o_proj)
      layers.N.mlp          (MLP with .gate_proj, .up_proj, .down_proj)
      layers.N.self_attn_norm, .cross_attn_norm, .mlp_norm, .scale_shift_table

    Examples:
      lycoris_layers_0_cross_attn_k_proj  ->  diffusion_model.layers.0.cross_attn.k_proj
      lycoris_layers_0_self_attn_q_proj   ->  diffusion_model.layers.0.self_attn.q_proj
      lycoris_layers_0_mlp_gate_proj      ->  diffusion_model.layers.0.mlp.gate_proj
      lycoris_condition_embedder          ->  diffusion_model.condition_embedder
      lycoris_proj_in_1                   ->  diffusion_model.proj_in_1
    """
    if not base.startswith("lycoris_"):
        return None

    rest = base[len("lycoris_"):]

    # Handle layers_N_<module>
    m = re.match(r"^layers_(\d+)_(.+)$", rest)
    if m:
        layer_idx = m.group(1)
        module_rest = m.group(2)

        # Try matching known sub-module names to insert dot separator correctly.
        for sub in _ACESTEP_LAYER_SUBMODULES:
            sub_prefix = sub + "_"
            if module_rest.startswith(sub_prefix):
                leaf = module_rest[len(sub_prefix):]  # e.g. "k_proj", "gate_proj"
                return f"diffusion_model.layers.{layer_idx}.{sub}.{leaf}"
            if module_rest == sub:
                # Exact match: the whole thing IS the sub-module (no leaf)
                return f"diffusion_model.layers.{layer_idx}.{sub}"

        # Anything not matching a known sub-module (scale_shift_table, etc.)
        return f"diffusion_model.layers.{layer_idx}.{module_rest}"

    # Anything else: condition_embedder, proj_in_1, time_embed_*, etc.
    return f"diffusion_model.{rest}"


def normalize_acesteop_lycoris_keys(
    sd: Dict[str, Any],
    adapter_type: str,
    verbose: bool = False,
) -> int:
    """
    Rename lycoris_* prefixed keys to the diffusion_model.* format that ComfyUI
    expects for ACE-Step models.

    Runs for adapter_type in ("lokr", "loha"). No-op for lora/dora/unknown.

    Returns number of keys renamed.
    """
    if adapter_type not in ("lokr", "loha"):
        return 0

    renamed = 0
    collisions = 0
    examples: List[str] = []

    for key in list(sd.keys()):
        ks = str(key)
        if not ks.startswith("lycoris_"):
            continue

        # Split base from adapter suffix (everything after the *first* adapter dot)
        # Example: "lycoris_layers_0_cross_attn_k_proj.lokr_w1"
        #   base_part = "lycoris_layers_0_cross_attn_k_proj"
        #   suf_part  = ".lokr_w1"
        dot_idx = ks.index(".")
        base_part = ks[:dot_idx]
        suf_part = ks[dot_idx:]   # includes the leading dot

        new_base = _acesteop_lycoris_base_to_diffusion(base_part)
        if new_base is None:
            continue

        new_key = new_base + suf_part

        if new_key == ks:
            continue

        if new_key in sd:
            collisions += 1
            if len(examples) < 5:
                examples.append(f"COLLISION {ks} -> {new_key}")
            sd.pop(key, None)
            continue

        sd[new_key] = sd.pop(key)
        renamed += 1
        if len(examples) < 10:
            examples.append(f"{ks} -> {new_key}")

    if verbose and (renamed or collisions):
        _LOG.info(
            "[ACEStep Adapter Loader] lycoris key normalize: renamed=%d collisions=%d adapter_type=%s",
            renamed, collisions, adapter_type,
        )
        for ex in examples:
            _LOG.info("[ACEStep Adapter Loader]   %s", ex)

    return renamed


# ──────────────────────────────────────────────────────────────────────────────
# LoRA / DoRA key normalisations (gated: lora and dora only)
# ──────────────────────────────────────────────────────────────────────────────

import re as _re

_LORA_MAG_VECTOR_RE = _re.compile(
    r"^(?P<base>.+?)\.lora_magnitude_vector(?:\.(?P<adapter>[A-Za-z0-9_-]+))?(?:\.weight)?$"
)


def normalize_dora_magnitude_keys(
    sd: Dict[str, Any],
    adapter_type: str,
    verbose: bool = False,
) -> int:
    """
    Rename lora_magnitude_vector -> dora_scale.
    Only runs when adapter_type == "dora".
    """
    if adapter_type != "dora":
        return 0

    renamed = 0
    for key in list(sd.keys()):
        m = _LORA_MAG_VECTOR_RE.match(str(key))
        if not m:
            continue
        new_key = m.group("base") + ".dora_scale"
        if new_key in sd or new_key == key:
            sd.pop(key, None)
            continue
        sd[new_key] = sd.pop(key)
        renamed += 1

    if verbose and renamed:
        _LOG.info(
            "[ACEStep Adapter Loader] dora magnitude key normalize: renamed=%d", renamed
        )
    return renamed


_PEFT_RENAME_MAP = {
    ".lora_A.weight": ".lora_down.weight",
    ".lora_B.weight": ".lora_up.weight",
    ".lora_A.default.weight": ".lora_down.weight",
    ".lora_B.default.weight": ".lora_up.weight",
    ".lora_A": ".lora_down.weight",
    ".lora_B": ".lora_up.weight",
}


def normalize_diffusers_peft_keys(
    sd: Dict[str, Any],
    adapter_type: str,
    verbose: bool = False,
) -> int:
    """
    Rename lora_A / lora_B -> lora_down / lora_up (PEFT/Diffusers convention).
    Only runs when adapter_type in ("lora", "dora").
    """
    if adapter_type not in ("lora", "dora"):
        return 0

    renamed = 0
    for key in list(sd.keys()):
        ks = str(key)
        for old_suf, new_suf in _PEFT_RENAME_MAP.items():
            if ks.endswith(old_suf):
                new_key = ks[: -len(old_suf)] + new_suf
                if new_key not in sd and new_key != ks:
                    sd[new_key] = sd.pop(key)
                    renamed += 1
                break

    if verbose and renamed:
        _LOG.info(
            "[ACEStep Adapter Loader] PEFT key normalize: renamed=%d adapter_type=%s",
            renamed, adapter_type,
        )
    return renamed
