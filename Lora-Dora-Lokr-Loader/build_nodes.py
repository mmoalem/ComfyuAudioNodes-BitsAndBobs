"""
build_nodes.py — Patch script for ACEStep Universal Adapter Loader.

Downloads the upstream nodes.py from xmarre/ComfyUI-DoRA-Dynamic-LoRA-Loader,
applies all our modifications, and writes the result to nodes.py in this directory.

Run once during development setup, or whenever the upstream changes.
Usage:  python build_nodes.py
"""

import re
import sys
import textwrap
import urllib.request
from pathlib import Path

UPSTREAM_URL = (
    "https://raw.githubusercontent.com/xmarre/ComfyUI-DoRA-Dynamic-LoRA-Loader/main/nodes.py"
)
OUTPUT_FILE = Path(__file__).parent / "nodes.py"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_upstream(url: str) -> str:
    print(f"Fetching upstream from {url} ...")
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def patch(src: str, find: str, replace: str, *, count: int = 1, label: str = "") -> str:
    """String patch with occurrence count check."""
    n = src.count(find)
    if n == 0:
        raise RuntimeError(f"PATCH FAILED [{label}]: target string not found:\n{find[:120]!r}")
    if count != -1 and n != count:
        raise RuntimeError(
            f"PATCH FAILED [{label}]: expected {count} occurrence(s), found {n}:\n{find[:120]!r}"
        )
    result = src.replace(find, replace, 1 if count == 1 else (count if count > 0 else 0))
    print(f"  OK  {label or 'patch'}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Patch definitions
# ─────────────────────────────────────────────────────────────────────────────

# --- Patch 1: Insert ACEStep imports + 5 critical rules after `import torch` ---
P1_FIND = "import torch\n"

P1_REPLACE = """\
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# ACEStep Universal Adapter Loader — added on top of DoRA Power LoRA Loader fork
# RULE 1: adapter_type is detected ONCE per file, before any transform.
#          Every subsequent function checks adapter_type before running.
# RULE 2: The DoRA monkey-patch on weight_decompose must be a no-op when
#          dora_scale is None.  LoKr and LoHa never have dora_scale.
# RULE 3: Layer scaling bakes into patch *strength* (not tensor data) to keep
#          the downstream ComfyUI application path unchanged.
# RULE 4: Flux-specific transforms (adaLN, ZiT QKV, OneTrainer broadcast)
#          only run when _is_flux_model() returns True.
# RULE 5: Auto-strength skips LoKr and LoHa rather than computing wrong ratios.
# ═══════════════════════════════════════════════════════════════════════════════

from .adapter_utils import detect_adapter_type
from .key_transforms import (
    normalize_acesteop_lycoris_keys,
    normalize_dora_magnitude_keys as _kt_normalize_dora_magnitude_keys,
    normalize_diffusers_peft_keys,
)
from .layer_scale import scale_patches

"""


# --- Patch 2: Make weight_decompose a no-op for non-DoRA (LoKr / LoHa) ---
# Target: the TypeError raise when dora_scale is None inside weight_decompose_fixed
P2_FIND = (
    "        if dora_scale is None or weight is None or lora_diff is None or alpha is None:\n"
    "            raise TypeError(\"weight_decompose_fixed missing required arguments (dora_scale, weight, lora_diff, alpha)\")\n"
)
P2_REPLACE = (
    "        # RULE 2: no-op for LoKr / LoHa — they never have dora_scale.\n"
    "        if dora_scale is None:\n"
    "            return wa_base._dora_weight_decompose_orig_by_dora_loader(*args, **kwargs)\n"
    "        if weight is None or lora_diff is None or alpha is None:\n"
    "            raise TypeError(\"weight_decompose_fixed missing required arguments (weight, lora_diff, alpha)\")\n"
)


# --- Patch 3: Add _is_flux_model() helper after the two patch calls ---
P3_FIND = "_patch_comfy_lora_calculate_weight_fp32()\n"
P3_REPLACE = """\
_patch_comfy_lora_calculate_weight_fp32()


def _is_flux_model(model: object) -> bool:
    \"\"\"
    Return True when the model is a Flux / Flux2 architecture.
    Used to gate Flux-specific transforms (RULE 4).
    \"\"\"
    try:
        core = getattr(model, "model", model)
        return "flux" in type(core).__name__.lower()
    except Exception:
        return False

"""


# --- Patch 4: Extend _load_one() signature with layer-scale params ---
# The method signature ends with `auto_strength_ratio_ceiling: float,`
P4_FIND = (
    "        auto_strength_ratio_ceiling: float,\n"
    "    ):\n"
    "        auto_strength_report: Optional[Dict[str, Any]] = None\n"
)
P4_REPLACE = (
    "        auto_strength_ratio_ceiling: float,\n"
    "        # Layer-scale params (RULE 3)\n"
    "        layer_scale_enabled: bool = False,\n"
    "        self_attn_scale: float = 1.0,\n"
    "        cross_attn_scale: float = 1.0,\n"
    "        ffn_scale: float = 1.0,\n"
    "        other_scale: float = 1.0,\n"
    "    ):\n"
    "        auto_strength_report: Optional[Dict[str, Any]] = None\n"
)


# --- Patch 5: Add adapter detection + lycoris key normalize after lora_sd is settled ---
# Target: the line that calls _normalize_diffusers_dora_magnitude_keys
P5_FIND = "        _normalize_diffusers_dora_magnitude_keys(lora_sd, verbose=verbose)\n"
P5_REPLACE = """\
        # ── RULE 1: detect adapter type once, before any transform ──────────
        adapter_type = detect_adapter_type(lora_sd)
        if verbose:
            _LOG.info(
                "[ACEStep Adapter Loader] %s: detected adapter_type=%s",
                lora_name,
                adapter_type,
            )

        # ── RULE 4 gate: ACE-Step lycoris_ prefix → diffusion_model.* ───────
        # Only LoKr and LoHa files use the lycoris_ prefix in ACE-Step.
        n_lycoris = normalize_acesteop_lycoris_keys(lora_sd, adapter_type, verbose=verbose)
        if verbose and n_lycoris:
            _LOG.info(
                "[ACEStep Adapter Loader] %s: lycoris keys renamed: %d",
                lora_name,
                n_lycoris,
            )

        # ── Normalize PEFT / Diffusers LoRA keys (lora_A/B → down/up) ───────
        # Only for lora/dora; no-op for lokr/loha (gated inside the function).
        normalize_diffusers_peft_keys(lora_sd, adapter_type, verbose=verbose)

        _normalize_diffusers_dora_magnitude_keys(lora_sd, verbose=verbose)
"""


# --- Patch 6: Gate ZiT/Lumina2 compat behind _is_flux_model() ---
P6_FIND = (
    "        if zimage_lumina2_compat and model is not None:\n"
    "            _apply_zimage_lumina2_compat(\n"
    "                lora_sd=lora_sd,\n"
    "                model=model,\n"
    "                model_sd_keys=model_sd_keys,\n"
    "                key_map=key_map,\n"
    "                verbose=verbose,\n"
    "            )\n"
)
P6_REPLACE = (
    "        # RULE 4: ZiT/Lumina2 compat only applies to Flux / Lumina2 models.\n"
    "        if zimage_lumina2_compat and model is not None and _is_flux_model(model):\n"
    "            _apply_zimage_lumina2_compat(\n"
    "                lora_sd=lora_sd,\n"
    "                model=model,\n"
    "                model_sd_keys=model_sd_keys,\n"
    "                key_map=key_map,\n"
    "                verbose=verbose,\n"
    "            )\n"
)


# --- Patch 7: Gate Flux2/OneTrainer DoRA compat behind _is_flux_model() ---
P7_FIND = (
    "        auto_strength_logical_groups: Dict[str, Tuple[str, float]] = {}\n"
    "        if model is not None:\n"
    "            _apply_flux2_onetrainer_dora_compat(\n"
)
P7_REPLACE = (
    "        auto_strength_logical_groups: Dict[str, Tuple[str, float]] = {}\n"
    "        # RULE 4: Flux2 / OneTrainer broadcast only for Flux models.\n"
    "        if model is not None and _is_flux_model(model):\n"
    "            _apply_flux2_onetrainer_dora_compat(\n"
)


# --- Patch 8: Gate output-axis fix on adapter_type ---
P8_FIND = (
    "        _fix_onetrainer_output_axis_dora_mats(\n"
    "            lora_sd=lora_sd,\n"
    "            key_map=key_map,\n"
    "            model_state_dict=model_state_dict,\n"
    "            clip_state_dict=clip_state_dict,\n"
    "            verbose=verbose,\n"
    "        )\n"
)
P8_REPLACE = (
    "        # RULE 1/2: OneTrainer output-axis fix only applies to LoRA/DoRA.\n"
    "        if adapter_type in (\"lora\", \"dora\"):\n"
    "            _fix_onetrainer_output_axis_dora_mats(\n"
    "                lora_sd=lora_sd,\n"
    "                key_map=key_map,\n"
    "                model_state_dict=model_state_dict,\n"
    "                clip_state_dict=clip_state_dict,\n"
    "                verbose=verbose,\n"
    "            )\n"
)


# --- Patch 9: Gate auto-strength on adapter_type (skip for lokr/loha) ---
P9_FIND = (
    "        if auto_strength_enabled and (abs(float(strength_model)) > _AUTO_STRENGTH_EPS"
    " or abs(float(strength_clip)) > _AUTO_STRENGTH_EPS):\n"
)
P9_REPLACE = (
    "        # RULE 5: skip auto-strength for LoKr and LoHa (would compute garbage ratios).\n"
    "        if auto_strength_enabled and adapter_type in (\"lokr\", \"loha\"):\n"
    "            if verbose:\n"
    "                _LOG.info(\n"
    "                    \"[ACEStep Adapter Loader] %s: auto-strength skipped for adapter_type=%s\",\n"
    "                    lora_name,\n"
    "                    adapter_type,\n"
    "                )\n"
    "            auto_strength_enabled = False  # disable for the rest of _load_one\n"
    "\n"
    "        if auto_strength_enabled and (abs(float(strength_model)) > _AUTO_STRENGTH_EPS"
    " or abs(float(strength_clip)) > _AUTO_STRENGTH_EPS):\n"
)


# --- Patch 10: Add layer scaling after load_lora() ---
P10_FIND = "        _log_loaded_tensor_health(lora_name, loaded, verbose=verbose)\n"
P10_REPLACE = """\
        _log_loaded_tensor_health(lora_name, loaded, verbose=verbose)

        # RULE 3: per-layer-category strength scaling (bakes into strength scalar).
        if layer_scale_enabled:
            loaded = scale_patches(
                loaded,
                self_attn_scale=self_attn_scale,
                cross_attn_scale=cross_attn_scale,
                ffn_scale=ffn_scale,
                other_scale=other_scale,
                lora_name=lora_name,
                verbose=verbose,
            )
"""


# --- Patch 11: Add layer_scale params to INPUT_TYPES ---
P11_FIND = (
    "                    # Optional per-base auto-strength redistribution\n"
    "                    \"auto_strength_enabled\": (\"BOOLEAN\", {\"default\": False}),\n"
)
P11_REPLACE = """\
                    # Per-layer category scaling (RULE 3)
                    "layer_scale_enabled": ("BOOLEAN", {"default": False}),
                    "self_attn_scale": ("FLOAT", {
                        "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                        "tooltip": "Strength multiplier for self-attention layers",
                    }),
                    "cross_attn_scale": ("FLOAT", {
                        "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                        "tooltip": "Strength multiplier for cross-attention layers",
                    }),
                    "ffn_scale": ("FLOAT", {
                        "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                        "tooltip": "Strength multiplier for FFN/MLP layers",
                    }),
                    "other_scale": ("FLOAT", {
                        "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                        "tooltip": "Strength multiplier for projection and other layers",
                    }),

                    # Verbose / debug
                    "verbose": ("BOOLEAN", {"default": False}),
                    "log_unloaded_keys": ("BOOLEAN", {"default": False}),

                    # Optional per-base auto-strength redistribution
                    "auto_strength_enabled": ("BOOLEAN", {"default": False}),
"""


# --- Patch 12: Make clip optional (ACE-Step has no CLIP) ---
P12_FIND = (
    "            \"required\": {\n"
    "                \"model\": (\"MODEL\",),\n"
    "                \"clip\": (\"CLIP\",),\n"
    "            },\n"
)
P12_REPLACE = (
    "            \"required\": {\n"
    "                \"model\": (\"MODEL\",),\n"
    "            },\n"
    "            # clip is optional — ACE-Step models have no CLIP text encoder.\n"
    "            # Connect it only when using CLIP-based models (SD, Flux, etc.).\n"
    "            # When None, clip patches are skipped silently.\n"
)


# --- Patch 13: Extract layer_scale kwargs in load_loras() ---
# Target: the block that reads auto_strength_enabled
P13_FIND = (
    "        auto_strength_enabled = bool(kwargs.get(\"auto_strength_enabled\", False))\n"
)
P13_REPLACE = (
    "        auto_strength_enabled = bool(kwargs.get(\"auto_strength_enabled\", False))\n"
    "        layer_scale_enabled = bool(kwargs.get(\"layer_scale_enabled\", False))\n"
    "        try:\n"
    "            self_attn_scale  = float(kwargs.get(\"self_attn_scale\", 1.0))\n"
    "            cross_attn_scale = float(kwargs.get(\"cross_attn_scale\", 1.0))\n"
    "            ffn_scale        = float(kwargs.get(\"ffn_scale\", 1.0))\n"
    "            other_scale      = float(kwargs.get(\"other_scale\", 1.0))\n"
    "        except Exception:\n"
    "            self_attn_scale = cross_attn_scale = ffn_scale = other_scale = 1.0\n"
)


# --- Patch 14: Forward layer_scale params into _load_one() call ---
P14_FIND = (
    "                auto_strength_ratio_ceiling=auto_strength_ratio_ceiling,\n"
    "            )\n"
    "            did_analyze = isinstance(auto_strength_report, dict)\n"
)
P14_REPLACE = (
    "                auto_strength_ratio_ceiling=auto_strength_ratio_ceiling,\n"
    "                layer_scale_enabled=layer_scale_enabled,\n"
    "                self_attn_scale=self_attn_scale,\n"
    "                cross_attn_scale=cross_attn_scale,\n"
    "                ffn_scale=ffn_scale,\n"
    "                other_scale=other_scale,\n"
    "            )\n"
    "            did_analyze = isinstance(auto_strength_report, dict)\n"
)


# --- Patch 15: Rename class DoraPowerLoraLoader -> ACEStepUniversalAdapterLoader ---
P15_FIND = "class DoraPowerLoraLoader:\n"
P15_REPLACE = "class ACEStepUniversalAdapterLoader:\n"


# --- Patch 16: Remove CLIP from RETURN_TYPES and RETURN_NAMES ---
P16_FIND = (
    "    RETURN_TYPES = (\"MODEL\", \"CLIP\", \"STRING\", \"STRING\")\n"
    "    RETURN_NAMES = (\"MODEL\", \"CLIP\", \"auto_strength_report_json\", \"analysis_report\")\n"
)
P16_REPLACE = (
    "    RETURN_TYPES = (\"MODEL\", \"STRING\", \"STRING\")\n"
    "    RETURN_NAMES = (\"MODEL\", \"auto_strength_report_json\", \"analysis_report\")\n"
)


# --- Patch 17: Remove CLIP from load_loras signature and _load_one return ---
P17_FIND = "        return model, clip, auto_strength_report\n\n    def load_loras(self, model, clip, **kwargs):\n"
P17_REPLACE = "        return model, auto_strength_report\n\n    def load_loras(self, model, clip=None, **kwargs):\n"


# --- Patch 18: Remove CLIP from the two early-exit result tuples (model, clip, ...) ---
P18_FIND  = '"result": (model, clip, report_json, report_text),'
P18_REPLACE = '"result": (model, report_json, report_text),'


# --- Patch 19: Remove CLIP from the final result tuple + fix _load_one return unpack ---
P19_FIND = (
    "            new_model, new_clip, auto_strength_report = self._load_one(\n"
)
P19_REPLACE = (
    "            new_model, auto_strength_report = self._load_one(\n"
)


# --- Patch 21: Update _BASE_SUFFIXES to include LoKr/LoHa ---
P21_FIND = (
    "    \".set_weight\",\n"
    "    \".reshape_weight\",\n"
    "]"
)
P21_REPLACE = (
    "    \".set_weight\",\n"
    "    \".reshape_weight\",\n"
    "    # LoKr\n"
    "    \".lokr_w1\", \".lokr_w2\", \".lokr_w1_a\", \".lokr_w1_b\", \".lokr_w2_a\", \".lokr_w2_b\",\n"
    "    \".lokr_w1.weight\", \".lokr_w2.weight\", \".lokr_w1_a.weight\", \".lokr_w1_b.weight\", \".lokr_w2_a.weight\", \".lokr_w2_b.weight\",\n"
    "    # LoHa\n"
    "    \".hada_w1_a\", \".hada_w1_b\", \".hada_w2_a\", \".hada_w2_b\",\n"
    "    \".hada_w1_a.weight\", \".hada_w1_b.weight\", \".hada_w2_a.weight\", \".hada_w2_b.weight\",\n"
    "]"
)


# --- Patch 22: Add always-on base/mapping telemetry to _load_one ---
P22_FIND = (
    "        # Extract base module names from file keys (after compat rewrites/broadcast).\n"
    "        lora_bases = _extract_lora_bases(lora_sd.keys())\n"
    "        if verbose:\n"
    "            _LOG.info(\"[DoRA Power LoRA Loader] %s: bases in file: %s\", lora_name, len(lora_bases))\n"
)
P22_REPLACE = (
    "        # Extract base module names from file keys (after compat rewrites/broadcast).\n"
    "        lora_bases = _extract_lora_bases(lora_sd.keys())\n"
    "        # Always log base count for ACEStep troubleshooting\n"
    "        _LOG.info(\"[ACEStep Adapter Loader] %s: bases in file: %s\", lora_name, len(lora_bases))\n"
)

P23_FIND = (
    "        if verbose:\n"
    "            _LOG.info(\n"
    "                \"[DoRA Power LoRA Loader] %s: dynamic mappings added: %s, unresolved: %s\",\n"
    "                lora_name,\n"
    "                added,\n"
    "                len(unresolved),\n"
    "            )\n"
)

P23_REPLACE = (
    "        # Always log mapping success for ACEStep troubleshooting\n"
    "        _LOG.info(\n"
    "            \"[ACEStep Adapter Loader] %s: dynamic mappings added: %s, unresolved: %s\",\n"
    "            lora_name,\n"
    "            added,\n"
    "            len(unresolved),\n"
    "        )\n"
)


# --- Patch 20: Append ACEStepUniversalAdapterLoaderSimple subclass ---
P20_APPEND = """

class ACEStepUniversalAdapterLoaderSimple(ACEStepUniversalAdapterLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return ACEStepUniversalAdapterLoader.INPUT_TYPES()

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_loras"
    CATEGORY = "ACE-Step"

    def load_loras(self, model, clip=None, **kwargs):
        # Delegate to the advanced node logic
        res = super().load_loras(model, clip, **kwargs)
        # Strip out the 'ui' element to prevent the visualizer from triggering
        if "ui" in res:
            del res["ui"]
        # Reduce the outputs to strictly the model
        result_tuple = res["result"]
        res["result"] = (result_tuple[0],)
        return res
"""

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    src = fetch_upstream(UPSTREAM_URL)
    print(f"Upstream: {len(src)} chars, {src.count(chr(10))} lines")

    patches = [
        (P1_FIND,  P1_REPLACE,  "imports + critical rules"),
        (P2_FIND,  P2_REPLACE,  "weight_decompose no-op for non-DoRA"),
        (P3_FIND,  P3_REPLACE,  "_is_flux_model() helper"),
        (P4_FIND,  P4_REPLACE,  "_load_one() layer-scale signature"),
        (P5_FIND,  P5_REPLACE,  "adapter detection + lycoris normalize"),
        (P6_FIND,  P6_REPLACE,  "gate ZiT/Lumina2 on _is_flux_model"),
        (P7_FIND,  P7_REPLACE,  "gate Flux2/OneTrainer broadcast on _is_flux_model"),
        (P8_FIND,  P8_REPLACE,  "gate output-axis fix on adapter_type"),
        (P9_FIND,  P9_REPLACE,  "gate auto-strength skip for lokr/loha"),
        (P10_FIND, P10_REPLACE, "layer scaling after load_lora()"),
        (P11_FIND, P11_REPLACE, "layer_scale params in INPUT_TYPES"),
        (P12_FIND, P12_REPLACE, "clip optional in INPUT_TYPES"),
        (P13_FIND, P13_REPLACE, "extract layer_scale kwargs in load_loras()"),
        (P14_FIND, P14_REPLACE, "forward layer_scale params to _load_one()"),
        (P15_FIND, P15_REPLACE, "rename class to ACEStepUniversalAdapterLoader"),
        (P16_FIND, P16_REPLACE, "remove CLIP from RETURN_TYPES/RETURN_NAMES"),
        (P17_FIND, P17_REPLACE, "remove CLIP from load_loras signature and _load_one return"),
        (P18_FIND, P18_REPLACE, "remove CLIP from model/clip early-exit result tuples", 2),
        (P19_FIND, P19_REPLACE, "remove CLIP from final new_model result tuple + _load_one unpack"),
        (P21_FIND, P21_REPLACE, "update _BASE_SUFFIXES for LoKr/LoHa"),
        (P22_FIND, P22_REPLACE, "always-on base-count telemetry"),
        (P23_FIND, P23_REPLACE, "always-on mapping telemetry"),
    ]

    print("\nApplying patches:")
    for item in patches:
        find = item[0]
        replace = item[1]
        label = item[2]
        count = item[3] if len(item) > 3 else 1
        src = patch(src, find, replace, count=count, label=label)

    print("\nApplying custom appended classes...")
    src += P20_APPEND

    # Verify the result parses as valid Python
    import ast
    try:
        ast.parse(src)
        print("\nOK AST parse OK")
    except SyntaxError as e:
        print(f"\nERROR SyntaxError: {e}")
        print("  Writing anyway — check the output file manually.")

    OUTPUT_FILE.write_text(src, encoding="utf-8")
    print(f"\nOK Written to {OUTPUT_FILE}")
    print(f"  {len(src)} chars, {src.count(chr(10))} lines")


if __name__ == "__main__":
    main()
