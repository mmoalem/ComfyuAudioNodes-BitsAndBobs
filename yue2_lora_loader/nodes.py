"""
YUE2 Universal Adapter Loader — LoRA / LoKr / DoRA loader for YUE2 models.

YUE2 LoRAs store weights under two top-level prefixes:
  diffusion_model.model.layers.*  → model (AR / NAR diffusion transformer)
  text_encoders.model.layers.*    → clip  (Qwen2 text encoder)

ComfyUI's model_lora_keys_clip builds the key_map from cond_stage_model.state_dict(),
which exposes keys as `model.layers.*` (no leading text_encoders.). The dynamic key
matcher in the ACEStep loader already handles `text_encoders.` prefix stripping, so
we reuse it here rather than duplicating that logic.

Adapter support:
  lora  — standard LoRA (lora_A/B or lora_up/down)
  dora  — DoRA (lora_A/B + dora_scale / lora_magnitude_vector)
  lokr  — LoKr (lokr_w1/w2)
  loha  — LoHa (hada_w*)
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import comfy.lora
import comfy.lora_convert
import comfy.utils
import folder_paths
import torch

from ..lora_dora_lokr_loader.adapter_utils import detect_adapter_type
from ..lora_dora_lokr_loader.key_transforms import (
    normalize_diffusers_peft_keys,
    normalize_dora_magnitude_keys,
)
from ..lora_dora_lokr_loader.layer_scale import scale_patches
from ..lora_dora_lokr_loader.nodes import (
    _extract_lora_bases,
    _extend_key_map_with_dynamic_matches,
    _parse_lora_stack_kwargs,
    FlexibleOptionalInputType,
    any_type,
)

_LOG = logging.getLogger(__name__)


class YUE2UniversalAdapterLoader:
    """
    Power-LoRA-style multi-adapter loader for YUE2 models.

    Handles both the AR and NAR diffusion transformer (MODEL) and the Qwen2
    text encoder (CLIP).  Accepts LoRA, DoRA, LoKr, and LoHa adapters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
            "optional": FlexibleOptionalInputType(
                any_type,
                data={
                    "verbose": ("BOOLEAN", {"default": False}),
                    "log_unloaded_keys": ("BOOLEAN", {"default": False}),
                    "layer_scale_enabled": ("BOOLEAN", {"default": False}),
                    "self_attn_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                    "cross_attn_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                    "ffn_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                    "other_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                },
            ),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_loras"
    CATEGORY = "YUE2/Adapters"

    def _load_one(
        self,
        model,
        clip,
        lora_name: str,
        strength_model: float,
        strength_clip: float,
        verbose: bool,
        log_unloaded_keys: bool,
        layer_scale_enabled: bool,
        self_attn_scale: float,
        cross_attn_scale: float,
        ffn_scale: float,
        other_scale: float,
        model_sd_keys: Optional[Set[str]],
        model_sd_list: Optional[List[str]],
        clip_sd_keys: Optional[Set[str]],
        clip_sd_list: Optional[List[str]],
    ) -> None:
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            raise FileNotFoundError(f"LoRA not found: {lora_name}")

        try:
            lora_sd_raw = comfy.utils.load_torch_file(lora_path, safe_load=True)
        except TypeError:
            lora_sd_raw = comfy.utils.load_torch_file(lora_path)

        # Try convert_lora; bypass if it zeros lora_up
        lora_sd_conv = comfy.lora_convert.convert_lora(lora_sd_raw)
        raw_up_max = max(
            (float(v.abs().max()) for k, v in lora_sd_raw.items()
             if isinstance(v, torch.Tensor) and str(k).endswith(".lora_up.weight")),
            default=0.0,
        )
        conv_up_max = max(
            (float(v.abs().max()) for k, v in lora_sd_conv.items()
             if isinstance(v, torch.Tensor) and str(k).endswith(".lora_up.weight")),
            default=0.0,
        )
        if raw_up_max > 0.0 and conv_up_max == 0.0:
            _LOG.warning(
                "[YUE2 Adapter Loader] %s: convert_lora zeroed lora_up — bypassing conversion.",
                lora_name,
            )
            try:
                lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            except TypeError:
                lora_sd = comfy.utils.load_torch_file(lora_path)
        else:
            lora_sd = lora_sd_conv

        # Detect adapter type once before any transform
        adapter_type = detect_adapter_type(lora_sd)
        if verbose:
            _LOG.info("[YUE2 Adapter Loader] %s: adapter_type=%s", lora_name, adapter_type)

        # lora_A/B → lora_down/up (PEFT/Diffusers convention)
        normalize_diffusers_peft_keys(lora_sd, adapter_type, verbose=verbose)

        # lora_magnitude_vector → dora_scale
        normalize_dora_magnitude_keys(lora_sd, adapter_type, verbose=verbose)

        # Build standard ComfyUI key_map from model and clip
        key_map: Dict[str, Any] = {}
        if model is not None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        if clip is not None:
            key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)

        # Extend key_map with dynamic matches for bases not in the standard map.
        # _candidate_base_variants already handles text_encoders.* prefix stripping,
        # so YUE2 clip keys (text_encoders.model.layers.*) resolve to the clip sd.
        lora_bases = _extract_lora_bases(lora_sd.keys())
        added, unresolved = _extend_key_map_with_dynamic_matches(
            key_map=key_map,
            lora_bases=lora_bases,
            model_sd_keys=model_sd_keys,
            model_sd_list=model_sd_list,
            clip_sd_keys=clip_sd_keys,
            clip_sd_list=clip_sd_list,
            verbose=verbose,
        )

        if verbose:
            _LOG.info(
                "[YUE2 Adapter Loader] %s: key_map=%d dynamic_added=%d unresolved=%d lora_keys=%d",
                lora_name, len(key_map), added, len(unresolved), len(lora_sd),
            )

        try:
            loaded = comfy.lora.load_lora(lora_sd, key_map, log_missing=log_unloaded_keys)
        except TypeError:
            try:
                loaded = comfy.lora.load_lora(lora_sd, key_map, log_unloaded_keys)
            except TypeError:
                loaded = comfy.lora.load_lora(lora_sd, key_map)

        if verbose:
            _LOG.info("[YUE2 Adapter Loader] %s: loaded %d patches", lora_name, len(loaded))

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

        if model is not None:
            applied_m = model.add_patches(loaded, strength_model) or []
            if verbose:
                _LOG.info(
                    "[YUE2 Adapter Loader] %s: applied %d model patches (strength=%.3f)",
                    lora_name, len(applied_m), strength_model,
                )
        if clip is not None:
            applied_c = clip.add_patches(loaded, strength_clip) or []
            if verbose:
                _LOG.info(
                    "[YUE2 Adapter Loader] %s: applied %d clip patches (strength=%.3f)",
                    lora_name, len(applied_c), strength_clip,
                )

    def load_loras(self, model, clip, **kwargs):
        stack_enabled = bool(kwargs.get("stack_enabled", True))
        verbose = bool(kwargs.get("verbose", False))
        log_unloaded_keys = bool(kwargs.get("log_unloaded_keys", False))
        layer_scale_enabled = bool(kwargs.get("layer_scale_enabled", False))
        try:
            self_attn_scale  = float(kwargs.get("self_attn_scale", 1.0))
            cross_attn_scale = float(kwargs.get("cross_attn_scale", 1.0))
            ffn_scale        = float(kwargs.get("ffn_scale", 1.0))
            other_scale      = float(kwargs.get("other_scale", 1.0))
        except Exception:
            self_attn_scale = cross_attn_scale = ffn_scale = other_scale = 1.0

        if not stack_enabled:
            return (model, clip)

        entries = _parse_lora_stack_kwargs(kwargs)
        if not entries:
            return (model, clip)

        new_model = model.clone() if model is not None else None
        new_clip = clip.clone() if clip is not None else None

        model_sd_keys = model_sd_list = None
        clip_sd_keys = clip_sd_list = None

        if new_model is not None:
            model_sd_list = list(new_model.model.state_dict().keys())
            model_sd_keys = set(model_sd_list)

        if new_clip is not None:
            clip_sd_list = list(new_clip.cond_stage_model.state_dict().keys())
            clip_sd_keys = set(clip_sd_list)

        for e in entries:
            lora_name = e.get("lora")
            if not lora_name or lora_name in ("None", "NONE"):
                continue
            if not e.get("on", True):
                continue
            sm = float(e.get("strength_model", 1.0))
            sc = float(e.get("strength_clip", sm))
            try:
                self._load_one(
                    new_model,
                    new_clip,
                    lora_name=lora_name,
                    strength_model=sm,
                    strength_clip=sc,
                    verbose=verbose,
                    log_unloaded_keys=log_unloaded_keys,
                    layer_scale_enabled=layer_scale_enabled,
                    self_attn_scale=self_attn_scale,
                    cross_attn_scale=cross_attn_scale,
                    ffn_scale=ffn_scale,
                    other_scale=other_scale,
                    model_sd_keys=model_sd_keys,
                    model_sd_list=model_sd_list,
                    clip_sd_keys=clip_sd_keys,
                    clip_sd_list=clip_sd_list,
                )
            except Exception as exc:
                _LOG.warning("[YUE2 Adapter Loader] %s failed: %r", lora_name, exc)

        return (new_model, new_clip)
