"""
per_step_self_attn_inject_legacy.py
====================================
Standalone ComfyUI custom node: "Per Step Self Attention Inject (Legacy)"

Recovered from git history of ComfyuAudioNodes-BitsAndBobs
  β€' Source commit (last state BEFORE per-layer was added): 67f1eb9
  β€' Per-layer feature was introduced in commit:            5566647

This file is fully self-contained — all core helpers (model_utils,
hook_manager, tensor_ops) are inlined below.  Drop this file into any
ComfyUI custom_nodes folder (or directly into the BitsAndBobs pack) and
register it in your __init__.py NODE_CLASS_MAPPINGS dict as shown at the
bottom of this file.

Differences from the current "Per Step Self Attention Inject (Per Layer)" node:
  β€' Single `strength` float  (not 32 per-layer sliders)
  β€' Single `taper` dropdown  (layer-based taper over start_layer–end_layer range)
  β€' No step_taper / time_taper controls
  β€' Simpler — good for quick tests where per-layer granularity isn't needed
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# Inlined from ace_step_reference/core/tensor_ops.py  (commit 67f1eb9)
# =============================================================================

def _compute_layer_strength(
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
        factor = (1 - math.cos(normalized * math.pi)) / 2
    else:
        factor = normalized
    return base_strength * factor


# =============================================================================
# Inlined from ace_step_reference/core/model_utils.py  (commit 67f1eb9)
# =============================================================================

def _get_raw_model(model) -> nn.Module:
    """Walk ComfyUI wrappers until we reach the ACEStep nn.Module with a 'decoder'."""
    if isinstance(model, nn.Module) and hasattr(model, "decoder"):
        return model

    candidates = []
    frontier = [model]
    seen = set()

    for _ in range(5):
        next_frontier = []
        for current in frontier:
            obj_id = id(current)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            for attr in ("model", "diffusion_model"):
                if not hasattr(current, attr):
                    continue
                child = getattr(current, attr)
                if not isinstance(child, nn.Module):
                    continue
                child_id = id(child)
                if child_id in seen:
                    continue
                if hasattr(child, "decoder"):
                    return child
                candidates.append(child)
                next_frontier.append(child)
        if not next_frontier:
            break
        frontier = next_frontier

    if candidates:
        return candidates[0]

    raise RuntimeError(
        "[ACEStep-Legacy] Cannot unwrap model. Walked .model/.diffusion_model up to 5 "
        f"levels without finding an nn.Module with a 'decoder' attribute. "
        f"Top-level type: {type(model).__name__}"
    )


def _get_vae_device_dtype(vae):
    device = getattr(vae, "device", None)
    if device is None:
        try:
            device = next(vae.first_stage_model.parameters()).device
        except Exception:
            device = torch.device("cpu")

    dtype = getattr(vae, "dtype", None)
    if dtype is None:
        dtype = getattr(vae, "vae_dtype", None)
    if dtype is None:
        try:
            dtype = next(vae.first_stage_model.parameters()).dtype
        except Exception:
            dtype = torch.float32

    return device, dtype


# =============================================================================
# Inlined from ace_step_reference/core/hook_manager.py  (commit 67f1eb9)
# =============================================================================

class _HookManager:
    """Manages monkey-patching of AceStepAttention.forward for KV injection."""

    def __init__(self, raw_model: nn.Module, layer_range: tuple):
        self.raw_model = raw_model
        self.start_layer, self.end_layer = layer_range
        self._patched_modules: list = []
        self._layer_modules: list = []
        self._build_layer_index()

    def _build_layer_index(self) -> None:
        if hasattr(self.raw_model, "decoder") and hasattr(self.raw_model.decoder, "layers"):
            for idx, layer in enumerate(self.raw_model.decoder.layers):
                if self.start_layer <= idx <= self.end_layer:
                    if hasattr(layer, "self_attn"):
                        self._layer_modules.append((idx, layer.self_attn))
                    else:
                        print(f"[HookManager-Legacy] Warning: layer {idx} has no self_attn")
        else:
            print("[HookManager-Legacy] Warning: raw_model has no decoder.layers — no hooks registered")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all()
        return False

    def remove_all(self) -> None:
        for module in self._patched_modules:
            if hasattr(module, "forward"):
                delattr(module, "forward")
        self._patched_modules.clear()

    def _make_forward_wrapper(self, layer_idx, cache, strength, is_capture, taper):
        effective_strength = strength
        if taper != "none" and not is_capture:
            effective_strength = _compute_layer_strength(
                layer_idx, self.start_layer, self.end_layer, strength, taper
            )

        def custom_forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            position_embeddings=None,
        ):
            from comfy.ldm.ace.ace_step15 import apply_rotary_pos_emb
            from comfy.ldm.modules.attention import optimized_attention

            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            query_states = self.q_norm(query_states)
            query_states = query_states.transpose(1, 2)

            if self.is_cross_attention and encoder_hidden_states is not None:
                bsz_enc, kv_len, _ = encoder_hidden_states.size()
                key_states = self.k_proj(encoder_hidden_states)
                value_states = self.v_proj(encoder_hidden_states)
                key_states = key_states.view(bsz_enc, kv_len, self.num_kv_heads, self.head_dim)
                key_states = self.k_norm(key_states)
                value_states = value_states.view(bsz_enc, kv_len, self.num_kv_heads, self.head_dim)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)
            else:
                kv_len = q_len
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)
                key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
                key_states = self.k_norm(key_states)
                value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                # ---- HOOK INJECTION POINT ----
                if is_capture:
                    cache[layer_idx] = {
                        "k": key_states.detach().cpu(),
                        "v": value_states.detach().cpu(),
                    }
                else:
                    ref_data = cache.get(layer_idx)
                    if ref_data is not None and effective_strength > 0:
                        ref_device = key_states.device
                        ref_dtype = key_states.dtype

                        ref_k = ref_data["k"].to(device=ref_device, dtype=ref_dtype)
                        ref_v = ref_data["v"].to(device=ref_device, dtype=ref_dtype)

                        if ref_k.shape[0] == 1 and bsz > 1:
                            ref_k = ref_k.expand(bsz, -1, -1, -1)
                            ref_v = ref_v.expand(bsz, -1, -1, -1)

                        ref_k = ref_k * effective_strength
                        ref_v = ref_v * effective_strength

                        key_states = torch.cat([key_states, ref_k], dim=2)
                        value_states = torch.cat([value_states, ref_v], dim=2)
                        kv_len = key_states.shape[2]
                # ---- END HOOK ----

            n_rep = self.num_heads // self.num_kv_heads
            if n_rep > 1:
                key_states = key_states.repeat_interleave(n_rep, dim=1)
                value_states = value_states.repeat_interleave(n_rep, dim=1)

            attn_bias = None
            if self.sliding_window is not None and not self.is_cross_attention:
                indices_q = torch.arange(q_len, device=query_states.device)
                if kv_len > q_len:
                    indices_k_gen = torch.arange(q_len, device=query_states.device)
                    indices_k_ref = torch.arange(kv_len - q_len, device=query_states.device)
                    indices_k = torch.cat([indices_k_gen, indices_k_ref])
                else:
                    indices_k = torch.arange(kv_len, device=query_states.device)

                diff = indices_q.unsqueeze(1) - indices_k.unsqueeze(0)
                in_window = torch.abs(diff) <= self.sliding_window

                window_bias = torch.zeros((q_len, kv_len), device=query_states.device, dtype=query_states.dtype)
                min_value = torch.finfo(query_states.dtype).min
                window_bias.masked_fill_(~in_window, min_value)
                window_bias = window_bias.unsqueeze(0).unsqueeze(0)

                if attn_bias is not None:
                    if attn_bias.dtype == torch.bool:
                        base_bias = torch.zeros_like(window_bias)
                        base_bias.masked_fill_(~attn_bias, min_value)
                        attn_bias = base_bias + window_bias
                    else:
                        attn_bias = attn_bias + window_bias
                else:
                    attn_bias = window_bias

            attn_output = optimized_attention(
                query_states, key_states, value_states,
                self.num_heads, attn_bias,
                skip_reshape=True, low_precision_attention=False
            )
            attn_output = self.o_proj(attn_output)
            return attn_output

        return custom_forward

    def register_capture_hooks(self, cache: dict) -> None:
        for idx, self_attn in self._layer_modules:
            fn = self._make_forward_wrapper(idx, cache, strength=1.0, is_capture=True, taper="none")
            self_attn.forward = fn.__get__(self_attn, type(self_attn))
            self._patched_modules.append(self_attn)

    def register_injection_hooks(self, cache: dict, strength: float, taper: str) -> None:
        for idx, self_attn in self._layer_modules:
            fn = self._make_forward_wrapper(idx, cache, strength=strength, is_capture=False, taper=taper)
            self_attn.forward = fn.__get__(self_attn, type(self_attn))
            self._patched_modules.append(self_attn)


# =============================================================================
# The Node
# =============================================================================

class PerStepSelfAttentionInjectLegacy:
    """
    Per-step self-attention KV injection — legacy (pre-per-layer) version.

    Recovered from commit 67f1eb9 of ComfyuAudioNodes-BitsAndBobs (the last
    state before per-layer strength controls were added in commit 5566647).

    At every sampling step this node:
      1. Captures reference K/V from the reference audio at the matching noise level.
      2. Concatenates them into the generation forward pass.

    Controls are intentionally simpler than the current per-layer node:
      β€' A single `strength` float that applies uniformly (or with a layer taper)
        across the start_layer–end_layer range.
      β€' A single `taper` dropdown for layer-based tapering.

    WARNING: runs 2Γ— full forward passes per step — expect ~2Γ— slower sampling.
    Compatible with BF16, FP16, FP32, FP8, and GGUF checkpoints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "audio": ("AUDIO",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "Strength multiplier for reference Keys and Values during concatenation.",
                    },
                ),
                "taper": (
                    ["none", "linear", "cosine"],
                    {
                        "default": "none",
                        "tooltip": "Strength taper across the captured layer range.",
                    },
                ),
                "start_layer": ("INT", {"default": 0, "min": 0, "max": 31}),
                "end_layer": ("INT", {"default": 31, "min": 0, "max": 31}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject"
    CATEGORY = "ACE-Step/Reference"
    DESCRIPTION = (
        "LEGACY version of Per-step Self-Attention Inject (recovered from git history, "
        "before per-layer strength controls were added). "
        "Single global strength + optional layer taper. "
        "WARNING: runs 2x full forward passes per step — expect ~2x slower sampling."
    )

    def inject(self, model, vae, audio, strength, taper, start_layer, end_layer):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :, :]

        if sample_rate != 48000:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=48000)
            sample_rate = 48000

        waveform_for_vae = waveform.movedim(1, -1)

        with torch.no_grad():
            vae_device, vae_dtype = _get_vae_device_dtype(vae)
            vae_output = vae.encode(waveform_for_vae.to(vae_device, vae_dtype))
            if hasattr(vae_output, "latent_dist"):
                ref_latent = vae_output.latent_dist.mode()
            else:
                ref_latent = vae_output
        ref_latent = ref_latent.detach().cpu()

        patched = model.clone()
        raw_patched = _get_raw_model(patched)

        prev_wrapper = patched.model_options.get("model_function_wrapper", None)

        def _match_t_to_generation(ref, gen_t):
            ref_t = ref.shape[2]
            if ref_t == gen_t:
                return ref
            if ref_t > gen_t:
                return ref[:, :, :gen_t]
            repeats = (gen_t + ref_t - 1) // ref_t
            return ref.repeat(1, 1, repeats)[:, :, :gen_t]

        def _build_null_ref_wrap(wrap_kwargs, noised_ref, ref_timestep):
            ref_wrap = {}
            ref_wrap["input"] = noised_ref
            ref_wrap["timestep"] = ref_timestep
            ref_wrap["cond_or_uncond"] = [0]

            null_c = {}
            for k, v in wrap_kwargs.get("c", {}).items():
                if k == "c_crossattn" and isinstance(v, torch.Tensor):
                    null_c[k] = torch.zeros(
                        1, v.shape[1], v.shape[2],
                        device=v.device, dtype=v.dtype
                    )
                elif isinstance(v, torch.Tensor) and v.shape[0] > 1:
                    null_c[k] = v[:1]
                else:
                    null_c[k] = v
            ref_wrap["c"] = null_c
            return ref_wrap

        _t_logged = [False]

        def my_unet_wrapper(model_function, wrap_kwargs):
            input_t = wrap_kwargs["input"]
            ref_dev = input_t.device
            ref_dtype = input_t.dtype
            gen_t = input_t.shape[2]

            with torch.no_grad():
                ref_matched = _match_t_to_generation(
                    ref_latent.to(device=ref_dev, dtype=ref_dtype), gen_t
                )

                if not _t_logged[0] and ref_latent.shape[2] != gen_t:
                    print(
                        f"[PerStepSelfAttentionInjectLegacy] Reference T={ref_latent.shape[2]} != "
                        f"generation T={gen_t} — "
                        f"{'trimmed' if ref_latent.shape[2] > gen_t else 'repeat-padded'} "
                        f"to T={gen_t} for every capture pass."
                    )
                    _t_logged[0] = True

                noise = torch.randn_like(ref_matched)
                ref_timestep = wrap_kwargs["timestep"][:1]

                if hasattr(patched, "model") and hasattr(patched.model, "model_sampling"):
                    noised_ref = patched.model.model_sampling.noise_scaling(
                        ref_timestep, noise, ref_matched
                    )
                else:
                    noised_ref = ref_matched + noise * ref_timestep.view(-1, 1, 1).to(ref_dev, ref_dtype)

                ref_wrap = _build_null_ref_wrap(wrap_kwargs, noised_ref, ref_timestep)

                attn_cache = {}
                with _HookManager(raw_patched, (start_layer, end_layer)) as hm_capture:
                    hm_capture.register_capture_hooks(attn_cache)
                    if prev_wrapper is not None:
                        prev_wrapper(model_function, ref_wrap)
                    else:
                        model_function(
                            ref_wrap["input"],
                            ref_wrap["timestep"],
                            **ref_wrap.get("c", {})
                        )

            with torch.no_grad():
                with _HookManager(raw_patched, (start_layer, end_layer)) as hm_inject:
                    hm_inject.register_injection_hooks(attn_cache, strength, taper)

                    if prev_wrapper is not None:
                        return prev_wrapper(model_function, wrap_kwargs)
                    else:
                        return model_function(
                            wrap_kwargs["input"],
                            wrap_kwargs["timestep"],
                            **wrap_kwargs.get("c", {})
                        )

        patched.set_model_unet_function_wrapper(my_unet_wrapper)

        print(
            f"[PerStepSelfAttentionInjectLegacy] Attached — "
            f"layers={start_layer}-{end_layer}, strength={strength}, taper={taper}\n"
            f"[PerStepSelfAttentionInjectLegacy] WARNING: 2x forward passes per step. "
            f"Expect ~2x slower sampling."
        )

        return (patched,)


# =============================================================================
# ComfyUI registration
# =============================================================================
# To use this node, add to your pack's NODE_CLASS_MAPPINGS and
# NODE_DISPLAY_NAME_MAPPINGS in __init__.py:
#
#   from .per_step_self_attn_inject_legacy import (
#       PerStepSelfAttentionInjectLegacy,
#       NODE_CLASS_MAPPINGS as _LEGACY_MAPPINGS,
#       NODE_DISPLAY_NAME_MAPPINGS as _LEGACY_DISPLAY,
#   )
#   NODE_CLASS_MAPPINGS.update(_LEGACY_MAPPINGS)
#   NODE_DISPLAY_NAME_MAPPINGS.update(_LEGACY_DISPLAY)

NODE_CLASS_MAPPINGS = {
    "PerStepSelfAttentionInjectLegacy": PerStepSelfAttentionInjectLegacy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerStepSelfAttentionInjectLegacy": "Per Step Self Attention Inject (Legacy - pre per-layer)",
}
