"""
per_step_self_attn_inject_legacy2_step_limited.py
=================================================
ComfyUI custom node: "Per Step Self Attention Inject (step+time taper, step limited)"

Based on PerStepSelfAttentionInjectLegacy2 (commit 5566647), extended with the
step-limited taper window from PerStepSelfAttentionInjectPerLayerStepLimited:

  step_taper_total_steps — total KSampler step count (kept in sync manually)
  step_taper_end_step    — step by which the step_taper curve completes and holds

All core helpers are inlined (prefixed _l2sl_) so this file has zero relative
imports and can be dropped anywhere without touching other files.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Inlined helpers (based on ace_step_reference/core — prefixed _l2sl_)
# =============================================================================

def _l2sl_compute_layer_strength(
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
    elif taper == "cosine_fade_out":
        # 1.0 at start_layer, smooth cosine down to 0.0 at end_layer
        factor = (1 + math.cos(normalized * math.pi)) / 2
    else:
        factor = normalized
    return base_strength * factor


def _l2sl_compute_step_multiplier(t: float, mode: str, end_pct: float = 100.0) -> float:
    """Compute a strength multiplier from the current diffusion timestep.

    t runs from ~1.0 (first step / pure noise) to ~0.0 (last step / clean).
    end_pct compresses the curve into the first end_pct% of sampling progress;
    the curve's terminal value is held for the remainder of the run.
    """
    t = max(0.0, min(1.0, float(t)))
    progress = 1.0 - t  # 0.0 at start, 1.0 at end

    if mode == "none":
        return 1.0

    cutoff = max(0.0001, min(1.0, end_pct / 100.0))
    scaled = min(1.0, progress / cutoff)

    if mode == "fade_out":
        return 1.0 - scaled
    elif mode == "fade_in":
        return scaled
    elif mode == "cosine_fade_out":
        return (1.0 + math.cos(scaled * math.pi)) / 2.0
    elif mode == "cosine_fade_in":
        return (1.0 - math.cos(scaled * math.pi)) / 2.0
    elif mode == "cosine_bell":
        return math.sin(scaled * math.pi)
    else:
        return 1.0


def _l2sl_compute_time_multiplier(seq_len: int, mode: str, device, dtype) -> torch.Tensor:
    """Returns a [1, 1, seq_len, 1] multiplier that broadcasts over K/V tensors."""
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


def _l2sl_get_raw_model(model) -> nn.Module:
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
        "[ACEStep-Legacy2StepLimited] Cannot unwrap model. Walked .model/.diffusion_model "
        f"up to 5 levels without finding an nn.Module with a 'decoder' attribute. "
        f"Top-level type: {type(model).__name__}"
    )


def _l2sl_get_vae_device_dtype(vae):
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
# Inlined HookManager
# =============================================================================

class _L2SLHookManager:
    """Manages monkey-patching of AceStepAttention.forward for KV capture/inject."""

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
                        print(f"[HookManager-Legacy2SL] Warning: layer {idx} has no self_attn")
        else:
            print("[HookManager-Legacy2SL] Warning: raw_model has no decoder.layers — no hooks registered")

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

    def _make_forward_wrapper(
        self,
        layer_idx: int,
        cache: dict,
        strength: float,
        is_capture: bool,
        taper: str,
        time_taper: str = "none",
    ):
        effective_strength = strength
        if taper != "none" and not is_capture:
            effective_strength = _l2sl_compute_layer_strength(
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

                        if time_taper != "none":
                            ref_seq_len = ref_k.shape[2]
                            time_mult = _l2sl_compute_time_multiplier(
                                ref_seq_len, time_taper, ref_device, ref_dtype
                            )
                            ref_k = ref_k * time_mult
                            ref_v = ref_v * time_mult

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

                window_bias = torch.zeros(
                    (q_len, kv_len), device=query_states.device, dtype=query_states.dtype
                )
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
            fn = self._make_forward_wrapper(
                idx, cache, strength=1.0, is_capture=True, taper="none", time_taper="none"
            )
            self_attn.forward = fn.__get__(self_attn, type(self_attn))
            self._patched_modules.append(self_attn)

    def register_injection_hooks(
        self, cache: dict, strength: float, taper: str, time_taper: str = "none"
    ) -> None:
        for idx, self_attn in self._layer_modules:
            fn = self._make_forward_wrapper(
                idx, cache, strength=strength, is_capture=False, taper=taper, time_taper=time_taper
            )
            self_attn.forward = fn.__get__(self_attn, type(self_attn))
            self._patched_modules.append(self_attn)


# =============================================================================
# The Node
# =============================================================================

class PerStepSelfAttentionInjectLegacy2StepLimited:
    """
    Per Step Self Attention Inject (step+time taper, step limited).

    Same as Legacy 2 (single global strength, layer taper, step_taper,
    time_taper, start_layer/end_layer) but the step_taper curve can be
    compressed into the first N steps of the run via:

      step_taper_total_steps — total steps configured on your KSampler
      step_taper_end_step    — step by which the curve completes and holds

    Example: total_steps=35, end_step=25, step_taper=cosine_fade_out
    → injection fades to 0 by step 25, then stays at 0 for steps 25-35.

    WARNING: runs 2x full forward passes per step — expect ~2x slower sampling.
    Compatible with BF16, FP16, FP32, FP8, and GGUF checkpoints.
    """

    TAPER_OPTIONS = ["none", "linear", "cosine", "cosine_fade_out"]
    STEP_TAPER_OPTIONS = [
        "none", "fade_out", "fade_in", "cosine_fade_out", "cosine_fade_in", "cosine_bell"
    ]

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
                "layer_taper": (
                    cls.TAPER_OPTIONS,
                    {
                        "default": "none",
                        "tooltip": (
                            "Strength taper across the captured layer range (layer-based, not step-based).\n"
                            "none             — flat strength across all layers in range.\n"
                            "linear           — ramps 0→strength from start_layer to end_layer.\n"
                            "cosine           — smooth S-curve 0→strength from start_layer to end_layer.\n"
                            "cosine_fade_out  — smooth cosine strength→0 from start_layer to end_layer."
                        ),
                    },
                ),
                "step_taper": (
                    cls.STEP_TAPER_OPTIONS,
                    {
                        "default": "none",
                        "tooltip": (
                            "Scales injection strength across diffusion steps using the current timestep "
                            "(tβ‰ˆ1 at step 1, tβ‰ˆ0 at the final step).\n"
                            "none             — constant strength every step.\n"
                            "fade_out         — linear 1β†'0  (strong early, gone by end).\n"
                            "fade_in          — linear 0β†'1  (absent early, full by end).\n"
                            "cosine_fade_out  — smooth cosine 1β†'0.\n"
                            "cosine_fade_in   — smooth cosine 0β†'1.\n"
                            "cosine_bell      — peaks at the midpoint of sampling (0β†'1β†'0).\n"
                            "The curve is compressed into the window defined by "
                            "step_taper_total_steps / step_taper_end_step below."
                        ),
                    },
                ),
                "step_taper_total_steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 10000,
                        "tooltip": (
                            "Total number of sampling steps configured on your KSampler. "
                            "Keep this in sync with the KSampler's 'steps' value."
                        ),
                    },
                ),
                "step_taper_end_step": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 10000,
                        "tooltip": (
                            "The step (out of step_taper_total_steps) by which the step_taper curve "
                            "reaches its terminal value and holds there for the rest of sampling.\n"
                            "Example: total_steps=35, end_step=25 with step_taper=cosine_fade_out "
                            "means injection fades to 0 by step 25, then stays at 0 for steps 25-35.\n"
                            "Set equal to step_taper_total_steps to reproduce the original full-run "
                            "taper behaviour."
                        ),
                    },
                ),
                "time_taper": (
                    cls.STEP_TAPER_OPTIONS,
                    {
                        "default": "none",
                        "tooltip": (
                            "Temporal scaling: fades the injection strength across the length of the audio sequence.\n"
                            "fade_out = strong at the start of the audio clip, fading to 0 at the end.\n"
                            "fade_in  = 0 at the start of the clip, ramping up to full strength at the end."
                        ),
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
        "Legacy 2 Per-step Self-Attention Inject with a step-limited taper window. "
        "Same as Legacy 2 (single global strength, layer/step/time taper) but the "
        "step_taper curve completes by step_taper_end_step instead of always spanning "
        "the full run — useful for tapering injection to 0 before sampling finishes. "
        "WARNING: runs 2x full forward passes per step — expect ~2x slower sampling."
    )

    def inject(
        self,
        model,
        vae,
        audio,
        strength,
        layer_taper,
        step_taper,
        step_taper_total_steps,
        step_taper_end_step,
        time_taper,
        start_layer,
        end_layer,
    ):
        total_steps = max(1, int(step_taper_total_steps))
        end_step = max(1, min(int(step_taper_end_step), total_steps))
        step_taper_end_pct = (end_step / total_steps) * 100.0

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
            vae_device, vae_dtype = _l2sl_get_vae_device_dtype(vae)
            vae_output = vae.encode(waveform_for_vae.to(vae_device, vae_dtype))
            if hasattr(vae_output, "latent_dist"):
                ref_latent = vae_output.latent_dist.mode()
            else:
                ref_latent = vae_output
        ref_latent = ref_latent.detach().cpu()

        patched = model.clone()
        raw_patched = _l2sl_get_raw_model(patched)

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
                        f"[PerStepSelfAttentionInjectLegacy2StepLimited] Reference T={ref_latent.shape[2]} != "
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
                with _L2SLHookManager(raw_patched, (start_layer, end_layer)) as hm_capture:
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
                t_now = float(wrap_kwargs["timestep"][0].item())
                step_mult = _l2sl_compute_step_multiplier(t_now, step_taper, step_taper_end_pct)
                effective_strength = strength * step_mult

                with _L2SLHookManager(raw_patched, (start_layer, end_layer)) as hm_inject:
                    hm_inject.register_injection_hooks(
                        attn_cache, effective_strength, layer_taper, time_taper
                    )

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
            f"[PerStepSelfAttentionInjectLegacy2StepLimited] Attached — "
            f"layers={start_layer}-{end_layer}, strength={strength}, "
            f"layer_taper={layer_taper}, step_taper={step_taper} "
            f"(ends by step {end_step}/{total_steps}, {step_taper_end_pct:.1f}% of progress), "
            f"time_taper={time_taper}\n"
            f"[PerStepSelfAttentionInjectLegacy2StepLimited] WARNING: 2x forward passes per step. "
            f"Expect ~2x slower sampling."
        )

        return (patched,)


# =============================================================================
# ComfyUI registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "PerStepSelfAttentionInjectLegacy2StepLimited": PerStepSelfAttentionInjectLegacy2StepLimited,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerStepSelfAttentionInjectLegacy2StepLimited": "Per Step Self Attention Inject (step+time taper, step limited)",
}
