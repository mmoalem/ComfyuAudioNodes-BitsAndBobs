"""
DebugPerStepSAInject — mirrors PerStepSelfAttentionInjectPerLayer exactly
(same 32 per-layer strength sliders, same injection logic) but prints
K/V and attention statistics so you can compare ComfyUI vs C++.

Logs to stdout/ComfyUI console for the first DEBUG_STEPS denoising steps
for ALL active layers (strength > 0):

  [DBG step=N t=T L=LL] CAPTURE  K shape=... mean=... std=... min=... max=...
  [DBG step=N t=T L=LL] CAPTURE  V shape=... mean=... std=... min=... max=...
  [DBG step=N t=T L=LL] GEN_LIVE K shape=... mean=... std=... min=... max=...
  [DBG step=N t=T L=LL] REF      K (pre-scale) mean=... std=...  s=<strength>
  [DBG step=N t=T L=LL] CONCAT   K shape=...  gen_part: ...  ref_part: ...
  [DBG step=N t=T L=LL] ATTN_OUT mean=... std=... min=... max=...
"""

import torch
from ..core import model_utils
from ..core.hook_manager import HookManager

_NUM_LAYERS = 32
DEBUG_STEPS  = 3   # how many denoising steps to print stats for


def _s(t: torch.Tensor) -> str:
    """Compact stats string matching ComfyUI debug_inject.py format."""
    f = t.float()
    return (f"mean={f.mean().item():.4f} std={f.std().item():.4f} "
            f"min={f.min().item():.4f} max={f.max().item():.4f}")


def _build_input_types():
    """Build INPUT_TYPES with sliders for all 32 layers (matches production node)."""
    required = {
        "model": ("MODEL",),
        "vae":   ("VAE",),
        "audio": ("AUDIO",),
        "mode":  (["inject", "replace"], {"default": "inject"}),
    }
    for i in range(_NUM_LAYERS):
        required[f"layer_{i:02d}"] = (
            "FLOAT",
            {
                "default": 0.0,
                "min":     0.0,
                "max":     5.0,
                "step":    0.05,
                "tooltip": (
                    f"SA KV injection strength for layer {i}. "
                    "0.0 = layer skipped entirely."
                ),
            },
        )
    return {"required": required}


class DebugPerStepSAInject:
    """
    Debug wrapper around PerStepSelfAttentionInjectPerLayer.

    Identical behaviour to the production node — same 32 per-layer strength
    sliders, same inject/replace modes, same HookManager mechanics — but prints
    K/V statistics to the ComfyUI console for the first DEBUG_STEPS denoising
    steps on all active layers.

    Use this to compare ComfyUI's captured K/V values against the C++ engine
    to diagnose injection correlation issues.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _build_input_types()

    RETURN_TYPES  = ("MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "inject"
    CATEGORY      = "ACE-Step/Reference/Debug"
    DESCRIPTION   = (
        "Like PerStepSelfAttentionInjectPerLayer but prints K/V stats to the "
        "console for the first 3 steps on every active layer. "
        "Treats all layers identically."
    )

    def inject(self, model, vae, audio, mode, **kwargs):
        # Collect per-layer strengths — same logic as production node
        layer_strengths = {}
        for i in range(_NUM_LAYERS):
            s = float(kwargs.get(f"layer_{i:02d}", 0.0))
            if s > 0.0:
                layer_strengths[i] = s
        active_layers = list(layer_strengths.keys())

        if not active_layers:
            print("[DebugInject] No active layers. Returning unmodified model.")
            return (model,)

        # ── Encode reference audio → clean latent ────────────────────────────
        waveform    = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform[:1]
        if sample_rate != 48000:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=48000)
        with torch.no_grad():
            vae_device, vae_dtype = model_utils.get_vae_device_dtype(vae)
            vae_out    = vae.encode(
                waveform.movedim(1, -1).to(vae_device, vae_dtype))
            ref_latent = (vae_out.latent_dist.mode()
                          if hasattr(vae_out, "latent_dist") else vae_out)
        ref_latent = ref_latent.detach().cpu()

        patched      = model.clone()
        raw_patched  = model_utils.get_raw_model(patched)
        prev_wrapper = patched.model_options.get("model_function_wrapper", None)
        hook_range   = (0, _NUM_LAYERS - 1)
        _step_count  = [0]

        # ── UNet wrapper: runs once per denoising step ────────────────────────
        def my_unet_wrapper(model_function, wrap_kwargs):
            step      = _step_count[0]
            _step_count[0] += 1
            input_t   = wrap_kwargs["input"]
            ref_dev   = input_t.device
            ref_dtype = input_t.dtype
            gen_t     = input_t.shape[2]
            t_now     = float(wrap_kwargs["timestep"][0].item())
            do_log    = (step < DEBUG_STEPS)

            with torch.no_grad():
                # ── Match reference length to generation ─────────────────────
                ref = ref_latent.to(device=ref_dev, dtype=ref_dtype)
                if ref.shape[2] != gen_t:
                    if ref.shape[2] > gen_t:
                        ref = ref[:, :, :gen_t]
                    else:
                        rpt = (gen_t + ref.shape[2] - 1) // ref.shape[2]
                        ref = ref.repeat(1, 1, rpt)[:, :, :gen_t]

                # ── Noise reference to current timestep ──────────────────────
                noise        = torch.randn_like(ref)
                ref_timestep = wrap_kwargs["timestep"][:1]
                if hasattr(patched, "model") and hasattr(patched.model, "model_sampling"):
                    noised_ref = patched.model.model_sampling.noise_scaling(
                        ref_timestep, noise, ref)
                else:
                    noised_ref = ref + noise * ref_timestep.view(-1, 1, 1).to(ref_dev, ref_dtype)

                # ── Null conditioning for capture (zero text embeddings) ──────
                null_c = {}
                for k, v in wrap_kwargs.get("c", {}).items():
                    if k == "c_crossattn" and isinstance(v, torch.Tensor):
                        null_c[k] = torch.zeros(
                            1, v.shape[1], v.shape[2],
                            device=v.device, dtype=v.dtype)
                    elif isinstance(v, torch.Tensor) and v.shape[0] > 1:
                        null_c[k] = v[:1]
                    else:
                        null_c[k] = v

                ref_wrap = {
                    "input":          noised_ref,
                    "timestep":       ref_timestep,
                    "cond_or_uncond": [0],
                    "c":              null_c,
                }

                # ── CAPTURE PASS (uses HookManager exactly as production node) ─
                attn_cache = {}
                with HookManager(raw_patched, hook_range) as hm_cap:
                    hm_cap.register_capture_hooks(attn_cache)
                    if prev_wrapper is not None:
                        prev_wrapper(model_function, ref_wrap)
                    else:
                        model_function(ref_wrap["input"], ref_wrap["timestep"],
                                       **ref_wrap.get("c", {}))

                # Log captured K/V for all active layers
                if do_log:
                    for idx in active_layers:
                        d = attn_cache.get(idx)
                        if d is None:
                            continue
                        print(f"[DBG step={step} t={t_now:.3f} L={idx:02d}] "
                              f"CAPTURE  K shape={tuple(d['k'].shape)} {_s(d['k'])}")
                        print(f"[DBG step={step} t={t_now:.3f} L={idx:02d}] "
                              f"CAPTURE  V shape={tuple(d['v'].shape)} {_s(d['v'])}")

            # ── INJECT PASS ──────────────────────────────────────────────────
            # Build per-layer debug injection forwards using the same bound-method
            # pattern as HookManager._make_forward_wrapper / __get__.
            # All layers are treated IDENTICALLY — no layer-specific logic.
            with torch.no_grad():

                def make_inject_fwd(layer_idx, strength, eff_mode):
                    """Returns a bound-method-compatible custom_forward for one layer."""

                    def custom_forward(
                        self,                       # self_attn module
                        hidden_states,
                        encoder_hidden_states=None,
                        attention_mask=None,
                        position_embeddings=None,
                    ):
                        from comfy.ldm.ace.ace_step15 import apply_rotary_pos_emb
                        from comfy.ldm.modules.attention import optimized_attention

                        bsz, q_len, _ = hidden_states.size()

                        # ── Q, K, V projections (identical to HookManager) ────
                        query_states = self.q_proj(hidden_states)
                        query_states = query_states.view(
                            bsz, q_len, self.num_heads, self.head_dim)
                        query_states = self.q_norm(query_states)
                        query_states = query_states.transpose(1, 2)

                        kv_len       = q_len
                        key_states   = self.k_proj(hidden_states)
                        value_states = self.v_proj(hidden_states)
                        key_states   = key_states.view(
                            bsz, q_len, self.num_kv_heads, self.head_dim)
                        key_states   = self.k_norm(key_states)
                        value_states = value_states.view(
                            bsz, q_len, self.num_kv_heads, self.head_dim)
                        key_states   = key_states.transpose(1, 2)
                        value_states = value_states.transpose(1, 2)

                        if position_embeddings is not None:
                            cos, sin = position_embeddings
                            query_states, key_states = apply_rotary_pos_emb(
                                query_states, key_states, cos, sin)

                        if do_log:
                            print(f"[DBG step={step} t={t_now:.3f} L={layer_idx:02d}] "
                                  f"GEN_LIVE K shape={tuple(key_states.shape)} "
                                  f"{_s(key_states)}")

                        # ── Blend in reference K / V ──────────────────────────
                        ref_data = attn_cache.get(layer_idx)
                        if ref_data is not None and strength > 0:
                            ref_k = ref_data["k"].to(key_states.device, key_states.dtype)
                            ref_v = ref_data["v"].to(key_states.device, key_states.dtype)

                            # Trim / repeat to match generation length
                            rt, gt = ref_k.shape[2], key_states.shape[2]
                            if rt != gt:
                                if rt > gt:
                                    ref_k = ref_k[:, :, :gt, :]
                                    ref_v = ref_v[:, :, :gt, :]
                                else:
                                    rpt   = (gt + rt - 1) // rt
                                    ref_k = ref_k.repeat(1, 1, rpt, 1)[:, :, :gt, :]
                                    ref_v = ref_v.repeat(1, 1, rpt, 1)[:, :, :gt, :]

                            if ref_k.shape[0] == 1 and bsz > 1:
                                ref_k = ref_k.expand(bsz, -1, -1, -1)
                                ref_v = ref_v.expand(bsz, -1, -1, -1)

                            if do_log:
                                print(f"[DBG step={step} t={t_now:.3f} L={layer_idx:02d}] "
                                      f"REF      K (pre-scale) {_s(ref_k)}  s={strength}")

                            if eff_mode == "replace":
                                key_states   = ref_k * strength
                                value_states = ref_v * strength
                                if do_log:
                                    print(f"[DBG step={step} t={t_now:.3f} L={layer_idx:02d}] "
                                          f"REPLACE  K {_s(key_states)}")
                            else:  # inject
                                ref_k_s = ref_k * strength
                                ref_v_s = ref_v * strength
                                key_states   = torch.cat([key_states,   ref_k_s], dim=2)
                                value_states = torch.cat([value_states, ref_v_s], dim=2)
                                kv_len       = key_states.shape[2]
                                if do_log:
                                    print(f"[DBG step={step} t={t_now:.3f} L={layer_idx:02d}] "
                                          f"CONCAT   K shape={tuple(key_states.shape)} "
                                          f"gen_part {_s(key_states[:, :, :gt, :])} "
                                          f"ref_part {_s(key_states[:, :, gt:, :])}")

                        # ── GQA head repeat ────────────────────────────────────
                        n_rep = self.num_heads // self.num_kv_heads
                        if n_rep > 1:
                            key_states   = key_states.repeat_interleave(n_rep, dim=1)
                            value_states = value_states.repeat_interleave(n_rep, dim=1)

                        # ── Sliding-window bias (identical to HookManager) ─────
                        attn_bias = None
                        if self.sliding_window is not None:
                            indices_q = torch.arange(q_len, device=query_states.device)
                            if kv_len > q_len:
                                indices_k = torch.cat([
                                    torch.arange(q_len,          device=query_states.device),
                                    torch.arange(kv_len - q_len, device=query_states.device),
                                ])
                            else:
                                indices_k = torch.arange(kv_len, device=query_states.device)
                            diff      = indices_q.unsqueeze(1) - indices_k.unsqueeze(0)
                            in_win    = torch.abs(diff) <= self.sliding_window
                            bias      = torch.zeros(
                                (q_len, kv_len),
                                device=query_states.device, dtype=query_states.dtype)
                            bias.masked_fill_(~in_win, torch.finfo(query_states.dtype).min)
                            attn_bias = bias.unsqueeze(0).unsqueeze(0)

                        attn_out = optimized_attention(
                            query_states, key_states, value_states,
                            self.num_heads, attn_bias,
                            skip_reshape=True, low_precision_attention=False)
                        attn_out = self.o_proj(attn_out)

                        if do_log:
                            print(f"[DBG step={step} t={t_now:.3f} L={layer_idx:02d}] "
                                  f"ATTN_OUT {_s(attn_out)}")
                        return attn_out

                    return custom_forward

                with HookManager(raw_patched, hook_range) as hm_inj:
                    # Apply debug forward to each active layer using the same
                    # __get__ binding pattern as HookManager._make_forward_wrapper.
                    # All layers use identical logic — no layer-specific branching.
                    for idx, attn_mod in hm_inj._layer_modules:
                        s = layer_strengths.get(idx, 0.0)
                        if s <= 0.0:
                            continue
                        fn = make_inject_fwd(idx, s, mode)
                        attn_mod.forward = fn.__get__(attn_mod, type(attn_mod))
                        hm_inj._patched_modules.append(attn_mod)

                    if prev_wrapper is not None:
                        return prev_wrapper(model_function, wrap_kwargs)
                    else:
                        return model_function(
                            wrap_kwargs["input"], wrap_kwargs["timestep"],
                            **wrap_kwargs.get("c", {}))

        patched.set_model_unet_function_wrapper(my_unet_wrapper)
        print(f"[DebugInject] Active layers: {active_layers}. "
              f"Printing stats for first {DEBUG_STEPS} steps on all active layers.")
        return (patched,)
