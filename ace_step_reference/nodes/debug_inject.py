"""
DebugPerStepSAInject — identical to PerStepSelfAttentionInjectPerLayer but prints
K/V statistics at each step so you can compare what ComfyUI produces vs C++.

Logs (to stdout/ComfyUI console):
  [DBG step=N t=T layer=L] cap  K shape=... mean=... std=...
  [DBG step=N t=T layer=L] live K mean=... std=...
  [DBG step=N t=T layer=L] ref  K mean=... std=...  (after * strength)
  [DBG step=N t=T layer=L] cat  K shape=...         (gen + ref concatenated)
  [DBG step=N t=T layer=L] attn_out mean=... std=...

Only prints for layers in DEBUG_LAYERS and first DEBUG_STEPS steps.
"""

import torch
from ..core import model_utils
from ..core.hook_manager import HookManager

_NUM_LAYERS = 32
DEBUG_LAYERS = [1, 5, 6, 7]   # layers to print
DEBUG_STEPS  = 3               # how many steps to print for


class DebugPerStepSAInject:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model":   ("MODEL",),
            "vae":     ("VAE",),
            "audio":   ("AUDIO",),
            "mode":    (["inject", "replace"], {"default": "inject"}),
            "layer_01_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            "layer_05_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            "layer_06_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            "layer_07_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject"
    CATEGORY = "ACE-Step/Reference/Debug"
    DESCRIPTION = "Like PerStepSelfAttentionInjectPerLayer but prints K/V stats to console for debugging."

    def inject(self, model, vae, audio, mode, layer_01_strength, layer_05_strength, layer_06_strength, layer_07_strength):
        layer_strengths = {1: layer_01_strength, 5: layer_05_strength, 6: layer_06_strength, 7: layer_07_strength}
        active_layers = [l for l, s in layer_strengths.items() if s > 0.0]
        if not active_layers:
            print("[DebugInject] No active layers. Returning unmodified model.")
            return (model,)

        # Encode reference audio
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform[:1]
        if sample_rate != 48000:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=48000)
        waveform_for_vae = waveform.movedim(1, -1)
        with torch.no_grad():
            vae_device, vae_dtype = model_utils.get_vae_device_dtype(vae)
            vae_out = vae.encode(waveform_for_vae.to(vae_device, vae_dtype))
            ref_latent = vae_out.latent_dist.mode() if hasattr(vae_out, "latent_dist") else vae_out
        ref_latent = ref_latent.detach().cpu()

        patched     = model.clone()
        raw_patched = model_utils.get_raw_model(patched)
        prev_wrapper = patched.model_options.get("model_function_wrapper", None)
        hook_layer_range = (0, _NUM_LAYERS - 1)

        _step_count = [0]

        def _s(t): return f"mean={t.float().mean().item():.4f} std={t.float().std().item():.4f} min={t.float().min().item():.4f} max={t.float().max().item():.4f}"

        def my_unet_wrapper(model_function, wrap_kwargs):
            step   = _step_count[0]
            _step_count[0] += 1
            input_t    = wrap_kwargs["input"]
            ref_dev    = input_t.device
            ref_dtype  = input_t.dtype
            gen_t      = input_t.shape[2]
            t_now      = float(wrap_kwargs["timestep"][0].item())

            with torch.no_grad():
                # Match ref length to generation
                ref_matched = ref_latent.to(device=ref_dev, dtype=ref_dtype)
                if ref_matched.shape[2] != gen_t:
                    if ref_matched.shape[2] > gen_t:
                        ref_matched = ref_matched[:, :, :gen_t]
                    else:
                        rpt = (gen_t + ref_matched.shape[2] - 1) // ref_matched.shape[2]
                        ref_matched = ref_matched.repeat(1, 1, rpt)[:, :, :gen_t]

                noise = torch.randn_like(ref_matched)
                ref_timestep = wrap_kwargs["timestep"][:1]

                if hasattr(patched, "model") and hasattr(patched.model, "model_sampling"):
                    noised_ref = patched.model.model_sampling.noise_scaling(ref_timestep, noise, ref_matched)
                else:
                    noised_ref = ref_matched + noise * ref_timestep.view(-1, 1, 1).to(ref_dev, ref_dtype)

                # Build null-conditioned capture wrap
                null_c = {}
                for k, v in wrap_kwargs.get("c", {}).items():
                    if k == "c_crossattn" and isinstance(v, torch.Tensor):
                        null_c[k] = torch.zeros(1, v.shape[1], v.shape[2], device=v.device, dtype=v.dtype)
                    elif isinstance(v, torch.Tensor) and v.shape[0] > 1:
                        null_c[k] = v[:1]
                    else:
                        null_c[k] = v

                ref_wrap = {"input": noised_ref, "timestep": ref_timestep, "cond_or_uncond": [0], "c": null_c}

                # CAPTURE — with per-layer debug logging
                attn_cache = {}

                class DebugCaptureHookManager(HookManager):
                    def register_capture_hooks(self, cache, is_cross=False):
                        targets = self._cross_attn_modules if is_cross else self._layer_modules
                        for idx, attn_mod in targets:
                            orig_fwd = self._make_forward_wrapper(idx, cache, strength=1.0, is_capture=True, taper="none")
                            def make_debug(i, fn, mod):
                                def debug_fwd(self_mod, hidden_states, encoder_hidden_states=None, attention_mask=None, position_embeddings=None):
                                    result = fn(mod, hidden_states, encoder_hidden_states, attention_mask, position_embeddings)
                                    if step < DEBUG_STEPS and i in DEBUG_LAYERS and i in cache:
                                        k = cache[i]["k"]
                                        v = cache[i]["v"]
                                        print(f"[DBG step={step} t={t_now:.3f} L={i:02d}] CAPTURE  K shape={tuple(k.shape)} {_s(k)}")
                                        print(f"[DBG step={step} t={t_now:.3f} L={i:02d}] CAPTURE  V shape={tuple(v.shape)} {_s(v)}")
                                    return result
                                return debug_fwd
                            patched_fn = make_debug(idx, orig_fwd, attn_mod)
                            import types
                            attn_mod.forward = types.MethodType(patched_fn, attn_mod)
                            self._patched_modules.append(attn_mod)

                with HookManager(raw_patched, hook_layer_range) as hm_capture:
                    hm_capture.register_capture_hooks(attn_cache)
                    if prev_wrapper is not None:
                        prev_wrapper(model_function, ref_wrap)
                    else:
                        model_function(ref_wrap["input"], ref_wrap["timestep"], **ref_wrap.get("c", {}))

            # INJECT with debug logging
            with torch.no_grad():
                step_scaled = {k: v for k, v in layer_strengths.items()}

                class DebugInjectHookManager(HookManager):
                    def register_injection_hooks_per_layer(self, cache, layer_str, mode="inject", time_taper="none", is_cross=False):
                        targets = self._cross_attn_modules if is_cross else self._layer_modules
                        for idx, attn_mod in targets:
                            s = layer_str.get(idx, 0.0)
                            if s <= 0.0:
                                continue
                            eff_mode = mode

                            def make_inject_debug(i, strength, eff_m):
                                def inject_fwd(self_mod, hidden_states, encoder_hidden_states=None, attention_mask=None, position_embeddings=None):
                                    from comfy.ldm.ace.ace_step15 import apply_rotary_pos_emb
                                    from comfy.ldm.modules.attention import optimized_attention
                                    bsz, q_len, _ = hidden_states.size()
                                    query_states = self_mod.q_proj(hidden_states)
                                    query_states = query_states.view(bsz, q_len, self_mod.num_heads, self_mod.head_dim)
                                    query_states = self_mod.q_norm(query_states)
                                    query_states = query_states.transpose(1, 2)

                                    kv_len = q_len
                                    key_states   = self_mod.k_proj(hidden_states)
                                    value_states = self_mod.v_proj(hidden_states)
                                    key_states   = key_states.view(bsz, q_len, self_mod.num_kv_heads, self_mod.head_dim)
                                    key_states   = self_mod.k_norm(key_states)
                                    value_states = value_states.view(bsz, q_len, self_mod.num_kv_heads, self_mod.head_dim)
                                    key_states   = key_states.transpose(1, 2)
                                    value_states = value_states.transpose(1, 2)
                                    if position_embeddings is not None:
                                        cos, sin = position_embeddings
                                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                                    if step < DEBUG_STEPS and i in DEBUG_LAYERS:
                                        print(f"[DBG step={step} t={t_now:.3f} L={i:02d}] GEN_LIVE K shape={tuple(key_states.shape)} {_s(key_states)}")

                                    ref_data = cache.get(i)
                                    if ref_data is not None and strength > 0:
                                        ref_k = ref_data["k"].to(key_states.device, key_states.dtype)
                                        ref_v = ref_data["v"].to(key_states.device, key_states.dtype)
                                        # trim/pad to gen length
                                        rt, gt = ref_k.shape[2], key_states.shape[2]
                                        if rt != gt:
                                            if rt > gt:
                                                ref_k, ref_v = ref_k[:,:,:gt,:], ref_v[:,:,:gt,:]
                                            else:
                                                rpt = (gt + rt - 1) // rt
                                                ref_k = ref_k.repeat(1,1,rpt,1)[:,:,:gt,:]
                                                ref_v = ref_v.repeat(1,1,rpt,1)[:,:,:gt,:]
                                        if ref_k.shape[0] == 1 and bsz > 1:
                                            ref_k = ref_k.expand(bsz,-1,-1,-1)
                                            ref_v = ref_v.expand(bsz,-1,-1,-1)

                                        if step < DEBUG_STEPS and i in DEBUG_LAYERS:
                                            print(f"[DBG step={step} t={t_now:.3f} L={i:02d}] REF      K (pre-scale) {_s(ref_k)}  s={strength}")

                                        if eff_m == "replace":
                                            key_states   = ref_k * strength
                                            value_states = ref_v * strength
                                            if step < DEBUG_STEPS and i in DEBUG_LAYERS:
                                                print(f"[DBG step={step} t={t_now:.3f} L={i:02d}] REPLACE  K {_s(key_states)}")
                                        else:
                                            ref_k_scaled = ref_k * strength
                                            ref_v_scaled = ref_v * strength
                                            key_states   = torch.cat([key_states, ref_k_scaled], dim=2)
                                            value_states = torch.cat([value_states, ref_v_scaled], dim=2)
                                            kv_len = key_states.shape[2]
                                            if step < DEBUG_STEPS and i in DEBUG_LAYERS:
                                                print(f"[DBG step={step} t={t_now:.3f} L={i:02d}] CONCAT   K shape={tuple(key_states.shape)} gen_part {_s(key_states[:,:,:gt,:])} ref_part {_s(key_states[:,:,gt:,:])}")

                                    n_rep = self_mod.num_heads // self_mod.num_kv_heads
                                    if n_rep > 1:
                                        key_states   = key_states.repeat_interleave(n_rep, dim=1)
                                        value_states = value_states.repeat_interleave(n_rep, dim=1)

                                    attn_bias = None
                                    if self_mod.sliding_window is not None:
                                        indices_q = torch.arange(q_len, device=query_states.device)
                                        if kv_len > q_len:
                                            indices_k = torch.cat([torch.arange(q_len, device=query_states.device),
                                                                    torch.arange(kv_len - q_len, device=query_states.device)])
                                        else:
                                            indices_k = torch.arange(kv_len, device=query_states.device)
                                        diff = indices_q.unsqueeze(1) - indices_k.unsqueeze(0)
                                        in_window  = torch.abs(diff) <= self_mod.sliding_window
                                        window_bias = torch.zeros((q_len, kv_len), device=query_states.device, dtype=query_states.dtype)
                                        window_bias.masked_fill_(~in_window, torch.finfo(query_states.dtype).min)
                                        attn_bias = window_bias.unsqueeze(0).unsqueeze(0)

                                    attn_output = optimized_attention(query_states, key_states, value_states,
                                                                      self_mod.num_heads, attn_bias, skip_reshape=True,
                                                                      low_precision_attention=False)
                                    attn_output = self_mod.o_proj(attn_output)
                                    if step < DEBUG_STEPS and i in DEBUG_LAYERS:
                                        print(f"[DBG step={step} t={t_now:.3f} L={i:02d}] ATTN_OUT {_s(attn_output)}")
                                    return attn_output
                                return inject_fwd

                            import types
                            fn = make_inject_debug(idx, s, eff_mode)
                            attn_mod.forward = types.MethodType(fn, attn_mod)
                            self._patched_modules.append(attn_mod)

                with DebugInjectHookManager(raw_patched, hook_layer_range) as hm_inject:
                    hm_inject.register_injection_hooks_per_layer(attn_cache, step_scaled, mode=mode)
                    if prev_wrapper is not None:
                        return prev_wrapper(model_function, wrap_kwargs)
                    else:
                        return model_function(wrap_kwargs["input"], wrap_kwargs["timestep"], **wrap_kwargs.get("c", {}))

        patched.set_model_unet_function_wrapper(my_unet_wrapper)
        print(f"[DebugInject] Attached. Active layers: {active_layers}. Will print for first {DEBUG_STEPS} steps on layers {DEBUG_LAYERS}.")
        return (patched,)
