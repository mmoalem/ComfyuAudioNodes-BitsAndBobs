import torch

from ..core import model_utils
from ..core.hook_manager import HookManager
from ..core.tensor_ops import compute_layer_strength, compute_step_multiplier

# Total number of decoder layers in ACE-Step 1.5 XL
_NUM_LAYERS = 32


def _build_input_types():
    """Build INPUT_TYPES dict programmatically so the 32 layer sliders are
    always in order 00-31 without repeating the definition 32 times."""
    required = {
        "model": ("MODEL",),
        "vae": ("VAE",),
        "audio": ("AUDIO",),
        "layer_taper": (
            ["none", "linear", "cosine"],
            {
                "default": "none",
                "tooltip": (
                    "Applies a taper curve as a multiplier on top of each layer's individual "
                    "strength value.  The curve spans taper_start_layer → taper_end_layer.\n"
                    "none   = per-layer strengths used exactly as entered.\n"
                    "linear = ramp from 0× at taper_start_layer to 1× at taper_end_layer.\n"
                    "cosine = smooth S-curve from 0× to 1× over the same range."
                ),
            },
        ),
        "taper_start_layer": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 31,
                "tooltip": "Layer at which the taper envelope begins (factor = 0 here).",
            },
        ),
        "taper_end_layer": (
            "INT",
            {
                "default": 31,
                "min": 0,
                "max": 31,
                "tooltip": "Layer at which the taper envelope reaches full strength (factor = 1 here).",
            },
        ),
        "___": (["----------------------------------------"],),
        "step_taper": (
            ["none", "fade_out", "fade_in", "cosine_fade_out", "cosine_fade_in", "cosine_bell"],
            {
                "default": "none",
                "tooltip": (
                    "Scales ALL layer strengths by a multiplier derived from the current diffusion "
                    "timestep (t≈1 at the first step, t≈0 at the last step).\n"
                    "none             — constant (multiplier = 1.0 every step).\n"
                    "fade_out         — linear 1→0  (strong early, gone by end).\n"
                    "fade_in          — linear 0→1  (absent early, full by end).\n"
                    "cosine_fade_out  — smooth cosine 1→0.\n"
                    "cosine_fade_in   — smooth cosine 0→1.\n"
                    "cosine_bell      — 0→1→0, peaking at the midpoint of sampling."
                ),
            },
        ),
        "time_taper": (
            ["none", "fade_out", "fade_in", "cosine_fade_out", "cosine_fade_in", "cosine_bell"],
            {
                "default": "none",
                "tooltip": (
                    "Temporal scaling: fades the injection strength across the length of the audio sequence.\n"
                    "fade_out = strong at the start of the audio clip, fading to 0 at the end.\n"
                    "fade_in  = 0 at the start of the clip, ramping up to full strength at the end."
                ),
            },
        ),
        "____": (["----------------------------------------"],),
    }
    for i in range(_NUM_LAYERS):
        required[f"layer_{i:02d}"] = (
            "FLOAT",
            {
                "default": 0.0,
                "min": 0.0,
                "max": 5.0,
                "step": 0.05,
                "tooltip": (
                    f"Self-attention KV injection strength for decoder layer {i}. "
                    "0.0 = this layer is skipped entirely (taper does not re-enable it)."
                ),
            },
        )
    return {"required": required}


class PerStepSelfAttentionInjectPerLayer:
    """Per-step self-attention KV injection with individual per-layer strength control.

    Identical to PerStepSelfAttentionInject but exposes a separate strength
    slider for each of the 32 decoder layers.  Layers set to 0 are completely
    bypassed — no monkey-patch is applied, so there is no overhead for unused layers.

    Use-cases
    ---------
    * Inject only a handful of specific layers (e.g. 1, 8, 10) without chaining
      multiple PerStepSelfAttentionInject nodes (chaining causes each injection to
      corrupt the cached KV used by the next one).
    * Fine-tune which layers contribute to the reference effect — early layers tend
      to control coarse structure / rhythm, late layers refine texture / timbre.
    * Asymmetric injection: boost key structural layers while leaving others clean.

    WARNING: Still runs 2× forward passes per step — expect ~2× slower sampling.
    Compatible with BF16, FP16, FP32, FP8, and GGUF checkpoints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _build_input_types()

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject"
    CATEGORY = "ACE-Step/Reference"
    DESCRIPTION = (
        "Per-step self-attention KV injection with individual strength per layer (0-31). "
        "Set a layer's strength to 0 to skip it entirely. "
        "Avoids chaining issues when you only want a specific subset of layers active. "
        "WARNING: 2× forward passes per step — ~2× slower sampling."
    )

    def inject(self, model, vae, audio, layer_taper, taper_start_layer, taper_end_layer, step_taper, time_taper, **kwargs):
        # Collect raw per-layer strengths from kwargs and apply taper as a multiplier.
        # A layer explicitly set to 0 is ALWAYS skipped — taper cannot re-enable it.
        raw_strengths = {}
        for i in range(_NUM_LAYERS):
            raw_strengths[i] = float(kwargs.get(f"layer_{i:02d}", 0.0))

        layer_strengths = {}
        active_layers = []
        for i in range(_NUM_LAYERS):
            raw = raw_strengths[i]
            if raw <= 0.0:
                layer_strengths[i] = 0.0
                continue
            if layer_taper == "none":
                effective = raw
            else:
                # compute_layer_strength returns base_strength × taper_factor.
                # We want just the taper_factor, so pass base_strength=1.0
                # then multiply by the user's raw value.
                taper_factor = compute_layer_strength(
                    i, taper_start_layer, taper_end_layer, 1.0, layer_taper
                )
                effective = raw * taper_factor
            layer_strengths[i] = effective
            if effective > 0.0:
                active_layers.append(i)

        if not active_layers:
            print(
                "[PerStepSelfAttentionInjectPerLayer] WARNING: all effective layer strengths are 0 "
                "(check taper settings or layer sliders). Returning unmodified model."
            )
            return (model,)

        # ----------------------------------------------------------------
        # Encode reference audio to latent
        # ----------------------------------------------------------------
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :, :]

        if sample_rate != 48000:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=48000
            )
            sample_rate = 48000

        waveform_for_vae = waveform.movedim(1, -1)

        with torch.no_grad():
            vae_device, vae_dtype = model_utils.get_vae_device_dtype(vae)
            vae_output = vae.encode(waveform_for_vae.to(vae_device, vae_dtype))
            if hasattr(vae_output, "latent_dist"):
                ref_latent = vae_output.latent_dist.mode()
            else:
                ref_latent = vae_output
        ref_latent = ref_latent.detach().cpu()

        # ----------------------------------------------------------------
        # Clone model and unwrap to raw nn.Module
        # ----------------------------------------------------------------
        patched = model.clone()
        raw_patched = model_utils.get_raw_model(patched)

        prev_wrapper = patched.model_options.get("model_function_wrapper", None)

        # HookManager needs a layer range covering all active layers.
        # We pass (0, 31) so _build_layer_index indexes everything;
        # register_injection_hooks_per_layer then selects which ones to patch.
        hook_layer_range = (0, _NUM_LAYERS - 1)

        # ----------------------------------------------------------------
        # Helpers — identical to PerStepSelfAttentionInject
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # UNet wrapper — runs at every sampling step
        # ----------------------------------------------------------------
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
                        f"[PerStepSelfAttentionInjectPerLayer] Reference T={ref_latent.shape[2]} != "
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

                # Capture pass — hook all layers so we have the full cache available
                attn_cache = {}
                with HookManager(raw_patched, hook_layer_range) as hm_capture:
                    hm_capture.register_capture_hooks(attn_cache)
                    if prev_wrapper is not None:
                        prev_wrapper(model_function, ref_wrap)
                    else:
                        model_function(
                            ref_wrap["input"],
                            ref_wrap["timestep"],
                            **ref_wrap.get("c", {}),
                        )

            # Injection pass — only patch layers with strength > 0 after step scaling
            with torch.no_grad():
                # Step-based multiplier: scale all layer strengths by a factor
                # derived from the current timestep (1.0 at start, 0.0 at end).
                t_now = float(wrap_kwargs["timestep"][0].item())
                step_mult = compute_step_multiplier(t_now, step_taper)
                step_scaled = {k: v * step_mult for k, v in layer_strengths.items()}

                with HookManager(raw_patched, hook_layer_range) as hm_inject:
                    hm_inject.register_injection_hooks_per_layer(attn_cache, step_scaled, time_taper)

                    if prev_wrapper is not None:
                        return prev_wrapper(model_function, wrap_kwargs)
                    else:
                        return model_function(
                            wrap_kwargs["input"],
                            wrap_kwargs["timestep"],
                            **wrap_kwargs.get("c", {}),
                        )

        patched.set_model_unet_function_wrapper(my_unet_wrapper)

        eff = {i: round(layer_strengths[i], 4) for i in active_layers}
        print(
            f"[PerStepSelfAttentionInjectPerLayer] Attached — "
            f"layer_taper={layer_taper} [{taper_start_layer}→{taper_end_layer}], "
            f"step_taper={step_taper}, time_taper={time_taper}, "
            f"active layers: {active_layers}\n"
            f"  base strengths (after layer taper): {eff}\n"
            f"[PerStepSelfAttentionInjectPerLayer] WARNING: 2x forward passes per step. "
            f"Expect ~2x slower sampling."
        )

        return (patched,)
