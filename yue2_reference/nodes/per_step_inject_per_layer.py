import torch

from ..core.hook_manager import YuE2HookManager
from ..core import model_utils
from ..core.tensor_ops import compute_layer_strength, compute_step_multiplier
from .kv_inject import _encode_audio

# Total decoder layers in YuE2-3B
_NUM_LAYERS = 28


def _build_input_types():
    required = {
        "model": ("MODEL",),
        "vae": ("VAE",),
        "audio": ("AUDIO",),
        "mode": (
            ["inject", "replace", "threshold"],
            {
                "default": "inject",
                "tooltip": (
                    "inject    — Scaled-concat: reference K/V appended to generation K/V.\n"
                    "replace   — Hard replace: generation K/V discarded, reference used directly.\n"
                    "threshold — inject if s <= 1.0, replace if s > 1.0."
                ),
            },
        ),
        "layer_taper": (
            ["none", "linear", "cosine"],
            {
                "default": "none",
                "tooltip": (
                    "Multiplier curve applied on top of each layer's individual strength value.\n"
                    "none   = per-layer strengths used as entered.\n"
                    "linear = ramp 0→1 from taper_start_layer to taper_end_layer.\n"
                    "cosine = smooth S-curve 0→1 over the same range."
                ),
            },
        ),
        "taper_start_layer": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": _NUM_LAYERS - 1,
                "tooltip": "Layer at which the taper envelope begins (factor = 0 here).",
            },
        ),
        "taper_end_layer": (
            "INT",
            {
                "default": _NUM_LAYERS - 1,
                "min": 0,
                "max": _NUM_LAYERS - 1,
                "tooltip": "Layer at which the taper envelope reaches full strength (factor = 1 here).",
            },
        ),
        "___": (["----------------------------------------"],),
        "step_taper": (
            ["none", "fade_out", "fade_in", "cosine_fade_out", "cosine_fade_in", "cosine_bell"],
            {
                "default": "none",
                "tooltip": (
                    "Scales ALL layer strengths by a factor derived from the current timestep.\n"
                    "none            = constant (1.0 every step).\n"
                    "fade_out        = linear 1→0 (strong early, gone by end).\n"
                    "fade_in         = linear 0→1 (absent early, full by end).\n"
                    "cosine_fade_out = smooth cosine 1→0.\n"
                    "cosine_fade_in  = smooth cosine 0→1.\n"
                    "cosine_bell     = 0→1→0, peaking at the midpoint of sampling."
                ),
            },
        ),
        "time_taper": (
            ["none", "fade_out", "fade_in", "cosine_fade_out", "cosine_fade_in", "cosine_bell"],
            {
                "default": "none",
                "tooltip": (
                    "Temporal scaling: fades injection strength across the audio sequence length.\n"
                    "fade_out = strong at the start of the clip, fading to 0 at the end.\n"
                    "fade_in  = 0 at the start, ramping to full strength at the end."
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
                    "0.0 = this layer is skipped entirely."
                ),
            },
        )
    return {"required": required}


class YuE2PerStepSAInjectPerLayer:
    """Per-step YuE2 self-attention KV injection with individual per-layer strength control.

    Runs 2× forward passes per sampling step (one reference capture, one generation inject).
    Accepts a reference LATENT directly — no VAE re-encoding needed.

    Use-cases
    ---------
    * Inject only specific layers to transfer structure (early layers) or
      timbre/texture (late layers) from a reference piece.
    * Combine layer-level and step-level tapers for fine-grained control.

    WARNING: ~2× slower sampling due to double forward passes.
    Compatible with BF16, FP16, FP32 checkpoints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _build_input_types()

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject"
    CATEGORY = "YUE2/Reference"
    DESCRIPTION = (
        "Per-step self-attention KV injection for YuE2 with individual strength per layer (0-27). "
        "Set a layer's strength to 0 to skip it entirely. "
        "WARNING: 2× forward passes per step — ~2× slower sampling."
    )

    def inject(
        self, model, vae, audio, mode, layer_taper, taper_start_layer, taper_end_layer,
        step_taper, time_taper, **kwargs
    ):
        # Collect and taper per-layer strengths
        raw_strengths = {i: float(kwargs.get(f"layer_{i:02d}", 0.0)) for i in range(_NUM_LAYERS)}
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
                taper_factor = compute_layer_strength(
                    i, taper_start_layer, taper_end_layer, 1.0, layer_taper
                )
                effective = raw * taper_factor
            layer_strengths[i] = effective
            if effective > 0.0:
                active_layers.append(i)

        if not active_layers:
            print(
                "[YuE2PerStepSAInjectPerLayer] WARNING: all effective layer strengths are 0 "
                "(check taper settings or layer sliders). Returning unmodified model."
            )
            return (model,)

        # Encode reference audio using the YuE2 VAE from the checkpoint
        ref_latent_cpu = _encode_audio(vae, audio)

        patched = model.clone()
        raw_patched = model_utils.get_raw_yue2_model(patched)
        prev_wrapper = patched.model_options.get("model_function_wrapper", None)
        hook_layer_range = (0, _NUM_LAYERS - 1)

        def _match_t(ref, gen_t):
            ref_t = ref.shape[2]
            if ref_t == gen_t:
                return ref
            if ref_t > gen_t:
                return ref[:, :, :gen_t]
            repeats = (gen_t + ref_t - 1) // ref_t
            return ref.repeat(1, 1, repeats)[:, :, :gen_t]

        def _build_null_ref_wrap(wrap_kwargs, noised_ref, ref_timestep):
            """Build a conditioning dict for the reference capture pass.

            We zero out the cross-attention conditioning so the reference pass
            uses neutral text conditioning, only inheriting structural yue2_chunks.
            """
            null_c = {}
            for k, v in wrap_kwargs.get("c", {}).items():
                if k == "context" and isinstance(v, torch.Tensor):
                    null_c[k] = torch.zeros_like(v[:1])
                elif isinstance(v, torch.Tensor) and v.shape[0] > 1:
                    null_c[k] = v[:1]
                else:
                    null_c[k] = v
            ref_wrap = {}
            ref_wrap["input"] = noised_ref
            ref_wrap["timestep"] = ref_timestep
            ref_wrap["cond_or_uncond"] = [0]
            ref_wrap["c"] = null_c
            return ref_wrap

        _t_logged = [False]

        def my_unet_wrapper(model_function, wrap_kwargs):
            input_t = wrap_kwargs["input"]
            ref_dev = input_t.device
            ref_dtype = input_t.dtype
            gen_t = input_t.shape[2]

            with torch.no_grad():
                ref_matched = _match_t(
                    ref_latent_cpu.to(device=ref_dev, dtype=ref_dtype), gen_t
                )
                if not _t_logged[0] and ref_latent_cpu.shape[2] != gen_t:
                    print(
                        f"[YuE2PerStepSAInjectPerLayer] Reference T={ref_latent_cpu.shape[2]} != "
                        f"generation T={gen_t} — "
                        f"{'trimmed' if ref_latent_cpu.shape[2] > gen_t else 'repeat-padded'} "
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
                with YuE2HookManager(raw_patched, hook_layer_range) as hm_capture:
                    hm_capture.register_capture_hooks(attn_cache)
                    if prev_wrapper is not None:
                        prev_wrapper(model_function, ref_wrap)
                    else:
                        model_function(
                            ref_wrap["input"],
                            ref_wrap["timestep"],
                            **ref_wrap.get("c", {}),
                        )

            # Injection pass with step-level scaling
            t_now = float(wrap_kwargs["timestep"][0].item())
            step_mult = compute_step_multiplier(t_now, step_taper)
            step_scaled = {k: v * step_mult for k, v in layer_strengths.items()}

            with YuE2HookManager(raw_patched, hook_layer_range) as hm_inject:
                hm_inject.register_injection_hooks_per_layer(
                    attn_cache, step_scaled, mode=mode, time_taper=time_taper
                )
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
            f"[YuE2PerStepSAInjectPerLayer] Attached — "
            f"mode={mode}, layer_taper={layer_taper} [{taper_start_layer}→{taper_end_layer}], "
            f"step_taper={step_taper}, time_taper={time_taper}, "
            f"active layers: {active_layers}\n"
            f"  base strengths (after layer taper): {eff}\n"
            f"[YuE2PerStepSAInjectPerLayer] WARNING: 2x forward passes per step. "
            f"Expect ~2x slower sampling."
        )
        return (patched,)
