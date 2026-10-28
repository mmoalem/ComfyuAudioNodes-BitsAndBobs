import torch

from ..core.hook_manager import YuE2HookManager
from ..core import model_utils
from ..core.tensor_ops import compute_layer_strength, compute_step_multiplier, compute_time_multiplier
from .kv_inject import _encode_audio

# Total decoder layers in YuE2-3B
_NUM_LAYERS = 28

TAPER_OPTIONS = ["none", "linear", "cosine", "cosine_fade_out"]
STEP_TAPER_OPTIONS = [
    "none", "fade_out", "fade_in", "cosine_fade_out", "cosine_fade_in", "cosine_bell"
]


class YuE2PerStepSAInjectStepLimited:
    """Per Step Self Attention Inject (step+time taper, step limited) for YuE2.

    Single global injection strength applied uniformly across the layer range,
    with optional layer taper, step taper, and time taper — matching the
    ACE-Step Legacy2StepLimited node but adapted for YuE2-3B (28 layers).

    The step_taper curve can be compressed into the first N steps of the run via:
      step_taper_total_steps — total steps on your KSampler
      step_taper_end_step    — step by which the taper curve completes and holds

    Example: total_steps=35, end_step=25, step_taper=cosine_fade_out
    → injection fades to 0 by step 25, stays at 0 for steps 25-35.

    WARNING: runs 2× full forward passes per step — expect ~2× slower sampling.
    Compatible with BF16, FP16, FP32 checkpoints.
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
                "layer_taper": (
                    TAPER_OPTIONS,
                    {
                        "default": "none",
                        "tooltip": (
                            "Strength taper across start_layer→end_layer (layer-based, not step-based).\n"
                            "none            — flat strength across all layers in range.\n"
                            "linear          — ramp 0→strength from start_layer to end_layer.\n"
                            "cosine          — smooth S-curve 0→strength from start_layer to end_layer.\n"
                            "cosine_fade_out — smooth cosine strength→0 from start_layer to end_layer."
                        ),
                    },
                ),
                "step_taper": (
                    STEP_TAPER_OPTIONS,
                    {
                        "default": "none",
                        "tooltip": (
                            "Scales injection strength across diffusion steps (t≈1 at step 1, t≈0 at end).\n"
                            "none            — constant strength every step.\n"
                            "fade_out        — linear 1→0 (strong early, gone by end).\n"
                            "fade_in         — linear 0→1 (absent early, full by end).\n"
                            "cosine_fade_out — smooth cosine 1→0.\n"
                            "cosine_fade_in  — smooth cosine 0→1.\n"
                            "cosine_bell     — peaks at the midpoint of sampling (0→1→0).\n"
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
                            "Total sampling steps on your KSampler. "
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
                            "Step (out of step_taper_total_steps) by which the step_taper curve "
                            "reaches its terminal value and holds there for the rest of sampling.\n"
                            "Example: total_steps=35, end_step=25 with step_taper=cosine_fade_out "
                            "→ injection fades to 0 by step 25, then stays at 0 for steps 25-35.\n"
                            "Set equal to step_taper_total_steps to reproduce full-run taper behaviour."
                        ),
                    },
                ),
                "time_taper": (
                    STEP_TAPER_OPTIONS,
                    {
                        "default": "none",
                        "tooltip": (
                            "Temporal scaling: fades injection strength across the audio sequence length.\n"
                            "fade_out = strong at the start of the clip, fading to 0 at the end.\n"
                            "fade_in  = 0 at the start of the clip, ramping to full strength at the end."
                        ),
                    },
                ),
                "start_layer": ("INT", {"default": 0, "min": 0, "max": _NUM_LAYERS - 1}),
                "end_layer": ("INT", {"default": _NUM_LAYERS - 1, "min": 0, "max": _NUM_LAYERS - 1}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject"
    CATEGORY = "YUE2/Reference"
    DESCRIPTION = (
        "Per-step self-attention KV injection for YuE2 with a single global strength, "
        "layer taper, step taper, and time taper. The step_taper curve can be compressed "
        "into the first N steps via step_taper_total_steps / step_taper_end_step — useful "
        "for fading injection to zero before sampling finishes. "
        "WARNING: 2× forward passes per step — expect ~2× slower sampling."
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

        ref_latent_cpu = _encode_audio(vae, audio)

        patched = model.clone()
        raw_patched = model_utils.get_raw_yue2_model(patched)
        prev_wrapper = patched.model_options.get("model_function_wrapper", None)
        hook_layer_range = (start_layer, end_layer)

        def _match_t(ref, gen_t):
            ref_t = ref.shape[2]
            if ref_t == gen_t:
                return ref
            if ref_t > gen_t:
                return ref[:, :, :gen_t]
            repeats = (gen_t + ref_t - 1) // ref_t
            return ref.repeat(1, 1, repeats)[:, :, :gen_t]

        def _build_null_ref_wrap(wrap_kwargs, noised_ref, ref_timestep):
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

        # Pre-compute per-layer effective strengths (layer taper only; step mult applied per step)
        layer_strengths = {}
        for i in range(start_layer, end_layer + 1):
            if layer_taper == "none":
                layer_strengths[i] = 1.0
            else:
                layer_strengths[i] = compute_layer_strength(
                    i, start_layer, end_layer, 1.0, layer_taper
                )

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
                        f"[YuE2PerStepSAInjectStepLimited] Reference T={ref_latent_cpu.shape[2]} != "
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

            t_now = float(wrap_kwargs["timestep"][0].item())
            step_mult = compute_step_multiplier(t_now, step_taper) if step_taper == "none" else \
                _compute_step_multiplier_limited(t_now, step_taper, step_taper_end_pct)
            effective_strength = strength * step_mult

            # Build per-layer strengths for this step (layer_taper * step_mult * strength)
            step_layer_strengths = {
                i: layer_strengths[i] * effective_strength
                for i in range(start_layer, end_layer + 1)
            }

            with YuE2HookManager(raw_patched, hook_layer_range) as hm_inject:
                hm_inject.register_injection_hooks_per_layer(
                    attn_cache, step_layer_strengths, mode="inject", time_taper=time_taper
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
        print(
            f"[YuE2PerStepSAInjectStepLimited] Attached — "
            f"layers={start_layer}-{end_layer}, strength={strength}, "
            f"layer_taper={layer_taper}, step_taper={step_taper} "
            f"(ends by step {end_step}/{total_steps}, {step_taper_end_pct:.1f}% of progress), "
            f"time_taper={time_taper}\n"
            f"[YuE2PerStepSAInjectStepLimited] WARNING: 2x forward passes per step. "
            f"Expect ~2x slower sampling."
        )
        return (patched,)


def _compute_step_multiplier_limited(t: float, mode: str, end_pct: float) -> float:
    """compute_step_multiplier with end_pct compression (matches the ACE-Step Legacy2SL version)."""
    import math
    t = max(0.0, min(1.0, float(t)))
    progress = 1.0 - t
    cutoff = max(0.0001, min(1.0, end_pct / 100.0))
    scaled = min(1.0, progress / cutoff)
    if mode == "none":
        return 1.0
    elif mode == "fade_out":
        return 1.0 - scaled
    elif mode == "fade_in":
        return scaled
    elif mode == "cosine_fade_out":
        return (1.0 + math.cos(scaled * math.pi)) / 2.0
    elif mode == "cosine_fade_in":
        return (1.0 - math.cos(scaled * math.pi)) / 2.0
    elif mode == "cosine_bell":
        return math.sin(scaled * math.pi)
    return 1.0
