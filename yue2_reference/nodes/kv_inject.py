import torch

from ..core.hook_manager import YuE2HookManager
from ..core import model_utils

# Total decoder layers in YuE2-3B
_NUM_LAYERS = 28
# YuE2 VAE sample rate (AudioOobleckVAE in the checkpoint)
_SAMPLE_RATE = 48000


def _encode_audio(vae, audio):
    """Encode reference audio waveform to YuE2 latent [1, 64, T]."""
    waveform = audio["waveform"]   # [B, C, samples] or [C, samples]
    sample_rate = audio["sample_rate"]

    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform[:1]

    if sample_rate != _SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=_SAMPLE_RATE
        )

    # VAE expects channel-last [B, T_samples, C] — same as ACE-Step
    waveform_for_vae = waveform.movedim(1, -1)

    vae_device = getattr(vae, "device", torch.device("cpu"))
    vae_dtype = getattr(vae, "dtype", None) or getattr(vae, "vae_dtype", torch.float32)
    if vae_dtype is None:
        vae_dtype = torch.float32

    with torch.no_grad():
        vae_out = vae.encode(waveform_for_vae.to(vae_device, vae_dtype))
        if hasattr(vae_out, "latent_dist"):
            latent = vae_out.latent_dist.mode()
        else:
            latent = vae_out

    return latent.detach().cpu().float()


class YuE2SelfAttentionCapture:
    """Encodes reference audio as a YuE2 latent for use by YuE2SelfAttentionInject.

    The actual self-attention KV capture happens inside the sampling loop on the
    first step that meets the sigma threshold.

    Connect KV_ACTIVATIONS_YUE2 → YuE2SelfAttentionInject.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "audio": ("AUDIO",),
                "capture_timestep_frac": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Fraction of the sigma range at which to perform the reference "
                            "KV capture pass (0=low noise, 1=high noise). "
                            "0.5 = midpoint balances structure and texture."
                        ),
                    },
                ),
                "start_layer": ("INT", {"default": 0, "min": 0, "max": _NUM_LAYERS - 1}),
                "end_layer": ("INT", {"default": _NUM_LAYERS - 1, "min": 0, "max": _NUM_LAYERS - 1}),
            },
        }

    RETURN_TYPES = ("KV_ACTIVATIONS_YUE2",)
    RETURN_NAMES = ("kv_activations",)
    FUNCTION = "capture"
    CATEGORY = "YUE2/Reference"
    DESCRIPTION = (
        "Encodes reference audio using the YuE2 VAE (included in the checkpoint). "
        "The actual self-attention KV capture runs inside the sampler at the first step "
        "meeting the sigma threshold. Connect to YuE2SelfAttentionInject."
    )

    def capture(self, model, vae, audio, capture_timestep_frac, start_layer, end_layer):
        ref_latent = _encode_audio(vae, audio)
        print(
            f"[YuE2SelfAttentionCapture] Encoded reference: shape={list(ref_latent.shape)}, "
            f"capture_frac={capture_timestep_frac:.2f}, layers={start_layer}-{end_layer}"
        )
        kv_activations = {
            "ref_latent": ref_latent,
            "capture_timestep_frac": capture_timestep_frac,
            "layer_range": (start_layer, end_layer),
            "attn_cache": {},
        }
        return (kv_activations,)


class YuE2SelfAttentionInject:
    """Injects reference self-attention KVs into YuE2 generation.

    At the first sampling step meeting capture_timestep_frac, runs a reference
    forward pass to capture per-layer post-RoPE K/V. From that point forward,
    concatenates the cached reference K/V onto the generation K/V at every step.

    NOTE: Runs 1 extra forward pass at capture time only (not every step).
    Compatible with BF16, FP16, FP32 checkpoints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "kv_activations": ("KV_ACTIVATIONS_YUE2",),
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
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject"
    CATEGORY = "YUE2/Reference"
    DESCRIPTION = (
        "Injects reference self-attention KVs into YuE2 generation via true concatenation. "
        "Captures KVs once at the target sigma fraction; injects at every subsequent step. "
        "Compatible with BF16, FP16, FP32 checkpoints."
    )

    def inject(self, model, kv_activations, strength, taper):
        ref_latent = kv_activations["ref_latent"]
        start_layer, end_layer = kv_activations["layer_range"]
        capture_timestep_frac = kv_activations["capture_timestep_frac"]

        kv_activations = kv_activations.copy()
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

        def _build_ref_wrap(wrap_kwargs, noised_ref, ref_timestep):
            cond_or_uncond = wrap_kwargs.get("cond_or_uncond", None)
            cond_idx = 0
            if cond_or_uncond is not None:
                for i, v in enumerate(cond_or_uncond):
                    if int(v) == 0:
                        cond_idx = i
                        break

            sliced_c = {}
            for k, v in wrap_kwargs.get("c", {}).items():
                if isinstance(v, torch.Tensor) and v.shape[0] > 1:
                    sliced_c[k] = v[cond_idx : cond_idx + 1]
                else:
                    sliced_c[k] = v

            ref_wrap = dict(wrap_kwargs)
            ref_wrap["input"] = noised_ref
            ref_wrap["timestep"] = ref_timestep
            ref_wrap["cond_or_uncond"] = [0]
            ref_wrap["c"] = sliced_c
            return ref_wrap

        state = {
            "attn_cache": None,
            "sigma_max": None,
            "capture_done": False,
        }

        def _reset_if_new_run(current_sigma):
            if state["sigma_max"] is not None and current_sigma > state["sigma_max"] * 1.05:
                print("[YuE2SelfAttentionInject] New sampling run — resetting state.")
                state["attn_cache"] = None
                state["sigma_max"] = None
                state["capture_done"] = False

        def my_unet_wrapper(model_function, wrap_kwargs):
            current_sigma = wrap_kwargs["timestep"].max().item()
            _reset_if_new_run(current_sigma)
            if state["sigma_max"] is None:
                state["sigma_max"] = current_sigma

            target_sigma = state["sigma_max"] * capture_timestep_frac
            sigmas = wrap_kwargs.get("c", {}).get("sigmas", None)
            if sigmas is not None and len(sigmas) >= 2:
                target_sigma = max(target_sigma, sigmas[-2].item())
            else:
                target_sigma = max(target_sigma, 0.01)

            input_tensor = wrap_kwargs["input"]
            ref_dev = input_tensor.device
            ref_dtype = input_tensor.dtype

            if not state["capture_done"] and current_sigma <= target_sigma:
                gen_t = input_tensor.shape[2]
                ref_matched = _match_t(
                    ref_latent.to(device=ref_dev, dtype=ref_dtype), gen_t
                )
                if ref_latent.shape[2] != gen_t:
                    print(
                        f"[YuE2SelfAttentionInject] Reference T={ref_latent.shape[2]} != "
                        f"generation T={gen_t} — "
                        f"{'trimmed' if ref_latent.shape[2] > gen_t else 'repeat-padded'} to T={gen_t}."
                    )

                noise = torch.randn_like(ref_matched)
                ref_timestep = wrap_kwargs["timestep"][:1]
                if hasattr(patched, "model") and hasattr(patched.model, "model_sampling"):
                    noised_ref = patched.model.model_sampling.noise_scaling(
                        ref_timestep, noise, ref_matched
                    )
                else:
                    noised_ref = ref_matched + noise * ref_timestep.view(-1, 1, 1).to(ref_dev, ref_dtype)

                ref_wrap = _build_ref_wrap(wrap_kwargs, noised_ref, ref_timestep)
                attn_cache = {}
                with YuE2HookManager(raw_patched, hook_layer_range) as hm:
                    hm.register_capture_hooks(attn_cache)
                    with torch.no_grad():
                        if prev_wrapper is not None:
                            prev_wrapper(model_function, ref_wrap)
                        else:
                            model_function(
                                ref_wrap["input"],
                                ref_wrap["timestep"],
                                **ref_wrap.get("c", {}),
                            )

                state["attn_cache"] = attn_cache
                kv_activations["attn_cache"] = attn_cache
                state["capture_done"] = True
                print(
                    f"[YuE2SelfAttentionInject] Captured {len(attn_cache)} layer KVs "
                    f"at sigma={current_sigma:.4f} (target={target_sigma:.4f})"
                )

            if (
                not state["capture_done"]
                and sigmas is not None
                and current_sigma <= sigmas[-2].item() * 1.1
            ):
                print(
                    f"[YuE2SelfAttentionInject] WARNING: approaching final step "
                    f"(sigma={current_sigma:.4f}) but KV capture has not fired. "
                    f"capture_timestep_frac={capture_timestep_frac} may be too low."
                )

            with YuE2HookManager(raw_patched, hook_layer_range) as hm_inject:
                if state["capture_done"] and state["attn_cache"]:
                    hm_inject.register_injection_hooks(state["attn_cache"], strength, taper)

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
            f"[YuE2SelfAttentionInject] Wrapper attached — "
            f"capture_frac={capture_timestep_frac}, "
            f"layers={start_layer}-{end_layer}, strength={strength}, taper={taper}"
        )
        return (patched,)
