import torch

from ..core import model_utils
from ..core.hook_manager import HookManager


class PerStepSelfAttentionInject:
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
        "Per-step real-time true concatenative KV injection. At every sampling step, "
        "captures reference KVs from the reference audio at the matching noise level, "
        "then injects them into the generation forward pass. "
        "WARNING: runs 2x full forward passes per step — expect ~2x slower sampling. "
        "Compatible with BF16, FP16, FP32, FP8, and GGUF checkpoints."
    )

    def inject(
        self,
        model,
        vae,
        audio,
        strength,
        taper,
        start_layer,
        end_layer,
    ):
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

        patched = model.clone()
        raw_patched = model_utils.get_raw_model(patched)

        prev_wrapper = patched.model_options.get("model_function_wrapper", None)

        def _match_t_to_generation(ref, gen_t):
            """
            Trim or repeat-pad ref latent along the time axis (dim=2) to match gen_t.

            The model's prepare_condition (ace_step15.py) concatenates src_latents
            (from the forward pass input) with chunk_masks along the time axis.
            If ref has a different T than the generation input this cat crashes —
            regardless of whether Generate Audio Codes is on or off.

            Trim:        ref T > gen_t → take first gen_t frames
            Repeat-pad:  ref T < gen_t → tile then trim to gen_t
                         (repeat avoids hard silence-boundary artefacts in captured KVs)
            """
            ref_t = ref.shape[2]
            if ref_t == gen_t:
                return ref
            if ref_t > gen_t:
                return ref[:, :, :gen_t]
            repeats = (gen_t + ref_t - 1) // ref_t
            return ref.repeat(1, 1, repeats)[:, :, :gen_t]

        def _build_null_ref_wrap(wrap_kwargs, noised_ref, ref_timestep):
            """
            Build a reference forward pass with:
              - batch size 1 (the reference latent, T-matched to generation)
              - all conditioning tensors sliced to batch 1
              - null (zeroed) cross-attention conditioning

            Using null conditioning for the reference pass ensures captured KVs
            represent the reference audio's acoustic structure only, not a blend
            with the generation prompt.
            """
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

        # Cache the T-mismatch log so we only print it once per run
        _t_logged = [False]

        def my_unet_wrapper(model_function, wrap_kwargs):
            input_t = wrap_kwargs["input"]
            ref_dev = input_t.device
            ref_dtype = input_t.dtype
            gen_t = input_t.shape[2]

            with torch.no_grad():
                # Match reference latent T to generation input T before noising.
                # Without this, prepare_condition inside ace_step15.py crashes when
                # it tries to cat src_latents (ref T) with chunk_masks (gen T).
                ref_matched = _match_t_to_generation(
                    ref_latent.to(device=ref_dev, dtype=ref_dtype), gen_t
                )

                if not _t_logged[0] and ref_latent.shape[2] != gen_t:
                    print(
                        f"[PerStepSelfAttentionInject] Reference T={ref_latent.shape[2]} != "
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
                with HookManager(raw_patched, (start_layer, end_layer)) as hm_capture:
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
                with HookManager(raw_patched, (start_layer, end_layer)) as hm_inject:
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
            f"[PerStepSelfAttentionInject] Attached — "
            f"layers={start_layer}-{end_layer}, strength={strength}, taper={taper}\n"
            f"[PerStepSelfAttentionInject] WARNING: 2x forward passes per step. "
            f"Expect ~2x slower sampling."
        )

        return (patched,)
