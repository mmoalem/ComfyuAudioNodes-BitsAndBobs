import torch
import torch.nn.functional as F

from ..core import model_utils
from ..core.hook_manager import HookManager


class SelfAttentionInject:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "kv_activations": ("KV_ACTIVATIONS",),
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
    CATEGORY = "ACE-Step/Reference"
    DESCRIPTION = (
        "Injects reference self-attention KVs into generation via true concatenation. "
        "Waits until the sampler hits the targeted sigma fraction, runs a reference "
        "forward pass to capture pre-rotated KVs, then concatenates them at every "
        "subsequent step.\n"
        "NOTE: This node handles self-attention KV injection only. "
        "For timbre conditioning, connect TimbreConditioningInject separately before this node. "
        "Compatible with BF16, FP16, FP32, FP8, and GGUF checkpoints."
    )

    def inject(self, model, kv_activations, strength, taper):
        ref_latent = kv_activations["ref_latent"]          # [1, 64, T_ref] CPU float32
        start_layer, end_layer = kv_activations["layer_range"]
        capture_timestep_frac = kv_activations["capture_timestep_frac"]

        current_hash = model_utils.compute_model_hash(model)
        if kv_activations.get("model_hash") != current_hash:
            print("[SelfAttentionInject] WARNING: model hash mismatch.")

        kv_activations = kv_activations.copy()

        patched = model.clone()
        raw_patched = model_utils.get_raw_model(patched)

        prev_wrapper = patched.model_options.get("model_function_wrapper", None)

        def _match_t_to_generation(ref, gen_t):
            """
            Trim or pad ref latent along the time axis (dim=2) to match gen_t.

            The model's prepare_condition concatenates src_latents (from the
            generation input) with chunk_masks along dim=-1 (time). If the
            reference latent has a different T than the generation latent, the
            cat fails. This happens when Generate Audio Codes is off and the
            reference audio is a different duration than the generation target.

            Trim:  ref T > gen_t  → take first gen_t frames (centre of audio)
            Pad:   ref T < gen_t  → repeat-pad on the right to reach gen_t
                   (repeat is better than zero-pad: avoids hard silence boundary
                    artefacts in the captured KVs)
            """
            ref_t = ref.shape[2]
            if ref_t == gen_t:
                return ref
            if ref_t > gen_t:
                # Trim: take the first gen_t frames
                return ref[:, :, :gen_t]
            # Pad: repeat-tile then trim to exact length
            repeats = (gen_t + ref_t - 1) // ref_t
            return ref.repeat(1, 1, repeats)[:, :, :gen_t]

        def _slice_conditioning_to_batch1(wrap_kw):
            """
            Slice all batch-dependent tensors in wrap_kw["c"] to batch size 1,
            selecting the first conditional item derived from cond_or_uncond.
            """
            cond_or_uncond = wrap_kw.get("cond_or_uncond", None)

            cond_idx = 0  # safe default — ComfyUI puts conditional first
            if cond_or_uncond is not None:
                for idx, v in enumerate(cond_or_uncond):
                    if int(v) == 0:
                        cond_idx = idx
                        break

            sliced_c = {}
            for k, v in wrap_kw.get("c", {}).items():
                if isinstance(v, torch.Tensor) and v.shape[0] > 1:
                    sliced_c[k] = v[cond_idx:cond_idx+1]
                else:
                    sliced_c[k] = v

            ref_wrap = dict(wrap_kw)
            ref_wrap["c"] = sliced_c
            ref_wrap["cond_or_uncond"] = [0]
            return ref_wrap

        # Mutable state for one sampling run
        state = {
            "attn_cache": None,
            "sigma_max": None,
            "capture_done": False,
        }

        def _reset_state_if_new_run(current_sigma):
            if state["sigma_max"] is not None and current_sigma > state["sigma_max"] * 1.05:
                print("[SelfAttentionInject] New sampling run detected — resetting state.")
                state["attn_cache"] = None
                state["sigma_max"] = None
                state["capture_done"] = False

        def my_unet_wrapper(model_function, wrap_kwargs):
            current_sigma = wrap_kwargs["timestep"].max().item()

            _reset_state_if_new_run(current_sigma)

            if state["sigma_max"] is None:
                state["sigma_max"] = current_sigma

            # Target sigma for KV capture
            target_sigma = state["sigma_max"] * capture_timestep_frac

            # Floor to minimum schedulable sigma
            sigmas = wrap_kwargs.get("c", {}).get("sigmas", None)
            if sigmas is not None and len(sigmas) >= 2:
                target_sigma = max(target_sigma, sigmas[-2].item())
            else:
                target_sigma = max(target_sigma, 0.01)

            input_tensor = wrap_kwargs["input"]
            ref_dev = input_tensor.device
            ref_dtype = input_tensor.dtype

            # === STEP 1: Capture reference KVs at targeted sigma ===
            if not state["capture_done"] and current_sigma <= target_sigma:
                # Match reference latent T to the generation input T.
                # The model's prepare_condition concatenates src_latents (from the
                # forward pass input) with chunk_masks along the time axis. If the
                # reference audio is a different duration than the generation target,
                # this cat fails. Trimming/padding here fixes the mismatch regardless
                # of whether Generate Audio Codes is on or off.
                gen_t = input_tensor.shape[2]
                ref_matched = _match_t_to_generation(
                    ref_latent.to(device=ref_dev, dtype=ref_dtype), gen_t
                )

                if ref_latent.shape[2] != gen_t:
                    print(
                        f"[SelfAttentionInject] Reference T={ref_latent.shape[2]} != "
                        f"generation T={gen_t} — "
                        f"{'trimmed' if ref_latent.shape[2] > gen_t else 'repeat-padded'} "
                        f"to T={gen_t} for capture pass."
                    )

                noise = torch.randn_like(ref_matched)
                ref_timestep = wrap_kwargs["timestep"][:1]

                if hasattr(patched, "model") and hasattr(patched.model, "model_sampling"):
                    noised_ref = patched.model.model_sampling.noise_scaling(
                        ref_timestep, noise, ref_matched
                    )
                else:
                    noised_ref = ref_matched + noise * ref_timestep.view(-1, 1, 1).to(ref_dev, ref_dtype)

                # Slice all conditioning to batch 1 for the reference forward pass
                ref_wrap = _slice_conditioning_to_batch1(wrap_kwargs)
                ref_wrap["input"] = noised_ref
                ref_wrap["timestep"] = ref_timestep

                attn_cache = {}
                with HookManager(raw_patched, (start_layer, end_layer)) as hm:
                    hm.register_capture_hooks(attn_cache)
                    with torch.no_grad():
                        if prev_wrapper is not None:
                            prev_wrapper(model_function, ref_wrap)
                        else:
                            model_function(
                                ref_wrap["input"],
                                ref_wrap["timestep"],
                                **ref_wrap.get("c", {})
                            )

                state["attn_cache"] = attn_cache
                kv_activations["attn_cache"] = attn_cache
                state["capture_done"] = True
                print(
                    f"[SelfAttentionInject] Captured {len(attn_cache)} layer KVs "
                    f"at sigma={current_sigma:.4f} (target={target_sigma:.4f})"
                )

            # Warn if approaching end of sampling without having captured
            if (
                not state["capture_done"]
                and sigmas is not None
                and current_sigma <= sigmas[-2].item() * 1.1
            ):
                print(
                    f"[SelfAttentionInject] WARNING: approaching final step "
                    f"(sigma={current_sigma:.4f}) but KV capture has not fired. "
                    f"capture_timestep_frac={capture_timestep_frac} may be too low "
                    f"for this scheduler. Try increasing it."
                )

            # === STEP 2: Generation forward pass with KV injection ===
            with HookManager(raw_patched, (start_layer, end_layer)) as hm_inject:
                if state["capture_done"] and state["attn_cache"] is not None:
                    hm_inject.register_injection_hooks(state["attn_cache"], strength, taper)
                elif state["capture_done"]:
                    print("[SelfAttentionInject] WARNING: capture_done but attn_cache is None")

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
            f"[SelfAttentionInject] Wrapper attached — "
            f"capture_frac={capture_timestep_frac}, "
            f"layers={start_layer}-{end_layer}, "
            f"strength={strength}, taper={taper}"
        )
        return (patched,)
