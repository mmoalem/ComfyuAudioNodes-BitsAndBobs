import torch

from ..core import model_utils


class TimbreConditioningInject:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "timbre_conditioning": ("TIMBRE_CONDITIONING",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": (
                            "0.0 = no timbre effect. "
                            "1.0 = full reference timbre at the CLS token. "
                            ">1.0 extrapolates beyond the reference for stronger emphasis "
                            "(works for all merge modes)."
                        ),
                    },
                ),
                "merge_mode": (
                    ["lerp", "replace", "add"],
                    {
                        "default": "lerp",
                        "tooltip": (
                            "lerp: interpolate/extrapolate between generated and reference "
                            "conditioning (strength=1 = full reference, >1 extrapolates). "
                            "replace: replace the first conditioning token with reference * strength. "
                            "add: add reference * strength on top of existing conditioning."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject"
    CATEGORY = "ACE-Step/Reference"
    DESCRIPTION = (
        "Patches MODEL to inject reference timbre into the conditioning sequence. "
        "Hooks into decoder.condition_embedder to project and blend reference tokens. "
        "Compatible with BF16, FP16, FP32, FP8, and GGUF checkpoints."
    )

    def inject(self, model, timbre_conditioning, strength, merge_mode):
        current_hash = model_utils.compute_model_hash(model)

        # Support both dict and Cache-object style TIMBRE_CONDITIONING
        if isinstance(timbre_conditioning, dict):
            metadata_hash = timbre_conditioning.get("model_hash")
            hidden_states = timbre_conditioning["hidden_states"]
        else:
            metadata_hash = getattr(timbre_conditioning, "model_hash", None)
            hidden_states = getattr(timbre_conditioning, "hidden_states", None)

        if hidden_states is None:
            raise ValueError("[TimbreConditioningInject] timbre_conditioning has no hidden_states.")

        if metadata_hash != current_hash:
            print(
                "[TimbreConditioningInject] WARNING: model hash mismatch. "
                "Timbre was encoded with a different model."
            )

        # hidden_states: [1, 1, 2048] stored on CPU — must be moved to the
        # active compute device inside the hook where we know the runtime device.

        m = model.clone()
        raw_patched = model_utils.get_raw_model(m)
        arch_model = getattr(raw_patched, "diffusion_model", raw_patched)

        try:
            embedder = arch_model.decoder.condition_embedder
        except AttributeError:
            print("[TimbreConditioningInject] ERROR: could not find decoder.condition_embedder")
            return (m,)

        embedder_device, embedder_dtype = model_utils.get_model_device_dtype(embedder)

        prev_wrapper = m.model_options.get("model_function_wrapper", None)

        _in_ref_projection = [False]

        def my_unet_wrapper(model_function, wrap_kwargs):

            def _hook(module, args, output):
                if _in_ref_projection[0]:
                    return output

                projected_gen = output  # [B, seq_len, 2560] on CUDA
                B, seq_len, H = projected_gen.shape

                cond_or_uncond = wrap_kwargs.get("cond_or_uncond", None)
                if cond_or_uncond is not None and len(cond_or_uncond) == B:
                    is_uncond = [int(v) == 1 for v in cond_or_uncond]
                else:
                    is_uncond = [False] * B

                # Project reference timbre through condition_embedder.
                # hidden_states is on CPU — move to embedder device/dtype first.
                _in_ref_projection[0] = True
                try:
                    with torch.no_grad():
                        ref_in = hidden_states.to(device=embedder_device, dtype=embedder_dtype)
                        ref_projected = module(ref_in)  # [1, 1, 2560]
                finally:
                    _in_ref_projection[0] = False

                # ref_projected may be on a different device/dtype than projected_gen
                # (embedder_device is probed at inject() time; the generation tensor
                # is on the active CUDA device at runtime). Always match projected_gen
                # exactly before any arithmetic — this is the fix for the CPU/CUDA error.
                ref_projected = ref_projected.to(
                    device=projected_gen.device,
                    dtype=projected_gen.dtype
                )
                ref_projected = ref_projected.expand(B, -1, -1)  # [B, 1, 2560]

                out = projected_gen.clone()
                for i, un in enumerate(is_uncond):
                    if un:
                        continue

                    ref_tok = ref_projected[i, 0, :]
                    gen_tok = projected_gen[i, 0, :]

                    if merge_mode == "replace":
                        out[i, 0, :] = ref_tok * strength
                    elif merge_mode == "lerp":
                        # strength=0 → gen, strength=1 → ref, strength>1 → extrapolate
                        out[i, 0, :] = gen_tok + strength * (ref_tok - gen_tok)
                    elif merge_mode == "add":
                        out[i, 0, :] = gen_tok + strength * ref_tok

                return out

            handle = embedder.register_forward_hook(_hook)
            try:
                if prev_wrapper is not None:
                    return prev_wrapper(model_function, wrap_kwargs)
                else:
                    return model_function(
                        wrap_kwargs["input"],
                        wrap_kwargs["timestep"],
                        **wrap_kwargs.get("c", {})
                    )
            finally:
                handle.remove()

        m.set_model_unet_function_wrapper(my_unet_wrapper)

        print(
            f"[TimbreConditioningInject] Hook registered on decoder.condition_embedder "
            f"(mode={merge_mode}, strength={strength:.2f}, "
            f"embedder_dtype={embedder_dtype})"
        )
        return (m,)
