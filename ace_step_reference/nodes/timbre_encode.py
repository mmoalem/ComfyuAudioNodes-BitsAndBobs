import torch

from ..core import model_utils
from ..core.cache import Cache


class AudioTimbreEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "audio": ("AUDIO",),
                "normalize": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Peak-normalize before encoding. Matches model training.",
                    },
                ),
                "use_encoder_norm": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Apply encoder.norm after timbre_encoder. "
                            "True matches ACEStep's production path exactly."
                        ),
                    },
                ),
                "use_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "True = VAE mode() — deterministic, same result every run. "
                            "False = VAE sample() — slight variation per run."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("TIMBRE_CONDITIONING",)
    RETURN_NAMES = ("timbre_conditioning",)
    FUNCTION = "encode"
    CATEGORY = "ACE-Step/Reference"
    DESCRIPTION = (
        "Encodes reference audio using ACEStep's timbre encoder. "
        "Outputs a [1, 1, 2048] pooled timbre embedding. "
        "Connect to TimbreConditioningInject. "
        "Compatible with BF16, FP16, FP32, FP8, and GGUF checkpoints."
    )

    def encode(self, model, vae, audio, normalize, use_encoder_norm, use_mode):
        raw_model = model_utils.get_raw_model(model)
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Step 1: Resample to 48kHz — VAE requires 48kHz
        if sample_rate != 48000:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=48000
            )
            sample_rate = 48000

        if normalize:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        # Ensure correct batch shape [B, C, samples]
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :, :]
            print("[AudioTimbreEncode] Warning: batch > 1, using first audio only")

        # ComfyUI VAE expects [B, T, C] (channel-last)
        waveform_for_vae = waveform.movedim(1, -1)

        # Resolve diffusion model (handles standard, GGUF, and other wrapper styles)
        arch_model = getattr(raw_model, "diffusion_model", raw_model)

        # Step 2: VAE encode → [B, 64, T] channel-first, typically float32
        with torch.no_grad():
            vae_device, vae_dtype = model_utils.get_vae_device_dtype(vae)
            vae_output = vae.encode(waveform_for_vae.to(vae_device, vae_dtype))
            if hasattr(vae_output, "latent_dist"):
                acoustic_hs = (
                    vae_output.latent_dist.mode()
                    if use_mode
                    else vae_output.latent_dist.sample()
                )
            else:
                acoustic_hs = vae_output

        # Step 3: Transpose to [B, T, 64]
        # timbre_encoder.embed_tokens is Linear(64→2048) — expects last dim = 64
        acoustic_hs_transposed = acoustic_hs.transpose(1, 2)  # [B, T, 64]

        # Step 4: Resolve model device and a safe compute dtype.
        # VAE outputs float32. Model weights may be BF16, FP16, FP8, or GGUF-quantized.
        # - BF16/FP16/FP32: cast input to match weight dtype directly.
        # - FP8/GGUF/quantized: weights are dequantized at runtime by PyTorch/llama.cpp;
        #   inputs must be a real float dtype (BF16) — not the quantized dtype itself.
        # get_model_device_dtype() handles this: it probes the encoder and promotes
        # any non-compute dtype (FP8, INT8, etc.) to BF16 automatically.
        model_device, model_dtype = model_utils.get_model_device_dtype(
            arch_model.encoder.timbre_encoder
        )
        acoustic_hs_transposed = acoustic_hs_transposed.to(
            device=model_device, dtype=model_dtype
        )

        # Step 5: Build order_mask — one scalar per batch item (not per frame)
        B = acoustic_hs_transposed.shape[0]
        order_mask = torch.zeros((B,), dtype=torch.long, device=model_device)

        # Step 6: Run timbre encoder
        # Input:  [B, T, 64]
        # Output: [B, T+1, 2048]  (+1 for prepended CLS special_token at index 0)
        with torch.no_grad():
            try:
                timbre_embs, _ = arch_model.encoder.timbre_encoder(
                    acoustic_hs_transposed, order_mask
                )
                print(f"[AudioTimbreEncode] timbre_encoder OK, shape: {list(timbre_embs.shape)}")
            except Exception as e:
                print(f"[AudioTimbreEncode] timbre_encoder FAILED: {e}")
                raise RuntimeError(
                    f"[AudioTimbreEncode] timbre_encoder failed: {e}\n"
                    f"Input shape={list(acoustic_hs_transposed.shape)}, "
                    f"dtype={acoustic_hs_transposed.dtype}, device={acoustic_hs_transposed.device}."
                ) from e

        # Step 7: Pool to CLS token → [B, 1, 2048]
        # timbre_encoder prepends special_token (CLS) at index 0
        if timbre_embs.shape[1] > 1:
            timbre_embs = timbre_embs[:, :1, :]

        # Step 8: Shared encoder norm (matches ACEStep production path)
        if use_encoder_norm and hasattr(arch_model.encoder, "norm"):
            with torch.no_grad():
                try:
                    timbre_embs = arch_model.encoder.norm(timbre_embs)
                except Exception as e:
                    print(f"[AudioTimbreEncode] encoder.norm failed (non-fatal): {e}")

        cache = Cache.create_timbre(
            hidden_states=timbre_embs.detach().cpu(),
            metadata={
                "model_hash": model_utils.compute_model_hash(model),
                "audio_duration_sec": waveform.shape[-1] / sample_rate,
                "sample_rate": sample_rate,
                "frame_count": timbre_embs.shape[1],
                "encoder_norm_applied": use_encoder_norm,
                "vae_mode": use_mode,
            },
        )

        print(
            f"[AudioTimbreEncode] Encoded {timbre_embs.shape[1]} timbre token(s), "
            f"duration: {waveform.shape[-1] / sample_rate:.2f}s, "
            f"shape: {list(timbre_embs.shape)}, dtype={timbre_embs.dtype}"
        )

        return (cache,)
