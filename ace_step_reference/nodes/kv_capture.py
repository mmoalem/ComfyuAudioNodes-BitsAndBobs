import torch

from ..core import model_utils
from ..core.hook_manager import HookManager


class SelfAttentionCapture:
    """Encodes reference audio as a VAE latent for use by SelfAttentionInject.

    The actual self-attention KV capture happens INSIDE the sampling loop
    on the first step that meets the sigma threshold inside SelfAttentionInject.

    Data flow:
        SelfAttentionCapture  →  kv_activations dict  →  SelfAttentionInject
            ref_latent              VAE latent of reference audio (CPU, float32)
            capture_timestep_frac   when in the sigma schedule to capture
            layer_range             which decoder layers to hook
            model_hash              sanity-check that model hasn't changed
            attn_cache              filled in by SelfAttentionInject at runtime

    Timbre conditioning is a separate concern handled by TimbreConditioningInject,
    which hooks condition_embedder directly in post-projection (2560-dim) space.
    Do not mix timbre conditioning through this node — the c_crossattn tensor
    that KV injection sees is 1024-dim (pre-text_projector), which is incompatible
    with the 2048-dim timbre encoder output.
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
                            "Fraction of the sampler's sigma range at which to perform "
                            "the reference KV capture pass (0=low noise, 1=high noise). "
                            "0.5 = midpoint — balances structure and texture capture. "
                            "Lower values capture finer timbral detail; higher values "
                            "capture more coarse structural information."
                        ),
                    },
                ),
                "start_layer": (
                    "INT",
                    {"default": 0, "min": 0, "max": 31},
                ),
                "end_layer": (
                    "INT",
                    {"default": 31, "min": 0, "max": 31},
                ),
            },
        }

    RETURN_TYPES = ("KV_ACTIVATIONS",)
    RETURN_NAMES = ("kv_activations",)
    FUNCTION = "capture"
    CATEGORY = "ACE-Step/Reference"
    DESCRIPTION = (
        "Encodes reference audio and stores the VAE latent. "
        "The actual self-attention KV capture happens inside the sampler on the "
        "first step that meets the sigma threshold. "
        "Connect KV_ACTIVATIONS output to SelfAttentionInject.\n"
        "For timbre conditioning, use AudioTimbreEncode → TimbreConditioningInject "
        "as a separate chain — do not mix timbre through this node. "
        "Compatible with BF16, FP16, FP32, FP8, and GGUF checkpoints."
    )

    def capture(
        self,
        model,
        vae,
        audio,
        capture_timestep_frac,
        start_layer,
        end_layer,
    ):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform[:1, :, :]

        # Resample to 48kHz — VAE requires 48kHz
        if sample_rate != 48000:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=48000
            )
            sample_rate = 48000

        # ComfyUI VAE expects channel-last [B, T, C]
        waveform_for_vae = waveform.movedim(1, -1)

        # VAE encode — stored as float32 on CPU regardless of model dtype.
        # Dtype reconciliation happens in SelfAttentionInject at runtime.
        with torch.no_grad():
            vae_device, vae_dtype = model_utils.get_vae_device_dtype(vae)
            vae_output = vae.encode(waveform_for_vae.to(vae_device, vae_dtype))
            if hasattr(vae_output, "latent_dist"):
                ref_latent = vae_output.latent_dist.mode()
            else:
                ref_latent = vae_output

        kv_activations = {
            "ref_latent": ref_latent.detach().cpu(),
            "capture_timestep_frac": capture_timestep_frac,
            "layer_range": (start_layer, end_layer),
            "model_hash": model_utils.compute_model_hash(model),
            "attn_cache": {},
        }

        print(
            f"[SelfAttentionCapture] Initialized capture target: "
            f"timestep_frac={capture_timestep_frac:.2f}, layers={start_layer}-{end_layer}"
        )
        return (kv_activations,)
