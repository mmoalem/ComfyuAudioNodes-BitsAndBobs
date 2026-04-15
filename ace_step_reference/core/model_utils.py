import torch
import torch.nn as nn
from typing import Optional

CONFIRMED_CONSTANTS = {
    "num_layers": 32,
    "hidden_size": 2560,
    "encoder_hidden": 2048,
    "num_q_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "dac_token_dim": 64,
    "conditioning_dim": 2560,
}


def get_raw_model(model) -> nn.Module:
    """Walk wrappers until we find the ACEStep nn.Module with a 'decoder' attribute.

    ComfyUI's wrapping chain for ACEStep GGUF models is:
        ModelPatcher (.model) → ACEStep15_XL_BaseModel (.diffusion_model)
                             → AceStepConditionGenerationModelXL  ← this has .decoder

    For standard (safetensors) models the chain is:
        ModelPatcher (.model) → AceStepConditionGenerationModel   ← this has .decoder

    Also accepts a bare nn.Module directly (idempotent).
    """
    # Already unwrapped — return as-is if it has the decoder we need
    if isinstance(model, nn.Module) and hasattr(model, "decoder"):
        return model

    # Walk up to 5 levels, checking both .model and .diffusion_model at each step
    candidates: list[nn.Module] = []
    frontier = [model]
    seen = set()

    for _ in range(5):
        next_frontier = []
        for current in frontier:
            obj_id = id(current)
            if obj_id in seen:
                continue
            seen.add(obj_id)

            for attr in ("model", "diffusion_model"):
                if not hasattr(current, attr):
                    continue
                child = getattr(current, attr)
                if not isinstance(child, nn.Module):
                    continue
                child_id = id(child)
                if child_id in seen:
                    continue
                # Best candidate: has the ACEStep decoder
                if hasattr(child, "decoder"):
                    return child
                candidates.append(child)
                next_frontier.append(child)

        if not next_frontier:
            break
        frontier = next_frontier

    # Fall back to the first nn.Module we encountered along any branch
    if candidates:
        return candidates[0]

    raise RuntimeError(
        "[ACEStep] Cannot unwrap model. Walked .model/.diffusion_model up to 5 levels "
        "without finding an nn.Module with a 'decoder' attribute. "
        f"Top-level type: {type(model).__name__}"
    )




def get_decoder_layers(model) -> nn.Module:
    raw = get_raw_model(model)
    if not hasattr(raw, "decoder"):
        raise AttributeError("[ACEStep] Model has no 'decoder' attribute")
    layers = raw.decoder.layers
    if not isinstance(layers, (list, nn.ModuleList)):
        raise TypeError(
            f"[ACEStep] decoder.layers is {type(layers)}, expected list or ModuleList"
        )
    if len(layers) == 0:
        raise ValueError("[ACEStep] decoder.layers is empty")
    return layers


def get_num_layers(model) -> int:
    return len(get_decoder_layers(model))


def get_self_attn(layer: nn.Module) -> nn.Module:
    if not hasattr(layer, "self_attn"):
        raise AttributeError(f"[ACEStep] Layer {layer} has no self_attn")
    return layer.self_attn


def get_cross_attn(layer: nn.Module) -> nn.Module:
    if not hasattr(layer, "cross_attn"):
        raise AttributeError(f"[ACEStep] Layer {layer} has no cross_attn")
    return layer.cross_attn


def get_condition_embedder(model) -> nn.Module:
    raw = get_raw_model(model)
    if not hasattr(raw.decoder, "condition_embedder"):
        raise AttributeError("[ACEStep] Model decoder has no condition_embedder")
    return raw.decoder.condition_embedder


def get_timbre_encoder(model) -> nn.Module:
    raw = get_raw_model(model)
    if not hasattr(raw.encoder, "timbre_encoder"):
        raise AttributeError("[ACEStep] Model encoder has no timbre_encoder")
    return raw.encoder.timbre_encoder


def get_encoder_norm(model) -> nn.Module:
    raw = get_raw_model(model)
    if not hasattr(raw.encoder, "norm"):
        raise AttributeError("[ACEStep] Model encoder has no norm")
    return raw.encoder.norm


def get_tokenizer(model) -> nn.Module:
    raw = get_raw_model(model)
    if not hasattr(raw, "tokenizer"):
        raise AttributeError("[ACEStep] Model has no tokenizer")
    return raw.tokenizer


def compute_model_hash(model) -> str:
    """Compute a short hash of the first few parameters.

    Accepts either a ModelPatcher wrapper or an already-unwrapped nn.Module.
    """
    raw = get_raw_model(model)
    if hasattr(raw, "_reference_hash"):
        return raw._reference_hash
    import hashlib

    parts = []
    for i, param in enumerate(raw.parameters()):
        if i >= 10:
            break
        parts.append(param.detach().flatten().cpu().to(torch.float32))
    hasher = torch.cat(parts) if parts else torch.tensor([], dtype=torch.float32)
    h = hashlib.sha256(hasher.numpy().tobytes()).hexdigest()[:16]
    raw._reference_hash = h
    return h


def get_vae_device_dtype(vae) -> tuple:
    """
    Safely return (device, dtype) from any ComfyUI VAE wrapper.

    Standard comfy.sd.VAE exposes  .device  and  .dtype.
    AceAudioOobleckVAEWrapper exposes  .device  and  .vae_dtype  (no .dtype).
    Fall back to inspecting first_stage_model parameters if both are absent.
    """
    # Device — almost always present
    device = getattr(vae, "device", None)
    if device is None:
        try:
            device = next(vae.first_stage_model.parameters()).device
        except Exception:
            import torch
            device = torch.device("cpu")

    # Dtype — varies by wrapper
    dtype = getattr(vae, "dtype", None)
    if dtype is None:
        dtype = getattr(vae, "vae_dtype", None)
    if dtype is None:
        try:
            dtype = next(vae.first_stage_model.parameters()).dtype
        except Exception:
            import torch
            dtype = torch.float32

    return device, dtype


def get_null_condition_emb(model) -> torch.Tensor:
    raw = get_raw_model(model)
    if not hasattr(raw, "null_condition_emb"):
        raise RuntimeError("[ACEStep] Model has no null_condition_emb")
    return raw.null_condition_emb

# FP8 dtypes introduced in PyTorch 2.1 — guard for older installs
_FP8_DTYPES = set()
for _name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    _dt = getattr(torch, _name, None)
    if _dt is not None:
        _FP8_DTYPES.add(_dt)
 
# Quantized / non-compute dtypes that cannot be used as matmul inputs
_NON_COMPUTE_DTYPES = _FP8_DTYPES | {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
 
 
def resolve_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Given a model weight dtype, return a safe dtype for compute (matmul inputs).
 
    - BF16, FP16, FP32 → returned as-is (all valid for matmul)
    - FP8, INT8, UINT8, INT32, etc. → promoted to BF16
      (FP8/quantized weights are dequantized internally by PyTorch/GGUF;
       inputs should be BF16 or FP16 so the dequantized multiply works)
    - Anything unrecognised → BF16 as a safe default
    """
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return dtype
    if dtype in _NON_COMPUTE_DTYPES:
        return torch.bfloat16
    # Unknown dtype (e.g. future formats, custom GGUF tensor dtype) → BF16
    return torch.bfloat16
 
 
def get_model_device_dtype(module: torch.nn.Module):
    """
    Probe an nn.Module for its actual device and a safe compute dtype.
 
    Works for standard nn.Parameter, FP8 checkpoints, and GGUF-wrapped
    tensors (which may expose a quantized dtype that is not safe for matmul).
 
    Returns:
        (device, compute_dtype)
    """
    try:
        param = next(module.parameters())
        device = param.device
        dtype  = resolve_compute_dtype(param.dtype)
        return device, dtype
    except StopIteration:
        # Module has no parameters (e.g. pure-functional wrapper)
        return torch.device("cpu"), torch.bfloat16