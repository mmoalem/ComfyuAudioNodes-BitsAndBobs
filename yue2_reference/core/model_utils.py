import torch
import torch.nn as nn


def get_raw_yue2_model(model) -> nn.Module:
    """Walk wrappers until we find the YuE2 nn.Module with model.model.layers.

    ComfyUI wrapping chain:
        ModelPatcher (.model) → YuE2  ← has .model.layers (ModuleList of TransformerBlock)

    Also accepts a bare nn.Module directly (idempotent).
    """
    if isinstance(model, nn.Module) and _is_yue2(model):
        return model

    frontier = [model]
    seen: set = set()

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
                if _is_yue2(child):
                    return child
                next_frontier.append(child)

        if not next_frontier:
            break
        frontier = next_frontier

    raise RuntimeError(
        "[YuE2] Cannot unwrap model. Walked .model/.diffusion_model up to 5 levels "
        "without finding a YuE2 nn.Module (expected .model.layers ModuleList). "
        f"Top-level type: {type(model).__name__}"
    )


def _is_yue2(m: nn.Module) -> bool:
    return (
        hasattr(m, "model")
        and hasattr(m.model, "layers")
        and isinstance(m.model.layers, nn.ModuleList)
        and len(m.model.layers) > 0
        and hasattr(m.model.layers[0], "self_attn")
    )


def get_yue2_layers(model) -> nn.ModuleList:
    raw = get_raw_yue2_model(model)
    return raw.model.layers


def get_num_layers(model) -> int:
    return len(get_yue2_layers(model))


def get_model_device_dtype(module: nn.Module):
    """Return (device, safe_compute_dtype) for any nn.Module."""
    try:
        param = next(module.parameters())
        device = param.device
        dtype = param.dtype
        # Promote quantized dtypes to bfloat16
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            dtype = torch.bfloat16
        return device, dtype
    except StopIteration:
        return torch.device("cpu"), torch.bfloat16
