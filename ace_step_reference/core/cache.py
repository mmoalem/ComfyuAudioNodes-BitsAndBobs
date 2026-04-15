import torch
from typing import Optional
from ..acestep_types import TIMBRE_CONDITIONING, KV_ACTIVATIONS


class Cache:
    @staticmethod
    def create_timbre(hidden_states: torch.Tensor, metadata: dict) -> dict:
        return {
            "type": TIMBRE_CONDITIONING,
            "hidden_states": hidden_states,
            "audio_duration_sec": metadata.get("audio_duration_sec", 0.0),
            "sample_rate": metadata.get("sample_rate", 44100),
            "frame_count": metadata.get("frame_count", 0),
            "model_hash": metadata.get("model_hash", ""),
            "encoder_norm_applied": metadata.get("encoder_norm_applied", True),
            "vae_mode": metadata.get("vae_mode", True),
        }

    @staticmethod
    def create_kv(layers_dict: dict, metadata: dict) -> dict:
        return {
            "type": KV_ACTIVATIONS,
            "model_hash": metadata.get("model_hash", ""),
            "capture_timestep": metadata.get("capture_timestep", 0),
            "layer_range": metadata.get("layer_range", (0, 31)),
            "layers": layers_dict,
            "audio_duration_sec": metadata.get("audio_duration_sec", 0.0),
            "timbre_used": metadata.get("timbre_used", False),
        }

    @staticmethod
    def validate_timbre(cache: dict) -> tuple[bool, list[str]]:
        errors = []
        required_keys = ["type", "hidden_states", "model_hash"]
        for key in required_keys:
            if key not in cache:
                errors.append(f"Missing required key: {key}")
        if cache.get("type") != TIMBRE_CONDITIONING:
            errors.append(
                f"Expected type {TIMBRE_CONDITIONING}, got {cache.get('type')}"
            )
        if "hidden_states" in cache:
            hs = cache["hidden_states"]
            if not isinstance(hs, torch.Tensor):
                errors.append("hidden_states must be a torch.Tensor")
            elif hs.ndim != 3:
                errors.append(f"hidden_states must be 3D, got {hs.ndim}D")
            elif hs.shape[-1] != 2048:
                errors.append(f"Expected last dim 2048, got {hs.shape[-1]}")
        return len(errors) == 0, errors

    @staticmethod
    def validate_kv(cache: dict) -> tuple[bool, list[str]]:
        errors = []
        required_keys = ["type", "model_hash", "layers"]
        for key in required_keys:
            if key not in cache:
                errors.append(f"Missing required key: {key}")
        if cache.get("type") != KV_ACTIVATIONS:
            errors.append(f"Expected type {KV_ACTIVATIONS}, got {cache.get('type')}")
        if "layers" in cache:
            layers = cache["layers"]
            if not isinstance(layers, dict):
                errors.append("layers must be a dict")
            elif len(layers) == 0:
                errors.append("layers dict is empty")
        return len(errors) == 0, errors

    @staticmethod
    def save(cache: dict, filepath: str) -> None:
        torch.save(cache, filepath, weights_only=False)

    @staticmethod
    def load(filepath: str) -> dict:
        cache = torch.load(filepath, weights_only=False)
        is_valid = False
        errors = []
        if cache.get("type") == TIMBRE_CONDITIONING:
            is_valid, errors = Cache.validate_timbre(cache)
        elif cache.get("type") == KV_ACTIVATIONS:
            is_valid, errors = Cache.validate_kv(cache)
        else:
            errors = [f"Unknown cache type: {cache.get('type')}"]
        if not is_valid:
            raise ValueError(f"Invalid cache file: {', '.join(errors)}")
        return cache

    @staticmethod
    def to_device(cache: dict, device, dtype) -> dict:
        result = {}
        for key, value in cache.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device=device, dtype=dtype)
            elif isinstance(value, dict):
                result[key] = {
                    k: v.to(device=device, dtype=dtype)
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in value.items()
                }
            else:
                result[key] = value
        return result

    @staticmethod
    def summary(cache: dict) -> str:
        lines = []
        cache_type = cache.get("type", "UNKNOWN")
        lines.append(f"Type: {cache_type}")
        if cache_type == TIMBRE_CONDITIONING:
            hs = cache.get("hidden_states")
            if hs is not None:
                lines.append(f"Shape: {list(hs.shape)}")
                lines.append(f"Duration: {cache.get('audio_duration_sec', 0):.2f}s")
                lines.append(f"Frame count: {cache.get('frame_count', 0)}")
                lines.append(f"Mean abs: {hs.abs().mean().item():.4f}")
                lines.append(f"Max abs: {hs.abs().max().item():.4f}")
                lines.append(f"Has NaN: {hs.isnan().any().item()}")
                lines.append(f"Has Inf: {hs.isinf().any().item()}")
        elif cache_type == KV_ACTIVATIONS:
            layers = cache.get("layers", {})
            lines.append(f"Layers captured: {len(layers)}")
            lines.append(f"Layer range: {cache.get('layer_range', (0, 31))}")
            lines.append(f"Capture timestep: {cache.get('capture_timestep', 0)}")
            lines.append(f"Duration: {cache.get('audio_duration_sec', 0):.2f}s")
            lines.append(f"Timbre used: {cache.get('timbre_used', False)}")
            if layers:
                first_key = sorted(layers.keys(), key=int)[0]
                first_k = layers[first_key].get("K")
                if first_k is not None:
                    lines.append(f"K shape: {list(first_k.shape)}")
        return "\n".join(lines)
