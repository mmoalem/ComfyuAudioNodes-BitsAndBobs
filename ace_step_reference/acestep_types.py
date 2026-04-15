TIMBRE_CONDITIONING = "TIMBRE_CONDITIONING"
KV_ACTIVATIONS = "KV_ACTIVATIONS"

TIMBRE_CONDITIONING_SCHEMA = {
    "hidden_states": "torch.Tensor",
    "audio_duration_sec": float,
    "sample_rate": int,
    "frame_count": int,
    "model_hash": str,
    "encoder_norm_applied": bool,
    "vae_mode": bool,
}

KV_ACTIVATIONS_SCHEMA = {
    "model_hash": str,
    "capture_timestep": int,
    "layer_range": tuple,
    "layers": dict,
    "audio_duration_sec": float,
    "timbre_used": bool,
}
