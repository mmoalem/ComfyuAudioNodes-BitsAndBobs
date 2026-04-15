from ..core.cache import Cache


class ReferenceInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "timbre_conditioning": ("TIMBRE_CONDITIONING",),
                "kv_activations": ("KV_ACTIVATIONS",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    OUTPUT_NODE = True
    FUNCTION = "inspect"
    CATEGORY = "ACE-Step/Reference"

    def inspect(self, timbre_conditioning=None, kv_activations=None):
        lines = []

        if timbre_conditioning is not None:
            hs = timbre_conditioning.get("hidden_states")
            if hs is not None:
                lines += [
                    "=== TIMBRE CONDITIONING ===",
                    f"Shape:          {list(hs.shape)}",
                    f"Dtype:          {hs.dtype}",
                    f"Duration:       {timbre_conditioning.get('audio_duration_sec', 0):.2f}s",
                    f"Frame count:    {timbre_conditioning.get('frame_count', 0)}",
                    f"Encoder norm:   {timbre_conditioning.get('encoder_norm_applied', False)}",
                    f"Mean abs value: {hs.abs().mean().item():.4f}",
                    f"Max abs value:  {hs.abs().max().item():.4f}",
                    f"Has NaN:        {hs.isnan().any().item()}",
                    f"Has Inf:        {hs.isinf().any().item()}",
                    f"Model hash:     {timbre_conditioning.get('model_hash', 'N/A')[:16]}...",
                ]

        if kv_activations is not None:
            attn_cache = kv_activations.get("attn_cache", {})
            num_layers = len(attn_cache)
            first_key = sorted(attn_cache.keys(), key=int)[0] if attn_cache else None
            first_k = attn_cache[first_key]["k"] if first_key else None

            lines += [
                "",
                "=== KV ACTIVATIONS ===",
                f"Layers captured: {num_layers}",
                f"Layer range:     {kv_activations.get('layer_range', (0, 31))}",
                f"K shape:         {list(first_k.shape) if first_k is not None else 'N/A'}",
                f"Dtype:           {first_k.dtype if first_k is not None else 'N/A'}",
                f"Timing Frac:     {kv_activations.get('capture_timestep_frac', 0.0):.2f}",
                f"Model hash:     {kv_activations.get('model_hash', 'N/A')[:16]}...",
            ]

            if attn_cache:
                lines.append("\nPer-layer K mean abs value:")
                for i in sorted(attn_cache.keys(), key=int):
                    k = attn_cache[i]["k"]
                    v = attn_cache[i]["v"]
                    lines.append(
                        f"  layer {int(i):02d}  K:{k.abs().mean():.4f}  V:{v.abs().mean():.4f}"
                    )

        summary = "\n".join(lines) if lines else "No inputs connected."
        print(f"[ReferenceInspector]\n{summary}")
        return {"ui": {"text": summary}, "result": (summary,)}
