import torch
import torch.nn as nn
from typing import Dict, Optional
from . import tensor_ops


class HookManager:
    """Manages dynamic monkey-patching of AceStepAttention.forward for true concatenative KV injection.
    
    Instead of interpolating self_attn output residuals, this intercepts K and V *after*
    post-rotation embeddings (RoPE) are applied.
    """

    def __init__(self, raw_model: nn.Module, layer_range: tuple):
        self.raw_model = raw_model
        self.start_layer, self.end_layer = layer_range
        self._patched_modules: list = []
        self._layer_modules: list = []        # self_attn monkey-patch targets
        self._cross_attn_modules: list = []   # cross_attn forward-hook targets
        self._hook_handles: list = []          # PyTorch hook handles (for cross-attn)
        self._build_layer_index()

    def _build_layer_index(self) -> None:
        if hasattr(self.raw_model, "decoder") and hasattr(self.raw_model.decoder, "layers"):
            for idx, layer in enumerate(self.raw_model.decoder.layers):
                if self.start_layer <= idx <= self.end_layer:
                    if hasattr(layer, "self_attn"):
                        self._layer_modules.append((idx, layer.self_attn))
                    else:
                        print(f"[HookManager] Warning: layer {idx} has no self_attn")
                    if hasattr(layer, "cross_attn"):
                        self._cross_attn_modules.append((idx, layer.cross_attn))
                    else:
                        print(f"[HookManager] Warning: layer {idx} has no cross_attn")
        else:
            print("[HookManager] Warning: raw_model has no decoder.layers — no hooks registered")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all()
        return False

    def remove_all(self) -> None:
        # Remove self-attn monkey-patches (instance-level forward attributes)
        for module in self._patched_modules:
            if hasattr(module, "forward"):  # Falls back to the class method
                delattr(module, "forward")
        self._patched_modules.clear()
        # Remove cross-attn forward hooks registered via register_forward_hook
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------
    # Capture & Inject — Dynamically binds replacement forward
    # ------------------------------------------------------------------

    def _make_forward_wrapper(
        self,
        layer_idx: int,
        cache: dict,
        strength: float,
        is_capture: bool,
        taper: str = "none",
        mode: str = "inject",
        time_taper: str = "none",
    ):
        """Creates a bound monkey-patched forward function for self_attn or cross_attn."""
        effective_strength = strength
        if taper != "none" and not is_capture:
            effective_strength = tensor_ops.compute_layer_strength(
                layer_idx, self.start_layer, self.end_layer, strength, taper
            )

        def custom_forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            position_embeddings=None,
        ):
            from comfy.ldm.ace.ace_step15 import apply_rotary_pos_emb
            from comfy.ldm.modules.attention import optimized_attention

            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            query_states = self.q_norm(query_states)
            query_states = query_states.transpose(1, 2)

            if self.is_cross_attention and encoder_hidden_states is not None:
                bsz_enc, kv_len, _ = encoder_hidden_states.size()
                key_states = self.k_proj(encoder_hidden_states)
                value_states = self.v_proj(encoder_hidden_states)

                key_states = key_states.view(bsz_enc, kv_len, self.num_kv_heads, self.head_dim)
                key_states = self.k_norm(key_states)
                value_states = value_states.view(bsz_enc, kv_len, self.num_kv_heads, self.head_dim)

                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)
            else:
                kv_len = q_len
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

                key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
                key_states = self.k_norm(key_states)
                value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # ==========================================================
            # Capture or Inject/Replace (Unified for self/cross)
            # ==========================================================
            if is_capture:
                cache[layer_idx] = {
                    "k": key_states.detach().cpu(),
                    "v": value_states.detach().cpu(),
                }
            else:
                ref_data = cache.get(layer_idx)
                if ref_data is not None and effective_strength > 0:
                    ref_device = key_states.device
                    ref_dtype = key_states.dtype

                    ref_k = ref_data["k"].to(device=ref_device, dtype=ref_dtype)
                    ref_v = ref_data["v"].to(device=ref_device, dtype=ref_dtype)

                    # Match sequence length: trim or repeat-pad to match gen length
                    ref_t = ref_k.shape[2]
                    gen_t = key_states.shape[2]
                    if ref_t != gen_t:
                        if ref_t > gen_t:
                            ref_k = ref_k[:, :, :gen_t, :]
                            ref_v = ref_v[:, :, :gen_t, :]
                        else:
                            repeats = (gen_t + ref_t - 1) // ref_t
                            ref_k = ref_k.repeat(1, 1, repeats, 1)[:, :, :gen_t, :]
                            ref_v = ref_v.repeat(1, 1, repeats, 1)[:, :, :gen_t, :]

                    # Expand B=1 reference cache to match generation batch size
                    if ref_k.shape[0] == 1 and bsz > 1:
                        ref_k = ref_k.expand(bsz, -1, -1, -1)
                        ref_v = ref_v.expand(bsz, -1, -1, -1)

                    # Apply temporal scaling if requested (fading across length of audio)
                    if time_taper != "none":
                        time_mult = tensor_ops.compute_time_multiplier(
                            ref_k.shape[2], time_taper, ref_device, ref_dtype
                        )
                        ref_k = ref_k * time_mult
                        ref_v = ref_v * time_mult

                    if mode == "replace":
                        key_states = ref_k * effective_strength
                        value_states = ref_v * effective_strength
                    else:  # inject (concat)
                        # Apply strength scaling prior to sequence concatenation
                        ref_k = ref_k * effective_strength
                        ref_v = ref_v * effective_strength
                        key_states = torch.cat([key_states, ref_k], dim=2)
                        value_states = torch.cat([value_states, ref_v], dim=2)

                    kv_len = key_states.shape[2]
            # ==========================================================

            n_rep = self.num_heads // self.num_kv_heads
            if n_rep > 1:
                key_states = key_states.repeat_interleave(n_rep, dim=1)
                value_states = value_states.repeat_interleave(n_rep, dim=1)

            attn_bias = None
            if self.sliding_window is not None and not self.is_cross_attention:
                indices_q = torch.arange(q_len, device=query_states.device)

                if kv_len > q_len:
                    # K sequence contains BOTH generation tokens and reference tokens
                    indices_k_gen = torch.arange(q_len, device=query_states.device)
                    indices_k_ref = torch.arange(kv_len - q_len, device=query_states.device)
                    indices_k = torch.cat([indices_k_gen, indices_k_ref])
                else:
                    indices_k = torch.arange(kv_len, device=query_states.device)

                diff = indices_q.unsqueeze(1) - indices_k.unsqueeze(0)
                in_window = torch.abs(diff) <= self.sliding_window

                window_bias = torch.zeros((q_len, kv_len), device=query_states.device, dtype=query_states.dtype)
                min_value = torch.finfo(query_states.dtype).min
                window_bias.masked_fill_(~in_window, min_value)

                window_bias = window_bias.unsqueeze(0).unsqueeze(0)

                if attn_bias is not None:
                    if attn_bias.dtype == torch.bool:
                        base_bias = torch.zeros_like(window_bias)
                        base_bias.masked_fill_(~attn_bias, min_value)
                        attn_bias = base_bias + window_bias
                    else:
                        attn_bias = attn_bias + window_bias
                else:
                    attn_bias = window_bias

            attn_output = optimized_attention(
                query_states,
                key_states,
                value_states,
                self.num_heads,
                attn_bias,
                skip_reshape=True,
                low_precision_attention=False,
            )
            attn_output = self.o_proj(attn_output)

            return attn_output

        return custom_forward

    def register_capture_hooks(self, cache: dict, is_cross: bool = False) -> None:
        """Monkey-patches the forward function to capture pre-rotated KV into cache."""
        targets = self._cross_attn_modules if is_cross else self._layer_modules
        for idx, attn_mod in targets:
            patched_fn = self._make_forward_wrapper(idx, cache, strength=1.0, is_capture=True, taper="none")
            attn_mod.forward = patched_fn.__get__(attn_mod, type(attn_mod))
            self._patched_modules.append(attn_mod)

    def register_injection_hooks(self, cache: dict, strength: float, taper: str, time_taper: str = "none") -> None:
        """Monkey-patches the forward function to concatenate cached pre-rotated reference KV with generated KV."""
        for idx, self_attn in self._layer_modules:
            patched_fn = self._make_forward_wrapper(
                idx, cache, strength=strength, is_capture=False, taper=taper, time_taper=time_taper
            )
            self_attn.forward = patched_fn.__get__(self_attn, type(self_attn))
            self._patched_modules.append(self_attn)

    def register_injection_hooks_per_layer(
        self,
        cache: dict,
        layer_strengths: dict,
        mode: str = "inject",
        time_taper: str = "none",
        is_cross: bool = False,
    ) -> None:
        """Monkey-patches only the layers that have a non-zero individual strength.

        Args:
            cache:           KV cache populated by register_capture_hooks.
            layer_strengths: dict mapping layer index → strength float.
                             Layers with strength <= 0 are skipped entirely —
                             no patch is applied and the original forward is used.
            mode:            "inject" (concat), "replace" (hard overwrite), or "threshold" (concat <=1, replace >1).
            time_taper:      Temporal taper pattern.
            is_cross:        If True, targets cross_attn modules. If False, targets self_attn modules.
        """
        targets = self._cross_attn_modules if is_cross else self._layer_modules
        for idx, attn_mod in targets:
            s = layer_strengths.get(idx, 0.0)
            if s <= 0.0:
                continue  # leave this layer completely unpatched

            if mode == "threshold":
                effective_mode = "replace" if s > 1.0 else "inject"
            else:
                effective_mode = mode

            patched_fn = self._make_forward_wrapper(
                idx,
                cache,
                strength=s,
                is_capture=False,
                taper="none",
                mode=effective_mode,
                time_taper=time_taper,
            )
            attn_mod.forward = patched_fn.__get__(attn_mod, type(attn_mod))
            self._patched_modules.append(attn_mod)

    # Keep stub for older node components
    def register_condition_embedder_hook(self, extra_hidden_states: torch.Tensor, strength: float) -> None:
        pass
