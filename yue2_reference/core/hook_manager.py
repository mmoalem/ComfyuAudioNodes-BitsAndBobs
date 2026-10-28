"""YuE2HookManager — monkey-patches Attention.forward per-layer for KV injection.

YuE2's self-attention is the standard Attention class from comfy/text_encoders/llama.py.
Its forward signature is:

    forward(hidden_states, attention_mask=None, freqs_cis=None,
            optimized_attention=None, past_key_value=None, sliding_window=None)

During the YuE2 forward loop, each layer is called as:

    state, _ = layer(state, freqs_cis=rope, optimized_attention=attention,
                     past_key_value=(prefix[index, 0], prefix[index, 1], ar_length))

where past_key_value carries the LLM-derived prefix KV. We must preserve this.

Injection strategy:
  - Capture: monkey-patch Attention.forward to save post-RoPE xk/xv for the reference pass.
  - Inject: monkey-patch Attention.forward to concatenate cached reference K/V onto the
    current generation K/V (after RoPE, before optimized_attention). The LLM prefix
    past_key_value tuple is left intact — it gets extended inside the normal path before
    our extra concatenation is appended.
"""

import torch
import torch.nn as nn
from comfy.ldm.modules.attention import optimized_attention_for_device
from . import tensor_ops


class YuE2HookManager:
    """Manages monkey-patching of Attention.forward for KV capture and injection."""

    def __init__(self, raw_model: nn.Module, layer_range: tuple):
        self.raw_model = raw_model
        self.start_layer, self.end_layer = layer_range
        self._patched_modules: list = []
        self._build_layer_index()

    def _build_layer_index(self) -> None:
        self._layer_modules: list = []
        layers = self.raw_model.model.layers
        for idx, layer in enumerate(layers):
            if self.start_layer <= idx <= self.end_layer:
                if hasattr(layer, "self_attn"):
                    self._layer_modules.append((idx, layer.self_attn))
                else:
                    print(f"[YuE2HookManager] Warning: layer {idx} has no self_attn")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all()
        return False

    def remove_all(self) -> None:
        for module in self._patched_modules:
            if hasattr(module, "forward"):
                delattr(module, "forward")
        self._patched_modules.clear()

    # ------------------------------------------------------------------
    # Forward wrapper factory
    # ------------------------------------------------------------------

    def _make_forward_wrapper(
        self,
        layer_idx: int,
        cache: dict,
        strength: float,
        is_capture: bool,
        mode: str = "inject",
        time_taper: str = "none",
    ):
        """Return a replacement forward function for an Attention module.

        The replacement replicates the real Attention.forward logic exactly up to
        and including the optimized_attention call, but intercepts K/V for capture
        or injection. The past_key_value (LLM prefix) path is preserved so the
        model's chunk-level auto-regressive conditioning keeps working.
        """

        def custom_forward(
            self_attn,
            hidden_states,
            attention_mask=None,
            freqs_cis=None,
            optimized_attention=None,
            past_key_value=None,
            sliding_window=None,
        ):
            from comfy.text_encoders.llama import apply_rope

            batch_size, seq_length, _ = hidden_states.shape

            # QKV projection — handles both merged and separate weights
            if self_attn.merged_qkv:
                xq, xk, xv = self_attn.qkv_proj(hidden_states).split(
                    (self_attn.inner_size, self_attn.kv_size, self_attn.kv_size), dim=-1
                )
            else:
                xq = self_attn.q_proj(hidden_states)
                xk = self_attn.k_proj(hidden_states)
                xv = self_attn.v_proj(hidden_states)

            xq = xq.view(batch_size, seq_length, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
            xk = xk.view(batch_size, seq_length, self_attn.num_kv_heads, self_attn.head_dim).transpose(1, 2)
            xv = xv.view(batch_size, seq_length, self_attn.num_kv_heads, self_attn.head_dim).transpose(1, 2)

            if self_attn.q_norm is not None:
                xq = self_attn.q_norm(xq)
            if self_attn.k_norm is not None:
                xk = self_attn.k_norm(xk)

            xq, xk = apply_rope(xq, xk, freqs_cis=freqs_cis)

            # ── LLM prefix (past_key_value) — unchanged from the real implementation ──
            present_key_value = None
            if past_key_value is not None and not isinstance(past_key_value, _FixedKVCheck):
                if len(past_key_value) > 0:
                    past_key, past_value, index = past_key_value
                    num_tokens = xk.shape[2]
                    if past_key.shape[2] >= (index + num_tokens):
                        past_key[:, :, index:index + xk.shape[2]] = xk
                        past_value[:, :, index:index + xv.shape[2]] = xv
                        xk = past_key[:, :, :index + xk.shape[2]]
                        xv = past_value[:, :, :index + xv.shape[2]]
                        present_key_value = (past_key, past_value, index + num_tokens)
                    else:
                        xk = torch.cat((past_key[:, :, :index], xk), dim=2)
                        xv = torch.cat((past_value[:, :, :index], xv), dim=2)
                        present_key_value = (xk, xv, index + num_tokens)
                else:
                    present_key_value = (xk, xv, xk.shape[2])

            # ── Capture or inject ──────────────────────────────────────────────
            if is_capture:
                cache[layer_idx] = {
                    "k": xk.detach().cpu(),
                    "v": xv.detach().cpu(),
                }
            else:
                ref_data = cache.get(layer_idx)
                if ref_data is not None and strength > 0:
                    ref_dev = xk.device
                    ref_dtype = xk.dtype
                    ref_k = ref_data["k"].to(device=ref_dev, dtype=ref_dtype)
                    ref_v = ref_data["v"].to(device=ref_dev, dtype=ref_dtype)

                    # Match sequence length of cached ref to current gen
                    ref_t = ref_k.shape[2]
                    gen_t = xk.shape[2]
                    if ref_t != gen_t:
                        if ref_t > gen_t:
                            ref_k = ref_k[:, :, :gen_t, :]
                            ref_v = ref_v[:, :, :gen_t, :]
                        else:
                            repeats = (gen_t + ref_t - 1) // ref_t
                            ref_k = ref_k.repeat(1, 1, repeats, 1)[:, :, :gen_t, :]
                            ref_v = ref_v.repeat(1, 1, repeats, 1)[:, :, :gen_t, :]

                    if ref_k.shape[0] == 1 and batch_size > 1:
                        ref_k = ref_k.expand(batch_size, -1, -1, -1)
                        ref_v = ref_v.expand(batch_size, -1, -1, -1)

                    if time_taper != "none":
                        time_mult = tensor_ops.compute_time_multiplier(
                            ref_k.shape[2], time_taper, ref_dev, ref_dtype
                        )
                        ref_k = ref_k * time_mult
                        ref_v = ref_v * time_mult

                    if mode == "replace":
                        xk = ref_k * strength
                        xv = ref_v * strength
                    else:  # inject (concat)
                        xk = torch.cat([xk, ref_k * strength], dim=2)
                        xv = torch.cat([xv, ref_v * strength], dim=2)
            # ──────────────────────────────────────────────────────────────────

            gqa_kwargs = {"enable_gqa": True} if self_attn.num_heads != self_attn.num_kv_heads else {}
            output = optimized_attention(
                xq, xk, xv, self_attn.num_heads,
                mask=attention_mask, skip_reshape=True,
                **gqa_kwargs,
            )
            return self_attn.o_proj(output), present_key_value

        return custom_forward

    # ------------------------------------------------------------------
    # Public registration helpers
    # ------------------------------------------------------------------

    def register_capture_hooks(self, cache: dict) -> None:
        for idx, attn_mod in self._layer_modules:
            fn = self._make_forward_wrapper(idx, cache, strength=1.0, is_capture=True)
            attn_mod.forward = fn.__get__(attn_mod, type(attn_mod))
            self._patched_modules.append(attn_mod)

    def register_injection_hooks(
        self, cache: dict, strength: float, taper: str, time_taper: str = "none"
    ) -> None:
        for idx, attn_mod in self._layer_modules:
            effective_strength = strength
            if taper != "none":
                effective_strength = tensor_ops.compute_layer_strength(
                    idx, self.start_layer, self.end_layer, strength, taper
                )
            if effective_strength <= 0.0:
                continue
            fn = self._make_forward_wrapper(
                idx, cache, strength=effective_strength,
                is_capture=False, mode="inject", time_taper=time_taper,
            )
            attn_mod.forward = fn.__get__(attn_mod, type(attn_mod))
            self._patched_modules.append(attn_mod)

    def register_injection_hooks_per_layer(
        self,
        cache: dict,
        layer_strengths: dict,
        mode: str = "inject",
        time_taper: str = "none",
    ) -> None:
        for idx, attn_mod in self._layer_modules:
            s = layer_strengths.get(idx, 0.0)
            if s <= 0.0:
                continue
            effective_mode = "replace" if (mode == "threshold" and s > 1.0) else (
                "inject" if mode == "threshold" else mode
            )
            fn = self._make_forward_wrapper(
                idx, cache, strength=s,
                is_capture=False, mode=effective_mode, time_taper=time_taper,
            )
            attn_mod.forward = fn.__get__(attn_mod, type(attn_mod))
            self._patched_modules.append(attn_mod)


# Sentinel to detect FixedKV objects without importing from llama (avoids circular import)
class _FixedKVCheck:
    pass
