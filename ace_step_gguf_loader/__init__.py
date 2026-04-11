"""
ComfyUI-AceStep-GGUF
Custom GGUF loaders for ACE-Step 1.5 models quantized with acestep.cpp.

Supports:
  - DiT  (acestep-v15-*.gguf / acestep-v15-xl-*.gguf)  → MODEL
  - Text encoder (Qwen3-Embedding-0.6B-*.gguf)          → CLIP
  - VAE  (vae-BF16.gguf)                                → VAE
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# ---------------------------------------------------------------------------
# Patch: fix PyTorch 2.9+ inplace memory-overlap crash in ACE-Step LM sampling
#
# ROOT CAUSE: ComfyUI-GGUF's GGMLTensor.clone() intentionally returns `self`
# (not a real copy) to avoid the overhead of copying compressed weight data.
# However, tensors flowing through a GGUF-loaded LM can inherit the GGMLTensor
# subclass type, so calls like `t[..., :-1].clone()` return the original tensor.
# PyTorch 2.9+ then detects that the "clone" still aliases the destination and
# raises "unsupported operation: some elements of the input tensor and the
# written-to tensor refer to a single memory location."
#
# FIX: Replace the upstream `sample_manual_loop_no_classes` function with a
# version that allocates fresh torch.zeros() bool tensors (guaranteed to be
# plain torch.Tensor, never GGMLTensor) for all masking operations, completely
# avoiding any .clone() calls on potentially-GGML tensors.
# ---------------------------------------------------------------------------

import logging
import torch


def _gguf_fixed_sample_loop(
    model,
    ids=None,
    execution_dtype=None,
    cfg_scale: float = 2.0,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = None,
    min_p: float = 0.000,
    seed: int = 1,
    min_tokens: int = 1,
    max_new_tokens: int = 2048,
    audio_start_id: int = 151669,
    audio_end_id: int = 215669,
    eos_token_id: int = 151645,
):
    """
    Drop-in replacement for comfy.text_encoders.ace15.sample_manual_loop_no_classes.

    Identical logic, but the top-p nucleus sampling block allocates masking tensors
    via torch.zeros(..., dtype=torch.bool) instead of relying on .clone() slices.
    This avoids the GGMLTensor.clone()-is-a-no-op aliasing crash in PyTorch 2.9+.
    """
    import comfy.model_management
    import comfy.utils

    if ids is None:
        return []

    device = model.execution_device

    if execution_dtype is None:
        if comfy.model_management.should_use_bf16(device):
            execution_dtype = torch.bfloat16
        else:
            execution_dtype = torch.float32

    embeds, attention_mask, num_tokens, embeds_info = model.process_tokens(ids, device)
    embeds_batch = embeds.shape[0]

    output_audio_codes = []
    past_key_values = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model_config = model.transformer.model.config
    past_kv_shape = [
        embeds_batch,
        model_config.num_key_value_heads,
        embeds.shape[1] + min_tokens,
        model_config.head_dim,
    ]

    for _ in range(model_config.num_hidden_layers):
        past_key_values.append((
            torch.empty(past_kv_shape, device=device, dtype=execution_dtype),
            torch.empty(past_kv_shape, device=device, dtype=execution_dtype),
            0,
        ))

    progress_bar = comfy.utils.ProgressBar(max_new_tokens)

    for step in comfy.utils.model_trange(max_new_tokens, desc="LM sampling"):
        outputs = model.transformer(
            None, attention_mask,
            embeds=embeds.to(execution_dtype),
            num_tokens=num_tokens,
            intermediate_output=None,
            dtype=execution_dtype,
            embeds_info=embeds_info,
            past_key_values=past_key_values,
        )
        next_token_logits = model.transformer.logits(outputs[0])[:, -1]
        past_key_values = outputs[2]

        if cfg_scale != 1.0:
            cond_logits   = next_token_logits[0:1]
            uncond_logits = next_token_logits[1:2]
            cfg_logits    = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
        else:
            cfg_logits = next_token_logits[0:1]

        # Cast to plain torch.Tensor in case a GGMLTensor subclass propagated
        # through the LM forward pass (GGMLTensor.clone() is a no-op, causing
        # PyTorch 2.9+ aliasing errors in all subsequent sampling operations).
        cfg_logits = cfg_logits.as_subclass(torch.Tensor)

        use_eos_score = (
            eos_token_id is not None
            and eos_token_id < audio_start_id
            and min_tokens < step
        )
        if use_eos_score:
            eos_score = cfg_logits[:, eos_token_id].clone()

        remove_logit_value = torch.finfo(cfg_logits.dtype).min

        # Restrict to audio token range
        cfg_logits[:, :audio_start_id] = remove_logit_value
        cfg_logits[:, audio_end_id:]   = remove_logit_value

        if use_eos_score:
            cfg_logits[:, eos_token_id] = eos_score

        # top-k filtering
        if top_k is not None and top_k > 0:
            top_k_vals, _ = torch.topk(cfg_logits, top_k)
            min_val = top_k_vals[..., -1, None]
            cfg_logits[cfg_logits < min_val] = remove_logit_value

        # min-p filtering
        if min_p is not None and min_p > 0:
            probs = torch.softmax(cfg_logits, dim=-1)
            p_max = probs.max(dim=-1, keepdim=True).values
            cfg_logits[probs < (min_p * p_max)] = remove_logit_value

        # top-p (nucleus) filtering — uses only fresh plain tensors, no .clone()
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(cfg_logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Allocate a brand-new plain bool tensor — guaranteed not a GGMLTensor.
            # Fill: mark a token for removal when the cumulative prob *before* it
            # already exceeded top_p (equivalent to the original right-shift logic).
            batch_sz, vocab_sz = cfg_logits.shape[0], cfg_logits.shape[1]
            remove_sorted = torch.zeros(
                batch_sz, vocab_sz, dtype=torch.bool, device=cfg_logits.device
            )
            remove_sorted[:, 1:] = cumulative_probs[:, :-1] > top_p

            # Scatter back to vocabulary order into another fresh plain tensor
            indices_to_remove = torch.zeros(
                batch_sz, vocab_sz, dtype=torch.bool, device=cfg_logits.device
            ).scatter_(1, sorted_indices, remove_sorted)

            cfg_logits[indices_to_remove] = remove_logit_value

        # Sample next token
        if temperature > 0:
            cfg_logits = cfg_logits / temperature
            next_token = torch.multinomial(
                torch.softmax(cfg_logits, dim=-1),
                num_samples=1,
                generator=generator,
            ).squeeze(1)
        else:
            next_token = torch.argmax(cfg_logits, dim=-1)

        token = next_token.item()

        if token == eos_token_id:
            break

        embed, _, _, _ = model.process_tokens([[token]], device)
        embeds = embed.repeat(embeds_batch, 1, 1)
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((embeds_batch, 1), device=device, dtype=attention_mask.dtype)],
            dim=1,
        )

        output_audio_codes.append(token - audio_start_id)
        progress_bar.update_absolute(step)

    return output_audio_codes


# Install the patch
try:
    import comfy.text_encoders.ace15
    comfy.text_encoders.ace15.sample_manual_loop_no_classes = _gguf_fixed_sample_loop
    logging.info(
        "AceStep-GGUF: patched sample_manual_loop_no_classes "
        "(avoids GGMLTensor.clone() no-op aliasing crash in PyTorch 2.9+)"
    )
except Exception as err:
    logging.warning(f"AceStep-GGUF: could not patch sample_manual_loop_no_classes: {err}")


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
