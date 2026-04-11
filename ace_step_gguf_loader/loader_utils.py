"""
loader_utils.py — Low-level GGUF loading helpers for ACE-Step models.

These functions read GGUF files whose general.architecture is one of:
  - 'acestep-dit'      : Diffusion Transformer
  - 'acestep-text-enc' : Qwen3-Embedding text encoder
  - 'acestep-vae'      : AutoencoderOobleck audio VAE

The tensor key names inside these GGUF files are already identical to
the original safetensors layouts, so no remapping is needed. The only
work required is:
  1. Bypassing ComfyUI-GGUF's architecture allowlist
  2. For the DiT: reshaping scale_shift_table from [6, D] → [1, 6, D]
     so it matches the safetensors convention ComfyUI's ACE model class expects.
  3. For the text encoder: extracting the BPE tokenizer vocab from GGUF
     metadata and packaging it as a 'bpe_model' byte tensor so ComfyUI's
     ACE CLIPType can build its tokenizer.
"""

import warnings
import logging
import struct
import json
import torch
import gguf

# ---------------------------------------------------------------------------
# Import City96's GGMLTensor and dequantize so we reuse the dequant engine
#
# ComfyUI-GGUF has a hyphenated folder name so normal Python import doesn't
# work. ComfyUI registers it in sys.modules under the key 'ComfyUI-GGUF'
# during startup. We look there first, then fall back to importlib.util.
# ---------------------------------------------------------------------------
import importlib.util as _ilu
import sys as _sys

def _import_from_gguf(submod_name: str):
    """Return a module from the ComfyUI-GGUF package using any available method."""
    # 1) Already registered by ComfyUI's node loader
    full_key = f"ComfyUI-GGUF.{submod_name}"
    if full_key in _sys.modules:
        return _sys.modules[full_key]

    # 2) Load via importlib from the sibling folder
    import os
    gguf_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ComfyUI-GGUF")
    submod_path = os.path.join(gguf_dir, f"{submod_name}.py")
    if not os.path.isfile(submod_path):
        raise ImportError(f"Cannot find ComfyUI-GGUF/{submod_name}.py at {submod_path}")

    # We need a synthetic parent package so relative imports inside work
    pkg_key = "ComfyUI-GGUF"
    if pkg_key not in _sys.modules:
        import types
        pkg = types.ModuleType(pkg_key)
        pkg.__path__ = [os.path.abspath(gguf_dir)]
        pkg.__package__ = pkg_key
        _sys.modules[pkg_key] = pkg
        # Also pre-load __init__ if present
        init_path = os.path.join(gguf_dir, "__init__.py")
        if os.path.isfile(init_path):
            spec = _ilu.spec_from_file_location(pkg_key, init_path,
                                                submodule_search_locations=[os.path.abspath(gguf_dir)])
            mod = _ilu.module_from_spec(spec)
            _sys.modules[pkg_key] = mod
            spec.loader.exec_module(mod)

    spec = _ilu.spec_from_file_location(full_key, submod_path)
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = pkg_key
    _sys.modules[full_key] = mod
    spec.loader.exec_module(mod)
    return mod

_gguf_ops    = _import_from_gguf("ops")
_gguf_dequant = _import_from_gguf("dequant")

GGMLTensor       = _gguf_ops.GGMLTensor
is_quantized     = _gguf_dequant.is_quantized
dequantize_tensor = _gguf_dequant.dequantize_tensor

# ---------------------------------------------------------------------------
# Known ACE-Step architecture strings
# ---------------------------------------------------------------------------
ACESTEP_ARCH_DIT      = "acestep-dit"
ACESTEP_ARCH_TEXT_ENC = "acestep-text-enc"
ACESTEP_ARCH_VAE      = "acestep-vae"
ACESTEP_ARCH_LM       = "acestep-lm"      # Audio-code language models (5Hz LMs)
ACESTEP_ARCH_ALL      = {ACESTEP_ARCH_DIT, ACESTEP_ARCH_TEXT_ENC, ACESTEP_ARCH_VAE, ACESTEP_ARCH_LM}


# ---------------------------------------------------------------------------
# Core GGUF reader — returns a raw state_dict of GGMLTensor objects
# ---------------------------------------------------------------------------
def _read_gguf_raw(path: str) -> tuple[dict, dict]:
    """
    Open a GGUF file and return (state_dict, metadata).

    state_dict : { tensor_name: GGMLTensor(...) }
    metadata   : { field_name: value } for simple scalar/string fields
    """
    reader = gguf.GGUFReader(path)

    # -- validate architecture -------------------------------------------------
    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        raise ValueError(
            f"GGUF file has no 'general.architecture' field — cannot determine model type.\n({path})"
        )
    arch_str = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")
    if arch_str not in ACESTEP_ARCH_ALL:
        raise ValueError(
            f"Unsupported architecture '{arch_str}'. "
            f"This loader only handles: {ACESTEP_ARCH_ALL}\n({path})"
        )

    # -- build state dict ------------------------------------------------------
    state_dict: dict[str, GGMLTensor] = {}
    qtype_dict: dict[str, int] = {}

    for tensor in reader.tensors:
        name = tensor.name

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)

        # Restore original shape (reversed in GGUF storage)
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)

        gmml_t = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

        # BF16 1D tensors → dequant immediately (avoids later issues)
        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            gmml_t = dequantize_tensor(gmml_t, dtype=torch.float32)

        state_dict[name] = gmml_t

        tname = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tname] = qtype_dict.get(tname, 0) + 1

    logging.info(
        f"AceStep-GGUF [{arch_str}] qtypes: "
        + ", ".join(f"{k}({v})" for k, v in qtype_dict.items())
    )

    # Mark the largest quantized tensor so ComfyUI can estimate VRAM
    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if qsd:
        max_key = max(qsd, key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    # -- collect metadata ------------------------------------------------------
    metadata: dict = {"arch_str": arch_str}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if field and len(field.types) == 1:
                t = field.types[0]
                if t == gguf.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif t in (gguf.GGUFValueType.INT32, gguf.GGUFValueType.UINT32,
                            gguf.GGUFValueType.INT64, gguf.GGUFValueType.UINT64,
                            gguf.GGUFValueType.UINT8):
                    metadata[field_name] = int(field.parts[field.data[-1]])
                elif t == gguf.GGUFValueType.FLOAT32:
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif t == gguf.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except Exception:
            pass

    return state_dict, metadata


# ---------------------------------------------------------------------------
# DiT loader
# ---------------------------------------------------------------------------
def load_dit_gguf(path: str) -> tuple[dict, dict]:
    """
    Load an ACE-Step DiT GGUF (acestep-v15-*.gguf / acestep-v15-xl-*.gguf).

    Returns (state_dict, metadata).

    Fixups applied to bridge GGUF storage → ComfyUI expected shapes:

    1. scale_shift_table: [6, D] → [1, 6, D]  (decoder and layer norms)
       also [2, D] → [1, 2, D]  (decoder scale_shift_table)

    2. Special-token parameters lose their leading batch dim in the GGUF:
         null_condition_emb              [D]    → [1, 1, D]
         encoder.timbre_encoder.special_token [D] → [1, 1, D]
         tokenizer.attention_pooler.special_token [D] → [1, 1, D]
         detokenizer.special_tokens      [5, D] → [1, 5, D]

    The metadata['acestep-dit.embedding_length'] contains the true hidden_size
    (2048 for standard, 2560 for XL) which the caller uses to build the right
    model config — see AceStepDiTLoaderGGUF.load_dit().
    """
    sd, meta = _read_gguf_raw(path)

    # ---- 1. scale_shift_table reshape -------------------------------------
    # stored as [N, D]; model expects [1, N, D]
    fixed_sst = 0
    for key in list(sd.keys()):
        if key.endswith("scale_shift_table"):
            t = sd[key]
            if is_quantized(t):
                t = dequantize_tensor(t, dtype=torch.bfloat16)
            if t.ndim == 2:          # [6,D] or [2,D]
                t = t.unsqueeze(0)   # → [1,6,D] or [1,2,D]
            sd[key] = t
            fixed_sst += 1
    if fixed_sst:
        logging.info(f"AceStep-GGUF DiT: reshaped {fixed_sst} scale_shift_table tensors → [1,N,D]")

    # ---- 2. special-token parameter reshape --------------------------------
    # GGUF drops the leading singleton batch dimension on nn.Parameter tensors.
    # We restore them so load_state_dict() doesn't raise RuntimeError.
    special_fixups = {
        # key suffix → expected final shape after unsqueeze
        "null_condition_emb": "unsqueeze_0_1",          # [D]    → [1, 1, D]
        "encoder.timbre_encoder.special_token": "unsqueeze_0_1",  # [D] → [1, 1, D]
        "tokenizer.attention_pooler.special_token": "unsqueeze_0_1",  # [D] → [1, 1, D]
        "detokenizer.special_tokens": "unsqueeze_0",    # [5, D] → [1, 5, D]
    }

    fixed_sp = 0
    for key in list(sd.keys()):
        for suffix, mode in special_fixups.items():
            if key == suffix or key.endswith(f".{suffix}"):
                t = sd[key]
                if is_quantized(t):
                    t = dequantize_tensor(t, dtype=torch.bfloat16)
                elif isinstance(t, GGMLTensor):
                    t = t.data.view(t.tensor_shape)
                # Convert to plain torch tensor for unsqueeze
                if hasattr(t, 'data') and hasattr(t, 'tensor_shape'):
                    t = t.data.view(t.tensor_shape)
                if mode == "unsqueeze_0_1":
                    if t.ndim == 1:         # [D] → [1, 1, D]
                        t = t.unsqueeze(0).unsqueeze(0)
                elif mode == "unsqueeze_0":
                    if t.ndim == 2:         # [5, D] → [1, 5, D]
                        t = t.unsqueeze(0)
                sd[key] = t
                fixed_sp += 1
                break
    if fixed_sp:
        logging.info(f"AceStep-GGUF DiT: fixed {fixed_sp} special-token tensors to restore batch dims")

    # ---- 3. BF16 physical PyTorch shape fix -------------------------------
    # GGUF models that contain native BF16 tensors (like the DiT patch proj) 
    # read them into uint8 numpy arrays. ComfyUI-GGUF skips viewing them as 
    # bfloat16, leaving them with PyTorch shapes like [2560, 192, 4].
    # PyTorch 2.9+ `assign=True` strict mode strictly compares PyTorch sizes.
    fixed_bf16 = 0
    import gguf
    for key, t in list(sd.items()):
        if getattr(t, "tensor_type", None) == gguf.GGMLQuantizationType.BF16:
            if hasattr(t, "data") and t.data.dtype == torch.uint8:
                try:
                    t.data = t.data.view(torch.bfloat16)
                    fixed_bf16 += 1
                except Exception as e:
                    logging.warning(f"Failed to view {key} as bfloat16: {e}")
    if fixed_bf16:
        logging.info(f"AceStep-GGUF DiT: fixed {fixed_bf16} BF16 tensors via .view(torch.bfloat16)")

    return sd, meta



# ---------------------------------------------------------------------------
# Text encoder loader (Qwen3-Embedding)
# ---------------------------------------------------------------------------
def _build_bpe_tokenizer_tensor(reader: gguf.GGUFReader) -> torch.Tensor | None:
    """
    Reconstruct a minimal BPE vocabulary JSON from GGUF tokenizer metadata
    and return it as a ByteTensor, compatible with ComfyUI's ACE text-encoder
    tokenizer loading path.

    Reads from GGUF KV metadata (standard llama.cpp format):
      tokenizer.ggml.model   – tokenizer type string (e.g. "gpt2")
      tokenizer.ggml.tokens  – full vocab as string array
      tokenizer.ggml.merges  – BPE merge rules as string array
      tokenizer.ggml.token_type – optional int array; absent in many Qwen3 GGUFs

    Returns None if the essential vocab isn't present.
    """
    def _get_list(field_name, cast=str):
        field = reader.get_field(field_name)
        if field is None:
            return None
        if cast == str:
            return [str(field.parts[i], "utf-8") for i in field.data]
        return [cast(field.parts[i][0]) for i in field.data]

    tokens = _get_list("tokenizer.ggml.tokens", str)
    if tokens is None:
        return None   # vocab is essential; nothing we can do without it

    # token_type is OPTIONAL — Qwen3 GGUFs typically omit it.
    # When absent, treat all tokens as regular (type 1).
    toktypes = _get_list("tokenizer.ggml.token_type", int)
    if toktypes is None:
        toktypes = [1] * len(tokens)  # 1 = normal, 3 = special/control

    # BPE merge rules (optional but enables proper tokenization)
    merges = _get_list("tokenizer.ggml.merges", str)

    # Build the bytes_to_unicode mapping (same as transformers GPT-2)
    import base64
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    decoder = {chr(c): b for b, c in zip(bs, cs)}

    data = {
        "config": {"num_vocab_tokens": len(tokens), "default_vocab_size": len(tokens)},
        "vocab": [],
        "special_tokens": [],
        "merges": merges if merges is not None else [],
    }
    for idx, (token, toktype) in enumerate(zip(tokens, toktypes)):
        if toktype == 3:  # special/control token
            data["special_tokens"].append(
                {"rank": idx, "token_str": token, "is_control": True}
            )
        else:
            try:
                tok_bytes = bytes([decoder.get(ch, 0) for ch in token])
                data["vocab"].append({
                    "rank": len(data["vocab"]),
                    "token_bytes": base64.b64encode(tok_bytes).decode("ascii"),
                    "token_str": tok_bytes.decode("utf-8", errors="replace"),
                })
            except Exception:
                pass

    logging.info(
        f"AceStep-GGUF TextEnc: built BPE vocab "
        f"({len(data['vocab'])} tokens + {len(data['special_tokens'])} special"
        f"{', ' + str(len(data['merges'])) + ' merges' if data['merges'] else ''})"
    )
    return torch.ByteTensor(list(json.dumps(data).encode("utf-8")))


def load_text_encoder_gguf(path: str) -> tuple[dict, dict]:
    """
    Load the Qwen3-Embedding-0.6B GGUF text encoder.

    The GGUF stores keys as `layers.X.*` / `embed_tokens.weight` / `norm.weight`
    but ComfyUI's detect_te_model() looks for `model.layers.X.*` etc.
    So we add the `model.` prefix to all relevant keys to match the HuggingFace
    safetensors layout that ComfyUI's detection logic expects.

    Returns (state_dict, metadata).
    """
    sd, meta = _read_gguf_raw(path)

    # --- Add `model.` prefix so detect_te_model() can identify QWEN3_06B ---
    # Keys that need the prefix:
    #   layers.*         → model.layers.*
    #   embed_tokens.*   → model.embed_tokens.*
    #   norm.*           → model.norm.*
    # Keys to leave as-is: bpe_model, etc.
    prefixed: dict = {}
    for k, v in sd.items():
        if k.startswith(("layers.", "embed_tokens.", "norm.")):
            prefixed[f"model.{k}"] = v
        else:
            prefixed[k] = v
    sd = prefixed
    logging.info("AceStep-GGUF TextEnc: added 'model.' prefix to layers/embed/norm keys")

    # --- Inject BPE tokenizer from GGUF metadata (if present) ---
    reader = gguf.GGUFReader(path)
    bpe_tensor = _build_bpe_tokenizer_tensor(reader)
    if bpe_tensor is not None:
        sd["bpe_model"] = bpe_tensor
        logging.info("AceStep-GGUF TextEnc: BPE tokenizer metadata extracted from GGUF")
    else:
        logging.warning(
            "AceStep-GGUF TextEnc: no BPE tokenizer metadata in GGUF. "
            "Tokenizer will fall back to ComfyUI's internal Qwen3 tokenizer if available."
        )

    # --- Dequantize the embedding table to avoid OOM during lookup ---
    emb_key = "model.embed_tokens.weight"
    if emb_key in sd and is_quantized(sd[emb_key]):
        logging.warning(f"AceStep-GGUF TextEnc: dequantizing {emb_key} to prevent OOM")
        sd[emb_key] = dequantize_tensor(sd[emb_key], dtype=torch.float16)

    return sd, meta


# ---------------------------------------------------------------------------
# VAE loader (AutoencoderOobleck)
# ---------------------------------------------------------------------------
def load_vae_gguf(path: str) -> dict:
    """
    Load the ACE-Step VAE GGUF (vae-BF16.gguf).

    Returns a plain state_dict with keys identical to the original
    diffusion_pytorch_model.safetensors (decoder.block.*, encoder.*).
    All BF16 tensors are dequantized immediately since the VAE is small and
    ComfyUI's VAE class doesn't use custom ops.

    Key facts about vae-BF16.gguf:
    - All tensors are BF16 stored as uint8 raw bytes (GGML BF16 type)
    - GGMLTensor.shape = reversed(GGUF_ne) = correct PyTorch tensor shape
    - data.shape has the last-dim doubled (2 uint8 bytes per bfloat16 element)
    - Must flatten+contiguous before view(bfloat16) to avoid mmap alignment errors
    """
    sd, meta = _read_gguf_raw(path)

    def _bf16_ggml_to_tensor(val) -> torch.Tensor:
        """Convert a BF16 GGMLTensor (uint8 raw bytes) to a plain bfloat16 tensor."""
        target_shape = val.tensor_shape  # correct PyTorch shape
        raw = val.data                   # uint8, shape may be (*, 2) where * = target_shape

        # Flatten to 1D uint8, convert to contiguous (safe for mmap'd arrays),
        # then reinterpret as bfloat16, then reshape to the target PyTorch shape.
        try:
            out = raw.flatten().contiguous().view(torch.bfloat16).reshape(target_shape)
            return out.clone()           # detach from mmap storage
        except Exception:
            # Fallback: use ComfyUI-GGUF dequantize path
            return dequantize_tensor(val, dtype=torch.bfloat16)

    out: dict[str, torch.Tensor] = {}
    for key, val in sd.items():
        if is_quantized(val):
            # Shouldn't happen for a BF16-only VAE, but handle gracefully
            out[key] = dequantize_tensor(val, dtype=torch.bfloat16)
        elif isinstance(val, GGMLTensor) and getattr(val, "tensor_type", None) == gguf.GGMLQuantizationType.BF16:
            out[key] = _bf16_ggml_to_tensor(val)
        elif isinstance(val, GGMLTensor):
            try:
                out[key] = dequantize_tensor(val, dtype=torch.bfloat16)
            except Exception:
                out[key] = val.data.flatten().contiguous().reshape(val.tensor_shape).to(torch.bfloat16)
        else:
            out[key] = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)

    logging.info(f"AceStep-GGUF VAE: loaded {len(out)} tensors from {path}")
    return out


# ---------------------------------------------------------------------------
# LM loader (acestep-5Hz-lm-1.7B, acestep-5Hz-lm-4B)
# ---------------------------------------------------------------------------
def load_lm_gguf(path: str) -> tuple[dict, dict]:
    """
    Load an ACE-Step 5Hz audio-code Language Model GGUF.

    Supports:
      - acestep-5Hz-lm-1.7B-Q8_0.gguf  (hidden=2048 → QWEN3_2B / qwen3_2b)
      - acestep-5Hz-lm-4B-Q8_0.gguf    (hidden=2560 → QWEN3_4B / qwen3_4b)

    Unlike the text encoder, these files already store keys with the
    'model.' prefix (e.g. model.layers.0.*), matching what ComfyUI's
    detect_te_model() and ACE15TEModel.load_sd() expect.
    No remapping required.

    The embedding table (model.embed_tokens.weight) is kept as a GGMLTensor
    so quantized ops work; only norm weights are dequantized immediately.
    """
    sd, meta = _read_gguf_raw(path)

    # Dequantize the embedding table for OOM safety (217k vocab × hidden_dim)
    emb_key = "model.embed_tokens.weight"
    if emb_key in sd and is_quantized(sd[emb_key]):
        logging.info(f"AceStep-GGUF LM: dequantizing {emb_key} ({sd[emb_key].shape})")
        # Keep as float16 to save VRAM vs float32
        sd[emb_key] = dequantize_tensor(sd[emb_key], dtype=torch.float16)

    # Log detected model size
    detect_key = "model.layers.0.post_attention_layernorm.weight"
    if detect_key in sd:
        hidden_dim = int(sd[detect_key].shape[0])
        if hidden_dim == 2048:
            lm_variant = "qwen3_2b (1.7B)"
        elif hidden_dim == 2560:
            lm_variant = "qwen3_4b (4B)"
        else:
            lm_variant = f"unknown (hidden={hidden_dim})"
        logging.info(
            f"AceStep-GGUF LM: detected {lm_variant} | "
            f"arch={meta.get('general.name', '?')} | "
            f"tensors={len(sd)}"
        )

    return sd, meta


# ---------------------------------------------------------------------------
# AudioOobleckVAE GGUF — direct load (bypasses comfy.sd.VAE auto-detection)
# ---------------------------------------------------------------------------

def is_oobleck_vae_gguf(sd: dict) -> bool:
    """Return True if *sd* looks like a vae-BF16.gguf (AudioOobleckVAE format)."""
    return "encoder.conv1.bias" in sd and "decoder.conv1.bias" in sd


def _remap_oobleck_gguf_to_comfy(raw: dict) -> dict:
    """
    Remap GGUF Oobleck VAE keys (encoder.block.N.res_unitM.*) to the key names
    produced by ComfyUI's AudioOobleckVAE state_dict() (encoder.layers.K.*).

    Two categories of changes:
      1. Structural prefix:
           encoder.conv1          → encoder.layers.0
           encoder.block.N.*      → encoder.layers.{N+1}.*
           encoder.snake1         → encoder.layers.6
           encoder.conv2          → encoder.layers.7
           (decoder mirrors the same pattern)

         Inside each block:
           block.N.res_unitM.snake1 → layers.{N+1}.layers.{M-1}.layers.0
           block.N.res_unitM.conv1  → layers.{N+1}.layers.{M-1}.layers.1
           block.N.res_unitM.snake2 → layers.{N+1}.layers.{M-1}.layers.2
           block.N.res_unitM.conv2  → layers.{N+1}.layers.{M-1}.layers.3
           block.N.snake1           → layers.{N+1}.layers.3   (encoder block)
           block.N.conv1            → layers.{N+1}.layers.4   (encoder strided conv)
           decoder block variant:
           block.N.snake1           → layers.{N+1}.layers.0
           block.N.conv_t1          → layers.{N+1}.layers.1
           block.N.res_unitM.*      → layers.{N+1}.layers.{M+1}.*

      2. Suffix for weight-normalised convolutions:
           .weight_g  → .parametrizations.weight.original0
           .weight_v  → .parametrizations.weight.original1
           .alpha/.beta tensors: squeezed from (1, C, 1) → (C,)
    """
    import re

    def _remap_suffix(suffix: str, val: torch.Tensor):
        """Map weight_g/v suffix and return (new_suffix, maybe_processed_val)."""
        if suffix == "weight_g":
            return "parametrizations.weight.original0", val
        if suffix == "weight_v":
            return "parametrizations.weight.original1", val
        if suffix in ("alpha", "beta") and val.ndim == 3:
            # GGUF stores alpha/beta as (1, C, 1); ComfyUI model expects (C,)
            val = val.squeeze(0).squeeze(-1)
        return suffix, val

    # Pre-compiled patterns
    _enc_block_ru = re.compile(r"^encoder\.block\.(\d+)\.(res_unit(\d+))\.(snake[12]|conv[12])\.(.+)$")
    _enc_block_other = re.compile(r"^encoder\.block\.(\d+)\.(snake1|conv1)\.(.+)$")
    _dec_block_ru = re.compile(r"^decoder\.block\.(\d+)\.(res_unit(\d+))\.(snake[12]|conv[12])\.(.+)$")
    _dec_block_other = re.compile(r"^decoder\.block\.(\d+)\.(snake1|conv_t1)\.(.+)$")

    # ResidualUnit internal layer index mapping
    # EncoderBlock layers: [ru1(0), ru2(1), ru3(2), snake(3), strided_conv(4)]
    # ResidualUnit layers:  snake1(0), conv1(1), snake2(2), conv2(3)
    _ru_map = {"snake1": "0", "conv1": "1", "snake2": "2", "conv2": "3"}
    # DecoderBlock layers: snake1(0), conv_t1(1), ru1(2), ru2(3), ru3(4)
    _dec_other_map = {"snake1": "0", "conv_t1": "1"}

    out = {}
    for key, val in raw.items():
        new_key = key
        new_val = val

        # ── encoder.conv1 ──────────────────────────────────────────────────
        if key.startswith("encoder.conv1.") or key.startswith("decoder.conv1."):
            # conv1 → layers.0
            prefix, rest_with_dot = key.split(".", 1)  # encoder or decoder
            _, rest = rest_with_dot.split(".", 1)       # conv1.<suffix>
            sfx, new_val = _remap_suffix(rest, val)
            new_key = f"{prefix}.layers.0.{sfx}"

        # ── encoder.snake1 / encoder.conv2 ─────────────────────────────────
        elif key.startswith("encoder.snake1.") or key.startswith("encoder.conv2."):
            _, rest_with_dot = key.split(".", 1)
            sub, sfx_raw = rest_with_dot.split(".", 1)
            layer_idx = "6" if sub == "snake1" else "7"
            sfx, new_val = _remap_suffix(sfx_raw, val)
            new_key = f"encoder.layers.{layer_idx}.{sfx}"

        # ── decoder.snake1 / decoder.conv2 ─────────────────────────────────
        elif key.startswith("decoder.snake1.") or key.startswith("decoder.conv2."):
            _, rest_with_dot = key.split(".", 1)
            sub, sfx_raw = rest_with_dot.split(".", 1)
            layer_idx = "6" if sub == "snake1" else "7"
            sfx, new_val = _remap_suffix(sfx_raw, val)
            new_key = f"decoder.layers.{layer_idx}.{sfx}"

        # ── encoder.block.N.res_unitM.snake/conv ───────────────────────────
        elif m := _enc_block_ru.match(key):
            N, _, M, sub_layer, sfx_raw = m.groups()
            block_li = str(int(N) + 1)         # encoder.layers index
            ru_li = str(int(M) - 1)             # ru inside block (0-based)
            inner_li = _ru_map[sub_layer]        # layers inside ResidualUnit
            sfx, new_val = _remap_suffix(sfx_raw, val)
            new_key = f"encoder.layers.{block_li}.layers.{ru_li}.layers.{inner_li}.{sfx}"

        # ── encoder.block.N.snake1 / encoder.block.N.conv1 (strided conv) ──
        elif m := _enc_block_other.match(key):
            N, sub_layer, sfx_raw = m.groups()
            block_li = str(int(N) + 1)
            inner_li = "3" if sub_layer == "snake1" else "4"
            sfx, new_val = _remap_suffix(sfx_raw, val)
            new_key = f"encoder.layers.{block_li}.layers.{inner_li}.{sfx}"

        # ── decoder.block.N.res_unitM.snake/conv ───────────────────────────
        elif m := _dec_block_ru.match(key):
            N, _, M, sub_layer, sfx_raw = m.groups()
            block_li = str(int(N) + 1)
            ru_li = str(int(M) + 1)             # res_unit1→layers.2, ru2→3, ru3→4
            inner_li = _ru_map[sub_layer]
            sfx, new_val = _remap_suffix(sfx_raw, val)
            new_key = f"decoder.layers.{block_li}.layers.{ru_li}.layers.{inner_li}.{sfx}"

        # ── decoder.block.N.snake1 / decoder.block.N.conv_t1 ────────────────
        elif m := _dec_block_other.match(key):
            N, sub_layer, sfx_raw = m.groups()
            block_li = str(int(N) + 1)
            inner_li = _dec_other_map[sub_layer]
            sfx, new_val = _remap_suffix(sfx_raw, val)
            new_key = f"decoder.layers.{block_li}.layers.{inner_li}.{sfx}"

        else:
            # Unchanged key (e.g. encoder/decoder.conv2.bias, unknown new keys)
            new_key = key
            new_val = val

        out[new_key] = new_val

    return out


def load_oobleck_vae_from_gguf(raw_sd: dict) -> "comfy.sd.VAE":
    """
    Directly instantiate AudioOobleckVAE from a GGUF-format state dict and
    wrap it in a comfy.sd.VAE shell (bypassing auto-detection).

    Architecture inferred from the vae-BF16.gguf tensor shapes:
      channels=128, latent_dim=64, c_mults=[1,2,4,8,16], strides=[2,4,4,6,10]
      in_channels=2, use_snake=True, antialias_activation=False

    The total compression ratio (product of strides) = 2*4*4*6*10 = 1920.
    """
    import math
    import comfy.sd
    import comfy.model_patcher
    import comfy.model_management as mm
    from comfy.ldm.audio.autoencoder import AudioOobleckVAE

    logging.info("AceStep-GGUF VAE: detected AudioOobleckVAE format — remapping keys")

    # 1. Remap GGUF keys → ComfyUI parameter names
    remapped = _remap_oobleck_gguf_to_comfy(raw_sd)

    # 2. Instantiate the model (registers weight_norm hook → parametrizations)
    model = AudioOobleckVAE(
        in_channels=2,
        channels=128,
        latent_dim=64,
        c_mults=[1, 2, 4, 8, 16],
        strides=[2, 4, 4, 6, 10],
        use_snake=True,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=False,
    )

    # 3. Load the remapped weights (strict=False — bias may be missing on some convs)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        logging.warning(f"AceStep-GGUF VAE: missing keys after remap ({len(missing)}): {missing[:5]}")
    if unexpected:
        logging.warning(f"AceStep-GGUF VAE: unexpected keys after remap ({len(unexpected)}): {unexpected[:5]}")

    # Cast everything to bfloat16: biases default to float32, but the model
    # runs in bfloat16, so a dtype mismatch would crash Conv1d forward.
    model.to(torch.bfloat16)

    # 4. Build a custom wrapper inheriting from comfy.sd.VAE
    # We do this because ComfyUI's ModelPatcher doesn't seem to natively walk
    # the nested Sequential blocks inside AudioOobleckVAE correctly, which
    # leaves some parameters on the CPU, causing device mismatch during forward().
    class AceAudioOobleckVAEWrapper(comfy.sd.VAE):
        def decode(self, samples_in, vae_options={}):
            # Ensure model parameters are actually moved to the GPU before decoding
            self.first_stage_model.to(device=self.device)
            audio_48k = super().decode(samples_in, vae_options)
            # The Oobleck VAE produces audio at 48000 Hz but ACE-Step 1.5's pipeline
            # (sampler + audio-save nodes) operates at 44100 Hz.
            # ComfyUI outputs [B, Time, Channels], so we move dimension to [B, Channels, Time] for torchaudio
            import torchaudio
            audio_44k = torchaudio.functional.resample(audio_48k.cpu().movedim(-1, -2), orig_freq=48000, new_freq=44100)
            return audio_44k.movedim(-2, -1)
            
        def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
            self.first_stage_model.to(device=self.device)
            audio_48k = super().decode_tiled(samples, tile_x, tile_y, overlap, tile_t, overlap_t)
            import torchaudio
            audio_44k = torchaudio.functional.resample(audio_48k.cpu().movedim(-1, -2), orig_freq=48000, new_freq=44100)
            return audio_44k.movedim(-2, -1)

        def encode(self, pixel_samples):
            self.first_stage_model.to(device=self.device)
            return super().encode(pixel_samples)

    offload_device = mm.unet_offload_device()
    # Skip standard __init__ which calls first_stage_model.load_state_dict, since
    # we already mapped and loaded the GGUF weights.
    vae_obj = object.__new__(AceAudioOobleckVAEWrapper)

    # Minimal attribute set copied from what sd.py sets for audio VAEs
    # shape is (batch, latent_channels, time_frames) — 3D for 1D audio VAEs
    vae_obj.memory_used_encode = lambda shape, dtype: (shape[2] * 640) * comfy.model_management.dtype_size(dtype)
    vae_obj.memory_used_decode = lambda shape, dtype: (shape[2] * 1920 * 1280) * comfy.model_management.dtype_size(dtype)
    vae_obj.downscale_ratio = 1920        # 2*4*4*6*10
    vae_obj.upscale_ratio   = 1920
    vae_obj.latent_dim      = 1           # 1D (time axis)
    vae_obj.latent_channels = 64
    vae_obj.output_channels = 2           # stereo
    vae_obj.process_input   = lambda audio: audio
    vae_obj.process_output  = lambda audio: audio
    vae_obj.working_dtypes  = [torch.bfloat16, torch.float32]
    vae_obj.vae_dtype       = torch.bfloat16
    vae_obj.first_stage_model = model
    vae_obj.device          = mm.vae_device()
    vae_obj.offload_device  = offload_device
    vae_obj.output_device   = mm.intermediate_device()

    vae_obj.patcher         = comfy.model_patcher.ModelPatcher(
        model, load_device=vae_obj.device, offload_device=offload_device
    )
    vae_obj.crop_input      = False
    vae_obj.upscale_index_formula = None
    vae_obj.downscale_index_formula = None
    vae_obj.size            = None
    vae_obj.pad_channel_value = None
    vae_obj.extra_1d_channel  = None
    vae_obj.disable_offload   = False

    logging.info(f"AceStep-GGUF VAE: loaded {len(remapped)} remapped tensors "
                 f"(loaded={len(remapped)-len(missing)}, missing={len(missing)})")
    return vae_obj


