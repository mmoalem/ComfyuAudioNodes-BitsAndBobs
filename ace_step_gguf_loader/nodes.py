"""
nodes.py — ComfyUI node classes for ACE-Step GGUF loaders.

Provides four nodes:
  - AceStepDiTLoaderGGUF          → MODEL
  - AceStepDualCLIPLoaderGGUF     → CLIP  (Qwen3-0.6B + 5Hz-LM, mirrors existing loader)
  - AceStepVAELoaderGGUF          → VAE
  - AceStepModelLoaderGGUF        → MODEL + CLIP + VAE (all-in-one convenience)

The dual-CLIP node mirrors the existing ScromfyAceStepModelLoader structure:
  text_encoder_1 = Qwen3-Embedding-0.6B (lyrics / caption encoder)
  text_encoder_2 = acestep-5Hz-lm-1.7B or 4B (audio-code language model)

These nodes appear in the ComfyUI node menu under: AceStep/GGUF Loaders/
"""

import os
import logging
import torch

import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.model_patcher

from .loader_utils import (
    load_dit_gguf, load_text_encoder_gguf, load_lm_gguf, load_vae_gguf,
    is_oobleck_vae_gguf, load_oobleck_vae_from_gguf,
    GGUF_AVAILABLE
)

# ---------------------------------------------------------------------------
# Import City96's GGMLOps and GGUFModelPatcher
# The hyphenated folder name means we need importlib (same as loader_utils.py)
# ---------------------------------------------------------------------------
import importlib.util as _ilu
import sys as _sys

def _get_gguf_mod(submod_name: str):
    """Return a module from the ComfyUI-GGUF package."""
    if not GGUF_AVAILABLE:
        return None
    full_key = f"ComfyUI-GGUF.{submod_name}"
    if full_key in _sys.modules:
        return _sys.modules[full_key]
    from .loader_utils import _import_from_gguf
    return _import_from_gguf(submod_name)

if GGUF_AVAILABLE:
    _gguf_ops_mod   = _get_gguf_mod("ops")
    _gguf_nodes_mod = _get_gguf_mod("nodes")
    _gguf_dequant   = _get_gguf_mod("dequant")

    GGMLOps          = _gguf_ops_mod.GGMLOps
    GGUFModelPatcher = _gguf_nodes_mod.GGUFModelPatcher
    is_quantized     = _gguf_dequant.is_quantized
else:
    GGMLOps          = None
    GGUFModelPatcher = None
    is_quantized     = None


def _check_gguf():
    if not GGUF_AVAILABLE:
        raise ImportError(
            "\n\n[ACEStep GGUF Loader] ERROR: 'ComfyUI-GGUF' custom node not found.\n"
            "This node is required to load and dequantize GGUF models.\n\n"
            "Please install it from: https://github.com/city96/ComfyUI-GGUF\n"
            "or via ComfyUI-Manager (search for 'GGUF').\n"
        )


# ---------------------------------------------------------------------------
# Folder registration — add GGUF-specific search paths
# ---------------------------------------------------------------------------
def _register_folder(key: str, primary_targets: list[str]):
    """Register a folder key that scans for .gguf files."""
    target_key = next(
        (t for t in primary_targets if t in folder_paths.folder_names_and_paths),
        primary_targets[0]
    )
    orig_dirs, _ = folder_paths.folder_names_and_paths.get(target_key, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig_dirs or [], {".gguf"})


_register_folder("acestep_dit_gguf",  ["diffusion_models", "unet"])
_register_folder("acestep_enc_gguf",  ["text_encoders", "clip"])   # 0.6B text encoder
_register_folder("acestep_lm_gguf",   ["text_encoders", "clip"])   # 1.7B / 4B audio LM
_register_folder("acestep_vae_gguf",  ["vae"])


# ---------------------------------------------------------------------------
# File listing / path resolution helpers
# ---------------------------------------------------------------------------
def _list_gguf_files(*folder_keys: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for key in folder_keys:
        try:
            for f in folder_paths.get_filename_list(key):
                if f not in seen:
                    seen.add(f)
                    result.append(f)
        except Exception:
            pass
    return sorted(result)


def _resolve_path(filename: str, *folder_keys: str) -> str:
    """Resolve a bare filename to a full path using the given folder keys."""
    if os.path.isabs(filename) and os.path.isfile(filename):
        return filename
    for key in folder_keys:
        try:
            p = folder_paths.get_full_path(key, filename)
            if p and os.path.isfile(p):
                return p
        except Exception:
            pass
    raise FileNotFoundError(
        f"Could not find '{filename}' in any of: {folder_keys}"
    )


# ===========================================================================
# Node 1: AceStepDiTLoaderGGUF
# ===========================================================================
class AceStepDiTLoaderGGUF:
    """
    Load an ACE-Step 1.5 Diffusion Transformer (DiT) from a GGUF file.

    Supports both the standard and XL variants in any quantization level.
    Output MODEL feeds directly into ACE-Step sampler nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        dit_files = _list_gguf_files("acestep_dit_gguf", "unet_gguf", "diffusion_models")
        if not dit_files:
            dit_files = ["[No .gguf files found — place in models/diffusion_models/]"]
        return {
            "required": {
                "dit_name": (dit_files, {
                    "tooltip": (
                        "ACE-Step DiT GGUF file (e.g. acestep-v15-xl-turbo-Q8_0.gguf). "
                        "Place in models/diffusion_models/ or models/unet/"
                    )
                }),
                "dequant_dtype": (
                    ["default", "float32", "float16", "bfloat16"],
                    {"default": "default",
                     "tooltip": "Precision used when dequantizing weights during the forward pass."}
                ),
            },
        }

    RETURN_TYPES  = ("MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "load_dit"
    CATEGORY      = "ACE-Step/GGUF Loaders"
    TITLE         = "AceStep DiT Loader (GGUF)"

    def load_dit(self, dit_name: str, dequant_dtype: str = "default"):
        _check_gguf()
        path = _resolve_path(dit_name, "acestep_dit_gguf", "unet_gguf", "diffusion_models", "unet")

        ops = GGMLOps()
        if dequant_dtype == "default":
            ops.Linear.dequant_dtype = None
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        sd, meta = load_dit_gguf(path)

        # ----------------------------------------------------------------
        # Build the correct ACEStep15 model config from GGUF metadata.
        #
        # Problem: ComfyUI's model_detection always returns hidden_size=2048
        # (the default in AceStepConditionGenerationModel) because it just
        # returns {"audio_model": "ace1.5"} without reading hidden_size.
        # For the XL model, hidden_size=2560 → all decoder layers mismatch.
        #
        # Solution: read 'acestep-dit.embedding_length' from the GGUF header
        # and inject the correct values into the unet_config before calling
        # get_model(), bypassing the auto-detection entirely.
        # ----------------------------------------------------------------
        import comfy.model_detection as _det
        import comfy.supported_models as _sm
        import comfy.model_management as _mm

        hidden_size    = meta.get("acestep-dit.embedding_length", 2048)
        
        # XL models sometimes carry default 2048 GGUF metadata. Pluck exact size from bias.
        proj_in_bias_key = "decoder.proj_in.1.bias"
        if proj_in_bias_key in sd:
            hidden_size = int(sd[proj_in_bias_key].shape[0])

        num_dit_layers = meta.get("acestep-dit.block_count", 24)
        num_heads      = hidden_size // 128
        num_kv_heads   = num_heads // 2
        head_dim       = 128
        intermediate   = hidden_size * 3

        # The XL turbo model has a patch size of 4, while the 1.5 base has 2!
        # Let's derive it directly from the tensor shape if possible, or assume 4 for XL.
        patch_size   = 2
        fsq_dim      = 2048
        # Check decoder.proj_in shape to accurately detect patch size
        proj_in_key = "decoder.proj_in.1.weight"
        if proj_in_key in sd:
            patch_size = list(sd[proj_in_key].shape)[-1]
        
        # Detect fsq_dim from the bias of project_out — biases are always shape
        # (output_dim,) with no ambiguity about row/col convention. project_out
        # maps codebook_dim → fsq_dim, so its bias has shape (fsq_dim,).
        fsq_key_bias = "tokenizer.quantizer.project_out.bias"
        fsq_key_weight = "tokenizer.quantizer.project_in.weight"
        if fsq_key_bias in sd:
            fsq_dim = int(sd[fsq_key_bias].shape[0])   # always = fsq_dim
        elif fsq_key_weight in sd:
            # Fallback: project_in weight reversed shape[0] gives fsq_dim
            # (GGUF reverses (codebook_dim, fsq_dim) → (fsq_dim, codebook_dim))
            fsq_dim = int(sd[fsq_key_weight].shape[0])


        logging.info(
            f"AceStep-GGUF DiT: hidden_size={hidden_size} "
            f"num_layers={num_dit_layers} heads={num_heads}/{num_kv_heads} "
            f"intermediate={intermediate}"
        )

        # Build unet_config that matches what ACEStep15.unet_config declares
        unet_config = {
            "audio_model":       "ace1.5",
            # These extra keys get passed to AceStepConditionGenerationModel:
            "hidden_size":       hidden_size,
            "num_dit_layers":    num_dit_layers,
            "num_heads":         num_heads,
            "num_kv_heads":      num_kv_heads,
            "head_dim":          head_dim,
            "intermediate_size": intermediate,
            "patch_size":        patch_size,
            "fsq_dim":           fsq_dim,
        }

        model_config = None
        if hidden_size == 2560:
            # Load our custom hybrid XL model config that fixes the 2048/2560 split
            from .xl_hybrid_model import ACEStep15_XL_Config
            model_config = ACEStep15_XL_Config(unet_config)
        else:
            # For 2048 models (1.7B etc.), standard detection is fine
            for cfg_cls in _sm.models:
                if cfg_cls.unet_config.get("audio_model") == "ace1.5":
                    model_config = cfg_cls(unet_config)
                    break

        if model_config is None:
            raise RuntimeError(
                "Could not find ACEStep15 in supported_models. "
                "Ensure your ComfyUI is up to date."
            )

        # Set dtype and custom ops
        load_device    = _mm.get_torch_device()
        offload_device = _mm.unet_offload_device()
        unet_dtype = _mm.unet_dtype(
            supported_dtypes=model_config.supported_inference_dtypes
        )
        model_config.set_inference_dtype(unet_dtype, None)
        model_config.custom_operations = ops

        # Instantiate model with correct config
        model = model_config.get_model(sd, "")
        # IMPORTANT: must use GGUFModelPatcher, not the standard ModelPatcher.
        # Standard ModelPatcher calls comfy.lora.calculate_weight on the raw GGMLTensor
        # whose .shape reports GGML convention (in, out) — the transpose of PyTorch (out, in).
        # LoKr/LoRA calculations that reference .shape then compute Kronecker products in the
        # wrong orientation, producing wildly mismatched weight shapes at runtime.
        # GGUFModelPatcher stores patches deferred on the GGMLTensor and applies them during
        # the GGML-aware forward pass after correct dequantization.
        model_patcher = GGUFModelPatcher(
            model, load_device=load_device, offload_device=offload_device
        )
        # Load weights
        model.load_model_weights(sd, "", assign=model_patcher.is_dynamic())
        model_patcher.patch_on_device = False

        logging.info(
            f"AceStep-GGUF DiT: loaded — {meta.get('general.name', '?')} | "
            f"hidden={hidden_size} | layers={num_dit_layers}"
        )
        return (model_patcher,)



# ===========================================================================
# Node 2: AceStepDualCLIPLoaderGGUF
# ===========================================================================
class AceStepDualCLIPLoaderGGUF:
    """
    Load the ACE-Step 1.5 dual text encoders from GGUF files.

    Mirrors the existing ScromfyAceStepModelLoader structure:

      text_encoder  = Qwen3-Embedding-0.6B GGUF
                      (lyrics / caption encoder — architecture: acestep-text-enc)

      audio_lm      = acestep-5Hz-lm-1.7B or 4B GGUF
                      (audio-code language model — architecture: acestep-lm)
                      1.7B → qwen3_2b mode  |  4B → qwen3_4b mode

    Both are packaged into a single CLIP output through CLIPType.ACE exactly
    as the safetensors loader does it — compatible with TextEncodeAceStepAudio1.5.
    """

    @classmethod
    def INPUT_TYPES(cls):
        enc_files = _list_gguf_files("acestep_enc_gguf", "text_encoders", "clip")
        lm_files  = _list_gguf_files("acestep_lm_gguf",  "text_encoders", "clip")

        if not enc_files: enc_files = ["[No .gguf found — place Qwen3-Embedding-0.6B in text_encoders/]"]
        if not lm_files:  lm_files  = ["[No .gguf found — place acestep-5Hz-lm-*.gguf in text_encoders/]"]

        return {
            "required": {
                "text_encoder": (enc_files, {
                    "tooltip": (
                        "Qwen3-Embedding-0.6B GGUF for lyrics/caption encoding.\n"
                        "Architecture: acestep-text-enc\n"
                        "File: Qwen3-Embedding-0.6B-Q8_0.gguf"
                    )
                }),
                "audio_lm": (lm_files, {
                    "tooltip": (
                        "ACE-Step 5Hz audio-code LM GGUF.\n"
                        "Architecture: acestep-lm\n"
                        "1.7B (recommended) → acestep-5Hz-lm-1.7B-Q8_0.gguf\n"
                        "4B (higher quality) → acestep-5Hz-lm-4B-Q8_0.gguf"
                    )
                }),
            },
        }

    RETURN_TYPES  = ("CLIP",)
    RETURN_NAMES  = ("clip",)
    FUNCTION      = "load_dual_clip"
    CATEGORY      = "ACE-Step/GGUF Loaders"
    TITLE         = "AceStep Dual CLIP Loader (GGUF)"

    def load_dual_clip(self, text_encoder: str, audio_lm: str):
        _check_gguf()
        # Resolve paths
        enc_path = _resolve_path(text_encoder, "acestep_enc_gguf", "text_encoders", "clip")
        lm_path  = _resolve_path(audio_lm,     "acestep_lm_gguf",  "text_encoders", "clip")

        # Load state dicts
        enc_sd, enc_meta = load_text_encoder_gguf(enc_path)   # adds model. prefix
        lm_sd,  lm_meta  = load_lm_gguf(lm_path)              # already has model. prefix

        # Detect LM variant for log
        detect_key = "model.layers.0.post_attention_layernorm.weight"
        lm_hidden  = int(lm_sd[detect_key].shape[0]) if detect_key in lm_sd else 0
        lm_variant = {2048: "qwen3_2b (1.7B)", 2560: "qwen3_4b (4B)"}.get(lm_hidden, f"unknown ({lm_hidden})")

        # ComfyUI's CLIPType.ACE with 2 state_dicts routes through:
        #   detect_te_model(enc_sd) → QWEN3_06B  (hidden=1024)
        #   detect_te_model(lm_sd)  → QWEN3_2B or QWEN3_4B
        # → ace15.te(lm_model="qwen3_2b" or "qwen3_4b")
        # → ACE15TEModel with qwen3_06b + lm sub-model
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=comfy.sd.CLIPType.ACE,
            state_dicts=[enc_sd, lm_sd],
            model_options={
                "custom_operations": GGMLOps,
                "initial_device": comfy.model_management.text_encoder_offload_device(),
            },
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        # Wrap patcher for quantized inference
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)

        logging.info(
            f"AceStep-GGUF Dual CLIP: "
            f"enc={enc_meta.get('general.name', text_encoder)} | "
            f"lm={lm_meta.get('general.name', audio_lm)} ({lm_variant})"
        )
        return (clip,)


# ===========================================================================
# Node 3: AceStepVAELoaderGGUF
# ===========================================================================
class AceStepVAELoaderGGUF:
    """
    Load the ACE-Step 1.5 audio VAE (AutoencoderOobleck) from a GGUF file.
    Weights are immediately dequantized to BF16 (VAE is ~320 MB so this is fine).
    """

    @classmethod
    def INPUT_TYPES(cls):
        vae_files = _list_gguf_files("acestep_vae_gguf", "vae")
        if not vae_files:
            vae_files = ["[No .gguf found — place vae-BF16.gguf in models/vae/]"]
        return {
            "required": {
                "vae_name": (vae_files, {
                    "tooltip": "ACE-Step VAE GGUF file (vae-BF16.gguf). Place in models/vae/"
                }),
            },
        }

    RETURN_TYPES  = ("VAE",)
    RETURN_NAMES  = ("vae",)
    FUNCTION      = "load_vae"
    CATEGORY      = "ACE-Step/GGUF Loaders"
    TITLE         = "AceStep VAE Loader (GGUF)"

    def load_vae(self, vae_name: str):
        _check_gguf()
        path = _resolve_path(vae_name, "acestep_vae_gguf", "vae")
        sd   = load_vae_gguf(path)
        if is_oobleck_vae_gguf(sd):
            # AudioOobleckVAE: comfy.sd.VAE has no detection for this format;
            # load directly with key remapping
            vae = load_oobleck_vae_from_gguf(sd)
        else:
            # Let ComfyUI detect other VAE formats normally
            vae = comfy.sd.VAE(sd=sd)
        logging.info(f"AceStep-GGUF VAE: {len(sd)} tensors from {os.path.basename(path)}")
        return (vae,)


# ===========================================================================
# Node 4: AceStepModelLoaderGGUF  (all-in-one convenience)
# ===========================================================================
class AceStepModelLoaderGGUF:
    """
    All-in-One loader: DiT + Dual CLIP (0.6B encoder + 5Hz LM) + VAE.

    Equivalent to wiring:
      AceStepDiTLoaderGGUF
      AceStepDualCLIPLoaderGGUF    ← note: text_encoder + audio_lm (not two encoders)
      AceStepVAELoaderGGUF

    Output (MODEL, CLIP, VAE) is compatible with all existing ACE-Step sampler nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        dit_files = _list_gguf_files("acestep_dit_gguf", "unet_gguf", "diffusion_models")
        enc_files = _list_gguf_files("acestep_enc_gguf", "text_encoders", "clip")
        lm_files  = _list_gguf_files("acestep_lm_gguf",  "text_encoders", "clip")
        vae_files = _list_gguf_files("acestep_vae_gguf", "vae")

        if not dit_files: dit_files = ["[No DiT .gguf — place in diffusion_models/]"]
        if not enc_files: enc_files = ["[No enc .gguf — place Qwen3-Embedding in text_encoders/]"]
        if not lm_files:  lm_files  = ["[No LM .gguf — place 5Hz-lm in text_encoders/]"]
        if not vae_files: vae_files = ["[No VAE .gguf — place vae-BF16.gguf in vae/]"]

        return {
            "required": {
                "dit_name": (dit_files, {
                    "tooltip": "ACE-Step DiT GGUF (e.g. acestep-v15-xl-turbo-Q8_0.gguf)"
                }),
                "text_encoder": (enc_files, {
                    "tooltip": "Qwen3-Embedding-0.6B GGUF (lyrics/caption encoder)"
                }),
                "audio_lm": (lm_files, {
                    "tooltip": "5Hz LM GGUF: 1.7B or 4B (audio-code generator)"
                }),
                "vae_name": (vae_files, {
                    "tooltip": "ACE-Step VAE GGUF (vae-BF16.gguf)"
                }),
                "dequant_dtype": (
                    ["default", "float32", "float16", "bfloat16"],
                    {"default": "default",
                     "tooltip": "Dequant precision for DiT forward pass."}
                ),
            },
            "optional": {
                "lora_stack": ("ACESTEP_LORA",),
            },
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES  = ("model", "clip", "vae")
    FUNCTION      = "load_all"
    CATEGORY      = "ACE-Step/GGUF Loaders"
    TITLE         = "AceStep All-in-One Loader (GGUF)"

    def load_all(
        self,
        dit_name: str,
        text_encoder: str,
        audio_lm: str,
        vae_name: str,
        dequant_dtype: str = "default",
        lora_stack=None,
    ):
        (model,) = AceStepDiTLoaderGGUF().load_dit(dit_name, dequant_dtype)
        (clip,)  = AceStepDualCLIPLoaderGGUF().load_dual_clip(text_encoder, audio_lm)
        (vae,)   = AceStepVAELoaderGGUF().load_vae(vae_name)

        # Apply LoRA stack if provided (same as existing scromfyUI-AceStep loader)
        if lora_stack is not None:
            for lora_spec in lora_stack:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_spec["lora_name"])
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, lora_data,
                    lora_spec["strength_model"], None
                )

        return (model, clip, vae)


# ===========================================================================
# Registration
# ===========================================================================
NODE_CLASS_MAPPINGS = {
    "AceStepDiTLoaderGGUF":       AceStepDiTLoaderGGUF,
    "AceStepDualCLIPLoaderGGUF":  AceStepDualCLIPLoaderGGUF,
    "AceStepVAELoaderGGUF":       AceStepVAELoaderGGUF,
    "AceStepModelLoaderGGUF":     AceStepModelLoaderGGUF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepDiTLoaderGGUF":       "AceStep DiT Loader (GGUF)",
    "AceStepDualCLIPLoaderGGUF":  "AceStep Dual CLIP Loader (GGUF)",
    "AceStepVAELoaderGGUF":       "AceStep VAE Loader (GGUF)",
    "AceStepModelLoaderGGUF":     "AceStep All-in-One Loader (GGUF)",
}
