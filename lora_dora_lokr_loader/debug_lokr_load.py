import sys
from unittest.mock import MagicMock
import os
import safetensors.torch as sf

# Mock ComfyUI environment
sys.modules["comfy"] = MagicMock()
sys.modules["comfy.lora"] = MagicMock()
sys.modules["comfy.lora_convert"] = MagicMock()
sys.modules["comfy.model_management"] = MagicMock()
sys.modules["comfy.utils"] = MagicMock()
sys.modules["folder_paths"] = MagicMock()
sys.modules["server"] = MagicMock()

# Ensure we can import modules from the current directory
sys.path.append(os.getcwd())

import adapter_utils
import key_transforms

# Manually populate sys.modules for relative imports
# (or just import the module as a top-level)
sys.modules["__main__"] = MagicMock() # not really needed
# To fix: from .adapter_utils import ... 
# We need to simulate that we are in a package.
package_name = "test_package"
sys.modules[package_name] = MagicMock()
sys.modules[f"{package_name}.adapter_utils"] = adapter_utils
sys.modules[f"{package_name}.key_transforms"] = key_transforms

import nodes
# Manually inject the imported names into nodes if relative import failed
# But actually, I'll just patch nodes.py temporarily if I have to, or just 
# run the functions directly in this script.

# Load LoKr
LORA_PATH = r"E:\AI\Lora-Dora-Lokr-node\lokr_weights.safetensors"
lora_sd = sf.load_file(LORA_PATH)

# Mock Model State Dict Keys
model_keys = [
    "diffusion_model.condition_embedder.weight",
    "diffusion_model.layers.0.cross_attn_k_proj.weight",
    "diffusion_model.layers.0.cross_attn_q_proj.weight",
    "diffusion_model.layers.0.self_attn_q_proj.weight",
]
model_sd_keys = set(model_keys)
model_sd_list = list(model_keys)

print(f"Detecting adapter type...")
adapter_type = adapter_utils.detect_adapter_type(lora_sd)
print(f"Detected: {adapter_type}")

print(f"Normalizing keys...")
n = key_transforms.normalize_acesteop_lycoris_keys(lora_sd, adapter_type, verbose=True)
print(f"Renamed {n} keys.")

print(f"Extracting bases...")
bases = nodes._extract_lora_bases(lora_sd.keys())
print(f"Extracted {len(bases)} bases.")

print(f"Building key map...")
key_map = {}
added, unresolved = nodes._extend_key_map_with_dynamic_matches(
    key_map, bases, model_sd_keys, model_sd_list, None, None, verbose=True
)
print(f"Added {added} mappings, {len(unresolved)} unresolved.")

if added > 0:
    for base, key in sorted(list(key_map.items()))[:5]:
        print(f"Mapping: {base} -> {key}")
else:
    print("FAILED TO MAP ANY KEYS")
    print("Example bases in LoRA sd after normalization:")
    for b in sorted(list(bases))[:5]:
        print(f"  {b}")
    print("Example model keys:")
    for k in sorted(list(model_sd_keys))[:5]:
        print(f"  {k}")
