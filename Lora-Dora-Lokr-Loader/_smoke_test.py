"""Smoke test for all three helper modules."""
import sys
sys.path.insert(0, r"E:\AI\Lora-Dora-Lokr-node")

from adapter_utils import detect_adapter_type, get_base_names
from key_transforms import normalize_acesteop_lycoris_keys
from layer_scale import classify_key, scale_patches
import safetensors.torch as sf

LORA_DIR = r"E:\AI\Lora-Dora-Lokr-node"

print("=== adapter_utils ===")
for fname in ["lokr_weights.safetensors", "lokr_weights-1.safetensors"]:
    sd = sf.load_file(f"{LORA_DIR}/{fname}")
    t = detect_adapter_type(sd)
    bases = get_base_names(sd)
    print(f"  {fname}: type={t!r}, bases={len(bases)}")
assert t == "lokr", f"Expected lokr, got {t!r}"

print("\n=== key_transforms ===")
sd = sf.load_file(f"{LORA_DIR}/lokr_weights.safetensors")
orig_keys = set(sd.keys())
n = normalize_acesteop_lycoris_keys(sd, "lokr", verbose=False)
print(f"  Keys renamed: {n}")
new_keys = set(sd.keys())
assert not any(k.startswith("lycoris_") for k in new_keys), "lycoris_ keys still present!"
assert all(k.startswith("diffusion_model.") for k in new_keys), "Unexpected prefix found"
sample = sorted(new_keys)[:5]
print("  First 5 keys after rename:")
for k in sample:
    print(f"    {k}")

# no-op for lora type
sd2 = sf.load_file(f"{LORA_DIR}/lokr_weights.safetensors")
n2 = normalize_acesteop_lycoris_keys(sd2, "lora", verbose=False)
assert n2 == 0, f"Expected 0 renames for lora, got {n2}"
print(f"  No-op test (lora type): renamed={n2} (correct)")

print("\n=== layer_scale classify_key ===")
tests = [
    ("diffusion_model.layers.0.cross_attn_k_proj.weight", "cross_attn"),
    ("diffusion_model.layers.0.cross_attn_o_proj.weight", "cross_attn"),
    ("diffusion_model.layers.0.self_attn_q_proj.weight",  "self_attn"),
    ("diffusion_model.layers.0.self_attn_k_proj.weight",  "self_attn"),
    ("diffusion_model.layers.0.mlp_gate_proj.weight",     "ffn"),
    ("diffusion_model.layers.0.mlp_up_proj.weight",       "ffn"),
    ("diffusion_model.layers.0.mlp_down_proj.weight",     "ffn"),
    ("diffusion_model.condition_embedder.weight",          "other"),
    ("diffusion_model.proj_in_1.weight",                   "other"),
    ("diffusion_model.time_embed_linear_1.weight",         "other"),
]
all_ok = True
for key, expected in tests:
    got = classify_key(key)
    ok = got == expected
    all_ok = all_ok and ok
    print(f"  {'OK' if ok else 'FAIL'} classify({key.rsplit('.', 2)[-2]!r}) = {got!r} (expected {expected!r})")
assert all_ok, "classify_key failures!"

print("\n=== layer_scale scale_patches ===")
dummy_patches = {
    "diffusion_model.layers.0.cross_attn_k_proj.weight": (1.0, "tensor_a"),
    "diffusion_model.layers.0.self_attn_q_proj.weight":  (1.0, "tensor_b"),
    "diffusion_model.layers.0.mlp_gate_proj.weight":     (1.0, "tensor_c"),
    "diffusion_model.condition_embedder.weight":          (1.0, "tensor_d"),
}
scaled = scale_patches(dummy_patches,
    self_attn_scale=0.5,
    cross_attn_scale=1.5,
    ffn_scale=2.0,
    other_scale=1.0,
    verbose=True,
)
assert scaled["diffusion_model.layers.0.cross_attn_k_proj.weight"][0] == 1.5
assert scaled["diffusion_model.layers.0.self_attn_q_proj.weight"][0] == 0.5
assert scaled["diffusion_model.layers.0.mlp_gate_proj.weight"][0] == 2.0
assert scaled["diffusion_model.condition_embedder.weight"][0] == 1.0
print("  scale_patches assertions passed")

print("\nAll smoke tests PASSED.")
