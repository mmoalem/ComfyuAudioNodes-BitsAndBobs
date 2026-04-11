from key_transforms import _acesteop_lycoris_base_to_diffusion as f
tests = [
    ("lycoris_layers_0_cross_attn_k_proj", "diffusion_model.layers.0.cross_attn.k_proj"),
    ("lycoris_layers_0_cross_attn_q_proj", "diffusion_model.layers.0.cross_attn.q_proj"),
    ("lycoris_layers_0_cross_attn_v_proj", "diffusion_model.layers.0.cross_attn.v_proj"),
    ("lycoris_layers_0_cross_attn_o_proj", "diffusion_model.layers.0.cross_attn.o_proj"),
    ("lycoris_layers_0_self_attn_q_proj",  "diffusion_model.layers.0.self_attn.q_proj"),
    ("lycoris_layers_0_mlp_gate_proj",     "diffusion_model.layers.0.mlp.gate_proj"),
    ("lycoris_layers_0_mlp_up_proj",       "diffusion_model.layers.0.mlp.up_proj"),
    ("lycoris_layers_0_mlp_down_proj",     "diffusion_model.layers.0.mlp.down_proj"),
    ("lycoris_layers_0_cross_attn_norm",   "diffusion_model.layers.0.cross_attn_norm"),
    ("lycoris_condition_embedder",         "diffusion_model.condition_embedder"),
    ("lycoris_proj_in_1",                  "diffusion_model.proj_in_1"),
]
all_ok = True
for base, expected in tests:
    result = f(base)
    ok = result == expected
    all_ok = all_ok and ok
    status = "OK" if ok else "FAIL"
    suffix = "" if ok else f" (expected {expected})"
    print(f"{status} {base} -> {result}{suffix}")
print()
print("ALL PASSED" if all_ok else "FAILURES DETECTED")
