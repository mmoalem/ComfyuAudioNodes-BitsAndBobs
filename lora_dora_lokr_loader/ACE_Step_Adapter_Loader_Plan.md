# ACE-Step Universal Adapter Loader — Vibe Coding Plan

## Project Goal

Fork `xmarre/ComfyUI-DoRA-Dynamic-LoRA-Loader` and extend it into a universal
adapter loader that correctly handles **LoRA, DoRA, LoKr, and LoHa** files,
with **per-layer-category strength scaling** (self-attention, cross-attention,
FFN/MLP, other), targeted at **ACE-Step audio generation workflows** while
remaining usable with other models.

---

## Repository Setup

```
Fork: https://github.com/xmarre/ComfyUI-DoRA-Dynamic-LoRA-Loader
New name: ComfyUI-ACEStep-Universal-Adapter-Loader  (or your preferred name)
```

File structure to produce:

```
ComfyUI-ACEStep-Universal-Adapter-Loader/
├── __init__.py
├── nodes.py               ← main node class (fork + heavy edits)
├── adapter_utils.py       ← NEW: adapter type detection + math helpers
├── layer_scale.py         ← NEW: per-layer scaling logic
├── key_transforms.py      ← NEW: all key-remapping logic extracted here
├── web/
│   └── layer_scale_ui.js  ← NEW: optional JS panel (Phase 3)
├── pyproject.toml
└── README.md
```

---

## Phase 1 — Fork Cleanup and Adapter Detection

**Goal:** Strip out Flux-specific code that doesn't apply to ACE-Step, add a
robust adapter-type detector that runs before any other transform, and gate
all existing fixes to only fire on their correct adapter type.

### Step 1.1 — Create `adapter_utils.py`

Write a function `detect_adapter_type(state_dict: dict) -> str` that inspects
the keys of a loaded safetensors file and returns one of:

- `"lora"` — has `lora_up` / `lora_down` / `lora_A` / `lora_B` keys, no
  Kronecker or Hadamard keys
- `"dora"` — same as lora but also has `dora_scale` or `lora_magnitude_vector`
  keys
- `"lokr"` — has `lokr_w1` / `lokr_w2` / `lokr_w1_a` / `lokr_w1_b` /
  `lokr_w2_a` / `lokr_w2_b` keys
- `"loha"` — has `hada_w1_a` / `hada_w1_b` / `hada_w2_a` / `hada_w2_b` keys
- `"unknown"` — fallback

Detection logic: strip the base name prefix from any key and look at the
suffix. Sample 20 keys for speed rather than scanning all keys.

Also write a helper `get_base_names(state_dict: dict) -> set` that returns
the unique base names (everything before the last `.`-separated adapter suffix
like `.lokr_w1`, `.lora_up`, etc.).

### Step 1.2 — Create `key_transforms.py`

Move all key-remapping functions from `nodes.py` into this file, organised
by which adapter type they apply to. Each function must accept
`(state_dict, adapter_type, model_key_map)` and return a transformed
`state_dict`. Functions to extract:

- `normalize_dora_magnitude_keys(sd)` — renames `lora_magnitude_vector` →
  `dora_scale`. **Only runs when `adapter_type == "dora"`.**
- `normalize_diffusers_peft_keys(sd)` — renames `lora_A` / `lora_B` to
  `lora_down` / `lora_up`. **LoRA and DoRA only.**
- `normalize_onetrainer_keys(sd)` — renames
  `transformer.time_guidance_embed.*` → `transformer.time_text_embed.*`.
  **Flux-targeted; skip for ACE-Step by checking model class.**
- `normalize_acesteop_lycoris_keys(sd)` — **NEW**: renames `lycoris_*` prefixed
  keys to the format ComfyUI's key mapper expects for ACE-Step. This replicates
  the fix from ComfyUI PR #12665 so the node works even on older ComfyUI
  versions. Pattern: `lycoris_layers_N_cross_attn_k_proj` →
  `diffusion_model.layers.N.cross_attn.k_proj` (inspect actual model
  `state_dict()` keys to confirm the exact target names).
- `fix_direction_matrix_orientation(sd, key_map)` — the output-axis swap fix.
  **DoRA and LoRA only, skip for LoKr/LoHa.**

**Critical rule written as a comment at the top of this file:**

```python
# Every transform function MUST be gated on adapter_type.
# Never run a LoRA/DoRA-specific fix on a LoKr or LoHa file.
# The adapter_type is determined ONCE at load time before any transform runs.
```

### Step 1.3 — Strip Flux-only code paths from `nodes.py`

Remove or gate behind a `_is_flux_model()` check:

- The `broadcast_modulations` / OneTrainer broadcast logic
- The ZiT/Lumina2 QKV fusion (keep the toggle but default it OFF and add a
  warning that it is not for ACE-Step)
- The `adaLN swap_scale_shift` fix (Flux2 only)
- The Flux2 `dora_slice_fix` for offset patches

These should not be deleted in case the user uses the node with Flux, but they
must not run on ACE-Step models. Gate them with:

```python
if self._is_flux_model(model):
    # run Flux-specific transforms
```

`_is_flux_model` checks `model.__class__.__name__` for `"Flux"`.

---

## Phase 2 — LoKr and LoHa Loading

**Goal:** Make LoKr and LoHa files actually apply their math correctly, not
just map keys.

### Step 2.1 — Understand what ComfyUI already handles

ComfyUI's `comfy.weight_adapter` module contains:
- `lora.py` — handles `lora_up`/`lora_down` and `dora_scale`
- `lokr.py` — handles `lokr_w1`/`lokr_w2` and the Kronecker product math
- `loha.py` — handles `hada_w1_a` etc. and the Hadamard product math

**Check your ComfyUI version.** If it is v0.16.0 or newer, ACE-Step LoKr key
aliases are already in core ComfyUI. The node still needs to handle the adapter
math correctly, but the key mapping is done.

Test: load one of your ACE-Step LoKr files with `Verbose` and `Log Unloaded
Keys` enabled. If zero keys are unloaded, the key mapping is working and the
issue is purely in strength / math correctness.

### Step 2.2 — Fix the monkey-patch scope in `nodes.py`

The xmarre node monkey-patches `comfy.weight_adapter.base.weight_decompose`
at import time. This patch applies DoRA fp32 normalization logic. **This must
not affect LoKr or LoHa adapters.**

Modify the monkey-patched `weight_decompose` to check whether `dora_scale` is
present in the patch before applying DoRA math:

```python
def patched_weight_decompose(weights, lora_diff, function, dora_scale, ...):
    if dora_scale is None:
        # Not a DoRA — call original implementation unchanged
        return original_weight_decompose(weights, lora_diff, function, None, ...)
    # DoRA path: apply fp32 normalization as before
    ...
```

This makes the patch a true no-op for LoKr and LoHa, which never have
`dora_scale`.

### Step 2.3 — Verify LoKr math path end-to-end

After key normalization, `comfy.lora.load_lora()` should correctly identify
`lokr_*` keys and route them through `comfy.weight_adapter.lokr`. To verify:

- Enable `Verbose` logging in the node
- Load an ACE-Step LoKr
- Confirm log shows `applied(model)=N` where N > 0
- Confirm no shape errors or "lora key not loaded" for `lokr_*` keys
- Compare audio output with and without the LoKr at strength 1.0 — should
  sound noticeably different

### Step 2.4 — LoHa key normalization

LoHa files for ACE-Step may also use the `lycoris_` prefix. Add
`normalize_acesteop_lycoris_keys` (from Step 1.2) to also handle LoHa key
names — the prefix normalization is the same regardless of whether the adapter
type is LoKr or LoHa.

---

## Phase 3 — Per-Layer Category Scaling

**Goal:** Add four strength multipliers — self-attention, cross-attention,
FFN/MLP, and other — that scale the effective strength of patches before they
are handed to `model.add_patches()`.

### Step 3.1 — Create `layer_scale.py`

#### ACE-Step key classification

Based on the actual ACE-Step model key structure (confirmed from the LoKr
issue debug logs):

```python
ACESTEOP_LAYER_PATTERNS = {
    "cross_attn": [
        "cross_attn",
        "cross_attention",
        "attn2",
    ],
    "self_attn": [
        "self_attn",
        "self_attention",
        "attn1",
        # bare .attn. not followed by a digit or 'cross' — check carefully
    ],
    "ffn": [
        ".ff.",
        ".ffn.",
        ".mlp.",
        "fc1",
        "fc2",
        "linear_in",
        "linear_out",
        "net.0",
        "net.2",
    ],
}

def classify_key(key: str) -> str:
    """
    Returns 'cross_attn', 'self_attn', 'ffn', or 'other'.
    Classification is done on the full patch key (model weight key),
    not the LoRA file key.
    Order matters: cross_attn check must come before self_attn check
    because cross-attention keys often contain 'attn'.
    """
    k = key.lower()
    for label, patterns in ACESTEOP_LAYER_PATTERNS.items():
        if any(p in k for p in patterns):
            return label
    return "other"
```

**Important:** classification runs on the **mapped model key** (the destination
weight name), not the raw LoRA file key. The model key is what's in the patches
dict returned by `load_lora()`.

#### Scaling function

```python
def scale_patches(
    patches: dict,
    self_attn_scale: float,
    cross_attn_scale: float,
    ffn_scale: float,
    other_scale: float,
    verbose: bool = False,
) -> dict:
    """
    Walk every key in patches, classify it, and multiply the patch tensor(s)
    by the corresponding scale factor.

    patches dict structure (from ComfyUI's load_lora):
      key -> (strength, (adapter_tensors...))

    We do NOT modify strength directly — we bake the multiplier into the
    first tensor of the adapter tuple (lora_up / lokr_w1 / hada_w1_a)
    to keep the application path unchanged.
    """
    scale_map = {
        "cross_attn": cross_attn_scale,
        "self_attn": self_attn_scale,
        "ffn": ffn_scale,
        "other": other_scale,
    }
    scaled = {}
    counts = {"cross_attn": 0, "self_attn": 0, "ffn": 0, "other": 0}

    for key, patch in patches.items():
        category = classify_key(key)
        factor = scale_map[category]
        counts[category] += 1

        if factor == 1.0:
            scaled[key] = patch
        else:
            # patch is (strength, adapter_tuple)
            # multiply strength by factor
            strength, adapter = patch[0], patch[1:]
            scaled[key] = (strength * factor,) + adapter

    if verbose:
        print(f"[LayerScale] {counts}")

    return scaled
```

**Note on when to apply scaling:** scaling runs AFTER `load_lora()` builds the
patches dict but BEFORE `model.add_patches()`. This is the same slot used by
auto-strength. If auto-strength is also enabled, apply auto-strength ratios
first, then apply layer-category scaling on top. The two are independent.

### Step 3.2 — Add UI parameters to the node

In `nodes.py`, inside `INPUT_TYPES`, add a new group of global options:

```python
# Per-layer category scaling
"layer_scale_enabled": ("BOOLEAN", {"default": False}),
"self_attn_scale": ("FLOAT", {
    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
    "tooltip": "Strength multiplier for self-attention layers"
}),
"cross_attn_scale": ("FLOAT", {
    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
    "tooltip": "Strength multiplier for cross-attention layers"
}),
"ffn_scale": ("FLOAT", {
    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
    "tooltip": "Strength multiplier for FFN/MLP layers"
}),
"other_scale": ("FLOAT", {
    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
    "tooltip": "Strength multiplier for projection and other layers"
}),
```

In the `forward` / `load` method, after `load_lora()` returns patches:

```python
if layer_scale_enabled:
    patches = scale_patches(
        patches,
        self_attn_scale=self_attn_scale,
        cross_attn_scale=cross_attn_scale,
        ffn_scale=ffn_scale,
        other_scale=other_scale,
        verbose=verbose,
    )
```

### Step 3.3 — Verbose logging for layer scaling

When `verbose=True` and `layer_scale_enabled=True`, print a breakdown after
each LoRA is processed:

```
[LayerScale] my_lora.safetensors
  self_attn : 47 patches × 0.80
  cross_attn: 38 patches × 1.20
  ffn       : 62 patches × 1.00
  other     : 15 patches × 1.00
```

This is important because it lets the user verify the classification is
correct for their specific LoKr/LoHa files.

---

## Phase 4 — Auto-Strength Fix for LoKr and LoHa

**Goal:** Make auto-strength either work correctly for LoKr/LoHa or cleanly
skip them rather than silently computing garbage ratios.

### Step 4.1 — Gate auto-strength on adapter type

In the auto-strength analysis loop, add an early exit for non-LoRA/DoRA
adapters:

```python
if adapter_type in ("lokr", "loha"):
    if verbose:
        print(f"[AutoStrength] Skipping {lora_name}: "
              f"auto-strength not supported for {adapter_type}")
    # Load with uniform strength — do not attempt magnitude analysis
    patches = raw_patches
```

This is safer than attempting to compute `||kron(w1,w2)||_F` incorrectly.

### Step 4.2 — (Optional, advanced) Add LoKr magnitude estimation

If you want auto-strength to work for LoKr in a later iteration, the correct
magnitude estimator is:

```python
import torch
def lokr_effective_magnitude(w1, w2):
    """
    Approximates the Frobenius norm of kron(w1, w2).
    ||kron(A, B)||_F == ||A||_F * ||B||_F
    """
    return torch.linalg.norm(w1.float()) * torch.linalg.norm(w2.float())
```

This is mathematically exact for the simple two-factor form. For the mixed
form (`kron(w1, w2_a @ w2_b)`), compute `||w2_a @ w2_b||_F` first, then
multiply by `||w1||_F`. Mark this as a TODO for now.

---

## Phase 5 — Node UI Polish

### Parameters to expose in the final node

**Per-LoRA row (existing):**
- Enabled toggle
- LoRA file dropdown
- Weight (applies to model; clip is usually not relevant for ACE-Step)

**Adapter info (read-only display, new):**
- Detected type: `LoRA / DoRA / LoKr / LoHa / unknown`
- Bases loaded / total bases

**Layer scaling section (new, collapsible):**
- Layer scale enabled toggle
- Self-attention scale (float, default 1.0)
- Cross-attention scale (float, default 1.0)
- FFN / MLP scale (float, default 1.0)
- Other scale (float, default 1.0)

**DoRA / compatibility section (existing, keep as-is):**
- DoRA fp32 normalization (monkey-patch toggle, default ON)
- DoRA adaLN swap fix (default ON, Flux only)
- DoRA slice fix (default ON, Flux only)
- ZiT/Lumina2 QKV fix (default OFF for ACE-Step)

**Auto-strength section (existing, keep as-is):**
- Auto-strength enabled (default OFF for safety)
- Analysis device (auto / cpu / gpu)
- Ratio floor / ceiling

**Debug section:**
- Verbose toggle
- Log unloaded keys toggle
- Show layer scale breakdown (new, only visible when layer scale enabled)

---

## Phase 6 — Testing Checklist

Run each test case and verify with verbose logging before shipping.

### LoRA tests
- [ ] Plain LoRA (SimpleTuner format) loads, patches applied > 0, audio changes
- [ ] LoRA + DoRA scale fix does not change LoRA behavior (no `dora_scale` present)
- [ ] Layer scaling changes audio when self_attn_scale ≠ 1.0

### DoRA tests
- [ ] DoRA with `dora_scale` keys — all `dora_scale` keys loaded (not unloaded)
- [ ] DoRA with `lora_magnitude_vector` keys — renamed correctly, all applied
- [ ] fp32 normalization patch fires only for DoRA (verify via verbose)
- [ ] Layer scaling works for DoRA

### LoKr tests
- [ ] ACE-Step LoKr with `lycoris_` prefix — all `lokr_*` keys loaded
- [ ] Audio differs from baseline with LoKr at strength 1.0
- [ ] DoRA monkey-patch does NOT fire for LoKr (verify `dora_scale` is None)
- [ ] Auto-strength is skipped for LoKr (verify skip message in verbose log)
- [ ] Layer scaling works for LoKr

### LoHa tests
- [ ] LoHa with `lycoris_` prefix (if applicable) — all `hada_*` keys loaded
- [ ] Audio differs from baseline
- [ ] Layer scaling works for LoHa

### Stacking tests
- [ ] LoRA + LoKr stacked together — both apply, no interference
- [ ] DoRA + LoHa stacked — both apply
- [ ] Layer scaling with mixed stack — each adapter scaled independently

### Regression tests
- [ ] Flux LoRA still works (if tested)
- [ ] Flux DoRA still works with adaLN fix ON

---

## Implementation Order

Work through the phases in this order — each phase is independently testable:

1. **Phase 1** — adapter detection + key_transforms extraction. Tests: load
   each adapter type with verbose, confirm correct type detected, no
   unloaded keys.

2. **Phase 2** — LoKr/LoHa math path verification. Tests: audio changes
   with LoKr. Monkey-patch scope fix.

3. **Phase 4** — auto-strength gating for LoKr/LoHa. This is a safety fix
   that should land before the node is used in production.

4. **Phase 3** — per-layer scaling. Tests: verbose breakdown shows correct
   counts per category, audio responds to scale changes.

5. **Phase 5** — UI polish. Tests: all parameters appear in ComfyUI node,
   descriptions are clear.

6. **Phase 6** — full test checklist pass.

---

## Key Files to Read Before Starting

Before writing any code, read these files from the fork:

- `nodes.py` — understand the full load pipeline: load → key transforms →
  key map → auto-strength → `load_lora()` → `add_patches()`
- `comfy/lora.py` in your ComfyUI install — understand `model_lora_keys_unet`,
  `load_lora`, and the key map format
- `comfy/weight_adapter/base.py` — understand `weight_decompose` and where
  the monkey-patch inserts
- `comfy/weight_adapter/lokr.py` — understand how LoKr math is already
  implemented in ComfyUI
- `comfy/weight_adapter/loha.py` — same for LoHa

---

## Critical Rules for the Coding Session

Write these as comments at the top of `nodes.py`:

```python
# RULE 1: adapter_type is detected ONCE per file, before any transform.
#          Every subsequent function checks adapter_type before running.
#
# RULE 2: The DoRA monkey-patch on weight_decompose must be a no-op when
#          dora_scale is None. LoKr and LoHa never have dora_scale.
#
# RULE 3: Layer scaling bakes into patch strength (not tensor data) to keep
#          the downstream ComfyUI application path unchanged.
#
# RULE 4: Flux-specific transforms (adaLN, ZiT QKV, OneTrainer broadcast)
#          only run when _is_flux_model() returns True.
#
# RULE 5: Auto-strength skips LoKr and LoHa rather than computing wrong ratios.
```

---

## Prompt Template for Vibe Coding Sessions

Use this as your opening prompt each session:

```
We are building ComfyUI-ACEStep-Universal-Adapter-Loader, a fork of
xmarre/ComfyUI-DoRA-Dynamic-LoRA-Loader extended to support LoRA, DoRA, LoKr,
and LoHa adapters with per-layer-category strength scaling.

The plan document is ACE_Step_Adapter_Loader_Plan.md. We are currently on
Phase [N], Step [N.N]: [step title].

The 5 critical rules are:
1. Adapter type detected once before any transform, all functions gate on it
2. DoRA monkey-patch is no-op when dora_scale is None
3. Layer scaling bakes into patch strength value, not tensor data
4. Flux transforms only run when _is_flux_model() is True
5. Auto-strength skips LoKr and LoHa

Today's task: [describe specific step]
```
