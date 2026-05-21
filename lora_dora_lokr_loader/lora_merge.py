"""
lora_merge.py — Conflict-aware multi-LoRA merge for ACEStep Universal Adapter Loader.

CRITICAL RULE: Only activates for adapter_type in ("lora", "dora").
               LoKr and LoHa are NEVER merged here — they always fall back to stack.

Algorithm (per base, two-pass):

  Pass 1 — Analysis (cheap, scalar stats only, no tensors held):
    - Materialize delta_i = (alpha_i/rank_i) * up_i @ down_i  per adapter i
    - Compute pairwise sign conflict ratio across adapters
    - Discard delta tensors, keep only scalar statistics per base

  Pass 2 — Merge (strategy chosen per base from Pass 1 stats):
    - 1 adapter only        → copy low-rank factors directly (no merge overhead)
    - conflict ≤ threshold  → weighted_average (adapters agree, just average)
    - conflict > threshold  → TIES: Trim + Elect Sign + Merge

  DARE sparsification (optional, applied before conflict analysis in Pass 1):
    - Zeros out a fraction of each delta's elements before merging
    - Conflict-aware variant: only sparsifies positions where signs differ

  SVD re-compression:
    - Merged full-rank delta is decomposed back to low-rank via truncated SVD
    - rank = sum of input ranks (information-preserving for weighted_avg; lossy for TIES)

  DoRA handling:
    - dora_scale tensors are magnitude-weighted averaged across adapters that have them
    - Produces a valid DoRA output that Comfy's weight_decompose can handle normally

References:
  - TIES: Yadav et al. 2023 (https://arxiv.org/abs/2306.01708)
  - DARE: Yu et al. 2023 (https://arxiv.org/abs/2311.03099)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

_LOG = logging.getLogger(__name__)

# ─── Default thresholds ───────────────────────────────────────────────────────
DEFAULT_CONFLICT_THRESHOLD = 0.25   # sign conflict ratio above which TIES is used
DEFAULT_TRIM_RATIO         = 0.20   # TIES: keep top 20% by magnitude per adapter
DEFAULT_DARE_RATE          = 0.0    # DARE sparsification rate (0 = disabled)

_EPS = 1e-8

# ─── Known LoRA low-rank pair suffixes ───────────────────────────────────────
_LORA_PAIR_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    (".lora_up.weight",              ".lora_down.weight"),
    (".lora_B.weight",               ".lora_A.weight"),
    (".lora_B.default.weight",       ".lora_A.default.weight"),
    ("_lora.up.weight",              "_lora.down.weight"),
    (".lora.up.weight",              ".lora.down.weight"),
    (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight"),
    (".lora_B",                      ".lora_A"),   # mochi-style (no .weight)
)

_DIRECT_DELTA_SUFFIXES = (".diff", ".diff_b", ".set_weight", ".reshape_weight")
_DORA_SCALE_SUFFIXES   = (".dora_scale", ".w_norm", ".b_norm")


# ──────────────────────────────────────────────────────────────────────────────
# Delta extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_delta(
    sd: Dict[str, Any],
    base: str,
    strength: float = 1.0,
) -> Optional[torch.Tensor]:
    """
    Materialise the full-rank weight delta for a given base as a 2-D float32 tensor.

    For low-rank pairs:   delta = (alpha/rank) * up @ down  * strength
    For direct tensors:   delta = tensor * strength
    Returns None when no representable delta exists for this base.
    """
    for up_suf, down_suf in _LORA_PAIR_SUFFIXES:
        up   = sd.get(base + up_suf)
        down = sd.get(base + down_suf)
        if up is None or down is None:
            continue
        if not isinstance(up, torch.Tensor) or not isinstance(down, torch.Tensor):
            continue
        if up.ndim < 2 or down.ndim < 2:
            continue

        up_mat   = up.reshape(int(up.shape[0]), -1).float()
        down_mat = down.reshape(int(down.shape[0]), -1).float()

        if int(up_mat.shape[1]) != int(down_mat.shape[0]):
            continue  # shape mismatch — skip this pair

        rank = int(down_mat.shape[0])
        alpha_v = sd.get(base + ".alpha")
        if alpha_v is not None:
            try:
                alpha = float(alpha_v.item()) if isinstance(alpha_v, torch.Tensor) else float(alpha_v)
            except Exception:
                alpha = float(rank)
        else:
            alpha = float(rank)

        scale = (alpha / max(rank, 1)) * strength
        try:
            return (up_mat @ down_mat) * scale
        except Exception:
            continue

    # Fallback: direct delta tensors
    for suf in _DIRECT_DELTA_SUFFIXES:
        v = sd.get(base + suf)
        if isinstance(v, torch.Tensor):
            return v.float().reshape(int(v.shape[0]), -1) * strength

    return None


def _get_base_rank(sd: Dict[str, Any], base: str) -> int:
    """Return the LoRA rank stored for a base (0 if not a low-rank LoRA)."""
    for _, down_suf in _LORA_PAIR_SUFFIXES:
        down = sd.get(base + down_suf)
        if isinstance(down, torch.Tensor) and down.ndim >= 1:
            return int(down.shape[0])
    return 0


def _get_base_dtype(sd: Dict[str, Any], base: str) -> torch.dtype:
    """Return the storage dtype of the first tensor found for this base."""
    for up_suf, _ in _LORA_PAIR_SUFFIXES:
        up = sd.get(base + up_suf)
        if isinstance(up, torch.Tensor):
            return up.dtype
    for suf in _DIRECT_DELTA_SUFFIXES:
        v = sd.get(base + suf)
        if isinstance(v, torch.Tensor):
            return v.dtype
    return torch.float16


def _copy_base_keys(
    src: Dict[str, Any],
    base: str,
    dst: Dict[str, Any],
    scale: float = 1.0,
) -> None:
    """
    Copy all adapter keys for a base from src → dst.

    Scaling rule (to keep it linear, not quadratic):
      - If .alpha is present  → scale ONLY .alpha
      - Otherwise             → scale only .lora_up / .lora_B (the "up" side)
      - Never scale dora_scale, w_norm, b_norm
    """
    prefix   = base + "."
    has_alpha = any(str(k) == base + ".alpha" for k in src if str(k).startswith(prefix))

    for k, v in src.items():
        ks = str(k)
        if not ks.startswith(prefix):
            continue
        if isinstance(v, torch.Tensor):
            if abs(scale - 1.0) < _EPS:
                dst[ks] = v
            elif ks.endswith(".alpha"):
                dst[ks] = v * scale
            elif (not has_alpha) and ks.endswith((
                ".lora_up.weight", ".lora_A.weight", ".lora_B.weight",
                ".lora_A.default.weight", ".lora_B.default.weight",
                "_lora.up.weight", ".lora.up.weight",
                ".lora_linear_layer.up.weight",
                ".lora_B", ".lora_A",
            )):
                dst[ks] = v * scale
            else:
                dst[ks] = v
        else:
            dst[ks] = v


# ──────────────────────────────────────────────────────────────────────────────
# DARE sparsification
# ──────────────────────────────────────────────────────────────────────────────

def _dare_sparsify(
    delta: torch.Tensor,
    sparsify_rate: float,
    conflict_aware: bool = True,
    reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Zero out a fraction of delta elements (DARE / DELLA-lite).

    conflict_aware=True: only sparsify positions where delta and reference disagree on sign.
    conflict_aware=False: random Bernoulli masking over all elements.
    """
    if sparsify_rate <= 0.0:
        return delta
    if sparsify_rate >= 1.0:
        return torch.zeros_like(delta)

    if conflict_aware and reference is not None:
        # Only sparsify conflicting positions (same-sign positions are left untouched)
        conflict_mask = (
            (delta.sign() != reference.sign())
            & (delta.abs() > _EPS)
            & (reference.abs() > _EPS)
        )
        prob = conflict_mask.float() * sparsify_rate
    else:
        prob = torch.full_like(delta, sparsify_rate)

    mask = torch.bernoulli(prob.clamp(0.0, 1.0)).bool()
    return delta.masked_fill(mask, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Sign conflict analysis
# ──────────────────────────────────────────────────────────────────────────────

def _sign_conflict_ratio(deltas: List[torch.Tensor]) -> float:
    """
    Fraction of non-zero elements where at least one adapter's sign
    disagrees with the elected majority sign.

    Returns 0.0 for a single-adapter case.
    """
    if len(deltas) < 2:
        return 0.0

    # Elected sign = sign of the magnitude-weighted sum
    total = sum(deltas)  # element-wise sum in float32
    elected = total.sign()

    conflicts  = 0
    total_nz   = 0
    for d in deltas:
        nz          = d.abs() > _EPS
        conflicts  += int(((d.sign() != elected) & nz).sum().item())
        total_nz   += int(nz.sum().item())

    return float(conflicts) / float(max(1, total_nz))


# ──────────────────────────────────────────────────────────────────────────────
# Merge strategies
# ──────────────────────────────────────────────────────────────────────────────

def _weighted_average_merge(
    deltas: List[torch.Tensor],
    weights: List[float],
) -> torch.Tensor:
    """Simple magnitude-normalised weighted average."""
    total_w = sum(abs(w) for w in weights)
    if total_w < _EPS:
        return torch.zeros_like(deltas[0])
    return sum(d * w for d, w in zip(deltas, weights)) / total_w


def _ties_merge(
    deltas: List[torch.Tensor],
    weights: List[float],
    trim_ratio: float = DEFAULT_TRIM_RATIO,
) -> torch.Tensor:
    """
    TIES: Trim → Elect Sign → Merge.

    Step 1 — Trim: zero the bottom (1 - trim_ratio) fraction by magnitude per adapter.
    Step 2 — Elect sign: for each element choose the sign with greater total magnitude.
    Step 3 — Merge: average only the adapters whose sign agrees with the elected sign.
              Dissenters contribute nothing to the merged value.

    Reference: Yadav et al. 2023, "TIES-Merging: Resolving Interference When Merging Models"
    """
    if len(deltas) == 1:
        return deltas[0] * weights[0]

    trimmed: List[torch.Tensor] = []
    for delta, w in zip(deltas, weights):
        d = delta * w
        if trim_ratio < 1.0 and d.numel() > 1:
            k = max(1, int(d.numel() * trim_ratio))
            # threshold = (numel - k)-th smallest absolute value
            threshold = float(d.abs().flatten().kthvalue(max(1, d.numel() - k)).values.item())
            d = d.where(d.abs() >= threshold, torch.zeros_like(d))
        trimmed.append(d)

    # Elect sign per element
    pos_mass   = sum(d.clamp(min=0) for d in trimmed)
    neg_mass   = sum(d.clamp(max=0).abs() for d in trimmed)
    elected    = torch.where(pos_mass >= neg_mass,
                             torch.ones_like(pos_mass),
                             -torch.ones_like(pos_mass))

    # Merge: include only agreeing (or zero) contributions
    agreed: List[torch.Tensor] = []
    for d in trimmed:
        agree_mask = (d.sign() == elected) | (d.abs() < _EPS)
        agreed.append(d * agree_mask.float())

    # Divide by count of non-zero agreeing adapters per element
    n_agreed = sum((d.abs() > _EPS).float() for d in agreed).clamp(min=1.0)
    return sum(agreed) / n_agreed


# ──────────────────────────────────────────────────────────────────────────────
# SVD re-compression
# ──────────────────────────────────────────────────────────────────────────────

def _svd_compress(
    delta: torch.Tensor,        # (out_dim, in_dim_flat) float32
    rank: int,
    out_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Re-compress a full-rank delta back to a low-rank LoRA factorisation via truncated SVD.

    Returns:
        lora_up   : (out_dim, rank)  — equivalent to lora_up.weight
        lora_down : (rank, in_dim)   — equivalent to lora_down.weight
        alpha     : scalar tensor = rank  (normalises the product to 1× delta)

    The truncated SVD absorbs sqrt(Σ) into both sides for balanced magnitude.
    alpha is set to rank so that (alpha/rank) * lora_up @ lora_down == delta exactly
    (within float precision and truncation error).
    """
    out_dim, in_dim = delta.shape
    rank = max(1, min(rank, out_dim, in_dim))

    try:
        U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
        # Truncate to desired rank
        U   = U[:, :rank]                   # (out, rank)
        S   = S[:rank]                       # (rank,)
        Vh  = Vh[:rank, :]                   # (rank, in)

        # Absorb S into both sides symmetrically
        S_sqrt    = S.clamp(min=0.0).sqrt()
        lora_up   = (U * S_sqrt.unsqueeze(0)).to(out_dtype)     # (out, rank)
        lora_down = (Vh * S_sqrt.unsqueeze(1)).to(out_dtype)    # (rank, in)
    except Exception as exc:
        _LOG.warning("[ACEStep Merge] SVD failed (%r); using zero-init low-rank pair.", exc)
        lora_up   = torch.zeros(out_dim, rank, dtype=out_dtype)
        lora_down = torch.zeros(rank, in_dim, dtype=out_dtype)

    # alpha=rank ensures (alpha/rank)*up@down = up@down, which is exactly our delta
    alpha = torch.tensor(float(rank), dtype=torch.float32)
    return lora_up, lora_down, alpha


# ──────────────────────────────────────────────────────────────────────────────
# Merge report
# ──────────────────────────────────────────────────────────────────────────────

class MergeReport:
    def __init__(self) -> None:
        self.total_bases          = 0
        self.single_adapter_bases = 0
        self.weighted_avg_bases   = 0
        self.ties_bases           = 0
        self.unmergeable_bases    = 0
        self.conflict_ratios: List[float] = []
        self.strategy_log: List[Dict[str, Any]] = []   # per-base detail (verbose)

    def to_dict(self) -> Dict[str, Any]:
        cr = self.conflict_ratios
        return {
            "total_bases":          self.total_bases,
            "single_adapter_bases": self.single_adapter_bases,
            "weighted_avg_bases":   self.weighted_avg_bases,
            "ties_bases":           self.ties_bases,
            "unmergeable_bases":    self.unmergeable_bases,
            "mean_conflict_ratio":  float(sum(cr) / max(1, len(cr))),
            "max_conflict_ratio":   float(max(cr)) if cr else 0.0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def merge_lora_state_dicts(
    entries: List[Tuple[Dict[str, Any], float, float]],
    conflict_threshold: float = DEFAULT_CONFLICT_THRESHOLD,
    trim_ratio:         float = DEFAULT_TRIM_RATIO,
    dare_sparsify_rate: float = DEFAULT_DARE_RATE,
    dare_conflict_aware: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], MergeReport]:
    """
    Merge multiple LoRA/DoRA state dicts (already key-normalised) into one.

    Args:
        entries:              list of (sd, strength_model, strength_clip).
                              Strengths are the user-specified per-LoRA values.
        conflict_threshold:   sign conflict ratio above which TIES is applied
                              instead of weighted_average. Default 0.25.
        trim_ratio:           TIES step 1 — fraction of elements kept per adapter.
                              Default 0.20 (top 20% by magnitude survive trimming).
        dare_sparsify_rate:   DARE pre-sparsification rate applied before conflict
                              analysis. 0.0 = disabled (default).
        dare_conflict_aware:  If True, DARE only sparsifies conflicting positions
                              (more surgical than random Bernoulli).
        verbose:              Emit per-base strategy logs.

    Returns:
        (merged_sd, report)

    The merged_sd uses the canonical .lora_up.weight / .lora_down.weight / .alpha
    naming so ComfyUI's comfy.lora.load_lora() can process it directly. Strengths
    are baked in; the caller should pass strength=1.0 to add_patches().

    Single-LoRA input: returns a copy of the first SD (no materialization).
    """
    report = MergeReport()

    if not entries:
        return {}, report

    if len(entries) == 1:
        sd, sm, sc = entries[0]
        # Copy with strength baked into alpha / up side
        merged: Dict[str, Any] = {}
        from .adapter_utils import get_base_names
        for base in get_base_names(sd):
            _copy_base_keys(sd, base, merged, scale=sm)
        report.total_bases = len(get_base_names(sd))
        report.single_adapter_bases = report.total_bases
        return merged, report

    # ── Collect all base names ────────────────────────────────────────────────
    from .adapter_utils import get_base_names

    per_adapter_bases: List[Set[str]] = [get_base_names(sd) for sd, _, _ in entries]
    all_bases: Set[str] = set()
    for b in per_adapter_bases:
        all_bases.update(b)

    report.total_bases = len(all_bases)
    merged_sd: Dict[str, Any] = {}

    for base in sorted(all_bases):
        covering = [
            (i, entries[i])
            for i, b_set in enumerate(per_adapter_bases)
            if base in b_set
        ]

        # ── Single-adapter base: copy directly, strength baked in ─────────────
        if len(covering) == 1:
            i, (sd, sm, sc) = covering[0]
            _copy_base_keys(sd, base, merged_sd, scale=sm)
            report.single_adapter_bases += 1
            continue

        # ── Multi-adapter base: materialise deltas ────────────────────────────
        deltas:  List[torch.Tensor] = []
        weights: List[float]        = []
        dtypes:  List[torch.dtype]  = []
        ranks:   List[int]          = []

        for i, (sd, sm, sc) in covering:
            delta = _extract_delta(sd, base, strength=1.0)  # strengths applied via weights
            if delta is None:
                continue
            deltas.append(delta)
            weights.append(float(sm))
            dtypes.append(_get_base_dtype(sd, base))
            ranks.append(max(0, _get_base_rank(sd, base)))

        if not deltas:
            # Cannot materialise — copy first adapter's keys as safe fallback
            i, (sd, sm, sc) = covering[0]
            _copy_base_keys(sd, base, merged_sd, scale=sm)
            report.unmergeable_bases += 1
            if verbose:
                _LOG.warning("[ACEStep Merge] base=%s: cannot materialise delta; copying first adapter.", base)
            continue

        # ── DARE sparsification (before conflict analysis) ────────────────────
        if dare_sparsify_rate > 0.0:
            ref = deltas[0] if dare_conflict_aware and len(deltas) > 1 else None
            sparsified: List[torch.Tensor] = []
            for j, d in enumerate(deltas):
                r = ref if (dare_conflict_aware and j > 0) else None
                sparsified.append(_dare_sparsify(d, dare_sparsify_rate, dare_conflict_aware, r))
            deltas = sparsified

        # ── Pass 1: conflict analysis ─────────────────────────────────────────
        conflict_ratio = _sign_conflict_ratio(deltas)
        report.conflict_ratios.append(conflict_ratio)

        # ── Pass 2: choose strategy and merge ─────────────────────────────────
        if conflict_ratio <= conflict_threshold:
            merged_delta = _weighted_average_merge(deltas, weights)
            strategy     = "weighted_avg"
            report.weighted_avg_bases += 1
        else:
            merged_delta = _ties_merge(deltas, weights, trim_ratio=trim_ratio)
            strategy     = "ties"
            report.ties_bases += 1

        if verbose:
            _LOG.info(
                "[ACEStep Merge] base=%s adapters=%d conflict=%.3f strategy=%s",
                base, len(covering), conflict_ratio, strategy,
            )

        # ── SVD re-compress merged delta back to low-rank ─────────────────────
        total_rank = sum(r for r in ranks if r > 0) or 4
        out_dtype  = dtypes[0] if dtypes else torch.float16
        lora_up, lora_down, alpha = _svd_compress(merged_delta, total_rank, out_dtype)

        merged_sd[base + ".lora_up.weight"]   = lora_up
        merged_sd[base + ".lora_down.weight"] = lora_down
        merged_sd[base + ".alpha"]            = alpha

        # ── DoRA scale: magnitude-weighted average across adapters that have it ─
        dora_scales: List[torch.Tensor] = []
        dora_ws:     List[float]        = []
        for i, (sd, sm, sc) in covering:
            ds = sd.get(base + ".dora_scale")
            if isinstance(ds, torch.Tensor):
                dora_scales.append(ds.float())
                dora_ws.append(abs(float(sm)))
        if dora_scales:
            total_dw = sum(dora_ws)
            if total_dw > _EPS:
                merged_dora = sum(ds * dw for ds, dw in zip(dora_scales, dora_ws)) / total_dw
                merged_sd[base + ".dora_scale"] = merged_dora.to(out_dtype)

    if verbose:
        r = report.to_dict()
        _LOG.info(
            "[ACEStep Merge] complete: bases=%d single=%d weighted_avg=%d ties=%d unmergeable=%d "
            "mean_conflict=%.3f max_conflict=%.3f",
            r["total_bases"], r["single_adapter_bases"], r["weighted_avg_bases"],
            r["ties_bases"], r["unmergeable_bases"],
            r["mean_conflict_ratio"], r["max_conflict_ratio"],
        )

    return merged_sd, report
