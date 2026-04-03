"""
ComfyUI-AceStep-ChordCodes  (Native Workflow Edition)
=====================================================
Injects chord progression codes into ACE-Step conditioning.

Key discoveries:
  - vae.encode() expects [B, N, C] (channels LAST) via waveform.movedim(1,-1)
  - audio_codes format is [[int, int, ...]]
  - FSQ lives at model.diffusion_model.tokenizer.quantizer (ResidualFSQ)
  - ACE-Step VAE sample rate is 48000 Hz

Chord map format (lyrics-aware):
    [verse]  Am F C G
    [chorus] F C G Am
    [bridge] Dm Am Bb G
    default  C G Am F
"""

from __future__ import annotations
import copy, re
from typing import List, Optional, Tuple
import numpy as np

_SR        = 48_000
_FSQ_RATIO = 5        # 25 Hz → 5 Hz

# ─────────────────────────────────────────────────────────────────────────────
#  Music theory
# ─────────────────────────────────────────────────────────────────────────────

_NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
_NOTE_ALIASES = {
    "Db":"C#","Eb":"D#","Fb":"E","Gb":"F#","Ab":"G#","Bb":"A#","Cb":"B",
    "D♭":"C#","E♭":"D#","G♭":"F#","A♭":"G#","B♭":"A#",
}
_CHORD_INTERVALS = {
    "mM7":[0,3,7,11],"maj7":[0,4,7,11],"maj9":[0,4,7,11,14],
    "add9":[0,4,7,14],"dim7":[0,3,6,9],"m7b5":[0,3,6,10],
    "sus2":[0,2,7],"sus4":[0,5,7],"aug":[0,4,8],"dim":[0,3,6],
    "min":[0,3,7],"maj":[0,4,7],"m9":[0,3,7,10,14],"m7":[0,3,7,10],
    "m6":[0,3,7,9],"m":[0,3,7],"M7":[0,4,7,11],"9":[0,4,7,10,14],
    "13":[0,4,7,10,14,17,21],"11":[0,4,7,10,14,17],"7":[0,4,7,10],
    "6":[0,4,7,9],"5":[0,7],"":[0,4,7],
}


def _parse_chord(s: str) -> Optional[Tuple[int, List[int]]]:
    s = s.strip()
    if not s or s in {"-","N","N.C.","rest","x"}:
        return None
    m = re.match(r'^([A-Ga-g][b#♭♯]?)', s)
    if not m:
        return None
    root_raw = m.group(1)
    qual = s[len(root_raw):].split("/")[0].strip()
    rn = root_raw[0].upper() + root_raw[1:].replace("♭","b").replace("♯","#")
    rn = _NOTE_ALIASES.get(rn, rn)
    if rn not in _NOTE_NAMES:
        return None
    pc  = _NOTE_NAMES.index(rn)
    ivs = next((v for q,v in _CHORD_INTERVALS.items()
                if qual == q or (q and qual.startswith(q))),
               _CHORD_INTERVALS[""])
    return 48 + pc, ivs


def _midi_to_freq(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def _synth_note(freq: float, dur: float, kind: str, vel: float) -> np.ndarray:
    n   = max(1, int(_SR * dur))
    t   = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
    sig = np.zeros(n, dtype=np.float64)
    nyq = _SR / 2.0
    if kind == "piano":
        for p,a,d in zip([1,2,3,4,5,6,7,8],
                         [1.0,.50,.25,.15,.08,.05,.03,.02],
                         [2.5,3.5,5.0,6.5,8.0,9.5,11.,13.]):
            if freq*p >= nyq: break
            sig += a * np.exp(-t*d) * np.sin(2*np.pi*freq*p*t)
        att = min(int(_SR*.004), n)
        sig[:att] *= np.linspace(0, 1, att)
    elif kind == "organ":
        for p,a in [(1,1.0),(2,.8),(3,.6),(4,.4),(6,.2),(8,.1)]:
            if freq*p >= nyq: break
            sig += a * np.sin(2*np.pi*freq*p*t)
        att,rel = min(int(_SR*.012),n), min(int(_SR*.04),n)
        env = np.ones(n)
        env[:att]   = np.linspace(0,1,att)
        env[n-rel:] *= np.linspace(1,0,rel)
        sig *= env
    else:  # pad
        for p in [1,2,3]:
            if freq*p >= nyq: break
            sig += (1/p)*np.sin(2*np.pi*freq*p*t)
            sig += (.4/p)*np.sin(2*np.pi*freq*p*1.003*t)
        att,rel = min(int(_SR*.3),n//2), min(int(_SR*.3),n//2)
        env = np.ones(n)
        env[:att]   = np.linspace(0,1,att)
        env[n-rel:] *= np.linspace(1,0,rel)
        sig *= env
    return (sig * vel).astype(np.float32)


def _parse_chord_tokens(text: str, default_beats: float):
    """Parse 'Am:2 F C G' → [(chord, beats), ...]"""
    result = []
    for tok in text.split():
        if not tok: continue
        beats = None
        if ":" in tok:
            c, b = tok.split(":",1)
            try: beats = float(b)
            except ValueError: c = tok
        elif "(" in tok:
            m = re.match(r'^([^(]+)\(([0-9.]+)\)$', tok)
            if m:
                c = m.group(1)
                try: beats = float(m.group(2))
                except ValueError: c = tok
            else: c = tok
        else: c = tok
        result.append((c, beats if beats is not None else default_beats))
    return result


def _synthesise_region(chords, bpm, duration, kind, velocity):
    """Synthesise one region of audio for a given chord list and duration."""
    spb = 60.0 / bpm
    out = np.zeros(int(_SR * duration), dtype=np.float32)
    events, cur = [], 0.0
    for cs, beats in chords:
        events.append((cur, cs, beats * spb)); cur += beats * spb
    loop = cur
    if loop <= 0: return out
    offset = 0.0
    while offset < duration:
        for ev_start, cs, dur in events:
            abs_s = offset + ev_start
            if abs_s >= duration: break
            parsed = _parse_chord(cs)
            if parsed is None: continue
            root, ivs = parsed
            rend = min(dur + 0.25, duration - abs_s)
            ss   = int(abs_s * _SR)
            for iv in ivs:
                nm = root + iv
                while nm > 72: nm -= 12
                while nm < 36: nm += 12
                note = _synth_note(_midi_to_freq(nm), rend, kind, velocity)
                es   = min(ss + len(note), len(out))
                out[ss:es] += note[:es-ss]
        offset += loop
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Lyrics section parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_lyrics_sections(lyrics: str) -> List[Tuple[str, int]]:
    """
    Parse lyrics text → ordered list of (section_name, content_line_count).
    Sections marked by [verse], [chorus], [bridge], [intro], [outro] etc.
    Blank lines don't count toward line count.
    Returns e.g. [('intro',2), ('verse',6), ('chorus',4), ('verse',6), ...]
    """
    sections = []
    current_section = "default"
    current_lines   = 0
    for raw_line in lyrics.splitlines():
        line = raw_line.strip()
        m = re.match(r'^\[([^\]]+)\]', line)
        if m:
            if current_lines > 0 or sections:
                sections.append((current_section, max(current_lines, 1)))
            current_section = m.group(1).lower()
            current_lines   = 0
        elif line:
            current_lines += 1
    if current_lines > 0 or not sections:
        sections.append((current_section, max(current_lines, 1)))
    return sections


def _parse_chord_map(chord_map: str, default_beats: float) -> dict:
    """
    Parse chord map text → {section_name: [(chord, beats), ...]}

    Format:
        [verse]  Am F C G       ← bracketed section name
        [chorus] F C G Am
        default  Am F C G       ← bare word 'default' → stored as 'default'
        Am F C G                ← bare chords with no prefix → 'default'
    """
    result  = {}
    current = "default"
    for raw in chord_map.splitlines():
        line = raw.strip()
        if not line: continue
        m = re.match(r'^\[([^\]]+)\]\s*(.*)', line)
        if m:
            # [section] tag
            current = m.group(1).lower()
            rest    = m.group(2).strip()
            if rest:
                result[current] = _parse_chord_tokens(rest, default_beats)
        else:
            # Bare lines: check if first word is a known section keyword
            # Can't use "not a chord" since words like "default"/"bridge" parse as notes
            _SECTION_KEYWORDS = {
                "default","verse","chorus","bridge","intro","outro",
                "pre","post","hook","refrain","solo","interlude","tag",
            }
            parts = line.split(None, 1)
            first = parts[0].lower()
            rest  = parts[1].strip() if len(parts) > 1 else ""
            if first in _SECTION_KEYWORDS and rest:
                current = first
                result[current] = _parse_chord_tokens(rest, default_beats)
            else:
                # Pure chord line → append to current section
                tokens = _parse_chord_tokens(line, default_beats)
                if tokens:
                    result[current] = tokens
    return result


def _build_chord_audio(
    lyrics:        str,
    chord_map_txt: str,
    bpm:           float,
    beats_per_chord: float,
    total_dur:     float,
    synth_type:    str,
    velocity:      float,
) -> np.ndarray:
    """
    Build full-length chord audio, mapping chords to lyrics sections.

    If lyrics is empty or has no sections, falls back to the 'default'
    entry in chord_map (or whatever is there).
    """
    chord_map  = _parse_chord_map(chord_map_txt, beats_per_chord)
    sections   = _parse_lyrics_sections(lyrics) if lyrics.strip() else []

    print(f"  [chord] sections from lyrics: {sections}")
    print(f"  [chord] chord map keys: {list(chord_map.keys())}")

    # ── No lyrics / no sections → use default for full duration ──────────
    if not sections:
        default_chords = chord_map.get("default", list(chord_map.values())[0]
                                       if chord_map else [("C",4),("G",4),("Am",4),("F",4)])
        valid = [c for c in default_chords if _parse_chord(c[0]) is not None]
        print(f"  [chord] no sections, using default: {default_chords}")
        audio = _synthesise_region(default_chords, bpm, total_dur, synth_type, velocity)
        pk = np.max(np.abs(audio))
        if pk > 1e-6: audio *= 0.85 / pk
        return audio

    # ── Distribute duration proportionally by line count ─────────────────
    total_lines = max(sum(lc for _, lc in sections), 1)
    out = np.zeros(int(_SR * total_dur), dtype=np.float32)
    cursor = 0.0

    for i, (sec_name, line_count) in enumerate(sections):
        # Last section gets whatever time remains
        if i == len(sections) - 1:
            sec_dur = total_dur - cursor
        else:
            sec_dur = (line_count / total_lines) * total_dur

        if sec_dur <= 0:
            continue

        # Look up chords: exact match → base name match → default → fallback
        chords = None
        for key in [sec_name,
                    re.sub(r'\s*\d+$', '', sec_name),  # "verse 2" → "verse"
                    "default"]:
            if key in chord_map:
                chords = chord_map[key]
                break
        if chords is None:
            chords = list(chord_map.values())[0] if chord_map else [("C",4),("G",4),("Am",4),("F",4)]

        print(f"  [chord]   [{sec_name}] {sec_dur:.1f}s → {[c[0] for c in chords]}")
        seg = _synthesise_region(chords, bpm, sec_dur, synth_type, velocity)
        start = int(cursor * _SR)
        end   = min(start + len(seg), len(out))
        out[start:end] += seg[:end-start]
        cursor += sec_dur

    pk = np.max(np.abs(out))
    if pk > 1e-6: out *= 0.85 / pk
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  VAE encode  —  CORRECT API: vae.encode(waveform.movedim(1, -1))
#
#  ComfyUI VAEEncodeAudio source (nodes_audio.py):
#    t = vae.encode(waveform.movedim(1, -1))
#  where waveform is [B, C, N] → movedim(1,-1) → [B, N, C]  (channels last)
# ─────────────────────────────────────────────────────────────────────────────

def _vae_encode_audio(vae, audio_np: np.ndarray):
    """
    Encode float32 mono numpy audio → latent tensor [B, C, T] at 25 Hz.

    Uses the exact same call as ComfyUI's built-in VAEEncodeAudio node:
        vae.encode(waveform.movedim(1, -1))
    where waveform is [B, C, N] → [B, N, C] after movedim.
    """
    import torch
    import torchaudio

    # ACE-Step VAE uses 48 kHz — check if the VAE has audio_sample_rate attr
    vae_sr = int(getattr(vae, "audio_sample_rate", _SR))

    audio_t = torch.from_numpy(audio_np).float()          # [N]
    # Build stereo [1, 2, N] then resample if needed
    waveform = audio_t.unsqueeze(0).unsqueeze(0).expand(1, 2, -1).contiguous()  # [1, 2, N]
    if vae_sr != _SR:
        waveform = torchaudio.functional.resample(waveform, _SR, vae_sr)
        print(f"  [VAE] resampled {_SR}→{vae_sr} Hz")

    print(f"  [VAE] waveform {list(waveform.shape)}  → movedim(1,-1) → "
          f"{list(waveform.movedim(1,-1).shape)}")

    # vae.encode handles device/dtype internally (ComfyUI model management)
    try:
        with torch.no_grad():
            latents = vae.encode(waveform.movedim(1, -1))   # [B, N, C] channels-last
        if isinstance(latents, dict):
            latents = latents.get("samples", next(iter(latents.values())))
        print(f"  [VAE] ✓ latents {list(latents.shape)}")
        return latents.float().cpu()
    except Exception as exc:
        print(f"  [VAE] ✗ vae.encode: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  FSQ tokenisation  —  hardcoded path: model.diffusion_model.tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def _get_tokenizer(model):
    inner = getattr(model, "model", model)
    dm    = getattr(inner, "diffusion_model", None)
    if dm is None:
        print("  [FSQ] diffusion_model not found"); return None, None
    tok   = getattr(dm, "tokenizer", None)
    if tok is None:
        print("  [FSQ] tokenizer not found"); return None, None
    quant = getattr(tok, "quantizer", None)
    return tok, quant


def _unwrap_codes(out):
    import torch
    if out is None: return None
    items = [out] if isinstance(out, torch.Tensor) else \
            list(out) if isinstance(out, (tuple, list)) else []
    for c in items:
        if isinstance(c, torch.Tensor) and c.dtype in (torch.int64, torch.int32, torch.long):
            return c
    for c in items:
        if isinstance(c, torch.Tensor) and c.numel() > 0:
            cf = c.float()
            if cf.min() >= 0 and cf.max() < 200_000:
                return c.long()
    return None


def _get_fsq_vocab_size(tok):
    """
    Compute the FSQ codebook size = product of quantizer levels.
    For levels=[8,8,8,5,5,5] → 64000. Valid indices: 0..63999.
    """
    import math
    try:
        fsq_layer = tok.quantizer.layers[0]
        for attr in ("_levels", "levels", "codebook_size"):
            levels = getattr(fsq_layer, attr, None)
            if levels is not None:
                if hasattr(levels, "tolist"):
                    levels = levels.tolist()
                if isinstance(levels, (list, tuple)):
                    return math.prod(levels)
                if isinstance(levels, int):
                    return levels
    except Exception:
        pass
    return 64000   # fallback: 8*8*8*5*5*5


def _extract_fsq_codes(model, latents):
    """
    Encode VAE latents → 5 Hz FSQ integer codes.

    From ace_step15.py source (confirmed):
    - prepare_condition calls:
        lm_hints_5Hz = tokenizer.quantizer.get_output_from_indices(audio_codes, ...)
    - get_output_from_indices expects shape [B, T, num_quantizers] = [1, T, 1]
    - FSQ vocab size = 8*8*8*5*5*5 = 64000, valid range [0, 63999]
    - Values ≥ 64000 cause CUDA device-side assert (out-of-range codebook lookup)

    We inject audio_codes as [[[code], [code], ...]] so torch.tensor gives [1, T, 1].
    """
    import torch
    import comfy.model_management as mm

    gpu     = mm.get_torch_device()
    offload = mm.unet_offload_device()

    tok, quant = _get_tokenizer(model)
    if quant is None:
        print("  [FSQ] could not reach tokenizer.quantizer")
        return None

    vocab_size = _get_fsq_vocab_size(tok)
    max_code   = vocab_size - 1
    print(f"  [FSQ] tokenizer={type(tok).__name__}  quant={type(quant).__name__}")
    print(f"  [FSQ] vocab_size={vocab_size}  valid_range=[0,{max_code}]")

    tok.to(gpu)

    # latents [1, 64, T] → [1, T, 64] bfloat16
    inp = latents.transpose(1, 2).to(device=gpu, dtype=torch.bfloat16)
    print(f"  [FSQ] tokenizer input {list(inp.shape)} {inp.dtype}")

    result = None
    for method in ("encode", "tokenize", "forward", "__call__"):
        fn = getattr(tok, method, None) if method != "__call__" else tok
        if fn is None:
            continue
        try:
            with torch.no_grad():
                out = fn(inp)
            codes = _unwrap_codes(out)
            if codes is not None:
                flat = codes.reshape(-1).long()
                n_before = flat.numel()
                n_oob    = (flat > max_code).sum().item()
                flat     = flat.clamp(0, max_code)
                print(f"  [FSQ] ✓ tok.{method} → {list(codes.shape)} "
                      f"min={flat.min().item()} max={flat.max().item()}")
                if n_oob:
                    print(f"  [FSQ]   clamped {n_oob}/{n_before} out-of-range tokens")
                result = flat.tolist()
                break
        except Exception as exc:
            print(f"  [FSQ] ✗ tok.{method}: {exc}")

    tok.to(offload)
    if result is None:
        print("  [FSQ] all attempts failed")
    return result
# ─────────────────────────────────────────────────────────────────────────────
#  Conditioning patcher  —  format [[int, int, ...]]
# ─────────────────────────────────────────────────────────────────────────────

def _patch_conditioning(conditioning, codes_list):
    """
    Inject audio_codes in the format that get_output_from_indices expects:
      tensor shape [1, T, 1]  (B, time, num_quantizers)

    We store as [[[c0], [c1], ...]] so torch.tensor converts to [1, T, 1].
    Plain [[c0, c1, ...]] converts to [1, T] which unbinds incorrectly.
    """
    out = []
    for tensor, d in conditioning:
        nd = copy.copy(d)
        if codes_list is not None:
            # Wrap each token in a list: [[c]] per time step → [1, T, 1] tensor
            codes_3d = [[[c] for c in codes_list]]
            nd["audio_codes"] = codes_3d
            print(f"  [patch] ✓ audio_codes [[[c],...]] → tensor [1,{len(codes_list)},1]  "
                  f"first 8: {codes_list[:8]}")
        else:
            print("  [patch] ⚠ codes is None — conditioning unchanged")
        out.append([tensor, nd])
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 0: AceStepConditioningInspector
# ─────────────────────────────────────────────────────────────────────────────

class AceStepConditioningInspector:
    """
    ACE-Step ▸ Conditioning Inspector — passthrough that prints the dict.
    """
    CATEGORY = "ACE-Step"
    FUNCTION = "inspect"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "label": ("STRING", {"default":"COND_DUMP","multiline":False}),
        }}

    def inspect(self, conditioning, label):
        import torch
        sep = "=" * 70
        print(f"\n{sep}\n[{label}] ({len(conditioning)} entries)\n{sep}")
        for i, (tensor, d) in enumerate(conditioning):
            print(f"\n  ── Entry [{i}] ──")
            if isinstance(tensor, torch.Tensor):
                print(f"    main tensor: {list(tensor.shape)}  {tensor.dtype}")
            print(f"    dict keys  : {sorted(d.keys())}")
            for k in sorted(d.keys()):
                v = d[k]
                if isinstance(v, torch.Tensor):
                    vf = v.float()
                    print(f"      {k:35s}: Tensor {list(v.shape)} {v.dtype} "
                          f"min={vf.min():.4f} max={vf.max():.4f}")
                elif isinstance(v, list):
                    print(f"      {k:35s}: list[{len(v)}]", end="")
                    if v and isinstance(v[0], list):
                        print(f"  → list[{len(v[0])}]  first 6: {v[0][:6]}")
                    elif v and isinstance(v[0], torch.Tensor):
                        print(f"  → Tensor {list(v[0].shape)}")
                    else:
                        print(f"  = {repr(v)[:80]}")
                elif isinstance(v, (int,float,bool,str)):
                    print(f"      {k:35s}: {type(v).__name__} = {repr(v)}")
                else:
                    print(f"      {k:35s}: {type(v).__name__}")
        print(sep + "\n")
        return (conditioning,)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 1: AceStepChordConditioner  —  the main node
# ─────────────────────────────────────────────────────────────────────────────

_CHORD_MAP_PLACEHOLDER = """\
[intro]   Am F C G
[verse]   Am F C G
[chorus]  F C G Am
[bridge]  Dm Am Bb G
[outro]   Am F C G
default   Am F C G"""


class AceStepChordConditioner:
    """
    ACE-Step ▸ Chord Conditioner (Native)

    Maps chord progressions to lyrics sections, synthesises reference audio,
    encodes through the ACE-Step VAE and FSQ quantizer, and injects the
    resulting tokens into the conditioning dict.

    ── CHORD MAP FORMAT ──────────────────────────────────────────────────
    [verse]   Am F C G
    [chorus]  F C G Am
    [bridge]  Dm Am Bb G
    default   Am F C G

    Section names match the [tags] in your lyrics.  Sections not listed
    use the 'default' entry.  Beat counts: Am:2 F:2 C G (colon notation).

    ── WIRING ────────────────────────────────────────────────────────────
    TextEncodeAceStepAudio1.5 ──► [this node] ──► KSampler (positive)
    VAELoader                  ──►    ↑
    UNETLoader / DoRA LoRA     ──►    ↑  (model output)

    Set generate_audio_codes = FALSE in TextEncodeAceStepAudio1.5.
    """
    CATEGORY = "ACE-Step"
    FUNCTION = "generate"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae":   ("VAE",),
                "model": ("MODEL",),
                "chord_map": ("STRING", {
                    "multiline": True,
                    "default": _CHORD_MAP_PLACEHOLDER,
                    "tooltip": (
                        "Map section names to chord progressions.\n"
                        "Format:\n"
                        "  [verse]   Am F C G\n"
                        "  [chorus]  F C G Am\n"
                        "  default   Am F C G\n\n"
                        "Section names must match [tags] in your lyrics.\n"
                        "Beat counts: Am:2 F:2 C G"
                    ),
                }),
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Paste the same lyrics as in TextEncodeAceStepAudio1.5.\n"
                        "Used to calculate how long each section is.\n"
                        "Leave empty to use 'default' chords for the full duration."
                    ),
                }),
                "bpm": ("INT", {
                    "default": 120, "min": 40, "max": 240, "step": 1,
                    "tooltip": "Match the BPM in TextEncodeAceStepAudio1.5",
                }),
                "beats_per_chord": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 16.0, "step": 0.5,
                    "tooltip": "Default beats per chord when not specified inline",
                }),
                "duration": ("FLOAT", {
                    "default": 30.0, "min": 5.0, "max": 600.0, "step": 1.0,
                    "tooltip": "Must match duration in TextEncodeAceStepAudio1.5",
                }),
                "synth_type": (["piano", "organ", "pad"], {"default": "piano"}),
            },
            "optional": {
                "velocity": ("FLOAT", {"default": 0.65, "min": 0.1, "max": 1.0, "step": 0.05}),
                "clip":     ("CLIP",),   # unused, kept for workflow compat
            },
        }

    def generate(self, conditioning, vae, model, chord_map, lyrics,
                 bpm, beats_per_chord, duration, synth_type,
                 velocity=0.65, clip=None, **kwargs):

        print("\n[AceStepChord] ════════════════════════════════════════════")
        print(f"  bpm={bpm}  bpc={beats_per_chord}  dur={duration}s  synth={synth_type}")

        # ── 1. Build section-aware chord audio ────────────────────────────
        audio = _build_chord_audio(
            lyrics        = lyrics,
            chord_map_txt = chord_map,
            bpm           = float(bpm),
            beats_per_chord = beats_per_chord,
            total_dur     = duration,
            synth_type    = synth_type,
            velocity      = velocity,
        )
        print(f"  synth: {len(audio)/_SR:.2f}s  peak={np.max(np.abs(audio)):.3f}")

        # ── 2. VAE encode → latents ──────────────────────────────────────
        latents = _vae_encode_audio(vae, audio)
        if latents is None:
            print("  VAE encode failed — conditioning unchanged")
            return (conditioning,)
        print(f"  latents: {list(latents.shape)}")

        # ── 3. FSQ tokenise: latents [B,C,T] → transpose → [B,T,C] bfloat16 ──
        codes_list = _extract_fsq_codes(model, latents)
        if codes_list is None:
            print("  FSQ failed — conditioning unchanged")
            return (conditioning,)
        print(f"  codes: {len(codes_list)} tokens")

        # ── 4. Inject ─────────────────────────────────────────────────────
        out = _patch_conditioning(conditioning, codes_list)
        print("[AceStepChord] ════════════════════════════════════════════\n")
        return (out,)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE 2: AceStepChordPreview  —  hear the section-aware synth audio
# ─────────────────────────────────────────────────────────────────────────────

class AceStepChordPreview:
    """
    ACE-Step ▸ Chord Preview

    Renders the section-aware chord audio so you can hear exactly what
    harmonic structure is being injected before committing to a full generation.
    """
    CATEGORY = "ACE-Step"
    FUNCTION = "preview"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("chord_audio",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chord_map": ("STRING", {"multiline": True, "default": _CHORD_MAP_PLACEHOLDER}),
            "lyrics": ("STRING", {"multiline": True, "default": ""}),
            "bpm":             ("INT",   {"default":120,"min":40,"max":240,"step":1}),
            "beats_per_chord": ("FLOAT", {"default":4.0,"min":0.5,"max":16.0,"step":0.5}),
            "duration":        ("FLOAT", {"default":30.0,"min":5.0,"max":300.0,"step":1.0}),
            "synth_type":      (["piano","organ","pad"], {"default":"piano"}),
        }, "optional": {
            "velocity": ("FLOAT", {"default":0.65,"min":0.1,"max":1.0,"step":0.05}),
        }}

    def preview(self, chord_map, lyrics, bpm, beats_per_chord,
                duration, synth_type, velocity=0.65):
        audio = _build_chord_audio(lyrics, chord_map, float(bpm),
                                   beats_per_chord, duration, synth_type, velocity)
        import torch
        return ({"waveform": torch.from_numpy(audio).unsqueeze(0).unsqueeze(0),
                 "sample_rate": _SR},)


# ─────────────────────────────────────────────────────────────────────────────
#  NODE: AceStepSourceReader — reads ace_step15.py prepare_condition
# ─────────────────────────────────────────────────────────────────────────────

class AceStepSourceReader:
    """
    ACE-Step ▸ Source Reader

    Reads the prepare_condition method from ace_step15.py and also tries
    injecting VAE latents directly as precomputed_lm_hints_25Hz.

    Run once to print the source — paste output so we can understand the
    direct 25Hz latent injection path.
    """
    CATEGORY = "ACE-Step"
    FUNCTION = "read"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "vae":          ("VAE",),
            "model":        ("MODEL",),
        }}

    def read(self, conditioning, vae, model):
        import torch, os, inspect

        sep = "=" * 70

        # ── 1. Find and print prepare_condition source ────────────────────
        try:
            inner = getattr(model, "model", model)
            dm    = getattr(inner, "diffusion_model", None)
            if dm is not None:
                src = inspect.getsource(dm.__class__)
                # Find prepare_condition section
                idx = src.find("def prepare_condition")
                if idx >= 0:
                    chunk = src[idx:idx+3000]
                    print(f"""
{sep}
ace_step15 prepare_condition SOURCE:
{sep}""")
                    print(chunk[:3000])
                    print(sep)
                else:
                    print("prepare_condition not found in diffusion_model source")
                    
                # Also print detokenizer info
                detok = getattr(dm, "detokenizer", None)
                if detok is not None:
                    print(f"""
detokenizer: {type(detok).__name__}""")
                    embed = getattr(detok, "embed_tokens", None)
                    if embed is not None:
                        print(f"  embed_tokens: num_embeddings={embed.num_embeddings}  "
                              f"embedding_dim={embed.embedding_dim}")
                    print("detokenizer source:")
                    print(inspect.getsource(detok.__class__)[:2000])
            else:
                print("diffusion_model not found")
        except Exception as exc:
            print(f"source read error: {exc}")

        # ── 2. Try injecting VAE latents as precomputed_lm_hints_25Hz ─────
        print(f"""
{sep}
Trying precomputed_lm_hints_25Hz injection
{sep}""")
        try:
            import numpy as np
            import comfy.model_management as mm

            # Synthesise 5s of C major as a quick test
            sr = 48000
            t  = np.linspace(0, 5.0, 5*sr, dtype=np.float32)
            test_audio = (0.3*np.sin(2*np.pi*261.63*t) +   # C
                          0.2*np.sin(2*np.pi*329.63*t) +   # E
                          0.2*np.sin(2*np.pi*392.00*t))    # G
            test_audio = test_audio.astype(np.float32)

            wav = torch.from_numpy(test_audio).float()
            wav_stereo = wav.unsqueeze(0).unsqueeze(0).expand(1,2,-1).contiguous()
            with torch.no_grad():
                latents = vae.encode(wav_stereo.movedim(1,-1))
            print(f"  latents: {list(latents.shape)}")

            # Try as precomputed_lm_hints_25Hz
            out = []
            for tensor, d in conditioning:
                nd = dict(d)
                # Try the 25Hz hints key
                nd["precomputed_lm_hints_25Hz"] = latents.float()
                # Also remove audio_codes to avoid conflict
                nd.pop("audio_codes", None)
                out.append([tensor, nd])
                print(f"  injected precomputed_lm_hints_25Hz {list(latents.shape)}")
                print(f"  removed audio_codes")
            print(sep)
            return (out,)

        except Exception as exc:
            print(f"  precomputed_lm_hints_25Hz test failed: {exc}")
            print(sep)
            return (conditioning,)


# ─────────────────────────────────────────────────────────────────────────────
#  Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AceStepConditioningInspector": AceStepConditioningInspector,
    "AceStepChordConditioner":      AceStepChordConditioner,
    "AceStepChordPreview":          AceStepChordPreview,
    "AceStepSourceReader":          AceStepSourceReader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioningInspector": "ACE-Step Conditioning Inspector",
    "AceStepChordConditioner":      "ACE-Step Chord Conditioner (Native)",
    "AceStepChordPreview":          "ACE-Step Chord Preview (Audio)",
    "AceStepSourceReader":          "ACE-Step Source Reader (Debug)",
}
