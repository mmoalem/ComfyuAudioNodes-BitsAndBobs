# ComfyUI-AceStep-ChordCodes

Injects a chord progression as structural guidance into native ComfyUI
ACE-Step 1.5 workflows — no external server, no separate process.
Works directly with your existing `TextEncodeAceStepAudio1.5 → KSampler` chain.

---

## How it works

ACE-Step 1.5 normally uses the LM to generate a sequence of 5 Hz integer
tokens (audio codes) that act as a harmonic skeleton for the DiT:

```
Text prompt  →  [LM]  →  5 Hz audio codes  →  [DiT]  →  Audio
```

This node replaces the LM's role using the model objects already loaded in
your workflow:

```
Chord text  →  additive synthesis  →  vae.encode()  →  25 Hz latents
                                                              ↓
                                              CLIP model FSQ quantizer
                                                              ↓
                                                    5 Hz integer codes
                                                              ↓
                                          patched CONDITIONING dict
                                                              ↓
                                                          KSampler
```

The synthesised chord audio travels through the exact same VAE → attention
pool → FSQ pipeline that the robustini AceFlow technique uses — everything
just runs inside ComfyUI using the models you have already loaded.

---

## Installation

Drop the folder into `ComfyUI/custom_nodes/` and restart ComfyUI.
No extra dependencies — numpy is already present in every ComfyUI install.

Three nodes appear under the **ACE-Step** category:

| Node | Purpose |
|------|---------|
| **ACE-Step Chord Conditioner (Native)** | Main node — synthesises chords, encodes them, patches the conditioning dict |
| **ACE-Step Conditioning Inspector** | Debug utility — prints conditioning dict keys to the console |
| **ACE-Step Chord Preview (Audio)** | Lets you hear the synthesised chord audio before generating |

---

## Wiring

One small change to your existing workflow:

```
Before:
  TextEncodeAceStepAudio1.5 ─────────────────────────────► KSampler

After:
  TextEncodeAceStepAudio1.5 ──► AceStepChordConditioner ──► KSampler
   (set generate_audio_codes        ↑              ↑
    = false)                   VAELoader     DualCLIPLoader / DoRA LoRA
                                             (clip output)
```

Your VAE and CLIP are already loaded — just wire the same outputs into the
chord node alongside their existing connections.

**One setting to change:** in `TextEncodeAceStepAudio1.5` set
`generate_audio_codes = false`. This skips the LM step entirely (saves time
and VRAM) and leaves the audio-codes slot empty for this node to fill.

---

## First run — use the Inspector

On the very first run, insert **ACE-Step Conditioning Inspector** between
`TextEncodeAceStepAudio1.5` (with `generate_audio_codes = true`) and the
KSampler. It prints the conditioning dict to the console:

```
[DEBUG] Conditioning structure dump
  Entry [0]
    dict keys : ['audio_codes', 'bpm', 'key', ...]
      audio_codes  : Tensor [1, 150] dtype=torch.int64
```

This confirms the exact key the model uses. The chord conditioner already
tries all known key names automatically, but the Inspector output is useful
for verifying injection is working correctly.

---

## Chord notation

```
Am F C G                    ← space separated
Am F | C G | F C | G Am     ← bar notation
Am F                        ← one bar per line (multiline input)
C G
Am:2 F:2 C:4                ← colon = beats per chord
Am(2) F(2) C(4)             ← parens = beats per chord
Cmaj7 Am7 Fmaj7 G7          ← seventh chords
F#m Bm E A                  ← sharps
Bb Eb Ab Db                 ← flats
- Am F C                    ← rest / silence beat
```

Supported qualities: `m  maj7  m7  7  mM7  dim  dim7  aug  sus2  sus4
add9  6  m6  9  maj9  m9  11  13  5` plus plain major (no suffix).
Slash chords (`Am/C`) are supported — bass note is ignored.

---

## Parameters

| Parameter | Advice |
|-----------|--------|
| **duration** | Must exactly match `duration` in `TextEncodeAceStepAudio1.5` |
| **bpm** | Match your generation BPM for correct chord timing |
| **synth_type** | `piano` (default) gives the clearest harmonic content for FSQ encoding |
| **strength** | 0.85 recommended; lower values give the DiT more stylistic freedom |
| **generate_audio_codes** (TextEncode node) | Set to `false` when using this node |

---

## Troubleshooting

**"No valid chords found"**
Check spelling — e.g. `Am` not `Amin`, `Bb` not `Bb major`.

**FSQ extraction fails (console says "quantizer NOT found")**
Use the Inspector node to see the full CLIP model structure and open an
issue with the output. The node will still inject the raw 25 Hz VAE latents
as a fallback, which may provide partial harmonic guidance.

**Output doesn't follow the chords**
- Raise `strength` toward 1.0
- Confirm the Inspector node shows codes being injected under `audio_codes`
- Try `piano` synth type for clearest harmonic signal

**Token count mismatch in console**
Make sure `duration` in this node exactly matches `duration` in
`TextEncodeAceStepAudio1.5`.

---

## Credits

Technique from [robustini/ACE-Step-1.5](https://github.com/robustini/ACE-Step-1.5)
(AceFlow).  ACE-Step 1.5 by ACE Studio × StepFun.
