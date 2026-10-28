# ComfyuAudioNodes-BitsAndBobs

> [!WARNING]
> **WORK IN PROGRESS (WIP)**  
> This code is currently **experimental, untuned, and not yet optimized**. Use at your own risk. Expect breaking changes and potential bugs as development continues.

A collection of custom ComfyUI nodes for audio generation, comparison, and manipulation.

## Nodes in this Collection

### [Lora-Dora-Lokr-Loader](./Lora-Dora-Lokr-Loader)
A universal adapter loader for ACE-Step models. 
- Supports LoRA, DoRA, and LoKr/LoHa (LyCORIS) formats.
- Features per-layer category scaling (Self-Attention, Cross-Attention, FFN).
- Advanced auto-strength balancing for Flux-based models.
- Includes a "Simple" node variant for a streamlined UI.
- *Based on the [DoRA Power LoRA Loader](https://github.com/xmarre/ComfyUI-DoRA-Dynamic-LoRA-Loader) by xmarre.*

### [Ace-Step_chord_injector](./Ace-Step_chord_injector)
Tools for manipulating and injecting chord information into the ACE-Step generation pipeline.
> [!NOTE]
> This node currently produces an audible effect on the output, but it is **not yet performing its intended function** correctly. It is included here for ongoing development and testing.

### [preview_audio_multi_compare](./preview_audio_multi_compare)
A utility node for side-by-side comparison of multiple audio generation outputs within the ComfyUI interface.
- *Modified from components in the [ryanontheinside](https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside) repository.*

### [ace_step_reference](./ace_step_reference)
A set of nodes for injecting reference audio into ACE-Step generation via multiple pathways.
- **Timbre Encoding & Conditioning:** Encodes reference audio into a timbre embedding and injects it into the cross-attention pathway. This method is stable and generally works well for transferring vocal/instrumental characteristics.
- **KV Self-Attention Injection:** Captures K/V tensors from a reference forward pass and injects them into the generation. This provides higher fidelity style transfer but is currently **WIP (Work In Progress)** with mixed results.
- **Per-Step KV Injection:** Real-time capture and injection at every sampling step. This is the most computationally expensive method but allows for precise alignment.

> [!TIP]
> **Timbre Conditioning** and **KV Injection** can be chained inline to combine both methods for more comprehensive reference transfer. 
> 
> See the included example workflow: `example_workflows/three_injection_methods.json`.

- Includes a **Reference Inspector** debug node to verify capture output.

### [ace_step_gguf_loader](./ace_step_gguf_loader)
A custom GGUF and PyTorch bypass loader specifically designed for running quantized ACE-Step models natively inside ComfyUI.
- Supports ACE-Step 1.5 DiT `acestep` architectures missing from standard allowlists.
- Re-maps the GGUF `qwen3` embedding namespace back into HuggingFace format for ComfyUI detection.
- Includes a direct subclass wrapper for the `AudioOobleckVAE` architecture to fix cross-device dtype crashes and apply missing 48kHz to 44.1kHz resampling when used with ACE-Step 1.5.

### [yue2_abc_manipulator](./yue2_abc_manipulator)
An all-in-one ABC score manipulator node for YuE2 audio generation workflows.
- **Mode Selection:**
  - `bypass`: Passes the ABC notation through untouched.
  - `instrumentalize`: Moves Vocal melodies into resting Ins (instrumental) measures and mutes the Vocal notes.
  - `vocalize`: Moves Ins melodies into resting Vocal measures using clean, minimal key-signature-aware accidentals and mutes the Ins notes.
- **BPM Adjustment:** Updates the score's declared tempo (`Q:` field) to a target BPM or leaves it unchanged.
- **Time Signature Modification:** Updates meter declarations (`M:` field) to common signatures (`2/4`, `3/4`, `4/4`, `5/4`, `6/4`, `6/8`, `7/8`) or leaves them unchanged.
- **Pitch Transposition:** Chromatically transposes notes up or down by up to two octaves (±24 semitones). Accurately handles octave wrapping across ABC register boundaries (uppercase/lowercase/octave marks) while preserving ties, rests, chord labels, and comments.
- **Fully Self-Contained:** Zero external dependencies outside the package.

### [yue2_lora_loader](./yue2_lora_loader)
A dedicated multi-adapter loader for YuE2 models, built on the Power-LoRA architecture.
- Supports **LoRA**, **DoRA**, **LoKr**, and **LoHa** adapter formats.
- Correctly targets both the AR/NAR diffusion transformer (`diffusion_model.model.layers.*`) and the Qwen2 text encoder (`text_encoders.model.layers.*`).
- Features a frontend picker UI with subfolder filtering to easily locate and stack multiple YuE2 adapters.

### [yue2_reference](./yue2_reference)
A suite of reference audio injection and style transfer nodes designed for the YuE2-3B architecture (28 transformer layers, 48kHz AudioOobleckVAE).
- **YuE2 Self-Attention Capture & Inject:** Captures self-attention Key/Value activations from an encoded reference audio latent at a designated noise threshold and injects them into the generation pass across selected layer ranges.
- **YuE2 Per-Step SA Inject (Per Layer):** Performs real-time reference forward passes at each sampling step with granular per-layer strength sliders (layers 0–27), support for `inject`, `replace`, and `threshold` modes, and layer taper options (`linear`, `cosine`).
- **YuE2 Per-Step SA Inject (step+time taper, step limited):** Applies unified injection strength across layers with customizable step taper curves (`fade_out`, `cosine_fade_out`, `cosine_bell`), time-based weighting, and step cutoffs to fade out reference guidance after a set step threshold.

## Installation

### Method 1: ComfyUI-Manager (Recommended)
1. Search for `ComfyuAudioNodes-BitsAndBobs` in the ComfyUI-Manager.
2. Click **Install**.
3. **Important:** You must also install `ComfyUI-GGUF` from the manager for the GGUF loader to function.

### Method 2: Manual Installation
1. Clone this repository into your ComfyUI `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/mmoalem/ComfyuAudioNodes-BitsAndBobs.git
   ```
2. Install the required Python dependencies. If you are using the **ComfyUI Portable** version, run this from your ComfyUI root folder:
   ```bash
   .\python_embeded\python.exe -m pip install -r .\custom_nodes\ComfyuAudioNodes-BitsAndBobs\requirements.txt
   ```
   For standard Python installs:
   ```bash
   pip install -r requirements.txt
   ```
3. **Mandatory Dependency:** The GGUF loader requires the `ComfyUI-GGUF` nodes to be present in your `custom_nodes` folder:
   ```bash
   git clone https://github.com/city96/ComfyUI-GGUF
   ```
4. Restart ComfyUI.

## Credits
Special thanks to the original authors whose work served as the foundation for these nodes:
- **xmarre** ([DoRA Power LoRA Loader](https://github.com/xmarre/ComfyUI-DoRA-Dynamic-LoRA-Loader))
- **ryanontheinside** ([RyanOnTheInside ComfyUI Nodes](https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside))

---

## Support

If you find this project useful, consider supporting my work:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-orange?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/mmoalem)
