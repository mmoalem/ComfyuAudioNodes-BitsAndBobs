# ComfyuAudioNodes-BitsAndBobs

A collection of custom ComfyUI nodes for audio generation, comparison, and manipulation.

## Nodes in this Collection

### [Lora-Dora-Lokr-Loader](./Lora-Dora-Lokr-Loader)
A universal adapter loader for ACE-Step models. 
- Supports LoRA, DoRA, and LoKr/LoHa (LyCORIS) formats.
- Features per-layer category scaling (Self-Attention, Cross-Attention, FFN).
- Advanced auto-strength balancing for Flux-based models.
- Includes a "Simple" node variant for a streamlined UI.

### [Ace-Step_chord_injector](./Ace-Step_chord_injector)
Tools for manipulating and injecting chord information into the ACE-Step generation pipeline.

### [preview_audio_multi_compare](./preview_audio_multi_compare)
A utility node for side-by-side comparison of multiple audio generation outputs within the ComfyUI interface.

## Installation
1. Clone this repository into your ComfyUI `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/mmoalem/ComfyuAudioNodes-BitsAndBobs.git
   ```
2. Restart ComfyUI.
