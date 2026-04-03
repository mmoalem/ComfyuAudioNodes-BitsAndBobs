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

## Installation
1. Clone this repository into your ComfyUI `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/mmoalem/ComfyuAudioNodes-BitsAndBobs.git
   ```
2. Restart ComfyUI.

## Credits
Special thanks to the original authors whose work served as the foundation for these nodes:
- **xmarre** ([DoRA Power LoRA Loader](https://github.com/xmarre/ComfyUI-DoRA-Dynamic-LoRA-Loader))
- **ryanontheinside** ([RyanOnTheInside ComfyUI Nodes](https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside))
