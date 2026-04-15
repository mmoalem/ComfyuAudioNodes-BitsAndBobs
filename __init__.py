from .ace_step_chord_injector import NODE_CLASS_MAPPINGS as ace_chord_nodes, NODE_DISPLAY_NAME_MAPPINGS as ace_chord_names
from .lora_dora_lokr_loader import NODE_CLASS_MAPPINGS as lora_loader_nodes, NODE_DISPLAY_NAME_MAPPINGS as lora_loader_names
from .preview_audio_multi_compare import NODE_CLASS_MAPPINGS as compare_nodes, NODE_DISPLAY_NAME_MAPPINGS as compare_names
from .ace_step_gguf_loader import NODE_CLASS_MAPPINGS as ace_gguf_nodes, NODE_DISPLAY_NAME_MAPPINGS as ace_gguf_names
from .ace_step_reference import NODE_CLASS_MAPPINGS as ace_ref_nodes, NODE_DISPLAY_NAME_MAPPINGS as ace_ref_names

NODE_CLASS_MAPPINGS = {
    **ace_chord_nodes,
    **lora_loader_nodes,
    **compare_nodes,
    **ace_gguf_nodes,
    **ace_ref_nodes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **ace_chord_names,
    **lora_loader_names,
    **compare_names,
    **ace_gguf_names,
    **ace_ref_names,
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
