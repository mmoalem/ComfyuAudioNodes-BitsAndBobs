from .nodes import YUE2UniversalAdapterLoader

import folder_paths
from aiohttp import web
from server import PromptServer


@PromptServer.instance.routes.get("/yue2_adapter_loader/loras")
async def yue2_list_loras(request):
    """Return the list of LoRA filenames for the frontend dropdown."""
    return web.json_response(folder_paths.get_filename_list("loras"))


NODE_CLASS_MAPPINGS = {
    "YUE2 Universal Adapter Loader": YUE2UniversalAdapterLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YUE2 Universal Adapter Loader": "YUE2 Universal Adapter Loader",
}
