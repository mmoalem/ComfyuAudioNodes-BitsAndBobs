from .nodes import ACEStepUniversalAdapterLoader, ACEStepUniversalAdapterLoaderSimple

import folder_paths
from aiohttp import web
from server import PromptServer


@PromptServer.instance.routes.get("/acestep_adapter_loader/loras")
async def acestep_list_loras(request):
    """Return the list of LoRA filenames for the frontend dropdown."""
    return web.json_response(folder_paths.get_filename_list("loras"))


# Tell ComfyUI to serve our frontend extension (inherits the upstream web/ dir).
# WEB_DIRECTORY removed - now handled in root __init__.py

NODE_CLASS_MAPPINGS = {
    "ACEStep Universal Adapter Loader": ACEStepUniversalAdapterLoader,
    "ACEStep Universal Adapter Loader (Simple)": ACEStepUniversalAdapterLoaderSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACEStep Universal Adapter Loader": "ACEStep Universal Adapter Loader (Advanced)",
    "ACEStep Universal Adapter Loader (Simple)": "ACEStep Universal Adapter Loader (Simple)",
}
