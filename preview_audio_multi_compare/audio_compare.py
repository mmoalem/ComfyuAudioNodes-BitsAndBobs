import folder_paths
import os
import av


def _save_audio_temp(audio, prefix):
    """Save audio dict to a temp flac file, return SavedResult-style dict."""
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_temp_directory()
    )
    file = f"{filename}_{counter:05}_.flac"
    output_path = os.path.join(full_output_folder, file)

    waveform = audio["waveform"].cpu().squeeze(0)  # [channels, samples]
    sample_rate = audio["sample_rate"]
    layout = "mono" if waveform.shape[0] == 1 else "stereo"

    container = av.open(output_path, "w")
    stream = container.add_stream("flac", rate=sample_rate, layout=layout)

    frame = av.AudioFrame.from_ndarray(
        waveform.movedim(0, 1).reshape(1, -1).float().numpy(),
        format="flt",
        layout=layout,
    )
    frame.sample_rate = sample_rate
    frame.pts = 0

    container.mux(stream.encode(frame))
    container.mux(stream.encode(None))
    container.close()

    return {"filename": file, "subfolder": subfolder, "type": "temp"}


SLOT_LABELS = ["A", "B", "C", "D", "E", "F"]


class PreviewAudioMultiCompare:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for label in SLOT_LABELS:  # all slots A-F are optional
            optional[f"audio_{label.lower()}"] = ("AUDIO",)
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ()
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def compare(self, **kwargs):
        # Return individual outputs expected by the JS file (e.g. a_audio, b_audio...)
        # instead of grouping them together in a single "audio" list
        ui_res = {}
        for label in SLOT_LABELS:
            key = f"audio_{label.lower()}"
            if kwargs.get(key) is not None:
                ui_res[f"{label.lower()}_audio"] = [
                    _save_audio_temp(kwargs[key], f"comfy_mc_{label.lower()}")
                ]

        return {"ui": ui_res}


NODE_CLASS_MAPPINGS = {
    "PreviewAudioMultiCompare": PreviewAudioMultiCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewAudioMultiCompare": "Preview Audio Multi Compare",
}
