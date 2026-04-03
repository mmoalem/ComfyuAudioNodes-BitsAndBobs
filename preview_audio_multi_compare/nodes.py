import folder_paths
import os
import json
import hashlib

MAX_AUDIO_INPUTS = 6

class PreviewAudioMultiCompare:
    """
    A drop-in replacement for PreviewAudioCompare that supports
    up to 6 audio inputs for side-by-side comparison.

    Drop this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
    The node will appear as "Preview Audio Multi Compare" in the audio category.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "audio_1": ("AUDIO",),
            },
            "optional": {},
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
        # Slots 2-6 are optional
        for i in range(2, MAX_AUDIO_INPUTS + 1):
            inputs["optional"][f"audio_{i}"] = ("AUDIO",)
        # Optional label widgets — give each track a custom name
        for i in range(1, MAX_AUDIO_INPUTS + 1):
            inputs["optional"][f"label_{i}"] = ("STRING", {
                "default": f"Track {i}",
                "multiline": False,
            })
        return inputs

    RETURN_TYPES = ()
    FUNCTION = "preview_compare"
    OUTPUT_NODE = True
    CATEGORY = "audio"
    DESCRIPTION = (
        "Preview and compare up to 6 audio outputs side by side. "
        "Connect audio_1 through audio_6 and optionally set label names. "
        "Unconnected slots are ignored."
    )

    def preview_compare(self, audio_1, prompt=None, extra_pnginfo=None, **kwargs):
        """
        Saves each connected audio input as a temp file and returns
        their file references so the JS frontend can render them.
        """
        output_dir = folder_paths.get_temp_directory()
        results = []

        # Collect all connected audio inputs in order
        audio_slots = {"audio_1": audio_1}
        for i in range(2, MAX_AUDIO_INPUTS + 1):
            key = f"audio_{i}"
            if key in kwargs and kwargs[key] is not None:
                audio_slots[key] = kwargs[key]

        # Collect labels
        labels = {}
        for i in range(1, MAX_AUDIO_INPUTS + 1):
            label_key = f"label_{i}"
            labels[f"audio_{i}"] = kwargs.get(label_key, f"Track {i}") or f"Track {i}"

        for slot_key, audio in audio_slots.items():
            try:
                import torchaudio
                import torch

                waveform = audio["waveform"]
                sample_rate = audio["sample_rate"]

                # waveform shape: [batch, channels, samples] or [channels, samples]
                if waveform.dim() == 3:
                    waveform = waveform[0]  # take first item in batch

                # Build a stable filename from a hash of the audio data
                audio_hash = hashlib.md5(waveform.numpy().tobytes()).hexdigest()[:8]
                filename = f"multi_compare_{slot_key}_{audio_hash}.flac"
                full_path = os.path.join(output_dir, filename)

                torchaudio.save(full_path, waveform, sample_rate, format="flac")

                results.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "temp",
                    "label": labels[slot_key],
                })
            except Exception as e:
                print(f"[PreviewAudioMultiCompare] Error saving {slot_key}: {e}")

        return {"ui": {"audio_tracks": results}}


NODE_CLASS_MAPPINGS = {
    "PreviewAudioMultiCompare": PreviewAudioMultiCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewAudioMultiCompare": "Preview Audio Multi Compare",
}
