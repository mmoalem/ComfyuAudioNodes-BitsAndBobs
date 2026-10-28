from .abc_ops import transpose_abc, change_bpm, change_meter
from .instrumentalize import instrumentalize, vocalize


class YuE2ABCManipulator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["bypass", "instrumentalize", "vocalize"],),
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300,
                         "tooltip": "New BPM. 0 = keep original."}),
                "time_signature": (["unchanged", "2/4", "3/4", "4/4", "5/4", "6/4", "6/8", "7/8"],),
                "transpose_semitones": ("INT", {"default": 0, "min": -24, "max": 24,
                                        "tooltip": "Semitones to transpose. ±12 = one octave, ±24 = two octaves."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("abc", "report")
    FUNCTION = "process"
    CATEGORY = "YuE2/ABC"
    DESCRIPTION = (
        "All-in-one ABC manipulator for YuE2 workflows. "
        "Instrumentalize or vocalize the score, change BPM, "
        "change time signature, and transpose up/down by up to two octaves."
    )

    def process(self, abc, mode, bpm, time_signature, transpose_semitones):
        report_parts = []

        # 1. Mode: instrumentalize / vocalize / bypass
        if mode == "instrumentalize":
            try:
                abc, muted, moved, mode_report = instrumentalize(abc)
                report_parts.append(mode_report)
            except ValueError as e:
                report_parts.append(f"instrumentalize error: {e}")
        elif mode == "vocalize":
            try:
                abc, muted, moved, mode_report = vocalize(abc)
                report_parts.append(mode_report)
            except ValueError as e:
                report_parts.append(f"vocalize error: {e}")
        else:
            report_parts.append("Mode: bypass")

        # 2. BPM
        if bpm > 0:
            abc = change_bpm(abc, bpm)
            report_parts.append(f"BPM set to {bpm}")

        # 3. Time signature
        if time_signature != "unchanged":
            abc = change_meter(abc, time_signature)
            report_parts.append(f"Time signature set to {time_signature}")

        # 4. Transpose
        if transpose_semitones != 0:
            abc = transpose_abc(abc, transpose_semitones)
            direction = "up" if transpose_semitones > 0 else "down"
            report_parts.append(f"Transposed {direction} {abs(transpose_semitones)} semitones")

        report = "; ".join(report_parts) if report_parts else "No changes."
        return (abc, report)


NODE_CLASS_MAPPINGS = {
    "YuE2ABCManipulator": YuE2ABCManipulator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2ABCManipulator": "YuE2 ABC Manipulator",
}
