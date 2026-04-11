"""
Dump ACEStep model state dict keys to see the actual naming format.
Run this from ComfyUI's Python environment.
"""
import folder_paths
import comfy.utils
import sys

# Find the ACEStep model file
model_files = folder_paths.get_filename_list("checkpoints")
print("Available checkpoints:")
for f in model_files:
    print(f"  {f}")

# Load the first ACEStep model and inspect its state dict keys
for f in model_files:
    if "ace" in f.lower() or "acestep" in f.lower():
        path = folder_paths.get_full_path("checkpoints", f)
        print(f"\nLoading model: {f}")
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        print(f"Total keys: {len(sd)}")
        print("\nFirst 30 keys:")
        for k in sorted(sd.keys())[:30]:
            print(f"  {k}")
        break
