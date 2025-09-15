import requests
import os
import re
from PIL import Image
import folder_paths
import json
from typing import List, Tuple

import comfy
import comfy.sd
import comfy.utils

# ---------- Wan 2.2 LoRA JSON -> two stacks (High/Low) ----------

class AV_WanLoraListStacker:
    """
    Parse a JSON list of Wan2.2 LoRAs and produce two LORA_STACK outputs:
      - HIGH_LORA_STACK: list[ (lora_name_high, m, m) ]
      - LOW_LORA_STACK : list[ (lora_name_low,  m, m) ]

    Expected JSON item keys (all strings/numbers):
      name : required high-noise LoRA filename in models/loras (e.g. "*.safetensors")
      low  : optional low-noise LoRA filename (if omitted, LoRA is applied only to HIGH)
      m    : model strength (float). If missing, defaults to 1.0
      c    : clip strength (ignored for Wan, accepted for parity)

    Empty, "[]", or invalid items are skipped gracefully.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "high_stack": ("LORA_STACK",),
                "low_stack":  ("LORA_STACK",),
            },
        }

    CATEGORY = "Art Venture/Loaders"
    RETURN_TYPES = ("LORA_STACK", "LORA_STACK")
    RETURN_NAMES = ("high_lora_stack", "low_lora_stack")
    FUNCTION = "build_stacks"

    def _parse(self, data: str):
        data = (data or "").strip()
        if not data or data == "[]":
            return []

        try:
            cfg = json.loads(data)
            if not isinstance(cfg, list):
                print("[AV_WanLoraListStacker] 'data' must be a JSON array; got:", type(cfg))
                return []
            return cfg
        except Exception as e:
            print("[AV_WanLoraListStacker] JSON parse error:", e)
            return []

    def _available(self) -> set:
        # set of filenames visible to Comfy in the "loras" search path
        return set(folder_paths.get_filename_list("loras"))

    def build_stacks(self, data, high_stack=None, low_stack=None):
        cfg = self._parse(data)
        avail = self._available()

        high: List[Tuple[str, float, float]] = []
        low:  List[Tuple[str, float, float]] = []

        for i, item in enumerate(cfg):
            if not isinstance(item, dict):
                continue

            name = item.get("name")
            low_name = item.get("low")
            m = float(item.get("m", 1.0))

            # Skip zeroed entries
            if m == 0:
                continue

            # High LoRA required to consider the entry
            if not name or name not in avail:
                print(f"[AV_WanLoraListStacker] Missing or unavailable HIGH LoRA at index {i}: {name!r}")
                # still allow low-only entries? Wan typically expects high/low pairing; skip if no high
                continue

            high.append((name, m, m))

            # Optional low LoRA; apply only if present & exists
            if low_name and low_name in avail:
                low.append((low_name, m, m))

        # allow stacking with pre-existing stacks
        if high_stack is not None:
            high.extend([t for t in high_stack if t[0] != "None"])
        if low_stack is not None:
            low.extend([t for t in low_stack if t[0] != "None"])

        return (high, low)


# ---------- Wan 2.2 LoRA JSON -> apply to two models (High/Low) ----------

class AV_WanLoraListLoader(AV_WanLoraListStacker):
    """
    Apply Wan2.2 LoRAs directly to the two expert models (model-only).
    Inputs:
      model_high: MODEL (Wan high-noise expert)
      model_low : MODEL (Wan low-noise expert)
      data      : JSON string (same as stacker)

    Returns:
      (model_high_with_loras, model_low_with_loras)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_high": ("MODEL",),
                "model_low":  ("MODEL",),
                "data": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            }
        }

    CATEGORY = "Art Venture/Loaders"
    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("model_high", "model_low")
    FUNCTION = "load_list_lora"

    def _apply_lora_model_only(self, model, lora_path, strength: float):
        # Prefer comfy.sd.load_lora_for_model if present; else fall back.
        lora_file = comfy.utils.load_torch_file(lora_path)
        if hasattr(comfy.sd, "load_lora_for_model"):
            return comfy.sd.load_lora_for_model(model, lora_file, strength)
        else:
            # Fallback: use load_lora_for_models with clip=None and clip strength 0.0
            model, _ = comfy.sd.load_lora_for_models(model, None, lora_file, strength, 0.0)
            return model

    def load_list_lora(self, model_high, model_low, data):
        cfg = self._parse(data)
        if not cfg:
            return (model_high, model_low)

        avail = self._available()

        # Apply high first, then low. Missing low entries are tolerated.
        mh = model_high
        ml = model_low

        for i, item in enumerate(cfg):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            low_name = item.get("low")
            m = float(item.get("m", 1.0))
            if m == 0:
                continue

            if name and name in avail:
                path_h = folder_paths.get_full_path("loras", name)
                mh = self._apply_lora_model_only(mh, path_h, m)
            else:
                print(f"[AV_WanLoraListLoader] HIGH LoRA missing/unavailable at index {i}: {name!r}")

            if low_name and low_name in avail:
                path_l = folder_paths.get_full_path("loras", low_name)
                ml = self._apply_lora_model_only(ml, path_l, m)
            elif low_name:
                print(f"[AV_WanLoraListLoader] LOW LoRA missing/unavailable at index {i}: {low_name!r}")

        return (mh, ml)

class DownloadVideoAsOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_url": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "Downloaded"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "video"
    FUNCTION = "download_video"

    def download_video(self, video_url, filename_prefix="Downloaded", prompt=None, extra_pnginfo=None, unique_id=None):
        # Use the same output directory logic as VideoCombine
        output_dir = folder_paths.get_output_directory()
        
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        
        # Same counter logic as VideoCombine
        max_counter = 0
        matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
        for existing_file in os.listdir(full_output_folder):
            match = matcher.fullmatch(existing_file)
            if match:
                file_counter = int(match.group(1))
                if file_counter > max_counter:
                    max_counter = file_counter
        
        counter = max_counter + 1
        
        # Download the video
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        # Determine file extension from URL or content-type
        file_ext = "mp4"  # default
        if video_url.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            file_ext = video_url.split('.')[-1]
        
        video_filename = f"{filename}_{counter:05}.{file_ext}"
        video_path = os.path.join(full_output_folder, video_filename)
        
        # Save the video file
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Save metadata image (first frame placeholder)
        png_filename = f"{filename}_{counter:05}.png"
        png_path = os.path.join(full_output_folder, png_filename)
        
        # Create a simple placeholder image
        placeholder = Image.new('RGB', (512, 512), color='black')
        placeholder.save(png_path, compress_level=4)
        
        output_files = [png_path, video_path]
        
        # Return the same format as VideoCombine
        preview = {
            "filename": video_filename,
            "subfolder": subfolder,
            "type": "output",
            "format": f"video/{file_ext}",
            "frame_rate": 30,  # default
            "workflow": png_filename,
            "fullpath": video_path,
        }
        
        return {
            "ui": {"gifs": [preview]}, 
            "result": ((True, output_files),)
        }

NODE_CLASS_MAPPINGS = {
    "DownloadVideoAsOutput": DownloadVideoAsOutput,
    "AV_WanLoraListStacker": AV_WanLoraListStacker,
    "AV_WanLoraListLoader":  AV_WanLoraListLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadVideoAsOutput": "Download Video as Output",
    "AV_WanLoraListStacker": "Wan2.2 LoRA List Stacker (High/Low)",
    "AV_WanLoraListLoader":  "Wan2.2 LoRA List Loader (High/Low)",
}