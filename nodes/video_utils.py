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

# ---------- Simple LoRA JSON list -> LORA_STACK ----------
class LoraListStacker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            },
            "optional": {"lora_stack": ("LORA_STACK",)},
        }

    RETURN_TYPES = ("LORA_STACK",)
    FUNCTION = "load_list_lora"
    CATEGORY = "Custom Nodes/Loaders"

    def parse_lora_list(self, data: str):
        # data is a list of lora model (lora_name, strength_model, strength_clip, url) in json format
        # trim data
        data = data.strip()
        if data == "" or data == "[]" or data is None:
            return []

        print(f"Loading lora list: {data}")

        lora_list = json.loads(data)
        if len(lora_list) == 0:
            return []

        available_loras = folder_paths.get_filename_list("loras")

        lora_params = []
        for lora in lora_list:
            lora_name = lora["name"]
            strength_model = lora["m"]
            strength_clip = lora["c"]

            if strength_model == 0 and strength_clip == 0:
                continue

            if lora_name not in available_loras:
                print(f"Not found lora {lora_name}, skipping")
                continue

            lora_params.append((lora_name, strength_model, strength_clip))

        return lora_params

    def load_list_lora(self, data, lora_stack=None):
        loras = self.parse_lora_list(data)

        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        return (loras,)


class LoraListLoader(LoraListStacker):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "data": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")

    def load_list_lora(self, model, clip, data):
        lora_params = self.parse_lora_list(data)

        if len(lora_params) == 0:
            return (model, clip)

        def load_loras(lora_params, model, clip):
            for lora_name, strength_model, strength_clip in lora_params:
                lora_path = folder_paths.get_full_path("loras", lora_name)
                lora_file = comfy.utils.load_torch_file(lora_path)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora_file, strength_model, strength_clip)
            return model, clip

        lora_model, lora_clip = load_loras(lora_params, model, clip)

        return (lora_model, lora_clip)


# ---------- Wan 2.2 LoRA JSON -> two stacks (High/Low) ----------

class WanLoraListStacker:
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

    CATEGORY = "Custom Nodes/Loaders"
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
                print("[WanLoraListStacker] 'data' must be a JSON array; got:", type(cfg))
                return []
            return cfg
        except Exception as e:
            print("[WanLoraListStacker] JSON parse error:", e)
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
                print(f"[WanLoraListStacker] Missing or unavailable HIGH LoRA at index {i}: {name!r}")
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

class WanLoraListLoader(WanLoraListStacker):
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

    CATEGORY = "Custom Nodes/Loaders"
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
                print(f"[WanLoraListLoader] HIGH LoRA missing/unavailable at index {i}: {name!r}")

            if low_name and low_name in avail:
                path_l = folder_paths.get_full_path("loras", low_name)
                ml = self._apply_lora_model_only(ml, path_l, m)
            elif low_name:
                print(f"[WanLoraListLoader] LOW LoRA missing/unavailable at index {i}: {low_name!r}")

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

class VideoResolutionCap:
    """
    Caps the longest side of a resolution to max_long_side while preserving aspect ratio.

    Inputs:
      width          (INT)  : input width in pixels
      height         (INT)  : input height in pixels
      max_long_side  (INT)  : maximum allowed size for the longer edge

    Outputs:
      width, height (INT, INT) : possibly scaled down integers, same aspect ratio
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 576,  "min": 1, "max": 8192, "step": 1}),
                "max_long_side": ("INT", {"default": 720, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "cap"
    CATEGORY = "video"

    def cap(self, width: int, height: int, max_long_side: int):
        # No-op if already <= cap
        long_side = max(width, height)
        if long_side <= max_long_side:
            return (int(width), int(height))

        # Scale factor to bring longest side to exactly max_long_side
        scale = max_long_side / float(long_side)

        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))

        # Make sure rounding didn't exceed the cap due to ties
        if max(new_w, new_h) > max_long_side:
            # reduce by 1 pixel on the longer edge if necessary
            if new_w >= new_h and new_w > max_long_side:
                new_w = max_long_side
                new_h = max(1, int(round(new_w * (height / float(width)))))
            elif new_h > max_long_side:
                new_h = max_long_side
                new_w = max(1, int(round(new_h * (width / float(height)))))

        return (int(new_w), int(new_h))


class VideoFramesFromSeconds:
    """Convert a duration string and frame rate into a frame count."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("STRING", {"default": "1", "multiline": False, "dynamicPrompts": False}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 240, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "frames"
    CATEGORY = "video"

    _SECONDS_REGEX = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds)?\s*$", re.IGNORECASE)

    def _parse_seconds(self, value: str) -> float:
        if value is None:
            raise ValueError("seconds input must be a string")

        value = value.strip()
        if not value:
            return 0.0

        match = self._SECONDS_REGEX.match(value)
        if not match:
            raise ValueError(f"Invalid seconds format: '{value}'")

        number = float(match.group(1))
        return number

    def frames(self, seconds: str, frame_rate: int):
        secs = self._parse_seconds(seconds)
        frame_count = int(secs * frame_rate) + 1
        return (frame_count,)

NODE_CLASS_MAPPINGS = {
    "DownloadVideoAsOutput": DownloadVideoAsOutput,
    "WanLoraListStacker": WanLoraListStacker,
    "WanLoraListLoader":  WanLoraListLoader,
    "LoraListStacker": LoraListStacker,
    "LoraListLoader":  LoraListLoader,
    "VideoFramesFromSeconds": VideoFramesFromSeconds,
    "VideoResolutionCap": VideoResolutionCap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadVideoAsOutput": "Download Video as Output",
    "WanLoraListStacker": "Wan2.2 LoRA List Stacker (High/Low)",
    "WanLoraListLoader":  "Wan2.2 LoRA List Loader (High/Low)",
    "LoraListStacker": "LoRA List Stacker",
    "LoraListLoader":  "LoRA List Loader",
    "VideoFramesFromSeconds": "Video Frames From Seconds",
    "VideoResolutionCap": "Video Resolution Cap (Max Long Side)",
}