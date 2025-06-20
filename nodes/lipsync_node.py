import asyncio
import os
import tempfile

import cv2
import requests
import torch
from fal_client import AsyncClient

from .fal_utils import ApiHandler, FalConfig

# Initialize FalConfig
fal_config = FalConfig()


class SyncLipsyncV2Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": "", "multiline": False}),
                "audio_url": ("STRING", {"default": "", "multiline": False}),
                "sync_mode": (
                    ["cut_off", "loop", "bounce", "silence", "remap"],
                    {"default": "cut_off"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_lipsync"
    CATEGORY = "FAL/Lipsync"

    def generate_lipsync(self, video_url, audio_url, sync_mode):
        try:
            if not video_url or not audio_url:
                return ("Error: Video URL and Audio URL are required.",)

            arguments = {
                "video_url": video_url,
                "audio_url": audio_url,
                "sync_mode": sync_mode,
            }

            result = ApiHandler.submit_and_get_result(
                "fal-ai/sync-lipsync/v2", arguments
            )

            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error("sync-lipsync/v2", str(e))


class VeedLipsyncNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": "", "multiline": False}),
                "audio_url": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_lipsync"
    CATEGORY = "FAL/Lipsync"

    def generate_lipsync(self, video_url, audio_url, face_id=0):
        try:
            if not video_url or not audio_url:
                return ("Error: Video URL and Audio URL are required.",)

            arguments = {
                "video_url": video_url,
                "audio_url": audio_url,
            }
            result = ApiHandler.submit_and_get_result("veed/lipsync", arguments)

            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error("veed/lipsync", str(e))


class TavusHummingbirdLipsyncNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": "", "multiline": False}),
                "audio_url": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_lipsync"
    CATEGORY = "FAL/Lipsync"

    def generate_lipsync(self, video_url, audio_url):
        try:
            if not video_url or not audio_url:
                return ("Error: Video URL and Audio URL are required.",)

            arguments = {
                "video_url": video_url,
                "audio_url": audio_url,
            }

            result = ApiHandler.submit_and_get_result(
                "fal-ai/tavus/hummingbird-lipsync/v0", arguments
            )

            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "tavus/hummingbird-lipsync/v0", str(e)
            )


class KlingAudioToVideoLipsyncNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": "", "multiline": False}),
                "audio_url": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_lipsync"
    CATEGORY = "FAL/Lipsync"

    def generate_lipsync(self, video_url, audio_url):
        try:
            arguments = {
                "video_url": video_url,
                "audio_url": audio_url,
            }

            result = ApiHandler.submit_and_get_result(
                "fal-ai/kling-video/lipsync/audio-to-video", arguments
            )

            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "kling-video/lipsync/audio-to-video", str(e)
            )


class CombinedLipsyncNode:
    MODEL_MAP = {
        "Sync.so Lipsync v2": {
            "path": "fal-ai/sync-lipsync/v2",
            "uses_sync_mode": True,
        },
        "Veed.io Lipsync": {
            "path": "veed/lipsync",
            "uses_sync_mode": False,
        },
        "Tavus Hummingbird Lipsync": {
            "path": "fal-ai/tavus/hummingbird-lipsync/v0",
            "uses_sync_mode": False,
        },
        "Kling Audio-to-Video Lipsync": {
            "path": "fal-ai/kling-video/lipsync/audio-to-video",
            "uses_sync_mode": False,
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(cls.MODEL_MAP.keys()),{"default": "Sync.so Lipsync v2"}),
                "video_url": ("STRING", {"default": "", "multiline": False}),
                "audio_url": ("STRING", {"default": "", "multiline": False}),
                "sync_mode": (
                    ["cut_off", "loop", "bounce", "silence", "remap"],
                    {"default": "cut_off"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_lipsync"
    CATEGORY = "FAL/Lipsync"

    def generate_lipsync(self, model, video_url, audio_url, sync_mode):
        try:
            if not video_url or not audio_url:
                return ("Error: Video URL and Audio URL are required.",)

            model_info = self.MODEL_MAP[model]
            api_path = model_info["path"]

            arguments = {
                "video_url": video_url,
                "audio_url": audio_url,
            }

            if model_info["uses_sync_mode"]:
                arguments["sync_mode"] = sync_mode

            result = ApiHandler.submit_and_get_result(api_path, arguments)

            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            error_path = self.MODEL_MAP[model]["path"].replace("fal-ai/", "")
            return ApiHandler.handle_video_generation_error(error_path, str(e))


NODE_CLASS_MAPPINGS = {
    "SyncLipsyncV2_fal": SyncLipsyncV2Node,
    "VeedLipsync_fal": VeedLipsyncNode,
    "TavusHummingbirdLipsync_fal": TavusHummingbirdLipsyncNode,
    "KlingAudioToVideoLipsync_fal": KlingAudioToVideoLipsyncNode,
    "CombinedLipsync_fal": CombinedLipsyncNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SyncLipsyncV2_fal": "Sync.so Lipsync v2 (fal)",
    "VeedLipsync_fal": "Veed.io Lipsync (fal)",
    "TavusHummingbirdLipsync_fal": "Tavus Hummingbird Lipsync (fal)",
    "KlingAudioToVideoLipsync_fal": "Kling Audio-to-Video Lipsync (fal)",
    "CombinedLipsync_fal": "Combined Lipsync (fal)",
}