from .fal_utils import ApiHandler, FalConfig, ImageUtils


fal_config = FalConfig()


class WanSpeechToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "audio_url": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "num_frames": ("INT", {"default": 80, "min": 40, "max": 120, "step": 4}),
                "frames_per_second": ("INT", {"default": 16, "min": 4, "max": 60, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "resolution": (
                    ["480p", "580p", "720p"],
                    {"default": "480p"},
                ),
                "num_inference_steps": ("INT", {"default": 27, "min": 2, "max": 40}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                "shift": (
                    "FLOAT",
                    {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
                "video_quality": (
                    ["low", "medium", "high", "maximum"],
                    {"default": "high"},
                ),
                "video_write_mode": (
                    ["fast", "balanced", "small"],
                    {"default": "balanced"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    def generate_video(
        self,
        prompt,
        image,
        audio_url,
        negative_prompt="",
        num_frames=80,
        frames_per_second=16,
        seed=-1,
        resolution="480p",
        num_inference_steps=27,
        enable_safety_checker=False,
        guidance_scale=3.5,
        shift=5.0,
        video_quality="high",
        video_write_mode="balanced",
    ):
        audio_url = audio_url.strip()
        if not audio_url:
            return ("Error: Audio URL is required.",)

        if num_frames % 4 != 0:
            return ("Error: num_frames must be a multiple of 4.",)

        try:
            image_url = ImageUtils.upload_image(image)
            if not image_url:
                return ApiHandler.handle_video_generation_error(
                    "wan/v2.2-14b/speech-to-video", "Failed to upload image"
                )

            arguments = {
                "prompt": prompt,
                "image_url": image_url,
                "audio_url": audio_url,
                "num_frames": num_frames,
                "frames_per_second": frames_per_second,
                "resolution": resolution,
                "num_inference_steps": num_inference_steps,
                "enable_safety_checker": enable_safety_checker,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "video_quality": video_quality,
                "video_write_mode": video_write_mode,
            }

            if negative_prompt and negative_prompt.strip():
                arguments["negative_prompt"] = negative_prompt

            if seed != -1:
                arguments["seed"] = seed

            result = ApiHandler.submit_and_get_result(
                "fal-ai/wan/v2.2-14b/speech-to-video", arguments
            )
            video_url = result["video"]["url"]
            return (video_url,)
        except Exception as e:
            return ApiHandler.handle_video_generation_error(
                "wan/v2.2-14b/speech-to-video", str(e)
            )


NODE_CLASS_MAPPINGS = {
    "WanSpeechToVideo_fal": WanSpeechToVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSpeechToVideo_fal": "Wan 2.2 Speech-to-Video (fal)",
}
