from .fal_utils import ApiHandler, ImageUtils, ResultProcessor

MAX_INPUT_IMAGES = 10

IMAGE_SIZE_PRESETS = {
    "square_hd": {"width": 2048, "height": 2048},
    "square": {"width": 1024, "height": 1024},
    "portrait_4_3": {"width": 1536, "height": 2048},
    "portrait_16_9": {"width": 1440, "height": 2560},
    "landscape_4_3": {"width": 2048, "height": 1536},
    "landscape_16_9": {"width": 2560, "height": 1440},
    "ultra_hd_landscape": {"width": 3840, "height": 2160},
    "ultra_hd_portrait": {"width": 2160, "height": 3840},
    "custom": None,
}


def upload_image(image):
    """Upload image tensor to FAL and return URL."""
    return ImageUtils.upload_image(image)


def _expand_image_inputs(*images):
    """Split batched image tensors into single images before upload."""
    expanded = []
    for image in images:
        if image is None:
            continue
        # torch.Tensor exposes dim(); numpy arrays expose shape
        try:
            if hasattr(image, "dim") and image.dim() == 4 and image.shape[0] > 1:
                expanded.extend(image[i] for i in range(image.shape[0]))
                continue
        except Exception:
            pass
        try:
            shape = image.shape  # numpy.ndarray or torch.Size
            if hasattr(shape, "__len__") and len(shape) == 4 and shape[0] > 1:
                expanded.extend(image[i] for i in range(shape[0]))
                continue
        except Exception:
            pass
        if isinstance(image, (list, tuple)):
            for item in image:
                expanded.extend(_expand_image_inputs(item))
            continue
        expanded.append(image)
    return expanded


class Seedream40ImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_1": ("IMAGE",),
                "image_size": (
                    list(IMAGE_SIZE_PRESETS.keys()),
                    {"default": "square_hd"},
                ),
                "width": (
                    "INT",
                    {"default": 2048, "min": 1024, "max": 4096, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 2048, "min": 1024, "max": 4096, "step": 16},
                ),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "seed": ("INT", {"default": -1}),
                "sync_mode": ("BOOLEAN", {"default": False}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "FAL/Image"

    def edit_image(
        self,
        prompt,
        image_1,
        image_size,
        width,
        height,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        num_images=1,
        max_images=1,
        seed=-1,
        sync_mode=False,
        enable_safety_checker=True,
    ):
        input_images = _expand_image_inputs(
            image_1,
            image_2,
            image_3,
            image_4,
            image_5,
            image_6,
            image_7,
            image_8,
            image_9,
            image_10,
        )

        image_urls = []
        for idx, img in enumerate(input_images, 1):
            url = ImageUtils.upload_image(img)
            if not url:
                print(f"Error: Failed to upload image {idx} for Seedream 4.0 Edit")
                return ResultProcessor.create_blank_image()
            image_urls.append(url)
            if len(image_urls) >= MAX_INPUT_IMAGES:
                break

        if not image_urls:
            print("Error: At least one input image is required for Seedream 4.0 Edit")
            return ResultProcessor.create_blank_image()

        if len(input_images) > MAX_INPUT_IMAGES:
            print(
                "Warning: Seedream 4.0 Edit supports up to 10 input images. Extra images were ignored."
            )

        if num_images * max_images + len(image_urls) > 15:
            print(
                "Error: Total number of images (inputs + outputs) must not exceed 15 for Seedream 4.0 Edit"
            )
            return ResultProcessor.create_blank_image()

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "max_images": max_images,
            "sync_mode": sync_mode,
            "enable_safety_checker": enable_safety_checker,
        }

        if image_size == "custom":
            arguments["image_size"] = {"width": width, "height": height}
        else:
            preset = IMAGE_SIZE_PRESETS.get(image_size)
            if preset:
                arguments["image_size"] = preset

        if seed != -1:
            arguments["seed"] = seed

        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/bytedance/seedream/v4/edit", arguments
            )
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Seedream 4.0 Edit", e)


class NanoBananaImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_1": ("IMAGE",),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "sync_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "FAL/Image"

    def edit_image(
        self,
        prompt,
        image_1,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        num_images=1,
        output_format="jpeg",
        sync_mode=False,
    ):
        input_images = _expand_image_inputs(
            image_1,
            image_2,
            image_3,
            image_4,
            image_5,
            image_6,
            image_7,
            image_8,
            image_9,
            image_10,
        )

        image_urls = []
        for idx, img in enumerate(input_images, 1):
            url = ImageUtils.upload_image(img)
            if not url:
                print(f"Error: Failed to upload image {idx} for Nano Banana Edit")
                return ResultProcessor.create_blank_image()
            image_urls.append(url)
            if len(image_urls) >= MAX_INPUT_IMAGES:
                break

        if not image_urls:
            print("Error: At least one input image is required for Nano Banana Edit")
            return ResultProcessor.create_blank_image()

        if len(input_images) > MAX_INPUT_IMAGES:
            print(
                "Warning: Nano Banana Edit supports up to 10 input images. Extra images were ignored."
            )

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "output_format": output_format,
            "sync_mode": sync_mode,
        }

        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/nano-banana/edit", arguments
            )
            description = result.get("description")
            if description:
                print(f"Nano Banana Edit description: {description}")
            return ResultProcessor.process_image_result(result)
        except Exception as e:
            return ApiHandler.handle_image_generation_error("Nano Banana Edit", e)

NODE_CLASS_MAPPINGS = {
    "NanoBananaEdit_fal": NanoBananaImageEdit,
    "Seedream40Edit_fal": Seedream40ImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaEdit_fal": "Nano Banana Edit (fal)",
    "Seedream40Edit_fal": "Seedream 4.0 Edit (fal)",
}


