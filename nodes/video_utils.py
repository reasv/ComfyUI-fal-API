import requests
import os
import re
from PIL import Image
import folder_paths

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
    "DownloadVideoAsOutput": DownloadVideoAsOutput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadVideoAsOutput": "Download Video as Output"
}