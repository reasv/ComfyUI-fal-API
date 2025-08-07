import requests
import os
import re
from PIL import Image
import folder_paths
import cv2
import numpy as np
import base64
from io import BytesIO
import tempfile
import subprocess
import shutil

def overlay_image_on_video_objects(video_path, overlay_image_pil, output_path):
    """
    Overlay a PIL image with transparency on top of a video.
    
    Args:
        video_path (str): Path to the input video file
        overlay_image_pil (PIL.Image): PIL Image object with transparency (RGBA)
        output_path (str): Path for the output video file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Resize the overlay image to match video dimensions
        overlay_pil = overlay_image_pil.convert("RGBA")
        overlay_pil = overlay_pil.resize((width, height), Image.Resampling.LANCZOS)
        
        # Convert PIL image to OpenCV format
        overlay_rgba = np.array(overlay_pil)
        overlay_rgb = cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2RGB)
        overlay_alpha = overlay_rgba[:, :, 3] / 255.0  # Normalize alpha channel
        
        # Create temporary output path for OpenCV
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        
        # Try H.264 codec first (most compatible with browsers)
        codecs_to_try = [
            ('H264', cv2.VideoWriter_fourcc(*'H264')),
            ('h264', cv2.VideoWriter_fourcc(*'h264')),
            ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
            ('X264', cv2.VideoWriter_fourcc(*'X264')),
            ('x264', cv2.VideoWriter_fourcc(*'x264')),
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # Fallback to original
        ]
        
        out = None
        used_codec = None
        
        # Try different codecs until one works
        for codec_name, fourcc in codecs_to_try:
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {codec_name}")
                used_codec = codec_name
                break
            out.release()
        
        if not out or not out.isOpened():
            print("Failed to open video writer with any codec")
            return False
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to float for blending
            frame_float = frame.astype(float)
            overlay_float = overlay_rgb.astype(float)
            
            # Apply alpha blending
            # result = background * (1 - alpha) + overlay * alpha
            for c in range(3):  # RGB channels
                frame_float[:, :, c] = (
                    frame_float[:, :, c] * (1 - overlay_alpha) + 
                    overlay_float[:, :, c] * overlay_alpha
                )
            
            # Convert back to uint8
            result_frame = frame_float.astype(np.uint8)
            
            # Write the frame
            out.write(result_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Progress update every 30 frames
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # If we couldn't use H.264 codec and ffmpeg is available, re-encode for browser compatibility
        if used_codec == 'mp4v' and shutil.which('ffmpeg'):
            print("Re-encoding with ffmpeg for browser compatibility...")
            try:
                cmd = [
                    'ffmpeg',
                    '-i', temp_output,
                    '-c:v', 'libx264',  # H.264 codec
                    '-preset', 'medium',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',  # Required for browser compatibility
                    '-movflags', '+faststart',  # Enable fast start for web playback
                    '-y',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("Successfully re-encoded with ffmpeg")
                    os.remove(temp_output)
                else:
                    print(f"ffmpeg re-encoding failed: {result.stderr}")
                    os.rename(temp_output, output_path)
                    
            except Exception as e:
                print(f"Error during ffmpeg re-encoding: {e}")
                os.rename(temp_output, output_path)
        else:
            # Just rename the temp file to final output
            os.rename(temp_output, output_path)
        
        print(f"Video processing complete! Output saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error overlaying image on video: {e}")
        return False

def base64_url_to_pil_image(base64_url):
    """
    Convert a base64 URL string to a PIL Image object.
    
    Args:
        base64_url (str): Base64 URL string (e.g., "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...")
    
    Returns:
        PIL.Image: PIL Image object or None if conversion fails
    """
    try:
        # Remove the data URL prefix if present
        if base64_url.startswith('data:'):
            base64_data = base64_url.split(',', 1)[1]
        else:
            base64_data = base64_url
        
        # Decode base64
        image_data = base64.b64decode(base64_data)
        
        # Create PIL Image
        image = Image.open(BytesIO(image_data))
        return image
        
    except Exception as e:
        print(f"Error converting base64 to PIL Image: {e}")
        return None

class DownloadVideoWithOverlay:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_url": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "Downloaded"}),
                "overlay_base64": ("STRING", {"default": "", "multiline": True}),
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

    def download_video(self, video_url, filename_prefix="Downloaded", overlay_base64="", prompt=None, extra_pnginfo=None, unique_id=None):
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
        
        # Check if overlay is provided
        has_overlay = overlay_base64.strip() != ""
        
        if has_overlay:
            # Create temporary file for original video
            temp_video_path = os.path.join(tempfile.gettempdir(), f"temp_video_{counter}.{file_ext}")
            
            # Save the original video to temp location
            with open(temp_video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Convert base64 to PIL Image
            overlay_image = base64_url_to_pil_image(overlay_base64)
            
            if overlay_image is not None:
                # Apply overlay and save to final location
                success = overlay_image_on_video_objects(temp_video_path, overlay_image, video_path)
                
                if not success:
                    print("Failed to apply overlay, saving original video instead")
                    # Fallback: move temp video to final location
                    os.rename(temp_video_path, video_path)
                else:
                    # Clean up temp file
                    try:
                        os.remove(temp_video_path)
                    except:
                        pass
            else:
                print("Failed to decode overlay image, saving original video instead")
                # Fallback: move temp video to final location
                os.rename(temp_video_path, video_path)
        else:
            # No overlay, save video directly (UNCHANGED FROM ORIGINAL)
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
    "DownloadVideoWithOverlay": DownloadVideoWithOverlay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadVideoWithOverlay": "Download Video with Overlay"
}