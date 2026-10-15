"""
Image preprocessing pipeline for Handwritten Digit Recognition.
Handles format conversions, background detection/inversion, bounding box cropping,
aspect-ratio preserving resizing, centering, and pixel normalization for MNIST CNN input.
"""

from typing import Tuple, Optional, Union
import io
import numpy as np
from PIL import Image, ImageOps

from config import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS


def load_input_image(
    image_input: Union[Image.Image, np.ndarray, bytes, str, io.BytesIO]
) -> Image.Image:
    """
    Converts various input formats (PIL, numpy array, bytes, file path, BytesIO)
    into a standardized PIL Image.
    """
    if isinstance(image_input, Image.Image):
        return image_input
    elif isinstance(image_input, np.ndarray):
        # Canvas arrays are often RGBA (H, W, 4) uint8 or float
        if image_input.dtype != np.uint8:
            if np.max(image_input) <= 1.0:
                image_input = (image_input * 255).astype(np.uint8)
            else:
                image_input = image_input.astype(np.uint8)
        return Image.fromarray(image_input)
    elif isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, io.BytesIO):
        return Image.open(image_input)
    elif isinstance(image_input, str):
        return Image.open(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def is_image_blank(
    gray_image: Image.Image,
    min_content_variance: float = 12.0,
    min_active_pixels: int = 15,
    threshold: int = 25
) -> bool:
    """
    Determines whether a grayscale image contains a readable digit or is blank/empty.
    Evaluates both pixel intensity variance and active foreground pixel count.
    """
    arr = np.array(gray_image, dtype=np.float32)
    
    # Check overall variance
    if np.var(arr) < min_content_variance:
        return True

    # Count pixels significantly different from median background
    bg_level = np.median(arr)
    diff = np.abs(arr - bg_level)
    active_count = np.sum(diff > threshold)
    
    return active_count < min_active_pixels


def normalize_background(gray_image: Image.Image) -> Image.Image:
    """
    Ensures the digit is represented as white/bright strokes (high pixel values)
    on a black/dark background (near 0 pixel values), matching the MNIST training distribution.
    If the image has a light background, it inverts the colors.
    """
    img_array = np.array(gray_image, dtype=np.float32)
    
    # Estimate background intensity using corners and outer borders
    h, w = img_array.shape
    border_pixels = np.concatenate([
        img_array[0, :],         # top row
        img_array[h - 1, :],     # bottom row
        img_array[:, 0],         # left column
        img_array[:, w - 1]      # right column
    ])
    
    median_border = float(np.median(border_pixels))
    
    # If border/background is light (> 127), invert to match MNIST (white strokes on dark bg)
    if median_border > 127:
        return ImageOps.invert(gray_image)
    return gray_image


def crop_and_center_digit(
    digit_image: Image.Image,
    target_size: Tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
    inner_box_size: int = 20,
    active_threshold: int = 30
) -> Image.Image:
    """
    Crops the digit tightly around its bounding box, scales it preserving aspect ratio
    to fit inside an inner bounding box (default 20x20 per MNIST specification),
    and centers it on a black target_size (28x28) canvas.
    """
    arr = np.array(digit_image)
    
    # Identify coordinates where digit stroke is present
    active_coords = np.argwhere(arr > active_threshold)
    
    if len(active_coords) == 0:
        # Return blank black canvas if no active pixels found
        return Image.new("L", target_size, 0)
    
    min_y, min_x = active_coords.min(axis=0)
    max_y, max_x = active_coords.max(axis=0)
    
    # Crop to bounding box (add 1 pixel margin if within bounds)
    crop_x0 = max(0, min_x - 1)
    crop_y0 = max(0, min_y - 1)
    crop_x1 = min(digit_image.width, max_x + 2)
    crop_y1 = min(digit_image.height, max_y + 2)
    
    cropped = digit_image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    
    # Compute aspect-ratio preserved scaling to fit inside inner_box_size x inner_box_size
    box_w = cropped.width
    box_h = cropped.height
    
    if box_w >= box_h:
        new_w = inner_box_size
        new_h = max(1, int(round((box_h / box_w) * inner_box_size)))
    else:
        new_h = inner_box_size
        new_w = max(1, int(round((box_w / box_h) * inner_box_size)))
        
    resized_digit = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Place on 28x28 black canvas centered
    canvas = Image.new("L", target_size, 0)
    paste_x = (target_size[0] - new_w) // 2
    paste_y = (target_size[1] - new_h) // 2
    canvas.paste(resized_digit, (paste_x, paste_y))
    
    return canvas


def preprocess_image(
    image_input: Union[Image.Image, np.ndarray, bytes, str, io.BytesIO]
) -> Tuple[Optional[np.ndarray], Optional[Image.Image], Optional[str]]:
    """
    Full end-to-end preprocessing pipeline.
    
    Steps:
      1. Load input into PIL Image.
      2. Handle RGBA alpha transparency (composite over white or black).
      3. Convert to Grayscale ('L').
      4. Check for blank / empty image.
      5. Normalize background (invert light background so digit is white on black).
      6. Crop bounding box, preserve aspect ratio into 20x20, center in 28x28.
      7. Normalize pixel values from [0, 255] to [0.0, 1.0].
      8. Reshape to (1, 28, 28, 1) model tensor.
      
    Returns:
      (processed_tensor, display_image, error_message)
      If blank or invalid, processed_tensor is None and error_message is provided.
    """
    try:
        pil_img = load_input_image(image_input)
    except Exception as exc:
        return None, None, f"Failed to load image: {exc}"
        
    # Handle RGBA transparent backgrounds commonly found in canvas outputs
    if pil_img.mode == "RGBA":
        # Extract alpha channel
        alpha = np.array(pil_img.split()[-1])
        # If image is almost entirely transparent, treat as blank
        if np.all(alpha == 0):
            return None, None, "Canvas is empty. Please draw a digit before predicting."
        
        # Check background: composite over black or white depending on stroke color
        rgb_img = Image.new("RGB", pil_img.size, (0, 0, 0))
        rgb_img.paste(pil_img, mask=pil_img.split()[3])
        gray_img = rgb_img.convert("L")
    else:
        gray_img = pil_img.convert("L")
        
    # Check if image is blank
    if is_image_blank(gray_img):
        return None, None, "Blank or unreadable image detected. Please draw or upload a clear digit."
        
    # Step 5: Normalize background so digit is white (bright) on black (dark)
    normalized_bg_img = normalize_background(gray_img)
    
    # Step 6: Crop bounding box, preserve aspect ratio, center in 28x28
    centered_28x28 = crop_and_center_digit(normalized_bg_img)
    
    # Check if centered image ended up blank
    arr_28x28 = np.array(centered_28x28, dtype=np.float32)
    if np.max(arr_28x28) < 10.0:
        return None, None, "Digit stroke could not be detected. Please draw more firmly."
        
    # Step 7: Normalize pixel values [0, 255] -> [0.0, 1.0]
    normalized_tensor = arr_28x28 / 255.0
    
    # Step 8: Reshape to (1, 28, 28, 1) for CNN input
    model_input = normalized_tensor.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    
    return model_input, centered_28x28, None
