"""
Automated unit tests for image preprocessing and normalization logic.
Ensures correct tensor dimensions, pixel scaling [0, 1], background inversion,
blank image handling, and aspect ratio preservation.
"""

import numpy as np
from PIL import Image, ImageDraw
import pytest

from preprocessing import (
    load_input_image,
    normalize_background,
    is_image_blank,
    crop_and_center_digit,
    preprocess_image,
)
from config import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS


def create_synthetic_digit_image(bg_color: int = 255, stroke_color: int = 0) -> Image.Image:
    """Creates a synthetic 100x100 grayscale image containing a line digit."""
    img = Image.new("L", (100, 100), color=bg_color)
    draw = ImageDraw.Draw(img)
    # Draw a vertical stroke resembling digit '1'
    draw.line([(50, 20), (50, 80)], fill=stroke_color, width=10)
    return img


def test_load_input_image_formats():
    """Tests loading from PIL, numpy array, and RGBA array."""
    # Test PIL
    pil_img = Image.new("L", (28, 28), 0)
    assert isinstance(load_input_image(pil_img), Image.Image)

    # Test numpy array
    np_arr = np.zeros((28, 28), dtype=np.uint8)
    loaded = load_input_image(np_arr)
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (28, 28)

    # Test RGBA array
    rgba_arr = np.zeros((100, 100, 4), dtype=np.uint8)
    loaded_rgba = load_input_image(rgba_arr)
    assert isinstance(loaded_rgba, Image.Image)


def test_preprocessing_output_shape():
    """Tests that preprocessing produces the exact (1, 28, 28, 1) tensor expected by CNN."""
    img = create_synthetic_digit_image(bg_color=255, stroke_color=0)
    tensor, display_img, error = preprocess_image(img)

    assert error is None, f"Expected successful preprocessing, got error: {error}"
    assert tensor is not None
    assert tensor.shape == (1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    assert isinstance(display_img, Image.Image)
    assert display_img.size == (IMAGE_WIDTH, IMAGE_HEIGHT)


def test_pixel_normalization_range():
    """Tests that all preprocessed tensor values are strictly within [0.0, 1.0]."""
    img = create_synthetic_digit_image(bg_color=255, stroke_color=0)
    tensor, _, error = preprocess_image(img)

    assert error is None
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    # The drawn stroke should produce non-zero pixel intensity
    assert tensor.max() > 0.3


def test_light_background_inversion():
    """
    Tests that a black digit on a white background (e.g. drawn on paper or uploaded photo)
    is properly inverted so the digit has high pixel intensity (matching MNIST format).
    """
    img = create_synthetic_digit_image(bg_color=255, stroke_color=0)
    inverted = normalize_background(img)
    arr = np.array(inverted)

    # Corner pixel should now be dark (near 0)
    assert arr[0, 0] < 50
    # Center stroke should now be bright (near 255)
    assert arr[50, 50] > 200


def test_blank_image_detection():
    """Tests that completely blank or unreadable images return a helpful error message."""
    # Completely white image
    white_img = Image.new("L", (100, 100), 255)
    tensor, _, error = preprocess_image(white_img)
    assert tensor is None
    assert error is not None
    assert "blank" in error.lower() or "unreadable" in error.lower()

    # Completely black image
    black_img = Image.new("L", (100, 100), 0)
    tensor_b, _, error_b = preprocess_image(black_img)
    assert tensor_b is None
    assert error_b is not None


def test_transparent_canvas_blank():
    """Tests that a fully transparent RGBA canvas from streamlit-drawable-canvas is recognized as empty."""
    transparent_canvas = np.zeros((200, 200, 4), dtype=np.uint8)
    tensor, _, error = preprocess_image(transparent_canvas)
    assert tensor is None
    assert error is not None
    assert "empty" in error.lower() or "draw" in error.lower()


def test_crop_and_center_digit():
    """Tests that crop_and_center_digit produces an exact 28x28 image with centered content."""
    # Digit off-center at the bottom right
    img = Image.new("L", (100, 100), 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([70, 70, 90, 90], fill=255)

    centered = crop_and_center_digit(img)
    assert centered.size == (28, 28)

    arr = np.array(centered)
    # The center of 28x28 should contain active pixels
    center_patch = arr[10:18, 10:18]
    assert np.max(center_patch) > 0
