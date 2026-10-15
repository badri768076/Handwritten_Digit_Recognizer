"""
Inference module for Handwritten Digit Recognition.
Loads the trained CNN model, runs inference on preprocessed inputs,
and formats class probability distributions.
"""

from typing import Dict, Any, Optional, Union
import io
import numpy as np
from PIL import Image

from config import MODEL_PATH, NUM_CLASSES
from preprocessing import preprocess_image


# Module-level model cache to avoid reloading on every prediction call
_MODEL_CACHE = None


def is_model_available() -> bool:
    """Checks whether the trained model file exists on disk."""
    return MODEL_PATH.exists() and MODEL_PATH.is_file()


def load_digit_model(force_reload: bool = False):
    """
    Loads and caches the trained Keras CNN model.
    Raises FileNotFoundError if model has not been trained yet.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE

    if not is_model_available():
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please train the model first by running: python train.py"
        )

    # Lazy import tensorflow only when model is actually loaded
    import tensorflow as tf
    from tensorflow import keras

    _MODEL_CACHE = keras.models.load_model(str(MODEL_PATH))
    return _MODEL_CACHE


def predict_digit(
    image_input: Union[Image.Image, np.ndarray, bytes, str, io.BytesIO]
) -> Dict[str, Any]:
    """
    End-to-end digit prediction function.
    
    Args:
        image_input: Image in any supported format (PIL Image, array, bytes, filepath).
        
    Returns:
        dict containing:
            - 'success': bool
            - 'predicted_digit': int (0-9) if success
            - 'confidence': float (0.0 to 1.0) if success
            - 'probabilities': dict {int: float} for all digits 0-9
            - 'sorted_probabilities': list of tuples (digit, prob) sorted descending
            - 'processed_image': PIL Image (28x28) of normalized input
            - 'error': str (if success is False)
    """
    # Verify model presence before doing heavy work
    if not is_model_available():
        return {
            "success": False,
            "error": (
                "Trained model file 'models/mnist_cnn.keras' does not exist yet. "
                "Please run 'python train.py' or use the Train Model button in the sidebar."
            ),
        }

    # Preprocess the input image
    model_input, display_image, error = preprocess_image(image_input)
    if error is not None:
        return {
            "success": False,
            "error": error,
        }

    try:
        model = load_digit_model()
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to load CNN model: {exc}",
        }

    # Execute model prediction (verbose=0 suppresses stdout spam)
    raw_preds = model.predict(model_input, verbose=0)
    probabilities = raw_preds[0]

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    # Build full probability dictionary
    prob_dict = {digit: float(prob) for digit, prob in enumerate(probabilities)}
    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)

    return {
        "success": True,
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": prob_dict,
        "sorted_probabilities": sorted_probs,
        "processed_image": display_image,
        "input_tensor": model_input,
    }
