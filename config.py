"""
Configuration settings for Handwritten Digit Recognizer.
Defines hyper-parameters, directory paths, and reproducibility settings.
"""

from pathlib import Path

# Base Paths (Relative to project root for portability)
BASE_DIR: Path = Path(__file__).resolve().parent
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
ASSETS_DIR: Path = BASE_DIR / "assets"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Model Artifact Paths
MODEL_FILENAME: str = "mnist_cnn.keras"
MODEL_PATH: Path = MODELS_DIR / MODEL_FILENAME

# Report Artifact Paths
TRAINING_HISTORY_PLOT_PATH: Path = REPORTS_DIR / "training_history.png"
CONFUSION_MATRIX_PLOT_PATH: Path = REPORTS_DIR / "confusion_matrix.png"
TEST_METRICS_PATH: Path = REPORTS_DIR / "test_metrics.json"

# Dataset and Image Constants
IMAGE_WIDTH: int = 28
IMAGE_HEIGHT: int = 28
IMAGE_CHANNELS: int = 1
INPUT_SHAPE: tuple[int, int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
NUM_CLASSES: int = 10
DIGIT_CLASSES: list[int] = list(range(NUM_CLASSES))

# Training Hyperparameters
RANDOM_SEED: int = 42
BATCH_SIZE: int = 128
EPOCHS: int = 12
VALIDATION_SPLIT: float = 0.1
LEARNING_RATE: float = 0.001
EARLY_STOPPING_PATIENCE: int = 3
