"""
Model training pipeline for Handwritten Digit Recognition.
Loads official MNIST dataset from Keras, normalizes and reshapes data,
builds a Convolutional Neural Network (CNN), trains with validation split and early stopping,
evaluates on test data, and saves model artifacts and learning curve plots.
"""

import os
import sys
import json
import argparse
from typing import Tuple, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

# Ensure reproducible random seed
import config
from config import (
    MODEL_PATH,
    TRAINING_HISTORY_PLOT_PATH,
    TEST_METRICS_PATH,
    CONFUSION_MATRIX_PLOT_PATH,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS,
    INPUT_SHAPE,
    NUM_CLASSES,
    RANDOM_SEED,
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)


def load_and_preprocess_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads MNIST dataset from Keras and applies preprocessing:
      - Reshapes grayscale 28x28 arrays into 4D tensors: (N, 28, 28, 1)
      - Normalizes uint8 pixel values [0, 255] into float32 [0.0, 1.0]
      
    Returns:
      (X_train, y_train), (X_test, y_test)
    """
    from tensorflow import keras
    
    print("[INFO] Loading official MNIST dataset from Keras...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"[INFO] Raw training samples: {X_train.shape[0]}, Raw test samples: {X_test.shape[0]}")
    
    # Reshape images to (Height, Width, Channels) = (28, 28, 1)
    X_train = X_train.reshape((-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    X_test = X_test.reshape((-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    
    # Pixel normalization: Rescaling raw integer pixel intensities [0, 255] to [0.0, 1.0].
    # Normalization stabilizes gradient updates during backpropagation and accelerates convergence.
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    print(f"[INFO] Preprocessed X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"[INFO] Preprocessed X_test shape:  {X_test.shape}, dtype: {X_test.dtype}")
    
    return X_train, y_train, X_test, y_test


def build_cnn_model(input_shape: tuple = INPUT_SHAPE, num_classes: int = NUM_CLASSES):
    """
    Constructs a high-accuracy Convolutional Neural Network (CNN) architecture:
      - Conv2D (32 filters, 3x3 kernel, ReLU) + BatchNormalization + Conv2D (32 filters)
      - MaxPooling2D (2x2) + Dropout (0.25)
      - Conv2D (64 filters, 3x3 kernel, ReLU) + BatchNormalization + Conv2D (64 filters)
      - MaxPooling2D (2x2) + Dropout (0.25)
      - Flatten
      - Dense (128 units, ReLU) + BatchNormalization + Dropout (0.5)
      - Dense (10 units, Softmax) output
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        # Block 1: Low-level edge & stroke feature extraction
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1_1"),
        layers.BatchNormalization(name="bn1_1"),
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.Dropout(0.25, name="drop1"),

        # Block 2: Mid-level contour and loop detection
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2_1"),
        layers.BatchNormalization(name="bn2_1"),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.Dropout(0.25, name="drop2"),

        # Classification Head: High-level reasoning and probability output
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense1"),
        layers.BatchNormalization(name="bn_dense"),
        layers.Dropout(0.5, name="drop_dense"),
        layers.Dense(num_classes, activation="softmax", name="digit_output")
    ], name="MNIST_CNN_Classifier")
    
    return model


def plot_training_history(history, save_path=TRAINING_HISTORY_PLOT_PATH) -> None:
    """
    Plots training & validation accuracy and loss side-by-side,
    saving the high-resolution figure to reports/.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    # Subplot 1: Model Accuracy
    ax1.plot(epochs_range, history.history["accuracy"], label="Training Accuracy", color="#3b82f6", lw=2.5)
    if "val_accuracy" in history.history:
        ax1.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", color="#10b981", lw=2.5, linestyle="--")
    ax1.set_title("Model Accuracy Across Epochs", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True)
    
    # Subplot 2: Model Loss
    ax2.plot(epochs_range, history.history["loss"], label="Training Loss", color="#ef4444", lw=2.5)
    if "val_loss" in history.history:
        ax2.plot(epochs_range, history.history["val_loss"], label="Validation Loss", color="#f59e0b", lw=2.5, linestyle="--")
    ax2.set_title("Categorical Cross-Entropy Loss Across Epochs", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training history plots saved successfully to '{save_path}'")


def train_model(force_retrain: bool = False) -> Dict[str, Any]:
    """
    Executes the complete training workflow:
      1. Checks if a trained model already exists (unless force_retrain=True).
      2. Loads and preprocesses MNIST dataset.
      3. Builds and compiles CNN model.
      4. Trains with early stopping.
      5. Evaluates on test set and computes metrics.
      6. Saves model and report artifacts.
    """
    if MODEL_PATH.exists() and not force_retrain:
        print(f"[INFO] Trained model already found at '{MODEL_PATH}'.")
        print("[INFO] Use '--force' flag to trigger retraining if desired.")
        return {"status": "reused", "model_path": str(MODEL_PATH)}
        
    set_seed(RANDOM_SEED)
    
    from tensorflow import keras
    
    X_train, y_train, X_test, y_test = load_and_preprocess_mnist()
    
    model = build_cnn_model()
    model.summary()
    
    # Compile model using Adam optimizer and Sparse Categorical Crossentropy
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Early stopping callback to prevent overfitting and avoid unnecessary epochs
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("\n[INFO] Starting CNN training...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history plots
    plot_training_history(history)
    
    # Evaluate model on untouched MNIST test set
    print("\n[INFO] Evaluating model on official MNIST test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"==================================================")
    print(f" FINAL TEST EVALUATION RESULTS:")
    print(f" Test Loss:     {test_loss:.4f}")
    print(f" Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"==================================================")
    
    # Save the trained model
    model.save(str(MODEL_PATH))
    print(f"[INFO] Model saved successfully to '{MODEL_PATH}'")
    
    # Generate predictions for confusion matrix and metrics
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
        "final_val_accuracy": float(history.history["val_accuracy"][-1]) if "val_accuracy" in history.history else None,
        "classification_report": report_dict,
        "confusion_matrix": conf_mat,
    }
    
    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Test evaluation metrics saved to '{TEST_METRICS_PATH}'")
    
    # Generate and save confusion matrix heatmap
    _plot_confusion_matrix(np.array(conf_mat))
    
    return metrics


def _plot_confusion_matrix(conf_mat: np.ndarray, save_path=CONFUSION_MATRIX_PLOT_PATH) -> None:
    """Renders a polished confusion matrix heatmap and saves to reports/."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf_mat, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    classes = [str(i) for i in range(10)]
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted Digit", fontsize=12, labelpad=8, fontweight="bold")
    ax.set_ylabel("True Digit", fontsize=12, labelpad=8, fontweight="bold")
    ax.set_title("MNIST Test Set Confusion Matrix", fontsize=14, pad=12, fontweight="bold")
    
    # Annotate counts inside matrix cells
    thresh = conf_mat.max() / 2.0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            val = conf_mat[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=9)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix plot saved to '{save_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on MNIST dataset.")
    parser.add_argument("--force", action="store_true", help="Force retrain even if model already exists.")
    args = parser.parse_args()
    
    train_model(force_retrain=args.force)
