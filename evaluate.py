"""
Model evaluation module for Handwritten Digit Recognition.
Loads the trained CNN model, runs inference on the official MNIST test set,
computes classification metrics (loss, accuracy, precision, recall, F1),
and generates visualization artifacts (confusion matrix and report json).
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    MODEL_PATH,
    TEST_METRICS_PATH,
    CONFUSION_MATRIX_PLOT_PATH,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS,
)


def evaluate_saved_model():
    """
    Evaluates the saved CNN model on the MNIST test dataset.
    """
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Model file not found at: {MODEL_PATH}")
        print("Please train the model first by executing:")
        print("    python train.py\n")
        sys.exit(1)

    print(f"[INFO] Loading saved model from '{MODEL_PATH}'...")
    from tensorflow import keras
    model = keras.models.load_model(str(MODEL_PATH))

    print("[INFO] Loading official MNIST test dataset...")
    (_, _), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocessing identical to training
    X_test_norm = X_test.reshape((-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)).astype("float32") / 255.0

    print("[INFO] Running evaluation on 10,000 test images...")
    test_loss, test_accuracy = model.evaluate(X_test_norm, y_test, verbose=1)

    # Class predictions
    y_pred_probs = model.predict(X_test_norm, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report_text = classification_report(y_test, y_pred, digits=4)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 65)
    print("           OFFICIAL MNIST TEST SET EVALUATION SUMMARY")
    print("=" * 65)
    print(f" Test Loss:     {test_loss:.4f}")
    print(f" Test Accuracy: {test_accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(report_text)
    print("=" * 65)

    # Save metrics JSON
    metrics_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classification_report": report_dict,
        "confusion_matrix": conf_mat.tolist(),
    }
    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[INFO] Metrics saved to: {TEST_METRICS_PATH}")

    # Plot & Save Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf_mat, interpolation="nearest", cmap="Purples")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = [str(i) for i in range(10)]
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted Digit", fontsize=12, labelpad=8, fontweight="bold")
    ax.set_ylabel("True Digit", fontsize=12, labelpad=8, fontweight="bold")
    ax.set_title("MNIST Test Confusion Matrix", fontsize=14, pad=12, fontweight="bold")

    thresh = conf_mat.max() / 2.0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            val = conf_mat[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {CONFUSION_MATRIX_PLOT_PATH}")

    return metrics_summary


if __name__ == "__main__":
    evaluate_saved_model()
