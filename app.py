"""
Streamlit Web Application for Handwritten Digit Recognizer.
Features an interactive drawing canvas, image upload, synthetic sample testing,
real-time CNN inference, probability distribution visualizations,
educational preprocessed image view, and model evaluation reports.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import streamlit as st

# Configure page settings early
st.set_page_config(
    page_title="Handwritten Digit Recognizer | CNN",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Project module imports
import config
from config import (
    MODEL_PATH,
    TRAINING_HISTORY_PLOT_PATH,
    CONFUSION_MATRIX_PLOT_PATH,
    TEST_METRICS_PATH,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)
from preprocessing import preprocess_image
from predict import predict_digit, is_model_available


# Custom styling for a modern, sleek dark-charcoal UI with electric violet accents
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Overall page background accents */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #1a162b 0%, #0d0f17 50%, #07080c 100%);
        color: #f1f5f9;
    }

    /* Top banner card */
    .header-card {
        background: linear-gradient(135deg, rgba(30, 27, 46, 0.7) 0%, rgba(20, 24, 38, 0.7) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px rgba(139, 92, 246, 0.1);
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #ffffff 30%, #a78bfa 70%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 16px;
    }

    /* Info metric pill badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 8px;
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #c4b5fd;
        letter-spacing: 0.2px;
    }

    .pill-badge span.dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }

    /* Sleek card container */
    .glass-card {
        background: rgba(22, 27, 39, 0.65);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    /* Giant predicted digit display */
    .prediction-hero {
        background: linear-gradient(135deg, rgba(30, 27, 54, 0.8) 0%, rgba(17, 24, 39, 0.8) 100%);
        border: 2px solid rgba(139, 92, 246, 0.45);
        border-radius: 20px;
        padding: 30px 20px;
        text-align: center;
        box-shadow: 0 0 35px rgba(139, 92, 246, 0.2);
    }

    .prediction-number {
        font-size: 5.5rem;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(180deg, #ffffff 20%, #a78bfa 85%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
        text-shadow: 0 0 25px rgba(167, 139, 250, 0.4);
    }

    .confidence-label {
        font-size: 1.15rem;
        font-weight: 600;
        color: #38bdf8;
        letter-spacing: 0.5px;
    }

    /* Stylized action buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35);
        width: 100%;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        box-shadow: 0 6px 22px rgba(139, 92, 246, 0.55);
        transform: translateY(-1px);
        color: #ffffff;
    }

    /* Secondary / outline buttons */
    div.stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background: rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
    }

    /* Sidebar customization */
    section[data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(139, 92, 246, 0.25) !important;
        color: #ffffff !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def load_test_metrics_data() -> Optional[Dict[str, Any]]:
    """Loads saved test evaluation metrics from reports/test_metrics.json."""
    if TEST_METRICS_PATH.exists():
        try:
            with open(TEST_METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def render_sidebar():
    """Renders informative sidebar with model status, architecture details, and controls."""
    with st.sidebar:
        st.markdown("### 🤖 Model Operations")
        
        # Check model file status
        model_exists = is_model_available()
        metrics = load_test_metrics_data()
        
        if model_exists:
            st.success("✅ **CNN Model Loaded & Ready**", icon="🔥")
            if metrics and "test_accuracy" in metrics:
                acc_pct = metrics["test_accuracy"] * 100
                loss_val = metrics["test_loss"]
                st.metric(label="Verified Test Accuracy", value=f"{acc_pct:.2f}%", delta="MNIST Benchmark")
                st.caption(f"📉 Test Cross-Entropy Loss: **{loss_val:.4f}**")
        else:
            st.warning("⚠️ **Model Not Yet Trained**")
            st.info("Train the CNN model from your terminal with:\n```cmd\npython train.py\n```\nor click the button below.")

            if st.button("🚀 Train Model Now", help="Runs CNN training on MNIST dataset"):
                with st.spinner("Training CNN on MNIST dataset (~1-2 min)..."):
                    try:
                        from train import train_model
                        res = train_model(force_retrain=True)
                        st.success(f"Training Complete! Test Accuracy: {res['test_accuracy'] * 100:.2f}%")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")

        st.markdown("---")
        st.markdown("### 📐 CNN Architecture")
        st.markdown(
            """
            - **Input**: 28 × 28 × 1 Grayscale
            - **Conv Block 1**: 2× [Conv2D (32, 3x3) + BatchNorm] + MaxPool(2x2) + Dropout(0.25)
            - **Conv Block 2**: 2× [Conv2D (64, 3x3) + BatchNorm] + MaxPool(2x2) + Dropout(0.25)
            - **Classifier**: Flatten + Dense(128, ReLU) + BatchNorm + Dropout(0.5)
            - **Output**: Dense(10, Softmax)
            """
        )

        st.markdown("---")
        st.markdown("### 💡 Tips for Best Results")
        st.markdown(
            """
            - Draw the digit **large and centered** on the canvas.
            - Use a single stroke or clear thick lines.
            - If uploading an image, ensure the digit is clearly contrasting against its background.
            - The pipeline automatically centers and inverts colors to match MNIST!
            """
        )

        st.markdown("---")
        st.caption("Handwritten Digit Recognizer • Deep Learning & CNN")


def render_header():
    """Renders the top header card and specification pills."""
    st.markdown(
        """
        <div class="header-card">
            <div class="main-title">Handwritten Digit Recognizer</div>
            <div class="sub-title">Powered by Convolutional Neural Networks & TensorFlow/Keras</div>
            <div class="badge-container">
                <div class="pill-badge"><span class="dot"></span> Model: Deep CNN</div>
                <div class="pill-badge">Input: 28 × 28 Grayscale</div>
                <div class="pill-badge">Classes: 0 to 9</div>
                <div class="pill-badge">Dataset: MNIST (60,000 Train / 10,000 Test)</div>
                <div class="pill-badge">Loss: Sparse Categorical Cross-Entropy</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_results(prediction_result: Dict[str, Any]):
    """Renders the prediction results, probabilities chart, and preprocessed tensor view."""
    pred_digit = prediction_result["predicted_digit"]
    confidence = prediction_result["confidence"]
    probabilities = prediction_result["probabilities"]
    processed_img = prediction_result["processed_image"]

    st.markdown("### 🎯 Prediction Results")
    
    col1, col2 = st.columns([1, 1.4])

    with col1:
        # Hero card for predicted digit
        conf_pct = confidence * 100
        st.markdown(
            f"""
            <div class="prediction-hero">
                <div style="font-size: 0.95rem; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: 1.5px;">
                    Predicted Digit
                </div>
                <div class="prediction-number">{pred_digit}</div>
                <div class="confidence-label">Confidence: {conf_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.progress(min(1.0, float(confidence)), text=f"Model Certainty: {conf_pct:.1f}%")

        # Educational 28x28 normalized image preview
        st.markdown("#### 🔬 Network Input (28 × 28)")
        st.caption("This is the exact centered, inverted, aspect-ratio-preserved 28x28 grayscale image fed into the CNN:")
        if processed_img is not None:
            st.image(
                processed_img.resize((160, 160), Image.Resampling.NEAREST),
                caption="Preprocessed MNIST-compatible Input",
                use_container_width=False,
            )

    with col2:
        st.markdown("#### 📊 Class Probability Distribution")
        
        # Format chart data
        digits = list(range(10))
        scores = [probabilities.get(d, 0.0) * 100 for d in digits]
        
        chart_data = {
            "Digit": [str(d) for d in digits],
            "Probability (%)": scores,
        }
        
        # Display top 3 predictions
        sorted_probs = prediction_result.get("sorted_probabilities", [])
        if sorted_probs:
            top_cols = st.columns(3)
            for idx, (d, prob) in enumerate(sorted_probs[:3]):
                with top_cols[idx]:
                    rank_label = ["🥇 Top 1", "🥈 Top 2", "🥉 Top 3"][idx]
                    st.metric(label=f"{rank_label} (Digit {d})", value=f"{prob * 100:.1f}%")

        st.bar_chart(
            chart_data,
            x="Digit",
            y="Probability (%)",
            color="#8b5cf6",
            use_container_width=True,
        )


def main():
    render_sidebar()
    render_header()

    # Main workflow tabs
    tab_interact, tab_eval, tab_info = st.tabs([
        "✍️ Recognize Digits",
        "📈 Model Evaluation & Metrics",
        "📚 How It Works & Architecture",
    ])

    with tab_interact:
        input_col, result_col = st.columns([1.1, 1], gap="large")

        prediction_input = None

        with input_col:
            st.markdown("#### 🎨 Choose Input Method")
            input_mode = st.radio(
                "Select input format:",
                ["Draw on Canvas", "Upload Image File"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if input_mode == "Draw on Canvas":
                st.caption("Draw any digit from **0 to 9** inside the canvas box below:")
                
                # Check if streamlit-drawable-canvas is available
                try:
                    from streamlit_drawable_canvas import st_canvas

                    canvas_stroke_width = st.slider("Stroke Width", min_value=8, max_value=30, value=18, step=2)
                    
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 255, 255, 0.0)",
                        stroke_width=canvas_stroke_width,
                        stroke_color="#ffffff",
                        background_color="#000000",
                        height=280,
                        width=280,
                        drawing_mode="freedraw",
                        return_image_data=True,
                        key="digit_canvas",
                    )

                    if canvas_result is not None and canvas_result.image_data is not None:
                        # Only set input if user has drawn something
                        rgba_data = canvas_result.image_data
                        if np.any(rgba_data[:, :, :3] > 0):
                            prediction_input = rgba_data

                except ImportError:
                    st.error("`streamlit-drawable-canvas` is not installed. Please install it using `pip install streamlit-drawable-canvas`.")
                    st.info("You can still use the **Upload Image File** tab!")

            else:
                st.caption("Upload a photograph or scan of a handwritten digit (PNG, JPG, JPEG):")
                uploaded_file = st.file_uploader(
                    "Upload digit image",
                    type=["png", "jpg", "jpeg"],
                    label_visibility="collapsed",
                )

                if uploaded_file is not None:
                    upload_img = Image.open(uploaded_file)
                    st.image(upload_img, caption="Uploaded Original Image", width=180)
                    prediction_input = upload_img

            st.write("")
            predict_clicked = st.button("🔮 Predict Digit", type="primary")

        with result_col:
            if predict_clicked:
                if prediction_input is None:
                    st.warning("Please draw a digit or upload an image before clicking Predict.")
                else:
                    with st.spinner("Analyzing strokes and running CNN inference..."):
                        pred_result = predict_digit(prediction_input)
                        
                        if pred_result["success"]:
                            render_prediction_results(pred_result)
                        else:
                            st.error(f"Prediction Error: {pred_result.get('error')}")
            else:
                st.info("👈 Draw a digit on the left or upload an image, then click **Predict Digit** to view classification and probabilities.")
                
                # If a model exists, show a quick snapshot of the model ready state
                if is_model_available():
                    st.markdown(
                        """
                        <div class="glass-card">
                            <h4 style="margin-top:0; color:#38bdf8;">⚡ High-Accuracy CNN Ready</h4>
                            <p style="color:#94a3b8; font-size:0.9rem;">
                                The model has been trained on the official 60,000 MNIST training images.
                                Advanced preprocessing handles automatic inversion, background subtraction,
                                and centering to match MNIST test conditions.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    with tab_eval:
        st.markdown("### 📊 Model Evaluation on Official MNIST Test Set")
        st.caption("Untouched benchmark evaluation over 10,000 official test samples.")

        metrics = load_test_metrics_data()

        if metrics:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0) * 100:.2f}%")
            with m2:
                st.metric("Test Loss", f"{metrics.get('test_loss', 0):.4f}")
            with m3:
                epochs_val = metrics.get("epochs_trained", "N/A")
                st.metric("Trained Epochs", f"{epochs_val}")
            with m4:
                clf_report = metrics.get("classification_report", {})
                weighted_f1 = clf_report.get("weighted avg", {}).get("f1-score", 0)
                st.metric("Weighted F1-Score", f"{weighted_f1:.4f}" if weighted_f1 else "N/A")

            st.markdown("---")

            col_plot1, col_plot2 = st.columns(2)

            with col_plot1:
                st.markdown("#### 🎯 Confusion Matrix")
                if CONFUSION_MATRIX_PLOT_PATH.exists():
                    st.image(str(CONFUSION_MATRIX_PLOT_PATH), use_container_width=True)
                else:
                    st.info("Run `python evaluate.py` to generate the confusion matrix plot.")

            with col_plot2:
                st.markdown("#### 📉 Training & Validation Learning Curves")
                if TRAINING_HISTORY_PLOT_PATH.exists():
                    st.image(str(TRAINING_HISTORY_PLOT_PATH), use_container_width=True)
                else:
                    st.info("Run `python train.py` to generate training history curves.")

            # Tabular breakdown
            if "classification_report" in metrics:
                st.markdown("#### 📋 Per-Digit Precision, Recall, and F1-Scores")
                clf_data = []
                for d in range(10):
                    str_d = str(d)
                    if str_d in metrics["classification_report"]:
                        info = metrics["classification_report"][str_d]
                        clf_data.append({
                            "Digit": d,
                            "Precision": f"{info['precision']:.4f}",
                            "Recall": f"{info['recall']:.4f}",
                            "F1-Score": f"{info['f1-score']:.4f}",
                            "Test Samples": int(info["support"]),
                        })
                st.dataframe(clf_data, use_container_width=True)

        else:
            st.warning("Evaluation metrics have not been generated yet.")
            st.markdown(
                """
                To evaluate the model and produce report visualizations, run:
                ```cmd
                python train.py
                python evaluate.py
                ```
                Once complete, this tab will automatically populate with verified test metrics, confusion matrix, and learning curves.
                """
            )

    with tab_info:
        st.markdown("### 🧠 How It Works")
        st.markdown(
            """
            This project implements an end-to-end Machine Learning pipeline for handwritten digit recognition using
            Deep Convolutional Neural Networks (CNNs).

            #### 1. Image Preprocessing Pipeline
            Raw user inputs (canvas strokes or uploaded images) rarely look like standard MNIST samples.
            Our robust preprocessing module transforms arbitrary inputs to match MNIST characteristics:
            - **Grayscale Conversion**: Eliminates irrelevant color variations.
            - **Background Detection & Inversion**: MNIST digits are white strokes on a black background.
              The pipeline detects the border background level and automatically inverts light backgrounds.
            - **Bounding Box Crop**: Isolates the digit strokes from excessive padding.
            - **Aspect-Ratio Preserved Resizing**: Resizes the bounding box so the maximum dimension fits in a 20×20 box,
              matching the official NIST/MNIST specification.
            - **Centering on 28×28 Canvas**: Places the 20×20 digit centered on a 28×28 canvas.
            - **Pixel Normalization**: Converts pixel intensities from `[0, 255]` to `[0.0, 1.0]`.

            #### 2. Convolutional Neural Network (CNN) Architecture
            Unlike simple Multilayer Perceptrons (MLPs), CNNs exploit spatial structure and translational invariance:
            - **Convolutional Layers (Conv2D)**: Apply 3×3 learnable filter kernels to extract edges, curves, and loops.
            - **Batch Normalization**: Stabilizes and speeds up network convergence.
            - **Max Pooling (MaxPooling2D)**: Halves spatial dimensions (from 28×28 down to 14×14 and 7×7),
              retaining dominant activations while providing scale invariance.
            - **Dropout Layers (0.25 & 0.50)**: Randomly deactivate neurons during training to prevent co-adaptation and overfitting.
            - **Dense Classifier**: Flattens feature maps and computes dense cross-feature activations.
            - **Softmax Layer**: Outputs a normalized probability distribution across the 10 digit classes (0–9).

            #### 3. Key Differences Between Confidence and Test Accuracy
            - **Prediction Confidence**: The Softmax probability assigned to the top class for a single input image.
              It reflects the model's certainty on that specific input.
            - **Test Accuracy**: The percentage of correct classifications across 10,000 independent, unseen test images
              from the MNIST test benchmark.
            """
        )


if __name__ == "__main__":
    main()
