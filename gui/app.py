"""
gui/app.py
==========
Interactive Gradio web interface for Structural Damage Image Classification.

Features:
- Image upload (JPG/PNG) and preview resized to 224x224.
- Model selector: 6 models + 'All models' option with side-by-side view.
- Color-coded output: Green for UNDAMAGED (ok), Red for DAMAGED (broken).
- Confidence/probability metrics for each model.
- Majority-vote consensus summary in 'All models' mode.
- Model validation accuracies displayed beside model names.
- Prominent disclaimer note regarding smoke test limitations.
- Lazy model loading to minimize startup latency and RAM usage.
- Robust handling of edge cases (None, grayscale, RGBA, corrupt images).
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set thread and logging limits before TensorFlow / NumPy imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
from PIL import Image
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"
SAVED_MODELS_DIR = PROJECT_ROOT / "gui" / "saved_models"
MODEL_INFO_FILE = SAVED_MODELS_DIR / "model_info.json"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "gui" / "sample_images"

# Global lazy model cache
_LOADED_MODELS: Dict[str, Any] = {}
_MODEL_METADATA: Optional[Dict[str, Any]] = None

MODEL_KEYS = [
    "knn",
    "logistic_regression",
    "svm_rbf",
    "mobilenetv1",
    "inceptionv3",
    "inception_svm_hybrid",
]

DEFAULT_DISPLAY_NAMES = {
    "knn": "KNN (k=5)",
    "logistic_regression": "Logistic Regression (L2, C=1.0)",
    "svm_rbf": "SVM RBF (C=1.0, gamma=0.001)",
    "mobilenetv1": "MobileNetV1 (Transfer Learning)",
    "inceptionv3": "InceptionV3 (Transfer Learning)",
    "inception_svm_hybrid": "InceptionV3 (Layer 288) + SVM Hybrid",
}


def load_model_info() -> Dict[str, Any]:
    global _MODEL_METADATA
    if _MODEL_METADATA is not None:
        return _MODEL_METADATA

    if MODEL_INFO_FILE.exists():
        try:
            with open(MODEL_INFO_FILE, "r", encoding="utf-8") as f:
                _MODEL_METADATA = json.load(f)
                return _MODEL_METADATA
        except Exception as e:
            print(f"Warning: Failed to parse {MODEL_INFO_FILE}: {e}")

    _MODEL_METADATA = {
        "samples": 150,
        "models": {k: {"display_name": v, "validation_accuracy": 0.0} for k, v in DEFAULT_DISPLAY_NAMES.items()},
    }
    return _MODEL_METADATA


def get_model_choices() -> List[str]:
    info = load_model_info()
    models_dict = info.get("models", {})
    choices = ["All models (compare all side by side)"]
    for key in MODEL_KEYS:
        m_info = models_dict.get(key, {})
        name = m_info.get("display_name", DEFAULT_DISPLAY_NAMES.get(key, key))
        val_acc = m_info.get("validation_accuracy", None)
        if val_acc is not None:
            label = f"{name} [Val Acc: {val_acc*100:.1f}%]"
        else:
            label = name
        choices.append(label)
    return choices


def key_from_choice(choice: str) -> str:
    choice_lower = choice.lower()
    if "all models" in choice_lower:
        return "all"
    if "knn" in choice_lower:
        return "knn"
    if "logistic" in choice_lower:
        return "logistic_regression"
    if "hybrid" in choice_lower or "288" in choice_lower:
        return "inception_svm_hybrid"
    if "svm" in choice_lower:
        return "svm_rbf"
    if "mobilenet" in choice_lower:
        return "mobilenetv1"
    if "inception" in choice_lower:
        return "inceptionv3"
    return "all"


def get_lazy_model(model_key: str) -> Any:
    """
    Loads a model only when requested, caching it in memory.
    """
    if model_key in _LOADED_MODELS:
        return _LOADED_MODELS[model_key]

    if not SAVED_MODELS_DIR.exists():
        raise FileNotFoundError(
            f"Saved models directory not found at {SAVED_MODELS_DIR}. "
            "Please run 'python gui/train_and_save.py' first."
        )

    print(f"[LazyLoader] Loading model: {model_key}...")

    if model_key in ["knn", "logistic_regression", "svm_rbf"]:
        filename = f"{model_key}.joblib"
        path = SAVED_MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} not found. Run train_and_save.py first.")
        bundle = joblib.load(path)
        _LOADED_MODELS[model_key] = bundle
        return bundle

    elif model_key in ["mobilenetv1", "inceptionv3"]:
        import tensorflow as tf  # Lazy import
        filename = f"{model_key}.keras"
        path = SAVED_MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} not found. Run train_and_save.py first.")
        keras_model = tf.keras.models.load_model(path)
        _LOADED_MODELS[model_key] = keras_model
        return keras_model

    elif model_key == "inception_svm_hybrid":
        import tensorflow as tf  # Lazy import
        from tensorflow.keras import layers, models

        path = SAVED_MODELS_DIR / "inception_svm_hybrid.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} not found. Run train_and_save.py first.")
        bundle = joblib.load(path)

        # Lazy construct InceptionV3 layer 288 feature extractor
        if "inception_feature_extractor" not in _LOADED_MODELS:
            print("[LazyLoader] Building InceptionV3 layer 288 feature extractor...")
            full_inception = tf.keras.applications.InceptionV3(
                include_top=False, weights="imagenet", input_shape=(224, 224, 3)
            )
            target_layer = full_inception.layers[bundle.get("layer_idx", 288)]
            feature_output = layers.GlobalAveragePooling2D()(target_layer.output)
            extractor = models.Model(inputs=full_inception.input, outputs=feature_output)
            _LOADED_MODELS["inception_feature_extractor"] = extractor

        bundle["feature_extractor"] = _LOADED_MODELS["inception_feature_extractor"]
        _LOADED_MODELS[model_key] = bundle
        return bundle

    else:
        raise ValueError(f"Unknown model key: {model_key}")


def preprocess_uploaded_image(
    image_input: Union[None, str, Path, Image.Image, np.ndarray]
) -> Tuple[Image.Image, np.ndarray]:
    """
    Robust image preprocessing according to spec:
    1. Validate input is not None/empty.
    2. Convert to RGB (handles grayscale, RGBA, CMYK, Palette cleanly).
    3. Resize to 224x224.
    4. Convert to float32 and normalize: x = (x / 128.0) - 1.0 (range [-1, 1)).
    Returns:
      (preview_image: PIL.Image of size 224x224, norm_array: np.ndarray shape (224, 224, 3))
    """
    if image_input is None:
        raise ValueError("No image provided. Please upload an image file (JPG or PNG).")

    pil_img: Optional[Image.Image] = None

    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image path does not exist: {p}")
        try:
            pil_img = Image.open(p)
            pil_img.load()
        except Exception as e:
            raise ValueError(f"Could not open image file ({p.name}): {e}")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        try:
            # Handle grayscale or RGBA array
            if image_input.ndim == 2:
                pil_img = Image.fromarray(image_input, mode="L")
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                pil_img = Image.fromarray(image_input, mode="RGBA")
            elif image_input.ndim == 3 and image_input.shape[2] == 1:
                pil_img = Image.fromarray(image_input[:, :, 0], mode="L")
            else:
                pil_img = Image.fromarray(image_input.astype(np.uint8), mode="RGB")
        except Exception as e:
            raise ValueError(f"Invalid numpy image array: {e}")
    elif isinstance(image_input, (bytes, io.BytesIO)):
        stream = io.BytesIO(image_input) if isinstance(image_input, bytes) else image_input
        pil_img = Image.open(stream)
        pil_img.load()
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    if pil_img is None:
        raise ValueError("Failed to load image.")

    # Convert safely to RGB (removes alpha channel, converts grayscale to 3 identical channels)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize to 224x224
    resized_preview = pil_img.resize((224, 224), Image.Resampling.BILINEAR)

    # Convert to float32 [0.0, 255.0]
    img_float = np.array(resized_preview, dtype=np.float32)

    # Spec: x = (x / 128.0) - 1.0 (range [-1, 1))
    norm_array = (img_float / 128.0) - 1.0

    return resized_preview, norm_array


def predict_for_model(norm_array: np.ndarray, model_key: str) -> Dict[str, Any]:
    """
    Performs inference for a single model on a normalized (224, 224, 3) image.
    Returns structured result dict.
    """
    info = load_model_info()
    model_meta = info.get("models", {}).get(model_key, {})
    display_name = model_meta.get("display_name", DEFAULT_DISPLAY_NAMES.get(model_key, model_key))
    val_acc = model_meta.get("validation_accuracy", 0.0)

    # Ensure shape (1, 224, 224, 3)
    batch_img = np.expand_dims(norm_array, axis=0)

    model_obj = get_lazy_model(model_key)

    if model_key in ["knn", "logistic_regression", "svm_rbf"]:
        clf = model_obj["model"]
        scaler = model_obj["scaler"]
        x_flat = batch_img.reshape(1, -1).astype(np.float32)
        x_scaled = scaler.transform(x_flat)

        if model_key == "knn":
            pred = int(clf.predict(x_scaled.astype(np.float64))[0])
            probs = clf.predict_proba(x_scaled.astype(np.float64))[0]
            conf_val = float(probs[pred])
            votes = int(round(conf_val * 5))
            metric_type = "Neighbor vote share"
            metric_detail = f"{conf_val*100:.1f}% ({votes}/5 neighbor votes)"
            prob_damaged = float(probs[1]) if len(probs) > 1 else (1.0 if pred == 1 else 0.0)

        elif model_key == "logistic_regression":
            pred = int(clf.predict(x_scaled)[0])
            probs = clf.predict_proba(x_scaled)[0]
            conf_val = float(probs[pred])
            metric_type = "Sigmoid probability"
            metric_detail = f"{conf_val*100:.1f}% (P(Damaged) = {probs[1]*100:.1f}%)"
            prob_damaged = float(probs[1])

        else:  # svm_rbf
            pred = int(clf.predict(x_scaled)[0])
            dec_score = float(clf.decision_function(x_scaled)[0])
            metric_type = "decision_function"
            metric_detail = f"Score: {dec_score:+.4f} (threshold: 0.0, margin distance)"
            # Approximate logistic sigmoid probability for reporting
            prob_damaged = float(1.0 / (1.0 + np.exp(-dec_score)))

    elif model_key in ["mobilenetv1", "inceptionv3"]:
        keras_model = model_obj
        raw_preds = keras_model.predict(batch_img, verbose=0)[0]
        pred = int(np.argmax(raw_preds))
        conf_val = float(raw_preds[pred])
        metric_type = "Softmax probability"
        prob_damaged = float(raw_preds[1]) if len(raw_preds) > 1 else float(raw_preds[0])
        metric_detail = f"{conf_val*100:.1f}% (P(Damaged) = {prob_damaged*100:.1f}%)"

    elif model_key == "inception_svm_hybrid":
        svm = model_obj["model"]
        scaler = model_obj["scaler"]
        extractor = model_obj["feature_extractor"]

        features = extractor.predict(batch_img, verbose=0)
        feats_scaled = scaler.transform(features)
        pred = int(svm.predict(feats_scaled)[0])
        dec_score = float(svm.decision_function(feats_scaled)[0])
        metric_type = "decision_function"
        metric_detail = f"Score: {dec_score:+.4f} (threshold: 0.0, deep feature margin)"
        prob_damaged = float(1.0 / (1.0 + np.exp(-dec_score)))

    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    is_damaged = (pred == 1)
    label_text = "DAMAGED (broken)" if is_damaged else "UNDAMAGED (ok)"
    color = "#dc3545" if is_damaged else "#198754"

    return {
        "model_key": model_key,
        "display_name": display_name,
        "val_accuracy": val_acc,
        "prediction_int": pred,
        "is_damaged": is_damaged,
        "label": label_text,
        "color": color,
        "metric_type": metric_type,
        "metric_detail": metric_detail,
        "prob_damaged": prob_damaged,
    }


def format_single_result_html(res: Dict[str, Any]) -> str:
    color = res["color"]
    label = res["label"]
    name = res["display_name"]
    val_acc = res["val_accuracy"]
    metric_type = res["metric_type"]
    metric_detail = res["metric_detail"]

    badge_bg = "#fee2e2" if res["is_damaged"] else "#dcfce7"
    badge_border = "#ef4444" if res["is_damaged"] else "#22c55e"
    badge_text = "#991b1b" if res["is_damaged"] else "#166534"
    icon = "⚠️" if res["is_damaged"] else "✅"

    html = f"""
    <div style="border: 2px solid {badge_border}; border-radius: 12px; padding: 20px; background: #ffffff; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); font-family: system-ui, -apple-system, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e5e7eb; padding-bottom: 12px; margin-bottom: 16px;">
            <h3 style="margin: 0; color: #111827; font-size: 1.25rem; font-weight: 600;">{name}</h3>
            <span style="background: #f3f4f6; color: #374151; padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; font-weight: 500;">
                Val Acc: {val_acc*100:.1f}%
            </span>
        </div>
        
        <div style="background: {badge_bg}; border: 1.5px solid {badge_border}; border-radius: 8px; padding: 14px 18px; text-align: center; margin-bottom: 16px;">
            <span style="font-size: 1.35rem; font-weight: 700; color: {badge_text};">
                {icon} {label}
            </span>
        </div>

        <div style="background: #f9fafb; border-radius: 8px; padding: 12px 16px; font-size: 0.95rem; color: #374151;">
            <div style="margin-bottom: 6px;">
                <strong>Confidence / Metric ({metric_type}):</strong>
            </div>
            <div style="font-family: monospace; font-size: 1.05rem; color: #111827;">
                {metric_detail}
            </div>
        </div>
    </div>
    """
    return html


def format_all_results_html(results: List[Dict[str, Any]]) -> str:
    n_damaged = sum(1 for r in results if r["is_damaged"])
    n_undamaged = len(results) - n_damaged

    if n_damaged > n_undamaged:
        consensus_label = "DAMAGED (broken)"
        consensus_bg = "#fee2e2"
        consensus_border = "#ef4444"
        consensus_text = "#991b1b"
        icon = "⚠️"
    elif n_undamaged > n_damaged:
        consensus_label = "UNDAMAGED (ok)"
        consensus_bg = "#dcfce7"
        consensus_border = "#22c55e"
        consensus_text = "#166534"
        icon = "✅"
    else:
        consensus_label = "TIE (Equally Split)"
        consensus_bg = "#fef3c7"
        consensus_border = "#f59e0b"
        consensus_text = "#92400e"
        icon = "⚖️"

    rows_html = ""
    for r in results:
        bg = "#fef2f2" if r["is_damaged"] else "#f0fdf4"
        txt = "#991b1b" if r["is_damaged"] else "#166534"
        badge = f"""<span style="background: {bg}; color: {txt}; padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 0.9rem;">{r['label']}</span>"""
        rows_html += f"""
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 10px 12px; font-weight: 600; color: #111827;">{r['display_name']}</td>
            <td style="padding: 10px 12px; text-align: center; color: #4b5563;">{r['val_accuracy']*100:.1f}%</td>
            <td style="padding: 10px 12px; text-align: center;">{badge}</td>
            <td style="padding: 10px 12px; font-family: monospace; font-size: 0.85rem; color: #374151;">{r['metric_detail']}</td>
        </tr>
        """

    html = f"""
    <div style="font-family: system-ui, -apple-system, sans-serif; background: #ffffff; border-radius: 12px; border: 1px solid #e5e7eb; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
        <!-- Majority Vote Banner -->
        <div style="background: {consensus_bg}; border: 2px solid {consensus_border}; border-radius: 10px; padding: 14px 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; color: #4b5563; display: block;">Majority-Vote Consensus</span>
                <span style="font-size: 1.4rem; font-weight: 800; color: {consensus_text};">{icon} {consensus_label}</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 600; color: #1f2937; background: #ffffff; padding: 8px 14px; border-radius: 8px; border: 1px solid #e5e7eb;">
                <span style="color: #dc2525;">Damaged: <strong>{n_damaged}/6</strong></span> &nbsp;|&nbsp; 
                <span style="color: #16a34a;">Undamaged: <strong>{n_undamaged}/6</strong></span>
            </div>
        </div>

        <!-- Comparison Table -->
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; text-align: left;">
                <thead>
                    <tr style="background: #f8fafc; border-bottom: 2px solid #cbd5e1;">
                        <th style="padding: 10px 12px; font-size: 0.85rem; color: #475569; text-transform: uppercase;">Model</th>
                        <th style="padding: 10px 12px; font-size: 0.85rem; color: #475569; text-transform: uppercase; text-align: center;">Val Acc</th>
                        <th style="padding: 10px 12px; font-size: 0.85rem; color: #475569; text-transform: uppercase; text-align: center;">Prediction</th>
                        <th style="padding: 10px 12px; font-size: 0.85rem; color: #475569; text-transform: uppercase;">Confidence / Metric</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    """
    return html


def classify_image(
    image_input: Any, model_choice: str
) -> Tuple[Optional[Image.Image], str]:
    """
    Main prediction callback for Gradio and standalone tests.
    """
    try:
        preview_img, norm_arr = preprocess_uploaded_image(image_input)
    except Exception as e:
        error_html = f"""
        <div style="background: #fee2e2; border: 1.5px solid #ef4444; border-radius: 8px; padding: 16px; color: #991b1b; font-family: system-ui, sans-serif;">
            <h4 style="margin: 0 0 6px 0; font-size: 1.05rem;">❌ Input Error</h4>
            <p style="margin: 0;">{str(e)}</p>
        </div>
        """
        return None, error_html

    chosen_key = key_from_choice(model_choice)

    try:
        if chosen_key == "all":
            results = []
            for k in MODEL_KEYS:
                r = predict_for_model(norm_arr, k)
                results.append(r)
            output_html = format_all_results_html(results)
        else:
            res = predict_for_model(norm_arr, chosen_key)
            output_html = format_single_result_html(res)

        return preview_img, output_html

    except Exception as e:
        error_html = f"""
        <div style="background: #fee2e2; border: 1.5px solid #ef4444; border-radius: 8px; padding: 16px; color: #991b1b; font-family: system-ui, sans-serif;">
            <h4 style="margin: 0 0 6px 0; font-size: 1.05rem;">❌ Prediction Failure</h4>
            <p style="margin: 0;">{str(e)}</p>
        </div>
        """
        return preview_img, error_html


def build_app() -> gr.Blocks:
    """
    Constructs the Gradio application layout.
    """
    choices = get_model_choices()

    sample_examples = []
    if SAMPLE_IMAGES_DIR.exists():
        for fn in sorted(os.listdir(SAMPLE_IMAGES_DIR)):
            if fn.endswith((".png", ".jpg", ".jpeg")):
                sample_examples.append(str(SAMPLE_IMAGES_DIR / fn))

    with gr.Blocks(
        title="Structural Damage Classifier",
    ) as demo:
        gr.Markdown(
            """
            # 🏗️ Structural Damage Image Classification
            ### PEER Hub ImageNet Benchmark — Six Model Pipeline Demo
            """
        )

        # Required Honest Limitation Banner
        gr.HTML(
            """
            <div style="background: #fffbeb; border-left: 5px solid #f59e0b; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px; font-family: system-ui, sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.3rem;">⚠️</span>
                    <div>
                        <strong style="color: #b45309; font-size: 0.95rem;">DISCLAIMER & DEMO NOTE:</strong>
                        <p style="margin: 3px 0 0 0; color: #92400e; font-size: 0.9rem; line-height: 1.4;">
                            These models were trained on only <strong>150 images (smoke test)</strong>, so predictions are unreliable. 
                            This is a demo of the pipeline, not a real damage detector.
                        </p>
                    </div>
                </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Structural Image (JPG or PNG)",
                    sources=["upload", "clipboard"],
                )
                model_dropdown = gr.Dropdown(
                    choices=choices,
                    value=choices[0],
                    label="Select Model",
                    interactive=True,
                )
                classify_btn = gr.Button("🔍 Classify Damage State", variant="primary", size="lg")

                if sample_examples:
                    gr.Markdown("#### Sample Validation Images (Click to Test):")
                    gr.Examples(
                        examples=sample_examples,
                        inputs=image_input,
                        label="Sample Dataset Images",
                    )

            with gr.Column(scale=1):
                preview_output = gr.Image(
                    type="pil",
                    label="224 × 224 Preprocessed Preview (Input to Model)",
                    interactive=False,
                    height=240,
                )
                result_output = gr.HTML(
                    value="""
                    <div style="background: #f9fafb; border: 1px dashed #d1d5db; border-radius: 8px; padding: 30px; text-align: center; color: #6b7280; font-family: system-ui, sans-serif;">
                        Upload an image and click <strong>Classify Damage State</strong> to view predictions.
                    </div>
                    """
                )

        classify_btn.click(
            fn=classify_image,
            inputs=[image_input, model_dropdown],
            outputs=[preview_output, result_output],
        )

    return demo


def main():
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        theme=gr.themes.Soft(primary_hue="blue"),
    )


if __name__ == "__main__":
    main()
