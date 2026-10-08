"""
gui/app.py
==========
Interactive Gradio web interface for Structural Damage Image Classification.
Plain, student-built aesthetic: simple two-column layout, neutral grays,
plain text results, and minimal styling.
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
    "knn": "KNN",
    "logistic_regression": "Logistic Regression",
    "svm_rbf": "SVM",
    "mobilenetv1": "MobileNetV1",
    "inceptionv3": "InceptionV3",
    "inception_svm_hybrid": "InceptionV3 layer 288 + SVM",
}

MODEL_CHOICES = [
    "KNN",
    "Logistic Regression",
    "SVM",
    "MobileNetV1",
    "InceptionV3",
    "InceptionV3 layer 288 + SVM",
    "All models",
]

CUSTOM_CSS = """
footer { display: none !important; }
.gradio-container {
    max-width: 860px !important;
    margin: 24px auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}
"""


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


def key_from_choice(choice: str) -> str:
    choice_lower = choice.lower()
    if "all models" in choice_lower:
        return "all"
    if "knn" in choice_lower:
        return "knn"
    if "logistic" in choice_lower:
        return "logistic_regression"
    if "288" in choice_lower or "hybrid" in choice_lower:
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
    Image preprocessing:
    1. Validate input is not None.
    2. Convert to RGB (handles grayscale, RGBA).
    3. Resize to 224x224.
    4. Convert to float32 and normalize: x = (x / 128.0) - 1.0 (range [-1, 1)).
    """
    if image_input is None:
        raise ValueError("Please upload an image first.")

    pil_img: Optional[Image.Image] = None

    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image path does not exist: {p}")
        try:
            pil_img = Image.open(p)
            pil_img.load()
        except Exception:
            raise ValueError("Could not open image file.")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        try:
            if image_input.ndim == 2:
                pil_img = Image.fromarray(image_input, mode="L")
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                pil_img = Image.fromarray(image_input, mode="RGBA")
            elif image_input.ndim == 3 and image_input.shape[2] == 1:
                pil_img = Image.fromarray(image_input[:, :, 0], mode="L")
            else:
                pil_img = Image.fromarray(image_input.astype(np.uint8), mode="RGB")
        except Exception:
            raise ValueError("Invalid image array.")
    elif isinstance(image_input, (bytes, io.BytesIO)):
        try:
            stream = io.BytesIO(image_input) if isinstance(image_input, bytes) else image_input
            pil_img = Image.open(stream)
            pil_img.load()
        except Exception:
            raise ValueError("Could not open image file.")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    if pil_img is None:
        raise ValueError("Please upload an image first.")

    # Convert to RGB (removes alpha channel, replicates grayscale to 3 channels)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    resized_preview = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_float = np.array(resized_preview, dtype=np.float32)
    norm_array = (img_float / 128.0) - 1.0

    return resized_preview, norm_array


def predict_for_model(norm_array: np.ndarray, model_key: str) -> Dict[str, Any]:
    """
    Performs inference for a single model on a normalized (224, 224, 3) image.
    """
    info = load_model_info()
    model_meta = info.get("models", {}).get(model_key, {})
    display_name = DEFAULT_DISPLAY_NAMES.get(model_key, model_key)
    val_acc = model_meta.get("validation_accuracy", 0.0)

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
            metric_detail = f"{int(round(conf_val * 100))}% ({votes}/5 votes)"
            prob_damaged = float(probs[1]) if len(probs) > 1 else (1.0 if pred == 1 else 0.0)

        elif model_key == "logistic_regression":
            pred = int(clf.predict(x_scaled)[0])
            probs = clf.predict_proba(x_scaled)[0]
            conf_val = float(probs[pred])
            metric_detail = f"{int(round(conf_val * 100))}%"
            prob_damaged = float(probs[1])

        else:  # svm_rbf
            pred = int(clf.predict(x_scaled)[0])
            dec_score = float(clf.decision_function(x_scaled)[0])
            metric_detail = f"{dec_score:+.2f} (decision function)"
            prob_damaged = float(1.0 / (1.0 + np.exp(-dec_score)))

    elif model_key in ["mobilenetv1", "inceptionv3"]:
        keras_model = model_obj
        raw_preds = keras_model.predict(batch_img, verbose=0)[0]
        pred = int(np.argmax(raw_preds))
        conf_val = float(raw_preds[pred])
        metric_detail = f"{int(round(conf_val * 100))}%"
        prob_damaged = float(raw_preds[1]) if len(raw_preds) > 1 else float(raw_preds[0])

    elif model_key == "inception_svm_hybrid":
        svm = model_obj["model"]
        scaler = model_obj["scaler"]
        extractor = model_obj["feature_extractor"]

        features = extractor.predict(batch_img, verbose=0)
        feats_scaled = scaler.transform(features)
        pred = int(svm.predict(feats_scaled)[0])
        dec_score = float(svm.decision_function(feats_scaled)[0])
        metric_detail = f"{dec_score:+.2f} (decision function)"
        prob_damaged = float(1.0 / (1.0 + np.exp(-dec_score)))

    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    is_damaged = (pred == 1)
    label_text = "Damaged" if is_damaged else "Undamaged"
    color = "#990000" if is_damaged else "#006600"

    return {
        "model_key": model_key,
        "display_name": display_name,
        "val_accuracy": val_acc,
        "prediction_int": pred,
        "is_damaged": is_damaged,
        "label": label_text,
        "color": color,
        "metric_detail": metric_detail,
        "prob_damaged": prob_damaged,
    }


def format_single_result_html(res: Dict[str, Any]) -> str:
    """
    Renders single model prediction as plain text with only the result word colored.
    """
    color = "#990000" if res["is_damaged"] else "#006600"
    word = "Damaged" if res["is_damaged"] else "Undamaged"
    conf_str = res["metric_detail"]
    val_acc = res["val_accuracy"]
    val_acc_pct = f"{val_acc * 100:.1f}%" if val_acc is not None else "N/A"

    html = f"""
    <div style="font-size: 15px; line-height: 1.8; color: #111; margin-top: 4px;">
        <div>Result: <span style="color: {color}; font-weight: bold;">{word}</span></div>
        <div>Confidence: {conf_str}</div>
        <div style="color: #666; font-size: 13px; margin-top: 6px;">Model validation accuracy: {val_acc_pct}</div>
    </div>
    """
    return html.strip()


def format_all_results_html(results: List[Dict[str, Any]]) -> str:
    """
    Renders simple table: Model | Prediction | Confidence | Val accuracy
    with plain count line underneath.
    """
    n_damaged = sum(1 for r in results if r["is_damaged"])
    total_models = len(results)

    rows_html = ""
    for r in results:
        color = "#990000" if r["is_damaged"] else "#006600"
        word = "Damaged" if r["is_damaged"] else "Undamaged"
        val_acc = r["val_accuracy"]
        val_acc_pct = f"{val_acc * 100:.1f}%" if val_acc is not None else "N/A"
        rows_html += f"""
        <tr style="border-bottom: 1px solid #e5e5e5;">
            <td style="padding: 7px 10px;">{r['display_name']}</td>
            <td style="padding: 7px 10px;"><span style="color: {color}; font-weight: bold;">{word}</span></td>
            <td style="padding: 7px 10px;">{r['metric_detail']}</td>
            <td style="padding: 7px 10px;">{val_acc_pct}</td>
        </tr>
        """

    html = f"""
    <div style="font-size: 14px; color: #111; margin-top: 4px;">
        <table style="width: 100%; border-collapse: collapse; text-align: left; margin-bottom: 10px;">
            <thead>
                <tr style="border-bottom: 1px solid #ccc; background: #f9f9f9;">
                    <th style="padding: 7px 10px; font-weight: 600;">Model</th>
                    <th style="padding: 7px 10px; font-weight: 600;">Prediction</th>
                    <th style="padding: 7px 10px; font-weight: 600;">Confidence</th>
                    <th style="padding: 7px 10px; font-weight: 600;">Val accuracy</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <p style="margin: 8px 0 0 0; font-size: 14px; color: #222;">{n_damaged} of {total_models} models say damaged.</p>
    </div>
    """
    return html.strip()


def classify_image(
    image_input: Any, model_choice: str
) -> Tuple[Optional[Image.Image], str]:
    """
    Prediction callback: handles validation, inference, and plain text formatting.
    """
    if image_input is None:
        return None, '<p style="color: #990000; margin: 0;">Please upload an image first.</p>'

    try:
        preview_img, norm_arr = preprocess_uploaded_image(image_input)
    except Exception as e:
        msg = str(e)
        if "No image provided" in msg:
            msg = "Please upload an image first."
        if not msg.endswith("."):
            msg += "."
        return None, f'<p style="color: #990000; margin: 0;">{msg}</p>'

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
        return preview_img, f'<p style="color: #990000; margin: 0;">Error during classification: {e}.</p>'


def build_app() -> gr.Blocks:
    """
    Constructs the plain, simple Gradio application layout.
    """
    with gr.Blocks(title="Structural Damage Classifier") as demo:
        gr.Markdown(
            """
            # Structural Damage Classifier
            Upload a photo of a structure to check whether it is damaged.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload image",
                    sources=["upload", "clipboard"],
                )
                model_dropdown = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="All models",
                    label="Model",
                    interactive=True,
                )
                classify_btn = gr.Button("Classify", variant="primary")
                preview_output = gr.Image(
                    type="pil",
                    label="224x224 preview",
                    interactive=False,
                    height=224,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Result")
                result_output = gr.HTML(
                    value="<p style='color: #666; margin: 0;'>Select an image and click Classify.</p>"
                )

        gr.HTML(
            """
            <div style="margin-top: 24px; padding-top: 12px; border-top: 1px solid #e5e5e5; font-size: 12px; color: #777;">
                Trained on a small sample (150 images), so results are not reliable. This is a demo of the pipeline.
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
        inbrowser=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(primary_hue="slate", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()
