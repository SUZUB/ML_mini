"""
gui/run_gui.py
==============
Single-command launcher for the Structural Damage Classification GUI.

Usage:
  python gui/run_gui.py

Behavior:
1. Verifies that all 6 models and metadata exist in gui/saved_models/.
2. If missing or incomplete, automatically runs gui/train_and_save.py first.
3. Verifies sample validation images exist in gui/sample_images/.
4. Launches the Gradio web server and prints the local access URL.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Thread and logging management
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

GUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
SAVED_MODELS_DIR = GUI_DIR / "saved_models"
SAMPLE_IMAGES_DIR = GUI_DIR / "sample_images"

REQUIRED_ARTIFACTS = [
    "knn.joblib",
    "logistic_regression.joblib",
    "svm_rbf.joblib",
    "mobilenetv1.keras",
    "inceptionv3.keras",
    "inception_svm_hybrid.joblib",
    "model_info.json",
]


def check_models_complete() -> bool:
    if not SAVED_MODELS_DIR.exists():
        return False
    for filename in REQUIRED_ARTIFACTS:
        target = SAVED_MODELS_DIR / filename
        if not target.exists() or target.stat().st_size == 0:
            return False
    return True


def ensure_models():
    if not check_models_complete():
        print("=" * 70)
        print("  NOTICE: Saved models in gui/saved_models/ are missing or incomplete.")
        print("  Running train_and_save.py now to train all six models.")
        print("  This takes approximately 1 to 2 minutes on CPU. Please wait...")
        print("=" * 70 + "\n")

        train_script = GUI_DIR / "train_and_save.py"
        res = subprocess.run(
            [sys.executable, str(train_script), "--samples", "150"],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if res.returncode != 0:
            sys.exit(f"Error: Model training failed with exit code {res.returncode}.")
        print("\nAll models trained and saved successfully.\n")


def ensure_sample_images():
    if not SAMPLE_IMAGES_DIR.exists() or len(list(SAMPLE_IMAGES_DIR.glob("*.png"))) < 4:
        print("Exporting sample validation images for GUI testing...")
        export_script = GUI_DIR / "export_samples.py"
        subprocess.run([sys.executable, str(export_script)], cwd=PROJECT_ROOT, check=False)


def main():
    print("=" * 70)
    print("  STARTING STRUCTURAL DAMAGE CLASSIFIER GUI")
    print("=" * 70)

    # 1. Verify / prepare models
    ensure_models()

    # 2. Verify / prepare sample images
    ensure_sample_images()

    # 3. Import and launch app
    print("Starting Gradio Web Application...")
    import gradio as gr
    from app import build_app, CUSTOM_CSS

    demo = build_app()
    server_port = 7860
    server_name = "127.0.0.1"
    url = f"http://{server_name}:{server_port}"

    print(f"\n=======================================================")
    print(f"  Local GUI is running and accessible at:")
    print(f"  >> {url} <<")
    print(f"  Press Ctrl+C to stop the server.")
    print(f"=======================================================\n")

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        inbrowser=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(primary_hue="slate", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()
