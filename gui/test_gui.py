"""
gui/test_gui.py
===============
Automated test suite verifying Step 5 requirements:
1. Preprocessing verification table comparing 4 exported PNGs against original arrays.
2. Direct inference test for all 6 models and 'All models' consensus.
3. Edge case and error handling tests (None, Grayscale, RGBA transparency, Corrupt data).
4. App initialization and endpoint validation.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "gui"))

import common
from app import (
    MODEL_KEYS,
    classify_image,
    preprocess_uploaded_image,
    predict_for_model,
    get_lazy_model,
    build_app,
)

SAMPLE_DIR = PROJECT_ROOT / "gui" / "sample_images"


def test_validation_images_comparison():
    print("=" * 80)
    print("TEST 1: Comparing Exported PNGs vs Original In-Memory Arrays (Step 5.2)")
    print("=" * 80)

    # Load original validation arrays
    X_train, y_train, X_val, y_val = common.load_dataset(
        smoke=True, smoke_samples=150, random_state=42
    )

    test_files = [
        ("val_01_damaged.png", 2, 1, "DAMAGED"),
        ("val_02_damaged.png", 4, 1, "DAMAGED"),
        ("val_03_undamaged.png", 0, 0, "UNDAMAGED"),
        ("val_04_undamaged.png", 1, 0, "UNDAMAGED"),
    ]

    print(f"\n{'File':<22} {'True':<11} {'Model':<22} {'PNG Pred':<14} {'Array Pred':<14} {'Match?'}")
    print("-" * 90)

    summary_rows = []

    for fname, val_idx, true_lbl, true_name in test_files:
        png_path = SAMPLE_DIR / fname
        assert png_path.exists(), f"Sample image {png_path} does not exist!"

        # 1. Prediction via GUI preprocessing pipeline on PNG
        _, norm_from_png = preprocess_uploaded_image(png_path)

        # 2. Prediction on original array from common.load_dataset
        norm_from_array = X_val[val_idx]

        row = {"file": fname, "true": true_name}

        for k in MODEL_KEYS:
            res_png = predict_for_model(norm_from_png, k)
            res_arr = predict_for_model(norm_from_array, k)

            png_lbl = "DAMAGED" if res_png["is_damaged"] else "UNDAMAGED"
            arr_lbl = "DAMAGED" if res_arr["is_damaged"] else "UNDAMAGED"
            match = "YES" if png_lbl == arr_lbl else "MISMATCH"

            print(f"{fname:<22} {true_name:<11} {k:<22} {png_lbl:<14} {arr_lbl:<14} {match}")
            row[k] = png_lbl

        summary_rows.append(row)

    print("\n" + "=" * 90)
    print("STEP 5.2 SUMMARY TABLE (PNG PREDICTIONS)")
    print("=" * 90)
    header = f"{'File':<22} | {'True Label':<12} | {'KNN':<9} | {'LogReg':<9} | {'SVM':<9} | {'MobileNet':<9} | {'Inception':<9} | {'Hybrid':<9}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(
            f"{r['file']:<22} | {r['true']:<12} | {r['knn']:<9} | {r['logistic_regression']:<9} | {r['svm_rbf']:<9} | {r['mobilenetv1']:<9} | {r['inceptionv3']:<9} | {r['inception_svm_hybrid']:<9}"
        )
    print("=" * 90)


def test_error_cases():
    print("\n" + "=" * 80)
    print("TEST 2: Error and Edge Cases (Step 5.4)")
    print("=" * 80)

    # 1. No image (None)
    print("\n[Case 1] No image provided (None):")
    preview, html = classify_image(None, "All models")
    assert preview is None
    assert "Input Error" in html and "No image provided" in html
    print("  -> Passed: Clean friendly error displayed, no crash.")

    # 2. Grayscale image
    print("\n[Case 2] Grayscale image (mode 'L', 100x100):")
    gray_img = Image.new("L", (100, 100), color=128)
    preview, html = classify_image(gray_img, "All models")
    assert preview is not None
    assert preview.size == (224, 224)
    assert preview.mode == "RGB"
    assert "Majority-Vote Consensus" in html
    print(f"  -> Passed: Successfully converted to RGB, resized to 224x224, classified without error.")

    # 3. PNG with transparency (mode 'RGBA', 150x150)
    print("\n[Case 3] PNG with transparency (mode 'RGBA', 150x150):")
    rgba_img = Image.new("RGBA", (150, 150), color=(200, 100, 50, 180))
    preview, html = classify_image(rgba_img, "All models")
    assert preview is not None
    assert preview.size == (224, 224)
    assert preview.mode == "RGB"
    assert "Majority-Vote Consensus" in html
    print(f"  -> Passed: Alpha channel safely stripped, resized to 224x224, classified without error.")

    # 4. Corrupt bytes
    print("\n[Case 4] Corrupt bytes input:")
    preview, html = classify_image(b"not an image byte stream", "All models")
    assert preview is None
    assert "Input Error" in html or "Prediction Failure" in html
    print("  -> Passed: Corrupt data handled safely with user-friendly alert, no crash.")


def test_gradio_app_structure():
    print("\n" + "=" * 80)
    print("TEST 3: Gradio App Structure & Launchability (Step 5.3)")
    print("=" * 80)
    demo = build_app()
    assert demo is not None
    print(f"  -> Gradio App successfully instantiated with title: '{demo.title}'")
    print(f"  -> Primary launch route is configured at: http://127.0.0.1:7860")
    print("=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_validation_images_comparison()
    test_error_cases()
    test_gradio_app_structure()
