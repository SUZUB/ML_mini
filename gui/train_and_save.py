"""
gui/train_and_save.py
=====================
Retrains the six structural damage classification models using the smoke-150 setup
and saves the trained weights/artifacts to gui/saved_models/.

Models:
1. KNN (k=5) + StandardScaler -> gui/saved_models/knn.joblib
2. Logistic Regression (L2, C=1.0) + StandardScaler -> gui/saved_models/logistic_regression.joblib
3. SVM RBF (C=1.0, gamma=0.001) + StandardScaler -> gui/saved_models/svm_rbf.joblib
4. MobileNetV1 Transfer Model -> gui/saved_models/mobilenetv1.keras
5. InceptionV3 Baseline Transfer Model -> gui/saved_models/inceptionv3.keras
6. InceptionV3 (Layer 288) + SVM Hybrid -> gui/saved_models/inception_svm_hybrid.joblib

Metadata:
- Saved to gui/saved_models/model_info.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# Configure threads and quiet TensorFlow logs before importing TF
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Avoid Python built-in 'code' package collision
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import common  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers, models, optimizers  # noqa: E402


REFERENCE_RESULTS_150 = {
    "knn": {"train": 0.6889, "val": 0.6667},
    "logistic_regression": {"train": 1.0000, "val": 0.5333},
    "svm_rbf": {"train": 1.0000, "val": 0.5333},
    "mobilenetv1": {"train": 0.7333, "val": 0.6667},
    "inceptionv3": {"train": 0.5852, "val": 0.5333},
    "inception_svm_hybrid": {"train": 0.8741, "val": 0.7333},
}


def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=0.001):
    base_model = tf.keras.applications.MobileNet(
        input_shape=input_shape,
        alpha=1.0,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV1_Transfer")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_inceptionv3_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=0.01):
    base_model = tf.keras.applications.InceptionV3(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="InceptionV3_Baseline")
    optimizer = optimizers.SGD(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_inception_feature_extractor(layer_idx=288, input_shape=(224, 224, 3)):
    full_inception = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    target_layer = full_inception.layers[layer_idx]
    feature_output = layers.GlobalAveragePooling2D()(target_layer.output)
    feature_extractor = models.Model(
        inputs=full_inception.input, outputs=feature_output, name="InceptionV3_Layer288_Extractor"
    )
    return feature_extractor, target_layer.name


def train_all_models(samples: int = 150) -> dict:
    saved_dir = PROJECT_ROOT / "gui" / "saved_models"
    saved_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  RETRAINING & SAVING 6 MODELS (Samples: {samples}, Seed: 42)")
    print("=" * 65)
    print(f"Target directory: {saved_dir}\n")

    # Step 1: Load dataset
    print(f"[1/7] Loading dataset with {samples} samples (stratified split)...")
    X_train, y_train, X_val, y_val = common.load_dataset(
        smoke=True,
        smoke_samples=samples,
        random_state=42,
    )
    n_train = len(y_train)
    n_val = len(y_val)
    print(f"  Training set:   {n_train} images (Damaged: {int(sum(y_train))}, Undamaged: {int(n_train - sum(y_train))})")
    print(f"  Validation set: {n_val} images (Damaged: {int(sum(y_val))}, Undamaged: {int(n_val - sum(y_val))})\n")

    # Prepare classical flattened features and fitted scaler
    print("Preparing classical flattened features...")
    X_tr_flat = X_train.reshape(n_train, -1).astype(np.float32)
    X_va_flat = X_val.reshape(n_val, -1).astype(np.float32)
    classical_scaler = StandardScaler()
    X_tr_scaled = classical_scaler.fit_transform(X_tr_flat).astype(np.float32)
    X_va_scaled = classical_scaler.transform(X_va_flat).astype(np.float32)

    results = {}

    # Model 1: KNN (k=5)
    print("\n--- Training Model 1: KNN (k=5) ---")
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, algorithm="brute")
    knn.fit(X_tr_scaled.astype(np.float64), y_train)
    knn_tr_acc = float(accuracy_score(y_train, knn.predict(X_tr_scaled.astype(np.float64))))
    knn_va_acc = float(accuracy_score(y_val, knn.predict(X_va_scaled.astype(np.float64))))
    knn_path = saved_dir / "knn.joblib"
    joblib.dump({"model": knn, "scaler": classical_scaler}, knn_path)
    t_elapsed = time.time() - t0
    results["knn"] = {
        "display_name": "KNN (k=5)",
        "train_accuracy": knn_tr_acc,
        "validation_accuracy": knn_va_acc,
        "filename": "knn.joblib",
        "type": "joblib",
        "confidence_type": "probability",
        "confidence_note": "k-NN neighbor vote share",
        "elapsed_seconds": round(t_elapsed, 2),
    }
    print(f"  Train Acc: {knn_tr_acc*100:6.2f}% | Val Acc: {knn_va_acc*100:6.2f}% | Saved: {knn_path.name}")
    del knn
    gc.collect()

    # Model 2: Logistic Regression (L2, C=1.0)
    print("\n--- Training Model 2: Logistic Regression (L2, C=1.0) ---")
    t0 = time.time()
    logreg = LogisticRegression(C=1.0, solver="lbfgs", max_iter=100, random_state=42)
    logreg.fit(X_tr_scaled, y_train)
    logreg_tr_acc = float(accuracy_score(y_train, logreg.predict(X_tr_scaled)))
    logreg_va_acc = float(accuracy_score(y_val, logreg.predict(X_va_scaled)))
    logreg_path = saved_dir / "logistic_regression.joblib"
    joblib.dump({"model": logreg, "scaler": classical_scaler}, logreg_path)
    t_elapsed = time.time() - t0
    results["logistic_regression"] = {
        "display_name": "Logistic Regression (L2, C=1.0)",
        "train_accuracy": logreg_tr_acc,
        "validation_accuracy": logreg_va_acc,
        "filename": "logistic_regression.joblib",
        "type": "joblib",
        "confidence_type": "probability",
        "confidence_note": "Calibrated sigmoid probability",
        "elapsed_seconds": round(t_elapsed, 2),
    }
    print(f"  Train Acc: {logreg_tr_acc*100:6.2f}% | Val Acc: {logreg_va_acc*100:6.2f}% | Saved: {logreg_path.name}")
    del logreg
    gc.collect()

    # Model 3: SVM RBF (C=1.0, gamma=0.001)
    print("\n--- Training Model 3: SVM RBF (C=1.0, gamma=0.001) ---")
    t0 = time.time()
    svm_rbf = SVC(kernel="rbf", C=1.0, gamma=0.001, random_state=42)
    svm_rbf.fit(X_tr_scaled, y_train)
    svm_tr_acc = float(accuracy_score(y_train, svm_rbf.predict(X_tr_scaled)))
    svm_va_acc = float(accuracy_score(y_val, svm_rbf.predict(X_va_scaled)))
    svm_path = saved_dir / "svm_rbf.joblib"
    joblib.dump({"model": svm_rbf, "scaler": classical_scaler}, svm_path)
    t_elapsed = time.time() - t0
    results["svm_rbf"] = {
        "display_name": "SVM RBF (C=1.0, gamma=0.001)",
        "train_accuracy": svm_tr_acc,
        "validation_accuracy": svm_va_acc,
        "filename": "svm_rbf.joblib",
        "type": "joblib",
        "confidence_type": "decision_function",
        "confidence_note": "Signed distance to separating hyperplane (threshold: 0.0)",
        "elapsed_seconds": round(t_elapsed, 2),
    }
    print(f"  Train Acc: {svm_tr_acc*100:6.2f}% | Val Acc: {svm_va_acc*100:6.2f}% | Saved: {svm_path.name}")
    del svm_rbf, classical_scaler, X_tr_scaled, X_va_scaled, X_tr_flat, X_va_flat
    gc.collect()

    # Model 4: MobileNetV1
    print("\n--- Training Model 4: MobileNetV1 (Transfer Learning, 2 Epochs) ---")
    t0 = time.time()
    tf.keras.backend.clear_session()
    mobilenet = build_mobilenet_model(learning_rate=0.001)
    mobilenet.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=2,
        batch_size=16,
        verbose=0,
    )
    _, mb_tr_acc = mobilenet.evaluate(X_train, y_train, batch_size=16, verbose=0)
    _, mb_va_acc = mobilenet.evaluate(X_val, y_val, batch_size=16, verbose=0)
    mb_path = saved_dir / "mobilenetv1.keras"
    mobilenet.save(mb_path)
    t_elapsed = time.time() - t0
    results["mobilenetv1"] = {
        "display_name": "MobileNetV1 (Transfer Learning)",
        "train_accuracy": float(mb_tr_acc),
        "validation_accuracy": float(mb_va_acc),
        "filename": "mobilenetv1.keras",
        "type": "keras",
        "confidence_type": "probability",
        "confidence_note": "Softmax class probability",
        "elapsed_seconds": round(t_elapsed, 2),
    }
    print(f"  Train Acc: {mb_tr_acc*100:6.2f}% | Val Acc: {mb_va_acc*100:6.2f}% | Saved: {mb_path.name}")
    del mobilenet
    tf.keras.backend.clear_session()
    gc.collect()

    # Model 5: InceptionV3
    print("\n--- Training Model 5: InceptionV3 (SGD lr=0.01, 2 Epochs) ---")
    t0 = time.time()
    inception = build_inceptionv3_model(learning_rate=0.01)
    inception.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=2,
        batch_size=16,
        verbose=0,
    )
    _, inc_tr_acc = inception.evaluate(X_train, y_train, batch_size=16, verbose=0)
    _, inc_va_acc = inception.evaluate(X_val, y_val, batch_size=16, verbose=0)
    inc_path = saved_dir / "inceptionv3.keras"
    inception.save(inc_path)
    t_elapsed = time.time() - t0
    results["inceptionv3"] = {
        "display_name": "InceptionV3 (Transfer Learning)",
        "train_accuracy": float(inc_tr_acc),
        "validation_accuracy": float(inc_va_acc),
        "filename": "inceptionv3.keras",
        "type": "keras",
        "confidence_type": "probability",
        "confidence_note": "Softmax class probability",
        "elapsed_seconds": round(t_elapsed, 2),
    }
    print(f"  Train Acc: {inc_tr_acc*100:6.2f}% | Val Acc: {inc_va_acc*100:6.2f}% | Saved: {inc_path.name}")
    del inception
    tf.keras.backend.clear_session()
    gc.collect()

    # Model 6: InceptionV3 Layer 288 + SVM Hybrid
    print("\n--- Training Model 6: InceptionV3 Layer 288 + SVM Hybrid ---")
    t0 = time.time()
    feature_extractor, target_layer_name = build_inception_feature_extractor(layer_idx=288)
    print(f"  Extracting features from layer 288: '{target_layer_name}'...")
    X_tr_feat = feature_extractor.predict(X_train, batch_size=16, verbose=0)
    X_va_feat = feature_extractor.predict(X_val, batch_size=16, verbose=0)
    del feature_extractor
    tf.keras.backend.clear_session()
    gc.collect()

    hybrid_scaler = StandardScaler()
    X_tr_feat_scaled = hybrid_scaler.fit_transform(X_tr_feat).astype(np.float32)
    X_va_feat_scaled = hybrid_scaler.transform(X_va_feat).astype(np.float32)
    del X_tr_feat, X_va_feat
    gc.collect()

    hybrid_svm = SVC(kernel="rbf", C=1.0, gamma=0.001, random_state=42)
    hybrid_svm.fit(X_tr_feat_scaled, y_train)
    hyb_tr_acc = float(accuracy_score(y_train, hybrid_svm.predict(X_tr_feat_scaled)))
    hyb_va_acc = float(accuracy_score(y_val, hybrid_svm.predict(X_va_feat_scaled)))

    hybrid_path = saved_dir / "inception_svm_hybrid.joblib"
    joblib.dump(
        {
            "model": hybrid_svm,
            "scaler": hybrid_scaler,
            "layer_idx": 288,
            "layer_name": target_layer_name,
            "gamma": 0.001,
            "c_param": 1.0,
        },
        hybrid_path,
    )
    t_elapsed = time.time() - t0
    results["inception_svm_hybrid"] = {
        "display_name": "InceptionV3 (Layer 288) + SVM Hybrid",
        "train_accuracy": hyb_tr_acc,
        "validation_accuracy": hyb_va_acc,
        "filename": "inception_svm_hybrid.joblib",
        "type": "hybrid_joblib",
        "layer_idx": 288,
        "layer_name": target_layer_name,
        "confidence_type": "decision_function",
        "confidence_note": "Signed distance to separating hyperplane (threshold: 0.0)",
        "elapsed_seconds": round(t_elapsed, 2),
    }
    print(f"  Train Acc: {hyb_tr_acc*100:6.2f}% | Val Acc: {hyb_va_acc*100:6.2f}% | Saved: {hybrid_path.name}")
    del hybrid_svm, hybrid_scaler, X_tr_feat_scaled, X_va_feat_scaled
    gc.collect()

    # Save model_info.json
    info = {
        "samples": samples,
        "train_samples": n_train,
        "val_samples": n_val,
        "seed": 42,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": results,
    }
    info_path = saved_dir / "model_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"\nMetadata saved to: {info_path}")

    # Print summary comparison table
    print("\n" + "=" * 78)
    print(f"{'Model':<35} {'Train':>9} {'Val':>9} {'Ref Val (Smoke)':>18} {'Diff':>6}")
    print("-" * 78)
    for key, data in results.items():
        ref_val = REFERENCE_RESULTS_150.get(key, {}).get("val", 0.0)
        diff = data["validation_accuracy"] - ref_val
        diff_str = f"{diff:+.2%}" if abs(diff) > 1e-4 else "0.00%"
        print(
            f"{data['display_name']:<35} "
            f"{data['train_accuracy']*100:>8.2f}% "
            f"{data['validation_accuracy']*100:>8.2f}% "
            f"{ref_val*100:>17.2f}% "
            f"{diff_str:>6}"
        )
    print("=" * 78)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train and save all 6 models for GUI")
    parser.add_argument(
        "--samples",
        type=int,
        default=150,
        help="Number of stratified samples to use (default: 150)",
    )
    args = parser.parse_args()
    train_all_models(samples=args.samples)


if __name__ == "__main__":
    main()
