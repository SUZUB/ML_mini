"""
code/common.py: Shared Data Loading and Preprocessing Utilities
==============================================================
Based on Ho & Troncoso (2018), "Structural Damage Image Classification":
- Dataset: Task 2 Damage State dataset (PEER Hub ImageNet).
- Image Size: 224x224 RGB.
- Labels: Undamaged = 0, Damaged = 1.
- Normalization: x = (x / 128.0) - 1.0, mapping pixel values to [-1, 1).
- Split: Stratified 90% train / 10% validation split (fixed random seed = 42).
- Classical ML feature preparation: Flattening and StandardScaler fitted strictly on training data.
- Smoke testing: Stratified subset sampling for fast verification.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def find_dataset_paths():
    """
    Locates the dataset numpy files across potential folder structures:
    - Data/task2_damage_state_1 and Data/task2_damage_state_2
    - data/dataset_1 and data/dataset_2
    - data/task2_damage_state_1 and data/task2_damage_state_2
    """
    candidate_roots = [
        os.path.join(os.path.dirname(__file__), "..", "Data"),
        os.path.join(os.path.dirname(__file__), "..", "data"),
        os.path.abspath("Data"),
        os.path.abspath("data")
    ]
    
    x_train_path = None
    y_train_path = None
    x_test_path = None
    y_test_path = None

    for root in candidate_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f == "task2_X_train.npy":
                    x_train_path = os.path.join(dirpath, f)
                elif f == "task2_y_train.npy":
                    y_train_path = os.path.join(dirpath, f)
                elif f == "task2_X_test.npy":
                    x_test_path = os.path.join(dirpath, f)
                elif f == "task2_y_test.npy":
                    y_test_path = os.path.join(dirpath, f)

    if not x_train_path or not y_train_path:
        raise FileNotFoundError(
            f"Could not find task2_X_train.npy or task2_y_train.npy in candidate locations: {candidate_roots}"
        )

    return {
        "x_train": x_train_path,
        "y_train": y_train_path,
        "x_test": x_test_path,
        "y_test": y_test_path
    }


def preprocess_images(raw_x):
    """
    Converts raw PEER Hub ImageNet array to normalized RGB [-1, 1):
    1. The raw arrays contain Caffe-style zero-centered BGR values:
       BGR channel means: [103.939, 116.779, 123.68]
       Add mean offsets and reverse channels to obtain RGB [0, 255].
    2. Normalize to [-1, 1) using formula: x = (x / 128.0) - 1.0 as specified in Section 3.
    """
    bgr_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img_bgr = raw_x.astype(np.float32) + bgr_mean
    img_bgr = np.clip(img_bgr, 0.0, 255.0)
    
    # Convert BGR -> RGB (reverse last dimension)
    img_rgb = img_bgr[..., ::-1]
    
    # Normalize to [-1, 1)
    img_norm = (img_rgb / 128.0) - 1.0
    return img_norm.astype(np.float32)


def preprocess_labels(raw_y):
    """
    Converts raw labels to integer class labels:
    Raw PEER Task 2 format is one-hot (N, 2):
      column 0: Damaged (D)
      column 1: Undamaged (UD)
    Target requirement:
      undamaged = 0
      damaged = 1
    Therefore:
      If column 0 == 1 (Damaged) -> 1
      If column 1 == 1 (Undamaged) -> 0
    """
    raw_y = np.asarray(raw_y)
    if raw_y.ndim == 2 and raw_y.shape[1] == 2:
        # One-hot encoded [Damaged, Undamaged]
        labels = (raw_y[:, 0] == 1.0).astype(np.int64)
    elif raw_y.ndim == 1:
        # If raw integer labels where 0=Damaged, 1=Undamaged
        labels = (raw_y == 0).astype(np.int64)
    else:
        raise ValueError(f"Unexpected label shape: {raw_y.shape}")
    return labels


def load_dataset(smoke=False, smoke_samples=150, random_state=42):
    """
    Loads dataset, applies preprocessing, and performs 90/10 stratified train/val split.
    
    Args:
        smoke (bool): If True, loads only a small stratified subset (e.g. 150 samples).
        smoke_samples (int): Number of samples to use in smoke testing mode.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    paths = find_dataset_paths()
    
    # Load labels first (lightweight)
    y_raw = np.load(paths["y_train"])
    y_all = preprocess_labels(y_raw)
    n_total = len(y_all)

    if smoke:
        # Select stratified subset indices
        subset_size = min(smoke_samples, n_total)
        indices = np.arange(n_total)
        sub_idx, _ = train_test_split(
            indices,
            train_size=subset_size,
            stratify=y_all,
            random_state=random_state
        )
        # Sort indices for efficient memory-mapped / disk reading
        sub_idx = np.sort(sub_idx)
        
        # Load only selected images using memory mapping
        x_mmap = np.load(paths["x_train"], mmap_mode="r")
        x_sub = np.array(x_mmap[sub_idx])
        y_sub = y_all[sub_idx]
        
        # Preprocess images
        x_processed = preprocess_images(x_sub)
        
        # Stratified 90/10 split
        x_train, x_val, y_train, y_val = train_test_split(
            x_processed, y_sub,
            test_size=0.1,
            stratify=y_sub,
            random_state=random_state
        )
    else:
        # Load full array
        x_raw = np.load(paths["x_train"])
        x_processed = preprocess_images(x_raw)
        
        # Stratified 90/10 split
        x_train, x_val, y_train, y_val = train_test_split(
            x_processed, y_all,
            test_size=0.1,
            stratify=y_all,
            random_state=random_state
        )

    return x_train, y_train, x_val, y_val


def prepare_classical_features(X_train, X_val):
    """
    Flattens 224x224x3 image tensors and scales them using StandardScaler.
    The scaler is fitted ONLY on the training split to avoid data leakage.
    
    Args:
        X_train (np.ndarray): Shape (N_train, 224, 224, 3)
        X_val (np.ndarray): Shape (N_val, 224, 224, 3)
        
    Returns:
        tuple: (X_train_scaled, X_val_scaled) as float32 arrays of shape (N, 150528)
    """
    N_tr = X_train.shape[0]
    N_va = X_val.shape[0]
    
    X_tr_flat = X_train.reshape(N_tr, -1).astype(np.float32)
    X_va_flat = X_val.reshape(N_va, -1).astype(np.float32)
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_flat).astype(np.float32)
    X_va_scaled = scaler.transform(X_va_flat).astype(np.float32)
    
    return X_tr_scaled, X_va_scaled


def get_base_parser(description="Model training script"):
    """
    Standard argument parser for all model scripts.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run quick smoke test on a small subset of data (150 samples)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10, smoke mode overrides to 1-2)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--smoke_samples",
        type=int,
        default=150,
        help="Number of samples to use in smoke testing (default: 150)"
    )
    return parser


def record_results(model_name, mode, train_acc, val_acc, elapsed_seconds, extra_info=None, results_dir="results"):
    """
    Appends execution metrics to results/results_summary.csv and results/metrics_log.jsonl.
    """
    import json
    import csv
    from datetime import datetime

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "results_summary.csv")
    jsonl_path = os.path.join(results_dir, "metrics_log.jsonl")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "timestamp": timestamp,
        "model_name": model_name,
        "mode": mode,
        "train_accuracy": float(train_acc),
        "validation_accuracy": float(val_acc),
        "elapsed_seconds": round(float(elapsed_seconds), 2),
        "extra_info": extra_info or {}
    }

    # Write to JSONL
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Write to CSV
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp", "model_name", "mode", "train_accuracy", "validation_accuracy", "elapsed_seconds", "extra_info"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": timestamp,
            "model_name": model_name,
            "mode": mode,
            "train_accuracy": f"{train_acc:.4f}",
            "validation_accuracy": f"{val_acc:.4f}",
            "elapsed_seconds": f"{elapsed_seconds:.2f}",
            "extra_info": json.dumps(extra_info or {})
        })

