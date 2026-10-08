"""
gui/export_samples.py
=====================
Exports 4 representative validation images (2 damaged, 2 undamaged)
from the smoke dataset to gui/sample_images/ as RGB PNGs.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import common  # noqa: E402


def export_samples():
    output_dir = PROJECT_ROOT / "gui" / "sample_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading validation dataset from smoke setup (150 samples)...")
    X_train, y_train, X_val, y_val = common.load_dataset(
        smoke=True,
        smoke_samples=150,
        random_state=42,
    )

    damaged_indices = np.where(y_val == 1)[0]
    undamaged_indices = np.where(y_val == 0)[0]

    selected = [
        ("val_01_damaged.png", damaged_indices[0], 1, "DAMAGED (broken)"),
        ("val_02_damaged.png", damaged_indices[1], 1, "DAMAGED (broken)"),
        ("val_03_undamaged.png", undamaged_indices[0], 0, "UNDAMAGED (ok)"),
        ("val_04_undamaged.png", undamaged_indices[1], 0, "UNDAMAGED (ok)"),
    ]

    manifest = []
    for filename, idx, true_label, desc in selected:
        norm_img = X_val[idx]
        # Reconstruct RGB [0, 255] from normalized [-1, 1) image
        rgb_float = (norm_img + 1.0) * 128.0
        rgb_uint8 = np.clip(np.round(rgb_float), 0, 255).astype(np.uint8)

        img_path = output_dir / filename
        Image.fromarray(rgb_uint8, mode="RGB").save(img_path, format="PNG")
        print(f"Saved: {img_path.name} (val index: {idx}, true label: {true_label} - {desc})")
        manifest.append(
            {
                "filename": filename,
                "val_index": int(idx),
                "true_label": true_label,
                "description": desc,
            }
        )

    return manifest


if __name__ == "__main__":
    export_samples()
