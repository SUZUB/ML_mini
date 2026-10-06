"""
06_inception_svm_hybrid.py - Pretrained InceptionV3 (Layer 288) + SVM RBF Hybrid
================================================================================
Paper Reference:
  Ho & Troncoso (2018), Section 4.5 "Support Vector Machine Based on Activations
  Earlier in the InceptionV3 Network", Section 5.1 (Table 1), and Figure 3 (Gamma tuning curve).

Model Description:
  Extracts intermediate visual representations from layer index 288 of the pretrained
  InceptionV3 architecture. The spatial activations are pooled via GlobalAveragePooling2D
  (to yield fixed-length dense feature vectors), scaled with StandardScaler fitted strictly
  on the training set, and fed into an RBF-kernel Support Vector Machine (SVC).
  Performs a sweep over the kernel parameter gamma (1e-6 to 1e-1 on a log scale)
  and saves a validation curve plot replicating Figure 3 of the paper.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf

from common import load_dataset, get_base_parser, record_results


def extract_layer_288_features(X, base_model_layer_idx=288, batch_size=32):
    """
    Builds a feature extractor using InceptionV3 up to layer 288 with GlobalAveragePooling.
    """
    full_inception = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    
    total_layers = len(full_inception.layers)
    target_layer = full_inception.layers[base_model_layer_idx]
    print(f"Total InceptionV3 layers: {total_layers}")
    print(f"Extracting features from Layer index {base_model_layer_idx}: '{target_layer.name}' (output shape: {target_layer.output.shape})")

    # Connect target layer output to GlobalAveragePooling2D
    feature_output = tf.keras.layers.GlobalAveragePooling2D()(target_layer.output)
    feature_extractor = tf.keras.Model(inputs=full_inception.input, outputs=feature_output)

    print(f"Extracting features for {len(X)} images (batch_size={batch_size})...")
    features = feature_extractor.predict(X, batch_size=batch_size, verbose=1)
    return features


def main():
    parser = get_base_parser(description="Train InceptionV3 (Layer 288) + SVM RBF Hybrid")
    parser.add_argument("--layer_idx", type=int, default=288, help="Layer index in InceptionV3 (default: 288)")
    parser.add_argument("--c_param", type=float, default=1.0, help="SVM penalty C parameter (default: 1.0)")
    parser.add_argument("--default_gamma", type=float, default=0.001, help="Default gamma for final model evaluation (default: 0.001)")
    args = parser.parse_args()

    print(f"=== Running InceptionV3 (Layer {args.layer_idx}) + SVM Hybrid [Mode: {'SMOKE TEST' if args.smoke else 'FULL RUN'}] ===")
    start_time = time.time()

    # Step 1: Load preprocessed dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_dataset(
        smoke=args.smoke,
        smoke_samples=args.smoke_samples
    )
    print(f"Train samples: {len(y_train)} (Damaged: {sum(y_train)}, Undamaged: {len(y_train)-sum(y_train)})")
    print(f"Val samples:   {len(y_val)} (Damaged: {sum(y_val)}, Undamaged: {len(y_val)-sum(y_val)})")

    # Step 2: Extract intermediate CNN activations from layer 288
    print(f"Extracting InceptionV3 activations from layer index {args.layer_idx}...")
    X_tr_feat = extract_layer_288_features(X_train, base_model_layer_idx=args.layer_idx, batch_size=args.batch_size)
    X_va_feat = extract_layer_288_features(X_val, base_model_layer_idx=args.layer_idx, batch_size=args.batch_size)

    # Step 3: Standardize features (fitted on train split only)
    print("Standardizing extracted deep features...")
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_feat).astype(np.float32)
    X_va_scaled = scaler.transform(X_va_feat).astype(np.float32)

    # Step 4: Gamma sweep (1e-6 to 1e-1 on log scale) replicating paper's Figure 3
    gammas = np.logspace(-6, -1, 6)
    train_accs = []
    val_accs = []

    print("\n--- Performing Gamma Sweep (reproducing Figure 3) ---")
    for g in gammas:
        svm = SVC(kernel="rbf", C=args.c_param, gamma=g, random_state=42)
        svm.fit(X_tr_scaled, y_train)
        tr_acc = accuracy_score(y_train, svm.predict(X_tr_scaled))
        va_acc = accuracy_score(y_val, svm.predict(X_va_scaled))
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        print(f"  Gamma = {g:.1e} | Train Acc: {tr_acc*100:5.2f}% | Val Acc: {va_acc*100:5.2f}%")

    # Step 5: Plot validation curve (reproducing Figure 3)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    fig_path = os.path.join("figures", "figure3_inception_svm_gamma_sweep.png")

    plt.figure(figsize=(8, 5))
    plt.semilogx(gammas, [a * 100 for a in train_accs], "o-", label="Training Accuracy", color="#1f77b4", linewidth=2)
    plt.semilogx(gammas, [a * 100 for a in val_accs], "s--", label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    plt.xlabel("Gamma (RBF Kernel Coefficient)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Figure 3 Replication: Tuning of the $\\gamma$ parameter for InceptionV3-SVM", fontsize=13)
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nGamma sweep plot saved to: {fig_path}")

    # Step 6: Train final model with optimal / chosen gamma
    best_idx = int(np.argmax(val_accs))
    best_gamma = gammas[best_idx]
    print(f"\nOptimal gamma based on validation accuracy: {best_gamma:.1e} (Val Acc: {val_accs[best_idx]*100:.2f}%)")

    final_svm = SVC(kernel="rbf", C=args.c_param, gamma=best_gamma, random_state=42)
    final_svm.fit(X_tr_scaled, y_train)
    final_train_acc = accuracy_score(y_train, final_svm.predict(X_tr_scaled))
    final_val_acc = accuracy_score(y_val, final_svm.predict(X_va_scaled))

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"InceptionV3 Layer 288 + SVM Results (Elapsed: {elapsed:.2f}s):")
    print(f"  Best Gamma:          {best_gamma:.1e}")
    print(f"  Train Accuracy:      {final_train_acc * 100:.2f}% ({final_train_acc:.4f})")
    print(f"  Validation Accuracy: {final_val_acc * 100:.2f}% ({final_val_acc:.4f})")
    print("-" * 50)

    # Save to results/
    record_results(
        model_name="06_inception_svm_hybrid",
        mode="smoke" if args.smoke else "full",
        train_acc=final_train_acc,
        val_acc=final_val_acc,
        elapsed_seconds=elapsed,
        extra_info={"best_gamma": float(best_gamma), "layer_idx": args.layer_idx, "c_param": args.c_param}
    )


if __name__ == "__main__":
    main()
