"""
03_svm_rbf.py - Support Vector Machine with RBF Kernel (C=1.0, gamma=0.001)
==========================================================================
Paper Reference:
  Ho & Troncoso (2018), Section 4.3 "Support Vector Machine" & Section 5.1 (Table 1).

Model Description:
  Support Vector Classifier using the non-linear Radial Basis Function (RBF) kernel:
    K(x, x') = exp(-gamma * ||x - x'||^2)
  Hyperparameters from paper:
    - Penalty parameter C = 1.0
    - Kernel coefficient gamma = 0.001
  Images are flattened to 150,528 dimensions and scaled with StandardScaler fitted
  on training data.
"""

import sys
import os
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from common import load_dataset, prepare_classical_features, get_base_parser, record_results


def main():
    parser = get_base_parser(description="Train SVM (RBF kernel, C=1.0, gamma=0.001) on Damage State dataset")
    parser.add_argument("--gamma", type=float, default=0.001, help="RBF kernel coefficient gamma (default: 0.001)")
    parser.add_argument("--c_param", type=float, default=1.0, help="Penalty parameter C (default: 1.0)")
    args = parser.parse_args()

    print(f"=== Running SVM RBF (C={args.c_param}, gamma={args.gamma}) [Mode: {'SMOKE TEST' if args.smoke else 'FULL RUN'}] ===")
    start_time = time.time()

    # Step 1: Load and preprocess data (90/10 stratified split)
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_dataset(
        smoke=args.smoke,
        smoke_samples=args.smoke_samples
    )
    print(f"Train samples: {len(y_train)} (Damaged: {sum(y_train)}, Undamaged: {len(y_train)-sum(y_train)})")
    print(f"Val samples:   {len(y_val)} (Damaged: {sum(y_val)}, Undamaged: {len(y_val)-sum(y_val)})")

    # Step 2: Flatten and scale features (StandardScaler fitted strictly on training data)
    print("Flattening and scaling image features...")
    X_tr_flat, X_val_flat = prepare_classical_features(X_train, X_val)

    # Step 3: Initialize and train SVM RBF model
    print(f"Fitting SVC(kernel='rbf', C={args.c_param}, gamma={args.gamma})...")
    clf = SVC(
        kernel="rbf",
        C=args.c_param,
        gamma=args.gamma,
        random_state=42
    )
    clf.fit(X_tr_flat, y_train)

    # Step 4: Evaluate accuracies
    print("Evaluating predictions...")
    y_tr_pred = clf.predict(X_tr_flat)
    y_val_pred = clf.predict(X_val_flat)

    train_acc = accuracy_score(y_train, y_tr_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"SVM RBF Results (Elapsed: {elapsed:.2f}s):")
    print(f"  Train Accuracy:      {train_acc * 100:.2f}% ({train_acc:.4f})")
    print(f"  Validation Accuracy: {val_acc * 100:.2f}% ({val_acc:.4f})")
    print("-" * 50)

    # Save to results/
    record_results(
        model_name="03_svm_rbf",
        mode="smoke" if args.smoke else "full",
        train_acc=train_acc,
        val_acc=val_acc,
        elapsed_seconds=elapsed,
        extra_info={"C": args.c_param, "gamma": args.gamma, "kernel": "rbf"}
    )


if __name__ == "__main__":
    main()
