"""
01_knn.py - K-Nearest Neighbors Classifier (k=5)
================================================
Paper Reference:
  Ho & Troncoso (2018), Section 4.1 "K-nearest neighbors" & Section 5.1 (Table 1).
  
Model Description:
  Classical baseline model using k-nearest neighbors classification with k=5.
  Images of size 224x224x3 are flattened to 150,528 features and standardized
  using StandardScaler fitted exclusively on the training split.
"""

import sys
import os
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from common import load_dataset, prepare_classical_features, get_base_parser, record_results


def main():
    parser = get_base_parser(description="Train KNN (k=5) on Damage State dataset")
    args = parser.parse_args()

    print(f"=== Running KNN (k=5) [Mode: {'SMOKE TEST' if args.smoke else 'FULL RUN'}] ===")
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

    # Step 3: Initialize and train KNN model with k=5 (using brute algorithm and float64 for 150k dimensions)
    print("Fitting KNeighborsClassifier(n_neighbors=5, algorithm='brute')...")
    clf = KNeighborsClassifier(n_neighbors=5, algorithm="brute")
    clf.fit(X_tr_flat.astype(np.float64), y_train)

    # Step 4: Evaluate accuracies
    print("Evaluating predictions...")
    y_tr_pred = clf.predict(X_tr_flat.astype(np.float64))
    y_val_pred = clf.predict(X_val_flat.astype(np.float64))

    train_acc = accuracy_score(y_train, y_tr_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"KNN (k=5) Results (Elapsed: {elapsed:.2f}s):")
    print(f"  Train Accuracy:      {train_acc * 100:.2f}% ({train_acc:.4f})")
    print(f"  Validation Accuracy: {val_acc * 100:.2f}% ({val_acc:.4f})")
    print("-" * 50)

    # Save to results/
    record_results(
        model_name="01_knn",
        mode="smoke" if args.smoke else "full",
        train_acc=train_acc,
        val_acc=val_acc,
        elapsed_seconds=elapsed,
        extra_info={"k": 5, "algorithm": "brute"}
    )


if __name__ == "__main__":
    main()
