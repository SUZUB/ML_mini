"""
02_logistic_regression.py - Logistic Regression with L2 Regularization (C=1.0)
==============================================================================
Paper Reference:
  Ho & Troncoso (2018), Section 4.2 "Logistic Regression" & Section 5.1 (Table 1).

Model Description:
  Logistic regression classifier parameterized with L2 regularization penalty and C=1.0.
  Uses sigmoid probability estimation optimized by maximizing log-likelihood.
  Images are flattened to 150,528 dimensions and scaled with StandardScaler fitted
  on training data.
"""

import sys
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from common import load_dataset, prepare_classical_features, get_base_parser, record_results


def main():
    parser = get_base_parser(description="Train Logistic Regression (L2, C=1.0) on Damage State dataset")
    parser.add_argument("--max_iter", type=int, default=300, help="Maximum solver iterations (default: 300)")
    args = parser.parse_args()

    print(f"=== Running Logistic Regression (L2, C=1.0) [Mode: {'SMOKE TEST' if args.smoke else 'FULL RUN'}] ===")
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

    # Step 3: Initialize and train Logistic Regression model (L2 regularization with C=1.0)
    print(f"Fitting LogisticRegression(C=1.0, max_iter={args.max_iter})...")
    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=args.max_iter,
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
    print(f"Logistic Regression Results (Elapsed: {elapsed:.2f}s):")
    print(f"  Train Accuracy:      {train_acc * 100:.2f}% ({train_acc:.4f})")
    print(f"  Validation Accuracy: {val_acc * 100:.2f}% ({val_acc:.4f})")
    print("-" * 50)

    # Save to results/
    record_results(
        model_name="02_logistic_regression",
        mode="smoke" if args.smoke else "full",
        train_acc=train_acc,
        val_acc=val_acc,
        elapsed_seconds=elapsed,
        extra_info={"C": 1.0, "penalty": "l2", "solver": "lbfgs"}
    )


if __name__ == "__main__":
    main()
