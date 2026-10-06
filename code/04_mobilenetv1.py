"""
04_mobilenetv1.py - MobileNetV1 Transfer Learning Model
======================================================
Paper Reference:
  Ho & Troncoso (2018), Section 4.4 "MobileNetV1 and InceptionV3" & Section 5.1 (Table 1).

Model Description:
  Pretrained MobileNetV1 (alpha=1.0, 224x224 input, ImageNet weights, include_top=False).
  Base network layers are frozen. A new classification head is attached on top:
  GlobalAveragePooling2D -> Dense(2, activation='softmax').
  Trained with categorical/sparse cross-entropy.
"""

import sys
import os
import time
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from common import load_dataset, get_base_parser, record_results


def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=0.001):
    """
    Builds transfer learning model using MobileNet base.
    """
    # Load pretrained MobileNet without classification top
    base_model = tf.keras.applications.MobileNet(
        input_shape=input_shape,
        alpha=1.0,
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze base model weights as described in paper Section 4.4
    base_model.trainable = False

    # Add classification head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV1_Transfer")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    parser = get_base_parser(description="Train MobileNetV1 transfer learning model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    # In smoke test mode, use 1-2 epochs for speed
    epochs = 2 if args.smoke else args.epochs

    print(f"=== Running MobileNetV1 [Mode: {'SMOKE TEST' if args.smoke else 'FULL RUN'}, Epochs: {epochs}] ===")
    start_time = time.time()

    # Step 1: Load preprocessed dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_dataset(
        smoke=args.smoke,
        smoke_samples=args.smoke_samples
    )
    print(f"Train samples: {len(y_train)} (Damaged: {sum(y_train)}, Undamaged: {len(y_train)-sum(y_train)})")
    print(f"Val samples:   {len(y_val)} (Damaged: {sum(y_val)}, Undamaged: {len(y_val)-sum(y_val)})")

    # Step 2: Build model
    print("Building MobileNetV1 transfer learning model...")
    model = build_mobilenet_model(learning_rate=args.lr)
    model.summary(line_length=80)

    # Step 3: Train model
    print(f"Training for {epochs} epochs with batch size {args.batch_size}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    # Step 4: Evaluate accuracies
    print("Evaluating model performance...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"MobileNetV1 Results (Elapsed: {elapsed:.2f}s):")
    print(f"  Train Accuracy:      {train_acc * 100:.2f}% ({train_acc:.4f})")
    print(f"  Validation Accuracy: {val_acc * 100:.2f}% ({val_acc:.4f})")
    print("-" * 50)

    # Save to results/
    record_results(
        model_name="04_mobilenetv1",
        mode="smoke" if args.smoke else "full",
        train_acc=train_acc,
        val_acc=val_acc,
        elapsed_seconds=elapsed,
        extra_info={"epochs": epochs, "batch_size": args.batch_size, "lr": args.lr}
    )


if __name__ == "__main__":
    main()
