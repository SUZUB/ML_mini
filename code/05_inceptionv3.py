"""
05_inceptionv3.py - InceptionV3 Baseline Transfer Learning Model
===============================================================
Paper Reference:
  Ho & Troncoso (2018), Section 4.4 "MobileNetV1 and InceptionV3", Section 5.1 (Table 1),
  and Section 5.2 "Bias versus Variance".

Model Description:
  Pretrained InceptionV3 on ImageNet (include_top=False).
  Base network weights are frozen.
  A new classification head is added:
  GlobalAveragePooling2D -> Dense(2, activation='softmax').
  Trained using plain Stochastic Gradient Descent (SGD) with learning rate 0.01
  as specified in Section 5.2.
"""

import sys
import os
import time
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from common import load_dataset, get_base_parser, record_results


def build_inceptionv3_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=0.01):
    """
    Builds baseline transfer learning model using InceptionV3 base and plain SGD optimizer.
    """
    # Load pretrained InceptionV3 base
    base_model = tf.keras.applications.InceptionV3(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze base model layers (transfer learning)
    base_model.trainable = False

    # Classification head
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="InceptionV3_Baseline")
    
    # Plain SGD with learning rate 0.01 as specified in Section 5.2
    optimizer = optimizers.SGD(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    parser = get_base_parser(description="Train InceptionV3 baseline transfer learning model")
    parser.add_argument("--lr", type=float, default=0.01, help="SGD learning rate (default: 0.01)")
    args = parser.parse_args()

    # In smoke test mode, use 2 epochs
    epochs = 2 if args.smoke else args.epochs

    print(f"=== Running InceptionV3 Baseline [Mode: {'SMOKE TEST' if args.smoke else 'FULL RUN'}, Epochs: {epochs}] ===")
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
    print("Building InceptionV3 baseline model (frozen base, SGD lr=0.01)...")
    model = build_inceptionv3_model(learning_rate=args.lr)
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
    print(f"InceptionV3 Baseline Results (Elapsed: {elapsed:.2f}s):")
    print(f"  Train Accuracy:      {train_acc * 100:.2f}% ({train_acc:.4f})")
    print(f"  Validation Accuracy: {val_acc * 100:.2f}% ({val_acc:.4f})")
    print("-" * 50)

    # Save to results/
    record_results(
        model_name="05_inceptionv3",
        mode="smoke" if args.smoke else "full",
        train_acc=train_acc,
        val_acc=val_acc,
        elapsed_seconds=elapsed,
        extra_info={"epochs": epochs, "batch_size": args.batch_size, "lr": args.lr, "optimizer": "SGD"}
    )


if __name__ == "__main__":
    main()
