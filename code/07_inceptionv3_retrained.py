"""
07_inceptionv3_retrained.py - InceptionV3 Retrained with Flatten & Data Augmentation
===================================================================================
Paper Reference:
  Ho & Troncoso (2018), Section 5.4 "Experiments with Inceptionv3" & Figure 7.

Model Description:
  Section 5.4 InceptionV3 architecture variant:
  1. Base InceptionV3 network with top layer removed.
  2. The penultimate layer output is flattened (Flatten) rather than global pooled,
     producing a high-dimensional feature representation (102,402 dense parameters connected
     to Dense(2, activation='softmax')).
  3. Trainable parameter count is printed to demonstrate the capacity increase.
  4. Configurable data augmentation strategies (using Keras ImageDataGenerator):
     - 'none': No data augmentation (Figure 7 left)
     - 'shift_flip': Random horizontal flip + random translation/shifts (Figure 7 middle)
     - 'flip_zoom': Random horizontal flip + random zoom (Figure 7 right)
"""

import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from common import load_dataset, get_base_parser, record_results


def get_data_generator(aug_mode="none"):
    """
    Creates an ImageDataGenerator configured for the specified augmentation mode.
    """
    if aug_mode == "none":
        return ImageDataGenerator()
    elif aug_mode == "shift_flip":
        return ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode="reflect"
        )
    elif aug_mode == "flip_zoom":
        return ImageDataGenerator(
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode="reflect"
        )
    else:
        raise ValueError(f"Unknown augmentation mode: {aug_mode}. Choose from 'none', 'shift_flip', 'flip_zoom'.")


def build_inceptionv3_retrained(input_shape=(224, 224, 3), num_classes=2, aug_mode="none", learning_rate=0.001):
    """
    Builds the Section 5.4 InceptionV3 retrained model with Flattening of the penultimate layer.
    """
    # Pretrained InceptionV3 base without top classification layer
    base_model = tf.keras.applications.InceptionV3(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze base model features (or train top layers)
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)

    # Flatten penultimate layer (as specified in Section 5.4)
    x = layers.Flatten()(x)

    # Add fully connected layer with softmax activation
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"InceptionV3_Retrained_{aug_mode}")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    parser = get_base_parser(description="Train InceptionV3 Retrained (Section 5.4) with Data Augmentation")
    parser.add_argument(
        "--aug",
        type=str,
        default="none",
        choices=["none", "shift_flip", "flip_zoom"],
        help="Data augmentation strategy: 'none', 'shift_flip', or 'flip_zoom' (default: 'none')"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    # In smoke test mode, use 2 epochs
    epochs = 2 if args.smoke else args.epochs

    print(f"=== Running InceptionV3 Retrained (Section 5.4) [Mode: {'SMOKE TEST' if args.smoke else 'FULL RUN'}, Aug: {args.aug}, Epochs: {epochs}] ===")
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
    print(f"Building InceptionV3 retrained model with augmentation='{args.aug}'...")
    model = build_inceptionv3_retrained(aug_mode=args.aug, learning_rate=args.lr)

    # Step 3: Display parameter counts
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params
    print("\n" + "=" * 50)
    print(f"Model Parameter Statistics:")
    print(f"  Total Parameters:         {total_params:,}")
    print(f"  Trainable Parameters:     {trainable_params:,} (Penultimate Flatten -> Dense)")
    print(f"  Non-trainable Parameters: {non_trainable_params:,} (Frozen InceptionV3 Base)")
    print("=" * 50 + "\n")

    # Step 4: Setup data generator for training
    datagen = get_data_generator(args.aug)
    train_generator = datagen.flow(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    # Step 5: Train model
    print(f"Training for {epochs} epochs with batch size {args.batch_size}...")
    history = model.fit(
        train_generator,
        validation_data=(X_val, y_val),
        epochs=epochs,
        verbose=1
    )

    # Step 6: Evaluate accuracies
    print("Evaluating model performance...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"InceptionV3 Retrained ({args.aug}) Results (Elapsed: {elapsed:.2f}s):")
    print(f"  Train Accuracy:      {train_acc * 100:.2f}% ({train_acc:.4f})")
    print(f"  Validation Accuracy: {val_acc * 100:.2f}% ({val_acc:.4f})")
    print("-" * 50)

    # Save to results/
    record_results(
        model_name="07_inceptionv3_retrained",
        mode="smoke" if args.smoke else "full",
        train_acc=train_acc,
        val_acc=val_acc,
        elapsed_seconds=elapsed,
        extra_info={"aug": args.aug, "epochs": epochs, "batch_size": args.batch_size, "trainable_params": trainable_params}
    )


if __name__ == "__main__":
    main()
