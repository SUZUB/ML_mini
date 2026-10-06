# Project Assumptions and Replication Notes

This document records the design choices, architectural details, and assumptions made while replicating the paper **"Structural Damage Image Classification"** (Minnie Ho & Jorge Troncoso, 2018).

---

## 1. Dataset Organization and Label Mapping
- **Source**: PEER Hub ImageNet Challenge (PHI-Net) Task 2 Damage State.
- **Raw Files**:
  - `Data/task2_damage_state_1`: `task2_y_train.npy` (11,811 labels), `task2_X_test.npy` (1,460 images), `task2_y_test.npy` (1,460 labels).
  - `Data/task2_damage_state_2`: `task2_X_train.npy` (11,811 images).
- **Label Mapping**:
  - The raw dataset registers `0: Damaged state` and `1: Undamaged state` (with one-hot column 0 = Damaged, column 1 = Undamaged).
  - As requested in the project specification, integer labels are mapped to:
    - **`0`**: Undamaged
    - **`1`**: Damaged
- **Data Normalization & Pixel Conversion**:
  - The raw `.npy` arrays contain Caffe-style zero-centered BGR images with mean subtraction `[103.939, 116.779, 123.68]`.
  - In `code/common.py`, the channel means are restored, channels are converted to RGB, and pixel values are normalized to $[-1, 1)$ via $x = (x / 128.0) - 1.0$ as detailed in Section 3 of the paper.
- **Train/Validation Split**:
  - A stratified 90/10 train/validation split with fixed seed (`random_state=42`) is used (the paper used 90% train / 10% validation and no separate test set).

---

## 2. Classical Machine Learning Models (Sections 4.1, 4.2, 4.3)
- **Feature Preprocessing**:
  - Each 224x224x3 image is flattened to a 150,528-dimensional vector.
  - Standardized using `StandardScaler` fitted **strictly on the training split** to prevent data leakage.
- **Model Hyperparameters**:
  - **KNN (`01_knn.py`)**: $k=5$, Euclidean distance (`n_neighbors=5`).
  - **Logistic Regression (`02_logistic_regression.py`)**: L2 penalty, $C=1.0$, `lbfgs` solver, max iterations 300.
  - **SVM RBF (`03_svm_rbf.py`)**: RBF kernel with $C=1.0$, $\gamma=0.001$.

---

## 3. Deep Learning Baseline Models (Sections 4.4, 5.2)
- **MobileNetV1 (`04_mobilenetv1.py`)**:
  - Pretrained `MobileNet` ($\alpha=1.0$, input $224\times224\times3$, ImageNet weights).
  - Base layers frozen (`base_model.trainable = False`).
  - Top classification head: `GlobalAveragePooling2D` $\rightarrow$ `Dense(2, activation='softmax')`.
  - Optimizer: Adam with $\text{lr}=0.001$, `sparse_categorical_crossentropy`.
- **InceptionV3 Baseline (`05_inceptionv3.py`)**:
  - Pretrained `InceptionV3` (ImageNet weights, base frozen).
  - Top classification head: `GlobalAveragePooling2D` $\rightarrow$ `Dense(2, activation='softmax')`.
  - Optimizer: Plain Stochastic Gradient Descent (SGD) with $\text{lr}=0.01$ as explicitly specified in Section 5.2.

---

## 4. InceptionV3 (Layer 288) + SVM RBF Hybrid (Section 4.5, Figure 3)
- **Layer Selection**:
  - Keras `InceptionV3` has 311 total layers. Layer index 288 (`model.layers[288]`, named `conv2d_88` / `activation_88` / `mixed9_1`) is tapped for intermediate activations.
- **Pooling**:
  - `GlobalAveragePooling2D` is applied to layer 288's spatial activation map to form compact feature vectors.
  - Features are scaled via `StandardScaler` (fitted on train split).
- **Gamma Sweep**:
  - $\gamma \in [10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}]$ (log scale).
  - Plots validation curve and saves to `figures/figure3_inception_svm_gamma_sweep.png` replicating the paper's Figure 3.

---

## 5. InceptionV3 Retrained & Data Augmentation (Section 5.4, Figure 7)
- **Architecture**:
  - Top classification layer removed.
  - Penultimate layer output is **flattened** (`Flatten()`), producing a high-dimensional feature vector connected directly to `Dense(2, activation='softmax')`.
  - This matches the paper's mention of ~2,001,000+ dense parameters.
- **Augmentation Modes**:
  - `none`: Original images without transformation.
  - `shift_flip`: Random horizontal flips + random height/width translations (shifts).
  - `flip_zoom`: Random horizontal flips + random zoom.

---

## 6. Smoke Testing Mode
- Activated by `--smoke` flag in all scripts.
- Samples a stratified subset of 150 images (90% train = 135 samples, 10% validation = 15 samples).
- Runs 2 epochs for deep learning models, completing in seconds for fast verification.
