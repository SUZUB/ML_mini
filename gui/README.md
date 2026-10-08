# Structural Damage Classification — Web GUI

An interactive Gradio web application for uploading structural images (JPG or PNG) and inspecting whether the structure is **DAMAGED (broken)** or **UNDAMAGED (ok)** according to the six machine learning models benchmarked in this project.

---

## ⚠️ Important Limitation & Disclaimer

> **These models were trained on only 150 images (smoke test), so predictions are unreliable. This is a demonstration of the machine learning inference pipeline, not a certified structural damage detector.**

---

## Folder Layout

```text
gui/
├── app.py              # Main Gradio application with UI layout, lazy loader, & preprocessing
├── train_and_save.py   # Retrains all 6 models with smoke setup (150 samples) and saves them
├── run_gui.py          # Single launcher: auto-trains if missing and serves the Gradio UI
├── export_samples.py   # Utility to export representative validation images for quick testing
├── requirements.txt    # GUI-specific Python dependencies (Gradio, Pillow, etc.)
├── README.md           # Documentation and usage instructions
├── sample_images/      # 4 exported validation images (2 damaged, 2 undamaged)
│   ├── val_01_damaged.png
│   ├── val_02_damaged.png
│   ├── val_03_undamaged.png
│   └── val_04_undamaged.png
└── saved_models/       # (Ignored in git) Model weights, scalers, and metadata
    ├── knn.joblib
    ├── logistic_regression.joblib
    ├── svm_rbf.joblib
    ├── mobilenetv1.keras
    ├── inceptionv3.keras
    ├── inception_svm_hybrid.joblib
    └── model_info.json
```

---

## Quickstart Commands

### 1. Install Dependencies

Ensure your Python virtual environment is active, then install the dependencies:

**Windows PowerShell:**
```powershell
.\.venv\Scripts\python.exe -m pip install -r gui\requirements.txt
```

**Linux / macOS:**
```bash
source .venv/bin/activate
pip install -r gui/requirements.txt
```

### 2. Launch the Application

You can launch the entire GUI with a single command. If models are not yet trained, the launcher will automatically train them first before starting the web server.

**Windows PowerShell:**
```powershell
.\.venv\Scripts\python.exe gui\run_gui.py
```

**Linux / macOS:**
```bash
python gui/run_gui.py
```

Once started, open your web browser and navigate to:
```text
http://127.0.0.1:7860
```

---

## Individual Script Execution

### Retraining / Saving Models Separately
To train or update the models manually with a custom sample size:
```powershell
.\.venv\Scripts\python.exe gui\train_and_save.py --samples 150
```

### Running the App Directly
If models are already present in `gui/saved_models/`:
```powershell
.\.venv\Scripts\python.exe gui\app.py
```

---

## Features

1. **Image Upload & Preprocessing:**
   - Accepts any JPG or PNG image.
   - Cleans and converts color channels to RGB (handling grayscale, transparency, and palettes).
   - Resizes to $224 \times 224$ and normalizes pixels to $[-1, 1)$ via $x = (x / 128.0) - 1.0$.
2. **Model Selection:**
   - Individual model inspection:
     - KNN ($k=5$)
     - Logistic Regression ($L_2, C=1.0$)
     - SVM RBF ($C=1.0, \gamma=0.001$)
     - MobileNetV1 (Transfer Learning)
     - InceptionV3 (Transfer Learning)
     - InceptionV3 (Layer 288) + SVM Hybrid
   - **All models mode:** Side-by-side comparison table of all 6 models with majority-vote consensus.
3. **Color-Coded Output:**
   - 🟢 **UNDAMAGED (ok)**
   - 🔴 **DAMAGED (broken)**
4. **Confidence Metrics:**
   - KNN: Neighbor vote share (e.g., $4/5$ votes).
   - Logistic Regression, MobileNet, InceptionV3: Class probabilities.
   - SVMs: `decision_function` signed distance to the separating hyperplane.
5. **Fast & Lightweight:**
   - Models are loaded **lazily** on first use to minimize initial memory footprint.
