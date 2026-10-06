# Experimental Results Directory

This directory stores execution logs, metrics, performance summaries, and evaluation artifacts generated across both smoke testing and full training runs.

---

## Logged Files

1. **`results_summary.csv`**:
   - Tabular summary of each model run containing timestamp, model name, execution mode (`smoke` vs `full`), training accuracy, validation accuracy, runtime (in seconds), and key model parameters.
   
2. **`metrics_log.jsonl`**:
   - Machine-readable, line-delimited JSON log capturing detailed execution parameters and performance metrics for each training run.

3. **`../figures/`**:
   - Replicated figures and validation curves (e.g., `figure3_inception_svm_gamma_sweep.png`).

---

## Paper Benchmark Comparison (Table 1 from Ho & Troncoso, 2018)

For reference, the published benchmarks in Table 1 of the paper are:

| Model | Paper Train Accuracy | Paper Val/Test Accuracy | Replicated File |
| :--- | :---: | :---: | :--- |
| **K-Nearest Neighbors ($k=5$)** | 67.4% | 53.7% / 52.9% | [`code/01_knn.py`](../code/01_knn.py) |
| **Logistic Regression (L2, $C=1.0$)** | 99.4% | 54.7% | [`code/02_logistic_regression.py`](../code/02_logistic_regression.py) |
| **SVM RBF ($C=1.0, \gamma=0.001$)** | 99.7% | 50.7% | [`code/03_svm_rbf.py`](../code/03_svm_rbf.py) |
| **MobileNetV1 Transfer Learning** | 80.0% | 67.0% | [`code/04_mobilenetv1.py`](../code/04_mobilenetv1.py) |
| **InceptionV3 Baseline (SGD $\text{lr}=0.01$)** | 81.0% | 83.0% | [`code/05_inceptionv3.py`](../code/05_inceptionv3.py) |
| **InceptionV3 Layer 288 + SVM RBF** | 95.0% | 75.0% | [`code/06_inception_svm_hybrid.py`](../code/06_inception_svm_hybrid.py) |
| **InceptionV3 Retrained (Section 5.4)** | ~85-90% | ~84% | [`code/07_inceptionv3_retrained.py`](../code/07_inceptionv3_retrained.py) |
