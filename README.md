# Structural Damage Image Classification

Machine-learning experiments for classifying structural images as **undamaged** or **damaged**. The damaged class includes visible structural damage such as cracks; the supplied labels do not provide a separate crack-only category.

This project reproduces the classical, transfer-learning, and hybrid model families described in *Structural Damage Image Classification* by Minnie Ho and Jorge Troncoso (2018).

## Project Contents

```text
code/
	01_knn.py                       K-nearest neighbors, k=5
	02_logistic_regression.py       Logistic regression
	03_svm_rbf.py                   RBF-kernel SVM
	04_mobilenetv1.py               MobileNetV1 transfer learning
	05_inceptionv3.py               InceptionV3 transfer learning
	06_inception_svm_hybrid.py      InceptionV3 layer-288 features plus SVM
	07_inceptionv3_retrained.py     Optional retrained InceptionV3 experiment
	common.py                       Data loading and preprocessing
Data/                              NumPy dataset arrays
figures/                           Generated plots
results/                           CSV and JSONL experiment logs
```

## Dataset

The project uses the PEER Hub ImageNet Challenge Task 2 Damage State dataset. The expected files are:

```text
Data/task2_damage_state_1/task2_y_train.npy
Data/task2_damage_state_1/task2_X_test.npy
Data/task2_damage_state_1/task2_y_test.npy
Data/task2_damage_state_2/task2_X_train.npy
```

The supplied dataset contains 11,811 training images and 1,460 test images. Images are 224 x 224 RGB arrays stored in the source dataset's Caffe-style BGR representation. The loader restores the channel means, converts BGR to RGB, and normalizes pixels to approximately `[-1, 1)`.

Labels are mapped as follows:

| Project label | Meaning |
|---:|---|
| `0` | Undamaged |
| `1` | Damaged, including cracks and other damage |

The experiment scripts use a reproducible stratified 90/10 train-validation split with `random_state=42`. The current scripts report validation accuracy; they do not automatically evaluate the held-out `task2_X_test.npy` array.

## Setup

Use Python 3.12 or a compatible Python environment:

```powershell
python -m pip install -r requirements.txt
```

Run commands from the repository root, `ML_mini_source`.

## Low-Memory Sequential Run

The full dataset is large and the CNN models can use substantial memory. The following commands run the six Table 1 models one at a time on the same stratified 150-image sample. Wait for each command to finish before starting the next one.

```powershell
$env:OMP_NUM_THREADS="1"
$env:TF_NUM_INTRAOP_THREADS="1"
$env:TF_NUM_INTEROP_THREADS="1"
$env:TF_ENABLE_ONEDNN_OPTS="0"

python .\code\01_knn.py --smoke --smoke_samples 150
python .\code\02_logistic_regression.py --smoke --smoke_samples 150 --max_iter 100
python .\code\03_svm_rbf.py --smoke --smoke_samples 150
python .\code\04_mobilenetv1.py --smoke --smoke_samples 150 --batch_size 16
python .\code\05_inceptionv3.py --smoke --smoke_samples 150 --batch_size 16
python .\code\06_inception_svm_hybrid.py --smoke --smoke_samples 150 --batch_size 16
```

Smoke mode uses 135 training images and 15 validation images. The deep-learning scripts run for two epochs in smoke mode. Remove `--smoke` for a full training run only on a machine with sufficient memory and time.

## Recorded Results

These are the sequential smoke-mode results recorded on 2026-10-07:

| Model | Train accuracy | Validation accuracy | Runtime |
|---|---:|---:|---:|
| KNN (`k=5`) | 68.89% | 66.67% | 2.80 s |
| Logistic Regression | 100.00% | 53.33% | 3.10 s |
| SVM RBF | 100.00% | 53.33% | 5.76 s |
| MobileNetV1 | 76.30% | 73.33% | 16.28 s |
| InceptionV3 | 79.26% | 66.67% | 41.48 s |
| InceptionV3 layer 288 + SVM | 87.41% | **73.33%** | 14.45 s |

The Inception-SVM experiment tested gamma values from `1e-6` through `1e-1` and selected `gamma=1e-3` on the validation split.

Results are appended to:

- [`results/results_summary.csv`](results/results_summary.csv)
- [`results/metrics_log.jsonl`](results/metrics_log.jsonl)
- [`figures/figure3_inception_svm_gamma_sweep.png`](figures/figure3_inception_svm_gamma_sweep.png)

## Paper Benchmarks

The original paper reports the following reference values. They are provided for comparison and should not be confused with the capped smoke results above:

| Model | Paper train accuracy | Paper validation/test accuracy |
|---|---:|---:|
| KNN (`k=5`) | 67.4% | 53.7% / 52.9% |
| Logistic Regression | 99.4% | 54.7% |
| SVM RBF | 99.7% | 50.7% |
| MobileNetV1 | 80.0% | 67.0% |
| InceptionV3 | 81.0% | 83.0% |
| InceptionV3 layer 288 + SVM | 95.0% | 75.0% |

Differences are expected because the recorded runs use only 150 samples for safe execution.

## Reproducibility Notes

- Preprocessing and label conversion are implemented in [`code/common.py`](code/common.py).
- Classical models flatten images and fit `StandardScaler` only on the training split.
- MobileNetV1 and InceptionV3 use frozen ImageNet feature extractors with a two-class head.
- The hybrid model extracts intermediate InceptionV3 activations, applies global average pooling, standardizes the features, and trains an RBF SVM.
- Model weights may be downloaded automatically by Keras during the first deep-learning run.
- Detailed design assumptions are documented in [`code/ASSUMPTIONS.md`](code/ASSUMPTIONS.md).