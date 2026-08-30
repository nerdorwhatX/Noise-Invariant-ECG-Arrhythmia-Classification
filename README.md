# Noise-Invariant ECG Arrhythmia Classification

This repository contains the source code for an advanced heartbeat classification system trained and evaluated on the **MIT-BIH Arrhythmia Database**. The project strictly adheres to the **de Chazal inter-patient evaluation protocol** to prevent data leakage, ensuring the models are evaluated on unseen patients. 

A major focus of this project is clinical reliability under extreme noise. By utilizing advanced Digital Signal Processing (DSP) feature engineering (including Wavelet transforms, phase-space geometry, and persistent homology), the final Hybrid model maintains high classification performance even when subjected to extreme noise levels (up to -6 dB SNR) from the **Noise Stress Test Database (NSTDB)**.

## Core Achievements
*   **Zero Data Leakage:** Built upon the strict DS1 (Train) and DS2 (Test) patient-wise split.
*   **Robust Feature Engineering:** Extracts 89 complex DSP features per beat, prioritizing signal morphology, frequency distribution, and contextual heart rate variability.
*   **Noise Invariance:** The final Hybrid LightGBM model maintains ~60% balanced accuracy and 78% standard accuracy on the NSTDB even at -6 dB (where noise power is 4x greater than the signal), significantly outperforming standard Deep Learning baselines.

## Project Architecture

```text
Project/
├── Datasets/              # Raw MIT-BIH & NSTDB databases (gitignored)
├── data/                  # Processed arrays and extracted features (gitignored)
├── models_saved/          # Trained model weights (gitignored)
├── results/               # JSON outputs with evaluation metrics and confusion matrices
├── src/
│   ├── data_preparation/  # ETL pipelines, data cleaning, and feature engineering
│   ├── data_utils/        # Utility scripts (e.g., WFDB parsing, Signal Augmenter)
│   ├── models/            # Classifier definitions (DSP Expert, ResNet1D, LightGBM)
│   ├── pipelines/         # High-level execution scripts for training and evaluation
│   └── scripts/           # Plot generation and result visualization
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Getting Started

### 1. Environment Setup
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
The project relies on the MIT-BIH Arrhythmia Database and the NSTDB. The build script automatically downloads the required records via the `wfdb` library, filters the signals (0.5 Hz - 45 Hz bandpass), segments the heartbeats, and partitions them into the DS1 and DS2 sets.
```bash
python src/data_preparation/build_dataset.py
```

### 3. Feature Engineering
Instead of relying solely on raw waveform amplitudes (which are highly susceptible to clinical noise), this script extracts 89 mathematically robust features for every heartbeat. These include:
*   **Wavelet Energy Ratios:** Frequency domain distribution.
*   **Phase-Space Geometry:** Convex hull area and perimeter of the electrical signal.
*   **Persistent Homology:** Topological data analysis (TDA) to measure structural holes in the signal.
*   **RR-Interval Context:** Preceding, subsequent, and local average heart rates.
```bash
python src/data_preparation/feature_engineering.py
```

## Running the Models

We evaluate three distinct modeling approaches on the unseen DS2 test set. 

### Model 1: DSP Expert (Rule-Based)
A purely deterministic, rule-based algorithm that uses hard-coded thresholds on the extracted DSP features (no machine learning involved). It acts as a baseline to demonstrate the limits of manual feature thresholding.
```bash
python src/pipelines/run_dsp_expert.py
```

### Model 2: ResNet1D-SE (Deep Learning Baseline)
A 1D Residual Neural Network with Squeeze-and-Excitation blocks trained directly on the raw ECG waveforms. This represents a standard modern deep learning approach.
```bash
python src/pipelines/run_resnet1d_se.py
```

### Model 3: Hybrid ML Ensemble (LightGBM)
The flagship model of this project. It trains a gradient boosting tree ensemble (LightGBM) on the 89 engineered DSP features rather than raw waveforms. 
```bash
python src/pipelines/run_hybrid_lgbm.py
```

## Noise Stress Testing (NSTDB)

To evaluate clinical viability, the models are subjected to real physiological noise (baseline wander, muscle artifact, electrode motion) at varying Signal-to-Noise Ratios (SNR: 24 dB down to -6 dB).
```bash
python src/pipelines/run_nstdb_stress_test.py
python src/pipelines/run_nstdb_resnet1d_se.py
python src/pipelines/run_nstdb_hybrid_lgbm.py
```
*(After running the stress tests, use `python src/scripts/generate_plots.py` to recreate the metric visualizations.)*

## Evaluation Metrics

Because the MIT-BIH database is heavily imbalanced (~87% of beats are Normal), standard accuracy is highly deceptive. A model can achieve 87% accuracy simply by predicting "Normal" for every beat. Therefore, this project evaluates models using:
1.  **Balanced Accuracy:** The unweighted average of recall across all classes, preventing majority-class bias.
2.  **Macro F2 Score:** Averages the F2 score of each class equally, heavily penalizing models that ignore minority arrhythmia classes (S, V, F).

### Results on Clean DS2 Test Set
| Model | Accuracy (%) | Balanced Acc (%) | Weighted F2 | Macro F2 |
| :--- | :--- | :--- | :--- | :--- |
| DSP Expert (Rule-based) | 65.82% | 29.01% | 0.6794 | 0.2608 |
| ResNet1D-SE (Deep Learning) | 64.57% | 48.83% | 0.6695 | 0.3841 |
| **Hybrid ML Ensemble (LGBM)** | **89.83%** | **58.36%** | **0.8992** | **0.5428** |

*Note: The Hybrid model significantly outperforms the deep learning baseline by combining the noise-resistant properties of engineered DSP features with the non-linear classification power of LightGBM.*
