# EEE 312: ECG Arrhythmia Classification

Heartbeat classification from the MIT-BIH Arrhythmia Database into 4 AAMI classes (N, S, V, F) using the strict de Chazal inter-patient evaluation protocol.

## Project Structure

```
Project/
├── Datasets/              # Raw MIT-BIH & NSTDB databases (gitignored)
├── data/                  # Processed arrays and extracted features (gitignored)
├── models_saved/          # Trained model weights (gitignored)
├── results/               # Evaluation JSON outputs
├── src/
│   ├── data_preparation/  # ETL, augmentation, feature engineering
│   ├── data_utils/        # Utility scripts (e.g., Signal Augmenter)
│   ├── models/            # One file per classifier (DSP, ResNet1D, LightGBM)
│   ├── pipelines/         # Execution & evaluation scripts
│   └── scripts/           # Plot generation and miscellaneous scripts
├── SCRATCH_KNOWLEDGE_BASE.md
├── requirements.txt
└── .gitignore
```

## Final Models & Results (DS2 Test Set)

| Model | Type | Accuracy | Weighted F2 | Status |
|-------|------|----------|-------------|--------|
| DSP Expert | Rule-based (no ML) | 65.02% | 0.6710 | ✅ Complete |
| Deep Learning (1D-CNN) | ResNet1D-SE | 75.72% | 0.7713 | ✅ Complete |
| Hybrid ML Ensemble | LightGBM + TDA | **92.35%** | **0.9209** | ✅ Complete |

## Quick Start

### 1. Build the Dataset
```bash
python src/data_preparation/build_dataset.py
```

### 2. Extract Features
```bash
python src/data_preparation/feature_engineering.py
```

### 3. Run Pipeline Evaluations
```bash
# Run DSP Expert
python src/pipelines/run_dsp_expert.py

# Train & Run Deep Learning Baseline
python src/pipelines/run_resnet1d_se.py

# Train & Run Hybrid ML Ensemble
python src/pipelines/run_hybrid_lgbm.py
```

### 4. Run NSTDB Stress Tests
```bash
python src/pipelines/run_nstdb_stress_test.py
python src/pipelines/run_nstdb_resnet1d_se.py
python src/pipelines/run_nstdb_hybrid_lgbm.py
```

## Evaluation Protocol

All models are evaluated using the **de Chazal inter-patient protocol**:
- **DS1** (22 patients): Training only
- **DS2** (22 patients): Testing only — no patient overlap

Primary metric: **Weighted F2 Score** (prioritizes recall over precision for clinical safety).

## Documentation

See `Summary/summary.pdf` for a comprehensive A-to-Z project guide with theory, code, figures, and results.
