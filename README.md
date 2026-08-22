# EEE 312: ECG Arrhythmia Classification

Heartbeat classification from the MIT-BIH Arrhythmia Database into 4 AAMI classes (N, S, V, F) using the strict de Chazal inter-patient evaluation protocol.

## Project Structure

```
Project/
├── Datasets/              # Raw MIT-BIH database (gitignored)
│   ├── mitdb/             # 48-patient PhysioNet records
│   └── nstdb/             # Noise Stress Test Database
├── data/                  # Processed arrays (gitignored)
├── results/               # Evaluation JSON outputs
├── src/
│   ├── data_preparation/  # ETL, augmentation, feature engineering
│   │   ├── build_dataset.py
│   │   ├── feature_engineering.py
│   │   └── generate_figures.py
│   ├── models/            # One file per classifier
│   │   └── dsp_expert.py
│   └── pipelines/         # Execution & evaluation scripts
│       └── run_dsp_expert.py
├── SCRATCH_KNOWLEDGE_BASE.md
└── .gitignore
```

## Models

| Model | Type | Weighted F2 | Status |
|-------|------|-------------|--------|
| DSP Expert | Rule-based (no ML) | 0.6710 | ✅ Complete |
| Deep Learning (1D-CNN) | ResNet1D-SE | — | 🔜 Planned |
| Hybrid ML Ensemble | LightGBM + TDA | — | 🔜 Planned |

## Quick Start

### 1. Build the Dataset
```bash
python src/data_preparation/build_dataset.py
```

### 2. Extract Features
```bash
python src/data_preparation/feature_engineering.py
```

### 3. Run the DSP Expert Evaluation
```bash
python src/pipelines/run_dsp_expert.py
```

## Evaluation Protocol

All models are evaluated using the **de Chazal inter-patient protocol**:
- **DS1** (22 patients): Training only
- **DS2** (22 patients): Testing only — no patient overlap

Primary metric: **Weighted F2 Score** (prioritizes recall over precision for clinical safety).

## Documentation

See `Summary/summary.pdf` for a comprehensive A-to-Z project guide with theory, code, figures, and results.
