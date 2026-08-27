# EEE 312: Scratch Knowledge Base & Cheat Sheet

This document serves as a comprehensive reference guide for models/agents working on the **EEE 312 ECG Arrhythmia Classification** project. It summarizes all structural designs, architectural choices, models, results, and critical engineering discoveries from the `Scratch/` folder experimentation phase.

> [!IMPORTANT]  
> **Core Philosophy**
> - **One Model = One File**: Do not create monolithic files. Each classifier must have a dedicated class in `src/models/` exposing `fit()`, `predict()`, `save()`, and `load()`.
> - **Train Once, Use Always**: Always use caching mechanisms. Models are cached to `models_saved/`. If a cached model exists, load it; do not retrain.

## 1. Architectural Organization

The project separates logic into distinct, reusable domain modules:
- `src/models/`: Formal definitions of every individual classifier (DSP rules, CNNs, LightGBM architectures).
- `src/pipelines/`: Orchestration scripts that build, train, and evaluate specific hybrid combinations.
- `src/data_utils/`: Datasets, preprocessing routines, and signal synthesizers (e.g., `signal_augmenter.py`, `noise_injection.py`).
- `src/features/`: Feature extraction scripts for Takens embeddings, TDA, and time-domain metrics.
- `evaluation/`: Scripts for evaluating strict de Chazal inter-patient protocols and reporting.

---

## 2. The 12-Model Inventory

The project features **12 distinct models** built across 3 primary methodologies (DSP, CNN, LightGBM), which assemble into **4 independent Hybrid Pipelines**.

### 2.1. Base Representational Models
- **Model 1: DSP-Expert** (`models/david_pipeline.py`): A rule-based DSP expert system utilizing Takens phase-space geometry and zero-phase FIR filtering.
- **Model 2: ResNet1D-SE** (`models/goliath_pipeline.py`): A deep 1D ResNet with Squeeze-and-Excitation (SE) blocks (`SEBlock`, `ResidualBlock1D`, `ResNet1D`) extracting morphological features.
- **Model 3: ResNet1D-DANN** (`models/goliath_pipeline.py`): Uses a Gradient Reversal Layer (`GradientReversalLayer`) for Domain Adversarial Neural Networks to achieve domain adaptation across patients.

### 2.2. Flat Hybrid Models
These models concatenate DSP, TDA, and Deep Learning representations into a single feature vector, training a flat LightGBM classifier (`FlatHybridLightGBM`).
- **Model 4: FlatHybridLightGBM (SMOTE)** (`models/flat_hybrid_lightgbm.py`): Handles class imbalance via feature-space SMOTE.
- **Model 5: FlatHybridLightGBM (Superposition)**: Swaps SMOTE for 40,000 synthetic F-beats generated via signal superposition.

### 2.3. DANN Hierarchical Cascade
Splits the 4-class problem into a tree of easier binary sub-tasks using DANN representations (`DannHierarchicalCascade`).
- **Model 6: Generalist Stage 1**: Binary LightGBM separating (Normal-like) vs. (Ventricular-like).
- **Model 7: S-Expert Stage 2a**: Binary LightGBM distinguishing Normal vs. Supraventricular.
- **Model 8: F-Expert Stage 2b**: Binary LightGBM distinguishing Ventricular vs. Fusion.

### 2.4. Component Hierarchical Ensemble
A gate-and-specialist architecture (`HierarchicalEnsemble`) using `BinarySpecialist` for extreme isolation of rare classes.
- **Model 9: Gate Classifier**: Detects Normal vs. Abnormal beats.
- **Model 10: S-Specialist**: Fires exclusively on S-class beats.
- **Model 11: V-Specialist**: Fires exclusively on V-class beats.
- **Model 12: F-Specialist**: Fires exclusively on F-class beats.

---

## 3. Pipelines & Execution

Pipelines are orchestrated in `src/pipelines/`. To run any of the combinations, execute the following from the root directory:

- **Pipeline 1 (SMOTE-based Flat Hybrid)**: `python src/pipelines/main.py`
- **Pipeline 2 (Signal Superposition Flat Hybrid)**: `python src/pipelines/main_advanced_theory.py`
- **Pipeline 3 (DANN-based Hierarchical Cascade)**: `python src/pipelines/main_dann_hierarchical.py`
- **Pipeline 4 (Component Hierarchical Ensemble)**: `python src/pipelines/main_component_hierarchical.py`
- **Full Run**: `.\run_all.ps1` (Rebuilds the entire cache and runs all sequentially).

---

## 4. Engineering Gotchas & Critical Discoveries

> [!WARNING]  
> **Numpy Object Arrays (Caching Issue)**
> When caching uneven, variable-length sequences as numpy templates, `np.save` quietly casts them to `dtype=object`. Attempting downstream DSP operations (like `np.corrcoef`) on these cached objects **will crash the pipeline**. 
> **Solution**: Explicitly downcast upon loading using `np.asarray(tmpl, dtype=float)`.

> [!TIP]  
> **Signal Superposition > SMOTE**
> Using theoretical math to synthesize artificial arrhythmias via time-domain signal superposition (`main_advanced_theory.py`) avoids synthetic feature artifacts inherent in SMOTE, yielding a superior Macro F2 score (0.456 vs. 0.435).

> [!TIP]  
> **Hierarchical > Flat**
> Breaking the multi-class problem into smaller binary trees (DANN Cascade) effectively isolates the extreme imbalance of the F-class better than a flat multi-classifier. DANN Cascade holds the highest Macro F2 score (0.461).

> [!NOTE]  
> **The Fall of DSP-Expert & Recovery via Hybridization**
> The DSP expert system (DSP-Expert) achieves near-perfect F2 (0.979) on Normal (N) beats, but **fails completely (0.000 F2)** on Ventricular (V) beats for unseen patients due to strict thresholding. However, hybridization successfully recovers this, detecting V-class beats with a high 0.813 F2 score.

---

## 5. Official Performance Results

Evaluations are performed using the strict de Chazal inter-patient protocol. 
Best overall architecture: **DANN Hierarchical Cascade**.

| Model Architecture | Macro F2 Score | Accuracy |
| :--- | :---: | :---: |
| **DSP-Expert (Rule-based System)** | 0.262 | 88.9% |
| **ResNet1D-SE (Pure 1D-CNN DL)** | ~0.412 | ~75.5% |
| **Hybrid 1: Flat SMOTE** | 0.435 | 84.8% |
| **Hybrid 2: Flat Superposition**| 0.456 | 85.1% |
| **Hybrid 3: DANN Cascade** | **0.461** | **86.5%** |
| **Hybrid 4: Component Ensemble**| 0.434 | 83.9% |

### Per-Class F2 Score Breakdown (Best vs Base)

| Class | DSP-Expert | ResNet1D-SE | Hybrid (DANN Cascade) |
| :--- | :---: | :---: | :---: |
| **N (Normal)** | **0.979** | 0.809 | 0.915 |
| **S (Supraventricular)**| 0.069 | 0.103 | **0.101** |
| **V (Ventricular)** | 0.000 | 0.723 | **0.813** |
| **F (Fusion)** | 0.000 | 0.011 | **0.014** |

## Conclusion & Next Steps
All subsequent model expansions, feature tuning, or new pipeline integrations must:
1. Adhere to the defined OOP structure and `models/` separation.
2. Exploit DANN architectures and Signal Superposition over flat feature engineering.
3. Utilize `models_saved/` caching mechanisms to respect compute constraints.
