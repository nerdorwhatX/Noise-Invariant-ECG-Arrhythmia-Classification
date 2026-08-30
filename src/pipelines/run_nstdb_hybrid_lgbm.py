import os
import sys
import json
import numpy as np
import logging
from sklearn.metrics import accuracy_score, fbeta_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.models.hybrid_lgbm import HybridLGBMClassifier
from src.data_preparation.build_dataset import extract_raw_beats
from src.data_preparation.feature_engineering import extract_all

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    model_path = os.path.join(project_root, "models_saved", "hybrid_lgbm.pkl")
    if not os.path.exists(model_path):
        logger.error(
            f"Trained model not found at {model_path}. Run run_hybrid_lgbm.py first."
        )
        sys.exit(1)

    logger.info("Loading trained Hybrid LGBM model...")
    model = HybridLGBMClassifier()
    model.load(model_path)

    nstdb_dir = os.path.join(project_root, "Datasets", "nstdb")
    snr_levels = ["24", "18", "12", "06", "00", "_6"]
    snr_numeric = [24, 18, 12, 6, 0, -6]
    records = ["118", "119"]

    results = {}
    logger.info("Starting NSTDB Stress Test for Hybrid LGBM...")

    for snr, snr_num in zip(snr_levels, snr_numeric):
        logger.info(f"\nEvaluating SNR: {snr_num} dB")
        X_all, y_true, rr_all = [], [], []

        for rec in records:
            record_name = f"{rec}e{snr}"
            path = os.path.join(nstdb_dir, record_name)
            try:
                X, y, rr = extract_raw_beats(path)
                X_all.append(X)
                y_true.append(y)
                rr_all.append(rr)
            except Exception as e:
                logger.warning(f"Skipping {record_name}: {e}")
                continue

        if not X_all:
            logger.warning(f"Skipping {snr_num} dB (files not found)")
            continue

        X_all = np.vstack(X_all)
        y_true = np.concatenate(y_true)
        rr_all = np.concatenate(rr_all)

        logger.info(
            f"Extracting TDA and DSP features for {len(X_all)} noisy beats on the fly..."
        )
        X_features = extract_all(X_all, y_true, rr_all, include_tda=True, n_jobs=4)

        y_pred = model.predict(X_features)

        acc = accuracy_score(y_true, y_pred)
        f2 = fbeta_score(y_true, y_pred, beta=2.0, average="weighted", zero_division=0)

        results[str(snr_num)] = {
            "accuracy": float(acc),
            "weighted_f2": float(f2),
            "total_beats": len(y_true),
        }

        logger.info(
            f"SNR {snr_num}dB -> Accuracy: {acc:.4f} | Weighted F2: {f2:.4f} ({len(y_true)} beats)"
        )

    results_path = os.path.join(
        project_root, "results", "nstdb_hybrid_lgbm_stress_test.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nResults saved to {results_path}")
