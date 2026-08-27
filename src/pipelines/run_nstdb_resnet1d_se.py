import os
import sys
import json
import numpy as np
import logging
from sklearn.metrics import accuracy_score, fbeta_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.resnet1d_se import ResNet1DClassifier
from src.data_preparation.build_dataset import extract_raw_beats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", ".."))
    nstdb_dir = os.path.join(project_root, "Datasets", "nstdb")
    model_path = os.path.join(project_root, "models_saved", "resnet1d_se.pth")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train it first.")
        sys.exit(1)

    snr_levels = ['24', '18', '12', '06', '00', '_6']
    snr_numeric = [24, 18, 12, 6, 0, -6]
    records = ['118', '119']

    # Load Model
    logger.info("Loading trained ResNet1D-SE model...")
    model = ResNet1DClassifier()
    model.load(model_path)

    results = {}
    logger.info("Starting NSTDB Stress Test for ResNet1D-SE...")
    
    for snr, snr_num in zip(snr_levels, snr_numeric):
        logger.info(f"\nEvaluating SNR: {snr_num} dB")
        X_all, y_true = [], []

        for rec in records:
            record_name = f"{rec}e{snr}"
            path = os.path.join(nstdb_dir, record_name)
            try:
                X, y, _ = extract_raw_beats(path)
                X_all.append(X)
                y_true.append(y)
            except Exception as e:
                logger.warning(f"Skipping {record_name}: {e}")
                continue

        if not X_all:
            logger.warning(f"Skipping {snr_num} dB (files not found)")
            continue

        X_all = np.vstack(X_all)
        y_true = np.concatenate(y_true)

        # Reshape data to (N, 1, Length) internally handled by predict()
        y_pred = model.predict(X_all)

        acc = accuracy_score(y_true, y_pred)
        f2 = fbeta_score(y_true, y_pred, beta=2.0,
                         average='weighted', zero_division=0)

        results[str(snr_num)] = {
            "accuracy": float(acc),
            "weighted_f2": float(f2),
            "total_beats": len(y_true)
        }

        logger.info(
            f"SNR {snr_num}dB -> Accuracy: {acc:.4f} | Weighted F2: {f2:.4f} ({len(y_true)} beats)")

    results_path = os.path.join(
        project_root, "results", "nstdb_resnet1d_se_stress_test.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nResults saved to {results_path}")
