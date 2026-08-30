import os
import sys
import numpy as np
import json
import logging
import time
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    fbeta_score,
)

# Ensure the root directory is accessible for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.models.resnet1d_se import ResNet1DClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting ResNet1D-SE (Pure DL Baseline) Training and Evaluation...")

    # 1. Load Datasets
    logger.info("Loading training dataset (DS1) and testing dataset (DS2)...")
    ds1_path = os.path.join(project_root, "data", "DS1_X_raw.npy")
    ds1_y_path = os.path.join(project_root, "data", "DS1_y.npy")
    ds2_path = os.path.join(project_root, "data", "DS2_X_raw.npy")
    ds2_y_path = os.path.join(project_root, "data", "DS2_y.npy")

    if not all(os.path.exists(p) for p in [ds1_path, ds1_y_path, ds2_path, ds2_y_path]):
        logger.error("Datasets not found! Please run build_dataset.py first.")
        return

    X_train = np.load(ds1_path)
    y_train = np.load(ds1_y_path)
    X_test = np.load(ds2_path)
    y_test = np.load(ds2_y_path)

    logger.info(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

    # 2. Initialize and Train Model
    model = ResNet1DClassifier(epochs=100, batch_size=256, lr=0.001)

    start_time = time.time()
    logger.info("Training ResNet1D-SE (This may take a while)...")
    model.fit(X_train, y_train, class_weights="balanced")
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds.")

    # Optional: Save the model
    model_dir = os.path.join(project_root, "models_saved")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "resnet1d_se.pth"))

    # 3. Evaluate Model
    logger.info("Evaluating on DS2...")
    eval_start_time = time.time()
    predictions = model.predict(X_test)
    eval_time = time.time() - eval_start_time

    # 4. Metrics
    accuracy = accuracy_score(y_test, predictions)
    weighted_f2 = fbeta_score(y_test, predictions, beta=2, average="weighted")
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    logger.info(f"Accuracy: {accuracy:.4%}")
    logger.info(f"Weighted F2: {weighted_f2:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")

    # 5. Save Results
    results = {
        "model": "ResNet1D-SE (Pure DL)",
        "dataset": "DS2",
        "accuracy": accuracy,
        "weighted_f2": weighted_f2,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "training_time_seconds": training_time,
        "execution_time_seconds": eval_time,
    }

    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "resnet1d_se_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
