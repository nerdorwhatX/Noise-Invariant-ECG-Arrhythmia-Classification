import os
import sys
import json
import time
import logging
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.models.hybrid_lgbm import HybridLGBMClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Hybrid (DSP+TDA+ML) Ensemble Training and Evaluation...")

    data_dir = os.path.join(project_root, "data")
    ds1_path = os.path.join(data_dir, "DS1_features.parquet")
    ds2_path = os.path.join(data_dir, "DS2_features.parquet")

    if not os.path.exists(ds1_path) or not os.path.exists(ds2_path):
        logger.error(
            "Feature files not found. Please run feature_engineering.py first."
        )
        return

    logger.info("Loading DS1 and DS2 features...")
    df_train = pd.read_parquet(ds1_path)
    df_test = pd.read_parquet(ds2_path)

    X_train_full = df_train.drop(columns=["label"]).values
    y_train_full = df_train["label"].values

    X_test = df_test.drop(columns=["label"]).values
    y_test = df_test["label"].values

    logger.info(f"Training shape: {X_train_full.shape}, Testing shape: {X_test.shape}")

    # Validation split for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    # Initialize and Train Model
    model = HybridLGBMClassifier(n_estimators=1500, learning_rate=0.05, max_depth=-1)

    start_time = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds.")

    # Save Model
    model_dir = os.path.join(project_root, "models_saved")
    model_path = os.path.join(model_dir, "hybrid_lgbm.pkl")
    model.save(model_path)

    # Evaluation
    logger.info("Evaluating on DS2...")
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f2 = fbeta_score(y_test, preds, beta=2, average="weighted")
    cm = confusion_matrix(y_test, preds)

    logger.info(f"Accuracy: {acc * 100:.4f}%")
    logger.info(f"Weighted F2: {f2:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Save Results
    results = {
        "model": "HybridLGBMClassifier",
        "accuracy": acc,
        "weighted_f2": f2,
        "confusion_matrix": cm.tolist(),
        "train_time_seconds": train_time,
    }

    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "hybrid_lgbm_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
