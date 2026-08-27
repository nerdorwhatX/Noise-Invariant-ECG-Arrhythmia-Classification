import os
import sys
import json
import time
import numpy as np
import logging
from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix, classification_report

# Add project root to path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.dsp_expert import DSPExpert

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Evaluating Pure DSP-Expert (Geometry Only) on DS2")
    logger.info("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    results_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    logger.info("Loading DS2 data...")
    X_test = np.load(os.path.join(data_dir, 'DS2_X_raw.npy'))
    y_test = np.load(os.path.join(data_dir, 'DS2_y.npy'))

    rr_path = os.path.join(data_dir, 'DS2_rr.npy')
    if os.path.exists(rr_path):
        rr_test = np.load(rr_path)
    else:
        logger.warning("DS2_rr.npy not found, using uniform RR=300")
        rr_test = np.full(len(X_test), 300.0, dtype=np.float32)

    logger.info(f"Loaded {len(X_test)} beats.")

    # Initialize the DSP Expert
    expert = DSPExpert(window_size=10)

    # We will track execution time because TDA is slow
    start_time = time.time()

    # Batch predict
    logger.info("Starting batch DSP evaluation...")

    y_pred = expert.predict(X_test, rr_test)
    rule_counts = expert.rule_counts
    total_time = time.time() - start_time
    logger.info(f"Evaluation complete in {total_time:.1f} seconds.")

    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    f2_weighted = fbeta_score(y_test, y_pred, beta=2,
                              average='weighted', zero_division=0)
    f2_per_class = fbeta_score(
        y_test, y_pred, beta=2, average=None, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[
                                   'N', 'S', 'V', 'F'], output_dict=True, zero_division=0)

    print("\n--- RESULTS ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Weighted F2: {f2_weighted:.4f}")
    print("Per-class F2:")
    for i, cls in enumerate(['N', 'S', 'V', 'F']):
        print(f"  {cls}: {f2_per_class[i]:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nRule Trigger Breakdown:")
    for rule, count in sorted(rule_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rule}: {count}")

    # Save to JSON
    results = {
        'model': 'DSP-Expert (Geometry Only)',
        'dataset': 'DS2',
        'accuracy': acc,
        'weighted_f2': f2_weighted,
        'per_class_f2': {cls: f2_per_class[i] for i, cls in enumerate(['N', 'S', 'V', 'F'])},
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'rule_triggers': rule_counts,
        'execution_time_seconds': total_time
    }

    out_path = os.path.join(results_dir, 'dsp_expert_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
