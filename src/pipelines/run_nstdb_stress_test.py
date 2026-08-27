import os
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score, fbeta_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_preparation.build_dataset import extract_raw_beats
from src.models.dsp_expert import DSPExpert

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    nstdb_dir = os.path.join(project_root, "Datasets", "nstdb")
    snr_levels = ['24', '18', '12', '06', '00', '_6']
    snr_numeric = [24, 18, 12, 6, 0, -6]
    records = ['118', '119']
    
    results = {}
    print("Starting NSTDB Stress Test for DSP Expert...")
    for snr, snr_num in zip(snr_levels, snr_numeric):
        print(f"\nEvaluating SNR: {snr_num} dB")
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
                print(f"  [Warning] Skipping {record_name}: {e}")
                continue
                
        if not X_all:
            print(f"Skipping {snr_num} dB (files not found)")
            continue
            
        X_all = np.vstack(X_all)
        y_true = np.concatenate(y_true)
        rr_all = np.concatenate(rr_all)
        
        # Run Expert
        expert = DSPExpert()
        y_pred = expert.predict(X_all, rr_all)
            
        acc = accuracy_score(y_true, y_pred)
        f2 = fbeta_score(y_true, y_pred, beta=2.0, average='weighted', zero_division=0)
        
        results[str(snr_num)] = {
            "accuracy": float(acc),
            "weighted_f2": float(f2),
            "total_beats": len(y_true)
        }
        
        print(f"SNR {snr_num}dB -> Accuracy: {acc:.4f} | Weighted F2: {f2:.4f} ({len(y_true)} beats)")
        
    results_path = os.path.join(project_root, "results", "nstdb_stress_test.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")
