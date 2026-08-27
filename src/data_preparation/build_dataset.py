import os
import sys
import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed

# Import augmentations
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from src.data_utils.signal_augmenter import augment_beat, synthesize_fusion_beats

# AAMI Mapping
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'S': 1, 'A': 1, 'a': 1, 'J': 1,          # Supraventricular
    'V': 2, 'E': 2,                          # Ventricular
    'F': 3,                                  # Fusion
}
VALID_SYMBOLS = set(AAMI_MAPPING.keys())

# Modified splits: Removed 118, 119 (moved to DS2/test) and swapped in 200, 202 (moved to DS1/train)
# to avoid data leakage when testing on NSTDB (which uses 118, 119).
DS1_PATIENTS = {'101', '106', '108', '109', '112', '114', '115', '116', '122', '124', '200', '202',
                '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'}

def extract_raw_beats(record_path, window_left=90, window_right=144):
    """Extract raw segments around R-peaks (no DSP filtering)."""
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')
    
    # Use only MLII lead (usually channel 0)
    sig_name = record.sig_name
    ch_idx = sig_name.index('MLII') if 'MLII' in sig_name else 0
    signal = record.p_signal[:, ch_idx]
    
    beats, labels, rrs = [], [], []
    ann_sym = np.array(annotation.symbol)
    ann_samp = np.array(annotation.sample)
    
    for i in range(1, len(ann_sym) - 1):
        sym = ann_sym[i]
        if sym in VALID_SYMBOLS:
            samp = ann_samp[i]
            # Ensure window is within signal bounds
            if samp - window_left >= 0 and samp + window_right < len(signal):
                beat = signal[samp - window_left : samp + window_right]
                rr_cur = ann_samp[i] - ann_samp[i-1]
                
                beats.append(beat)
                labels.append(AAMI_MAPPING[sym])
                rrs.append(rr_cur)
                
    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(rrs, dtype=np.float32)

def balance_nv(X, y, rr, seed=42):
    """Aggressively balances V-class to match N-class count via signal augmentation."""
    np.random.seed(seed)
    n_mask = (y == 0)
    v_mask = (y == 2)
    
    n_count = np.sum(n_mask)
    v_count = np.sum(v_mask)
    
    X_bal = list(X)
    y_bal = list(y)
    rr_bal = list(rr)
    
    if v_count > 0 and v_count < n_count:
        needed = n_count - v_count
        v_indices = np.where(v_mask)[0]
        
        print(f"    Augmenting V class: adding {needed} beats...")
        for _ in range(needed):
            idx = np.random.choice(v_indices)
            rr_cur = rr[idx]
            # using rr_cur for prev and next for simplicity in augmentation
            aug_beat = augment_beat(X[idx], rr_cur, rr_cur, rr_cur)
            
            X_bal.append(aug_beat)
            y_bal.append(2)
            rr_bal.append(rr_cur)
            
    return np.array(X_bal, dtype=np.float32), np.array(y_bal, dtype=np.int64), np.array(rr_bal, dtype=np.float32)

def main():
    mitdb_dir = os.path.join(project_root, "Datasets", "mitdb")
    out_dir = os.path.join(project_root, "data")
    os.makedirs(out_dir, exist_ok=True)
    
    records = [f.split('.')[0] for f in os.listdir(mitdb_dir) if f.endswith('.dat')]
    
    ds1_X, ds1_y, ds1_rr = [], [], []
    ds2_X, ds2_y, ds2_rr = [], [], []
    
    print("Extracting raw signals...")
    for rec in sorted(records):
        path = os.path.join(mitdb_dir, rec)
        try:
            X, y, rr = extract_raw_beats(path)
            if rec in DS1_PATIENTS:
                ds1_X.append(X); ds1_y.append(y); ds1_rr.append(rr)
            else:
                ds2_X.append(X); ds2_y.append(y); ds2_rr.append(rr)
        except Exception as e:
            print(f"Skipping {rec}: {e}")
            
    ds1_X = np.vstack(ds1_X)
    ds1_y = np.concatenate(ds1_y)
    ds1_rr = np.concatenate(ds1_rr)
    
    ds2_X = np.vstack(ds2_X)
    ds2_y = np.concatenate(ds2_y)
    ds2_rr = np.concatenate(ds2_rr)
    
    print(f"DS1 raw: N={np.sum(ds1_y==0)}, S={np.sum(ds1_y==1)}, V={np.sum(ds1_y==2)}, F={np.sum(ds1_y==3)}")
    print("Balancing DS1 (N vs V)...")
    ds1_X_bal, ds1_y_bal, ds1_rr_bal = balance_nv(ds1_X, ds1_y, ds1_rr)
    
    print(f"DS1 balanced: N={np.sum(ds1_y_bal==0)}, S={np.sum(ds1_y_bal==1)}, V={np.sum(ds1_y_bal==2)}, F={np.sum(ds1_y_bal==3)}")
    
    # Save raw arrays
    print("Saving .npy datasets...")
    np.save(os.path.join(out_dir, "DS1_X_raw.npy"), ds1_X_bal)
    np.save(os.path.join(out_dir, "DS1_y.npy"), ds1_y_bal)
    np.save(os.path.join(out_dir, "DS1_rr.npy"), ds1_rr_bal)
    
    np.save(os.path.join(out_dir, "DS2_X_raw.npy"), ds2_X)
    np.save(os.path.join(out_dir, "DS2_y.npy"), ds2_y)
    np.save(os.path.join(out_dir, "DS2_rr.npy"), ds2_rr)
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
