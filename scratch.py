import wfdb
import os
import numpy as np

mitdb_dir = "data/raw/mitdb"
records = [f.split('.')[0] for f in os.listdir(mitdb_dir) if f.endswith('.dat')]

DS1_PATIENTS_OLD = {'101', '106', '108', '109', '112', '114', '115', '116', '122', '124', '200', '202', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'}

AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'S': 1, 'A': 1, 'a': 1, 'J': 1,
    'V': 2, 'E': 2,
    'F': 3,
}

VALID_SYMBOLS = set(AAMI_MAPPING.keys())

ds2_beats = 0
ds2_records = []

for rec in records:
    if rec not in DS1_PATIENTS_OLD:
        ds2_records.append(rec)
        ann = wfdb.rdann(os.path.join(mitdb_dir, rec), 'atr')
        
        signal_len = wfdb.rdrecord(os.path.join(mitdb_dir, rec)).p_signal.shape[0]
        
        window_left = 90
        window_right = 144
        
        for i, sym in enumerate(ann.symbol):
            if sym in VALID_SYMBOLS:
                samp = ann.sample[i]
                if samp - window_left >= 0 and samp + window_right < signal_len:
                    ds2_beats += 1

print(f"Old DS2 records: {ds2_records}")
print(f"Old DS2 total valid beats: {ds2_beats}")
