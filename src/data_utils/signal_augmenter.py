# signal_augmenter.py
# This script adds small random changes (like time shifts and noise) to the ECG beats.
# We use this to generate more data for the rare heartbeat classes so our model trains better.

import numpy as np
from scipy.interpolate import interp1d
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def time_shift(beat, max_shift=10):
    """Shift the beat left or right by a random number of samples."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(beat, shift)


def amplitude_scale(beat, low=0.7, high=1.3):
    """Scale amplitude by a random factor — simulates inter-patient gain variation."""
    scale = np.random.uniform(low, high)
    return beat * scale


def time_warp(beat, low=0.85, high=1.15):
    """
    Non-uniformly stretch/compress the beat via cubic interpolation.
    Simulates natural heart-rate variability in beat duration.
    """
    n = len(beat)
    # Create a smooth random warping path
    n_knots = 5
    knot_positions = np.linspace(0, n - 1, n_knots)
    knot_warps = np.random.uniform(low, high, n_knots)
    knot_warps[0] = 1.0  # anchor start
    knot_warps[-1] = 1.0  # anchor end

    # Build cumulative warp
    warp_fn = interp1d(
        knot_positions, knot_warps, kind="linear", fill_value="extrapolate"
    )
    warp_factors = warp_fn(np.arange(n))
    warped_indices = np.cumsum(warp_factors)
    warped_indices = (
        warped_indices / warped_indices[-1] * (n - 1)
    )  # normalize to [0, n-1]

    # Resample
    original_fn = interp1d(np.arange(n), beat, kind="cubic", fill_value="extrapolate")
    return original_fn(warped_indices).astype(np.float32)


def additive_jitter(beat, sigma=0.02):
    """Add tiny Gaussian noise to smooth decision boundaries."""
    return beat + np.random.normal(0, sigma, len(beat)).astype(np.float32)


def augment_beat(beat, rr_current, rr_prev, rr_next):
    """
    Apply a random combination of augmentations to a single beat.
    Returns the augmented beat and UNCHANGED RR intervals
    (RR context is preserved from the original beat).
    """
    aug = beat.copy()

    # Apply 2-3 random transforms
    if np.random.rand() > 0.3:
        aug = time_shift(aug, max_shift=8)
    if np.random.rand() > 0.3:
        aug = amplitude_scale(aug, 0.75, 1.25)
    if np.random.rand() > 0.3:
        aug = time_warp(aug, 0.9, 1.1)
    if np.random.rand() > 0.5:
        aug = additive_jitter(aug, 0.015)

    return aug


def balance_dataset(X, y, rr, rec=None, target_ratio=1.0, seed=42):
    # Balances the dataset by making copies of the rare beats with slight variations.
    np.random.seed(seed)

    classes, counts = np.unique(y, return_counts=True)
    max_count = int(np.max(counts) * target_ratio)

    total_size = sum(max(c, max_count) for c in counts)

    X_bal = np.empty((total_size, X.shape[1]), dtype=np.float32)
    y_bal = np.empty(total_size, dtype=np.int64)
    rr_bal = np.empty(total_size, dtype=np.float32)
    rec_bal = np.empty(total_size, dtype=rec.dtype) if rec is not None else None

    # Copy original data
    orig_size = len(X)
    X_bal[:orig_size] = X
    y_bal[:orig_size] = y
    rr_bal[:orig_size] = rr
    if rec is not None:
        rec_bal[:orig_size] = rec

    idx = orig_size

    for cls in classes:
        cls_mask = y == cls
        cls_count = np.sum(cls_mask)
        if cls_count >= max_count:
            continue

        n_needed = max_count - cls_count
        cls_indices = np.where(cls_mask)[0]

        for _ in range(n_needed):
            rand_idx = np.random.choice(cls_indices)

            rr_cur = rr[rand_idx]
            rr_prev = rr[rand_idx - 1] if rand_idx > 0 else rr_cur
            rr_next = rr[rand_idx + 1] if rand_idx < len(rr) - 1 else rr_cur

            X_bal[idx] = augment_beat(X[rand_idx], rr_cur, rr_prev, rr_next)
            y_bal[idx] = cls
            rr_bal[idx] = rr_cur
            if rec is not None:
                rec_bal[idx] = rec[rand_idx]
            idx += 1

    if rec is not None:
        return X_bal, y_bal, rr_bal, rec_bal
    return X_bal, y_bal, rr_bal


def synthesize_fusion_beats(X, y, rr, rec, num_fusion_beats, seed=42):
    # Creates fake 'Fusion' beats by mathematically mixing a Normal beat and a Ventricular beat
    # from the exact same patient to keep it realistic.
    rng = np.random.RandomState(seed)

    synth_beats = []
    synth_labels = []
    synth_rrs = []
    synth_recs = []

    patients = np.unique(rec)

    patient_nv_indices = {}
    for p in patients:
        p_mask = rec == p
        n_idx = np.where(p_mask & (y == 0))[0]  # N
        v_idx = np.where(p_mask & (y == 2))[0]  # V
        if len(n_idx) > 0 and len(v_idx) > 0:
            patient_nv_indices[p] = (n_idx, v_idx)

    valid_patients = list(patient_nv_indices.keys())
    if not valid_patients:
        logger.warning("No patients have both N and V beats to synthesize F beats.")
        return X, y, rr, rec

    for _ in range(num_fusion_beats):
        p = rng.choice(valid_patients)
        n_idx, v_idx = patient_nv_indices[p]

        n_beat_idx = rng.choice(n_idx)
        v_beat_idx = rng.choice(v_idx)

        n_beat = X[n_beat_idx]
        v_beat = X[v_beat_idx]

        alpha = rng.uniform(0.3, 0.7)
        f_synth = alpha * n_beat + (1 - alpha) * v_beat
        rr_synth = alpha * rr[n_beat_idx] + (1 - alpha) * rr[v_beat_idx]

        synth_beats.append(f_synth.astype(np.float32))
        synth_labels.append(3)  # F
        synth_rrs.append(np.float32(rr_synth))
        synth_recs.append(p)

    return (
        np.vstack([X, np.array(synth_beats, dtype=np.float32)]),
        np.concatenate([y, np.array(synth_labels, dtype=np.int64)]),
        np.concatenate([rr, np.array(synth_rrs, dtype=np.float32)]),
        np.concatenate([rec, np.array(synth_recs, dtype=rec.dtype)]),
    )
