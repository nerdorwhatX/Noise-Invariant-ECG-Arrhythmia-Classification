"""
feature_engineering.py — Extracts ~89 mathematically derived features per beat.

Produces a tabular (Parquet) dataset from raw time-series waveforms.
Feature groups:
  1. Multi-Tau Phase-Space Geometry (τ=4,8,16):   24 features
  2. RR-Context Window (10-beat surrounding):       8 features
  3. Regional Morphology (P/QRS/T):                12 features
  4. Statistical Moments:                           8 features
  5. Template Correlation:                          4 features
  6. Wavelet Energy (4-level DWT):                  8 features
  7. Autocorrelation:                               4 features
  8. Takens' Phase-Space Excursion:                  1 feature
  9. TDA (Persistent Homology H0+H1):              20 features
  ─────────────────────────────────────────────────────
  Total:                                           ~89 features

Usage:
    python feature_engineering.py
"""
import os
import sys
import numpy as np
import pandas as pd
import logging
from scipy.stats import skew, kurtosis
from scipy.signal import correlate
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 360  # MIT-BIH sampling frequency
project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(project_root, "data")


# ============================================================================
# 1. Phase-Space Geometric Features (from DSP-Expert / david_pipeline.py)
# ============================================================================

def _extract_geometric_features(beat, tau=8):
    """Takens embedding → convex hull + PCA ellipse + angular dispersion."""
    if len(beat) <= tau + 2:
        return [0.0] * 8

    x = beat[:-tau]
    y = beat[tau:]
    points = np.column_stack((x, y))
    cx, cy = np.mean(x), np.mean(y)

    # Convex hull
    try:
        if len(np.unique(points, axis=0)) < 4:
            raise ValueError
        hull = ConvexHull(points)
        area, perimeter = hull.volume, hull.area
    except (QhullError, ValueError):
        area, perimeter = 0.0, 0.0

    # PCA ellipse
    try:
        cov_mat = np.cov(x, y)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov_mat))[::-1]
        sd2 = np.sqrt(max(eigenvalues[0], 1e-12))
        sd1 = np.sqrt(max(eigenvalues[1], 1e-12))
        eccentricity = np.sqrt(1.0 - (sd1**2 / sd2**2)) if sd2 > sd1 else 0.0
    except (np.linalg.LinAlgError, ValueError):
        sd1, sd2, eccentricity = 0.0, 0.0, 0.0

    # Angular dispersion
    angles = np.arctan2(y - cy, x - cx)
    dispersion = 1.0 - np.abs(np.mean(np.exp(1j * angles)))
    compactness = perimeter / (area + 1e-8)

    return [area, cx, cy, eccentricity, sd1, sd2, dispersion, compactness]


def phase_space_features(beat, taus=(4, 8, 16)):
    """8 features × 3 taus = 24 features."""
    feats = []
    for tau in taus:
        feats.extend(_extract_geometric_features(beat, tau=tau))
    return np.array(feats, dtype=np.float32)


# ============================================================================
# 2. RR-Context Window
# ============================================================================

def rr_context_features(rr_current, rr_prev, rr_next, rr_window):
    """8 features from the surrounding RR-interval context."""
    rr_mean = np.mean(rr_window) if len(rr_window) > 0 else rr_current
    rr_std = np.std(rr_window) if len(rr_window) > 1 else 0.0

    prematurity = rr_current / (rr_mean + 1e-6)
    compensatory = rr_next / (rr_mean + 1e-6)
    rr_ratio_prev = rr_current / (rr_prev + 1e-6)
    rr_ratio_next = rr_next / (rr_current + 1e-6)
    is_shortest = 1.0 if rr_current <= np.min(rr_window) else 0.0
    is_longest = 1.0 if rr_current >= np.max(rr_window) else 0.0

    return np.array([
        rr_current, prematurity, compensatory,
        rr_ratio_prev, rr_ratio_next,
        rr_std, is_shortest, is_longest
    ], dtype=np.float32)


# ============================================================================
# 3. Regional Morphology (P-wave, QRS, T-wave)
# ============================================================================

def regional_features(beat, fs=FS):
    """12 features: energy/peak/p2p for P, QRS, T regions + 3 ratios."""
    center = len(beat) // 2

    p_start = max(0, center - int(0.25 * fs))
    p_end = max(0, center - int(0.05 * fs))
    p_region = beat[p_start:p_end] if p_end > p_start else np.zeros(1)

    qrs_start = max(0, center - int(0.05 * fs))
    qrs_end = min(len(beat), center + int(0.05 * fs))
    qrs_region = beat[qrs_start:qrs_end] if qrs_end > qrs_start else np.zeros(
        1)

    t_start = min(len(beat), center + int(0.10 * fs))
    t_end = min(len(beat), center + int(0.30 * fs))
    t_region = beat[t_start:t_end] if t_end > t_start else np.zeros(1)

    def region_stats(region):
        return np.sum(region**2), np.max(np.abs(region)), np.max(region) - np.min(region)

    p_e, p_pk, p_p2p = region_stats(p_region)
    qrs_e, qrs_pk, qrs_p2p = region_stats(qrs_region)
    t_e, t_pk, t_p2p = region_stats(t_region)

    total_e = np.sum(beat**2) + 1e-8

    return np.array([
        p_e, p_pk, p_p2p,
        qrs_e, qrs_pk, qrs_p2p,
        t_e, t_pk, t_p2p,
        qrs_e / (p_e + 1e-8),
        qrs_e / (t_e + 1e-8),
        qrs_e / total_e
    ], dtype=np.float32)


# ============================================================================
# 4. Statistical Moments
# ============================================================================

def stat_features(beat):
    """8 features: mean, std, skew, kurtosis, p2p, energy, zero-crossings, form factor."""
    b_mean = np.mean(beat)
    b_std = np.std(beat) + 1e-8
    b_skew = skew(beat)
    b_kurt = kurtosis(beat)
    p2p = np.max(beat) - np.min(beat)
    energy = np.sum(beat**2)
    zc = np.sum(np.diff(np.sign(beat)) != 0)
    form_factor = np.std(np.diff(beat)) / b_std

    return np.array([b_mean, b_std, b_skew, b_kurt, p2p, energy, zc, form_factor], dtype=np.float32)


# ============================================================================
# 5. Template Correlation
# ============================================================================

def build_templates(X, y, num_classes=4):
    """Build per-class average beat templates from training data."""
    templates = []
    for cls in range(num_classes):
        cls_beats = X[y == cls]
        if len(cls_beats) > 0:
            templates.append(np.mean(cls_beats, axis=0).astype(np.float32))
        else:
            templates.append(np.zeros(X.shape[1], dtype=np.float32))
    return templates


def template_correlation(beat, templates):
    """4 features: Pearson correlation against each class template."""
    corrs = []
    for tmpl in templates:
        tmpl = np.asarray(tmpl, dtype=float)
        min_len = min(len(tmpl), len(beat))
        c = np.corrcoef(beat[:min_len], tmpl[:min_len])[0, 1]
        corrs.append(c if np.isfinite(c) else 0.0)
    return np.array(corrs, dtype=np.float32)


# ============================================================================
# 6. Wavelet Energy
# ============================================================================

def wavelet_features(beat, wavelet='db4', level=4):
    """8 features: normalized DWT energies + ratios + entropy."""
    try:
        import pywt
        coeffs = pywt.wavedec(np.array(beat), wavelet, level=level)
        energies = [np.sum(c**2) for c in coeffs]
        total = sum(energies) + 1e-8
        norm_e = [e / total for e in energies]
        approx_detail = energies[0] / (sum(energies[1:]) + 1e-8)
        high_low = (energies[1] + energies[2]) / (energies[3] +
                                                  energies[4] + 1e-8) if len(energies) >= 5 else 0.0
        detail_entropy = - \
            np.sum([e/total * np.log2(e/total + 1e-12) for e in energies[1:]])
        return np.array(norm_e + [approx_detail, high_low, detail_entropy], dtype=np.float32)
    except ImportError:
        return np.zeros(8, dtype=np.float32)


# ============================================================================
# 7. Autocorrelation
# ============================================================================

def autocorr_features(beat):
    """4 features: lag-1, zero-crossing, half-life, secondary peak."""
    beat_centered = beat - np.mean(beat)
    ac = correlate(beat_centered, beat_centered, mode='full')
    ac = ac[len(ac) // 2:]
    ac_norm = ac / (ac[0] + 1e-8)

    ac_lag1 = ac_norm[1] if len(ac_norm) > 1 else 0.0

    zero_cross = len(ac_norm)
    for i in range(1, len(ac_norm)):
        if ac_norm[i] <= 0:
            zero_cross = i
            break

    half_life = len(ac_norm)
    for i in range(1, len(ac_norm)):
        if ac_norm[i] <= 0.5:
            half_life = i
            break

    secondary_peak = np.max(ac_norm[5:]) if len(ac_norm) > 10 else 0.0

    return np.array([
        ac_lag1, zero_cross / len(beat), half_life / len(beat), secondary_peak
    ], dtype=np.float32)


# ============================================================================
# 8. Takens' Phase-Space Excursion
# ============================================================================

def takens_excursion(i, rr, window=10):
    """1 feature: Euclidean distance of 3D RR embedding from local centroid."""
    if i < 2:
        return 0.0
    v_i = np.array([rr[i], rr[i-1], rr[i-2]])
    embeddings = []
    for j in range(max(2, i - window), i):
        embeddings.append([rr[j], rr[j-1], rr[j-2]])
    if not embeddings:
        return 0.0
    centroid = np.mean(embeddings, axis=0)
    return float(np.linalg.norm(v_i - centroid))


# ============================================================================
# 9. TDA (Persistent Homology) — optional, requires ripser
# ============================================================================

def tda_features(beat, tau=8):
    """20 features: H0 (5) + H1 (9) + Cross (6) from persistence diagrams."""
    try:
        from ripser import ripser
    except ImportError:
        return np.zeros(20, dtype=np.float32)

    x = beat[:-tau]
    y = beat[tau:]
    pc = np.column_stack([x, y]).astype(np.float64)

    if len(pc) > 200:
        idx = np.linspace(0, len(pc) - 1, 200, dtype=int)
        pc = pc[idx]

    result = ripser(pc, maxdim=1, thresh=np.inf)
    dgms = result['dgms']

    h0 = dgms[0] if len(dgms) > 0 else np.array([]).reshape(0, 2)
    h1 = dgms[1] if len(dgms) > 1 else np.array([]).reshape(0, 2)

    def _stats(dgm):
        finite = dgm[np.isfinite(dgm[:, 1])] if len(
            dgm) > 0 else np.array([]).reshape(0, 2)
        if len(finite) == 0:
            return [0]*9
        lifetimes = np.maximum(finite[:, 1] - finite[:, 0], 0)
        total = np.sum(lifetimes) + 1e-12
        probs = lifetimes / total
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        midlife = np.mean((finite[:, 0] + finite[:, 1]) / 2.0)
        return [
            len(finite), float(np.max(lifetimes)), float(np.mean(lifetimes)),
            float(np.std(lifetimes)), float(total), float(entropy),
            float(np.max(finite[:, 0])), float(
                np.max(finite[:, 1])), float(midlife)
        ]

    s0 = _stats(h0)
    s1 = _stats(h1)

    # H0: 5 features
    f_h0 = s0[:5]
    # H1: 9 features
    f_h1 = s1[:9]
    # Cross: 6 features
    h1_h0_persist = s1[4] / (s0[4] + 1e-8)
    h1_h0_count = s1[0] / (s0[0] + 1e-8)
    entropy_ratio = s1[5] / (s0[5] + 1e-8) if len(s0) > 5 else 0.0
    total_features = s0[0] + s1[0]
    dominant = max(s0[1], s1[1])
    h1_finite = h1[np.isfinite(h1[:, 1])] if len(
        h1) > 0 else np.array([]).reshape(0, 2)
    if len(h1_finite) > 1:
        lt = h1_finite[:, 1] - h1_finite[:, 0]
        persist_range = float(np.max(lt) - np.min(lt))
    else:
        persist_range = 0.0
    f_cross = [h1_h0_persist, h1_h0_count, entropy_ratio,
               total_features, dominant, persist_range]

    return np.array(f_h0 + f_h1 + f_cross, dtype=np.float32)


# ============================================================================
# Master Feature Extraction
# ============================================================================

def extract_single_beat(i, X, rr, templates, include_tda=True):
    """Extract all features for beat i."""
    beat = X[i]
    n = len(X)
    rr_cur = rr[i]
    rr_prev = rr[i - 1] if i > 0 else rr_cur
    rr_next = rr[i + 1] if i < n - 1 else rr_cur

    # RR window
    half_w = 5
    w_start = max(0, i - half_w)
    w_end = min(n, i + half_w + 1)
    rr_window = rr[w_start:w_end]

    f1 = phase_space_features(beat)                               # 24
    f2 = rr_context_features(rr_cur, rr_prev, rr_next, rr_window)  # 8
    f3 = regional_features(beat)                                  # 12
    f4 = stat_features(beat)                                      # 8
    f5 = template_correlation(beat, templates)                    # 4
    f6 = wavelet_features(beat)                                   # 8
    f7 = autocorr_features(beat)                                  # 4
    f_takens = np.array([takens_excursion(i, rr)], dtype=np.float32)  # 1

    parts = [f1, f2, f3, f4, f5, f6, f7, f_takens]

    if include_tda:
        f8 = tda_features(beat, tau=8)                            # 20
        parts.append(f8)

    return np.concatenate(parts)


def extract_all(X, y, rr, include_tda=True, n_jobs=4):
    """Extract features for all beats and return a DataFrame."""
    templates = build_templates(X, y)

    tda_str = "+TDA" if include_tda else ""
    n_feat = "~89" if include_tda else "~69"
    logger.info(f"Extracting {len(X)} beats × {n_feat} DSP{tda_str} features using {n_jobs} cores...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(extract_single_beat)(i, X, rr, templates, include_tda)
            for i in range(len(X))
        )

    features = np.array(results, dtype=np.float32)

    # Replace NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"Feature matrix shape: {features.shape}")
    return features


def main():
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline")
    logger.info("=" * 60)

    # Check for ripser
    try:
        import ripser
        include_tda = True
        logger.info("ripser found — TDA features ENABLED")
    except ImportError:
        include_tda = False
        logger.info("ripser NOT found — TDA features DISABLED (69 features only)")

    for split in ["DS1", "DS2"]:
        logger.info(f"\n--- Processing {split} ---")
        X = np.load(os.path.join(DATA_DIR, f"{split}_X_raw.npy"))
        y = np.load(os.path.join(DATA_DIR, f"{split}_y.npy"))

        # RR intervals: for DS1 balanced, augmented beats reuse donor RR
        # We load or reconstruct RR from the raw data
        rr_path = os.path.join(DATA_DIR, f"{split}_rr.npy")
        if os.path.exists(rr_path):
            rr = np.load(rr_path)
        else:
            # Approximate: use constant RR for missing data
            logger.warning(f"{split}_rr.npy not found, using uniform RR=300")
            rr = np.full(len(X), 300.0, dtype=np.float32)

        features = extract_all(X, y, rr, include_tda=include_tda, n_jobs=4)

        # Save as Parquet
        out_path = os.path.join(DATA_DIR, f"{split}_features.parquet")
        df = pd.DataFrame(features)
        df['label'] = y
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved: {out_path} ({df.shape[0]} rows × {df.shape[1]} cols)")

        # Also save as .npy for convenience
        np.save(os.path.join(DATA_DIR, f"{split}_features.npy"), features)

    logger.info("[DONE] Feature engineering complete!")


if __name__ == "__main__":
    main()
