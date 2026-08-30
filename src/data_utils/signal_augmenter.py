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

