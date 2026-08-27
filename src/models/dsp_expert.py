"""
dsp_expert.py

Pure DSP-Expert Classification Model.

This module implements the classical Digital Signal Processing (DSP) approach 
for ECG Arrhythmia Classification without using Machine Learning.

It utilizes zero-phase digital filtering and Takens' delay embedding for phase-space 
geometry. It relies on a deterministic, patient-adaptive rule-based expert system 
based on handcrafted thresholds.
"""

import numpy as np
from scipy.signal import filtfilt, butter
from scipy.spatial import ConvexHull

def zero_phase_filter(beat, fs=360):
    """
    Applies a zero-phase Butterworth bandpass filter (0.5 - 40 Hz) to remove
    baseline wander and high-frequency noise without phase distortion.
    """
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 40.0 / nyq
    b, a = butter(3, [low, high], btype='band')
    return filtfilt(b, a, beat)

def extract_geometric_features(beat, tau=8):
    """
    Reconstructs the phase-space trajectory and extracts geometric properties.
    """
    x = beat[:-tau]
    y = beat[tau:]
    
    if len(x) < 3:
        return {'area': 0.0, 'centroid_x': 0.0, 'centroid_y': 0.0, 
                'eccentricity': 0.0, 'sd1': 0.0, 'sd2': 0.0, 
                'dispersion': 0.0, 'compactness': 0.0}
        
    points = np.column_stack((x, y))
    
    try:
        hull = ConvexHull(points)
        area = hull.volume  # In 2D, volume is area
        perimeter = hull.area # In 2D, area is perimeter length
    except Exception:
        area = 0.0
        perimeter = 0.0
        
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    
    cov = np.cov(x, y)
    if cov.shape == (2, 2):
        eigenvalues, _ = np.linalg.eigh(cov)
        sd1 = np.sqrt(max(eigenvalues[0], 0))
        sd2 = np.sqrt(max(eigenvalues[1], 0))
        eccentricity = np.sqrt(1 - (sd1**2 / (sd2**2 + 1e-8))) if sd2 > sd1 else 0.0
    else:
        sd1 = sd2 = eccentricity = 0.0
        
    dispersion = np.mean(np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2))
    compactness = (perimeter**2) / (4 * np.pi * area + 1e-8)
    
    return {
        'area': area,
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'eccentricity': eccentricity,
        'sd1': sd1,
        'sd2': sd2,
        'dispersion': dispersion,
        'compactness': compactness
    }


class DSPExpert:
    """
    Adaptive, per-patient rule-based classifier using relative phase-space ratios.
    """
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self._reset_reference()
        
        # Hardcoded empirical thresholds for geometric ratios
        self.theta_area_high = 2.0     
        self.theta_area_low = 0.3      
        self.theta_shift = 0.8         
        self.theta_ecc_dev = 0.5       
        self.theta_disp_dev = 0.3      
        self.theta_rr_irreg = 0.20     
        self.theta_afib = 0.5          

    def _reset_reference(self):
        """Clears the patient-specific sliding window."""
        self.ref_areas = []
        self.ref_centroids_x = []
        self.ref_centroids_y = []
        self.ref_eccs = []
        self.ref_sd1 = []
        self.ref_dispersions = []
        self.recent_ratio_areas = []
        self.rule_counts = {}

    def _update_reference(self, features):
        """Adds a normal beat to the sliding reference window."""
        self.ref_areas.append(features['area'])
        self.ref_centroids_x.append(features['centroid_x'])
        self.ref_centroids_y.append(features['centroid_y'])
        self.ref_eccs.append(features['eccentricity'])
        self.ref_sd1.append(features['sd1'])
        self.ref_dispersions.append(features['dispersion'])
        
        if len(self.ref_areas) > self.window_size:
            self.ref_areas.pop(0)
            self.ref_centroids_x.pop(0)
            self.ref_centroids_y.pop(0)
            self.ref_eccs.pop(0)
            self.ref_sd1.pop(0)
            self.ref_dispersions.pop(0)
            
    def _has_reference(self):
        return len(self.ref_areas) >= 3

    def classify_beat(self, raw_beat, rr_current, rr_prev):
        """
        Classifies a single raw ECG beat using purely mathematical DSP rules.
        """
        # Step 1: Classical DSP Filtering
        filt_beat = zero_phase_filter(raw_beat)
        
        # Step 2: Feature Extraction (Geometry Only)
        features = extract_geometric_features(filt_beat, tau=8)
        
        # Step 3: Rule-Based Evaluation
        if not self._has_reference():
            self._update_reference(features)
            return 'N', {'reason': 'warmup'}
            
        # -- Compute Geometry Medians --
        med_area = np.median(self.ref_areas)
        med_cx = np.median(self.ref_centroids_x)
        med_cy = np.median(self.ref_centroids_y)
        med_ecc = np.median(self.ref_eccs)
        med_disp = np.median(self.ref_dispersions)
        
        ratio_area = features['area'] / (med_area + 1e-8)
        shift = np.sqrt((features['centroid_x'] - med_cx)**2 + 
                        (features['centroid_y'] - med_cy)**2)
        ecc_dev = abs(features['eccentricity'] - med_ecc)
        disp_dev = abs(features['dispersion'] - med_disp)
        
        rr_ratio = abs(rr_current - rr_prev) / (rr_prev + 1e-8)
        rr_irregular = rr_ratio > self.theta_rr_irreg
        premature = rr_current < 0.7 * rr_prev if rr_prev > 0.3 else False
        
        self.recent_ratio_areas.append(ratio_area)
        if len(self.recent_ratio_areas) > 20:
            self.recent_ratio_areas.pop(0)
        area_var = np.var(self.recent_ratio_areas) if len(self.recent_ratio_areas) >= 10 else 0.0
        
        # ---- Decision Tree (Deterministic) ----
        pred = 'N'
        rule = 'N_default'
        
        # RULE: Geometric Anomalies
        if (ratio_area > self.theta_area_high or ratio_area < self.theta_area_low) and shift > self.theta_shift:
            pred, rule = 'V', 'V_Morphology_Distortion'
        elif ratio_area > 3.0 or ratio_area < 0.15:
            pred, rule = 'V', 'V_Extreme_Area'
        elif area_var > self.theta_afib:
            pred, rule = 'S', 'S_AFib_Variance'
        elif premature and (ecc_dev > self.theta_ecc_dev or disp_dev > self.theta_disp_dev):
            pred, rule = 'S', 'S_Premature_Deviant'
        elif rr_irregular and ecc_dev > self.theta_ecc_dev * 0.7:
            pred, rule = 'S', 'S_Irregular_Eccentric'
        elif (1.3 < ratio_area < self.theta_area_high) and shift > self.theta_shift * 0.5:
            pred, rule = 'F', 'F_Moderate_Distortion'
            
        # RULE: Normal
        elif 0.5 < ratio_area < 1.8 and shift < self.theta_shift:
            self._update_reference(features)
            pred, rule = 'N', 'N_Matched_Reference'
        elif premature or rr_irregular:
            pred, rule = 'S', 'S_Catchall_Premature'
        else:
            self._update_reference(features)
            
        features['rule'] = rule
        self.rule_counts[rule] = self.rule_counts.get(rule, 0) + 1
        return pred, features

    def predict(self, X_raw, rr_intervals):
        """
        Batch prediction method for pipeline integration.
        """
        self._reset_reference()
        
        class_map = {'N': 0, 'S': 1, 'V': 2, 'F': 3}
        predictions = []
        
        for i in range(len(X_raw)):
            rr_cur = rr_intervals[i]
            rr_prev = rr_intervals[i-1] if i > 0 else rr_cur
            
            pred_str, _ = self.classify_beat(X_raw[i], rr_cur, rr_prev)
            predictions.append(class_map[pred_str])
            
        return np.array(predictions)
