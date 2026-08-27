import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
matplotlib.use('Agg')

project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(project_root, "results")
SUMMARY_DIR = os.path.join(project_root, "Summary", "figures")

os.makedirs(SUMMARY_DIR, exist_ok=True)


def generate_confusion_matrix():
    # Confusion matrix from dsp_expert_results.json
    cm = np.array([
        [31542,  3180,  9916,     5],
        [1624,    98,   125,     0],
        [916,   918,  1227,     1],
        [367,     5,    13,     0]
    ])

    classes = ['N (Normal)', 'S (Supra-V)', 'V (Ventricular)', 'F (Fusion)']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Number of Beats'})
    plt.title('Confusion Matrix: DSP Expert System', fontsize=14, pad=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    path = os.path.join(SUMMARY_DIR, "cm_formal.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def generate_noise_stress_plot():
    # Data from nstdb_stress_test.json
    data = {
        "24": {"accuracy": 0.6987, "weighted_f2": 0.7264},
        "18": {"accuracy": 0.7062, "weighted_f2": 0.7323},
        "12": {"accuracy": 0.7196, "weighted_f2": 0.7443},
        "6": {"accuracy": 0.5947, "weighted_f2": 0.6338},
        "0": {"accuracy": 0.4310, "weighted_f2": 0.4633},
        "-6": {"accuracy": 0.3951, "weighted_f2": 0.4223}
    }

    snr_levels = [24, 18, 12, 6, 0, -6]
    accuracies = [data[str(snr)]["accuracy"] for snr in snr_levels]
    f2_scores = [data[str(snr)]["weighted_f2"] for snr in snr_levels]

    plt.figure(figsize=(9, 5))
    plt.plot(snr_levels, f2_scores, marker='o', linestyle='-',
             linewidth=2, color='#2ecc71', label='Weighted F2 Score')
    plt.plot(snr_levels, accuracies, marker='s', linestyle='--',
             linewidth=2, color='#3498db', label='Accuracy')

    plt.gca().invert_xaxis()  # 24 down to -6
    plt.title('Noise Stress Test (NSTDB)', fontsize=14, pad=15)
    plt.xlabel('Signal-to-Noise Ratio (SNR) in dB', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    path = os.path.join(SUMMARY_DIR, "nstdb_stress_test_plot.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


if __name__ == "__main__":
    generate_confusion_matrix()
    generate_noise_stress_plot()
