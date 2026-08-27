"""
generate_figures.py — Generates publication-quality figures for the LaTeX summary.

Creates:
  1. Class distribution bar chart (before & after balancing)
  2. Sample raw ECG waveforms for each AAMI class
  3. Pipeline flowchart (saved as a diagram)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUMMARY_DIR = os.path.join(project_root, "Summary", "figures")
DATA_DIR = os.path.join(project_root, "data")

CLASS_NAMES = ['N (Normal)', 'S (Supra-V)', 'V (Ventricular)', 'F (Fusion)']
CLASS_COLORS = ['#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']

os.makedirs(SUMMARY_DIR, exist_ok=True)


def fig1_class_distribution():
    """Bar chart: class counts before and after balancing."""
    # Before balancing (from build_dataset.py output)
    before = {'N': 45839, 'S': 943, 'V': 3788, 'F': 414}
    # After balancing
    after = {'N': 45839, 'S': 943, 'V': 45839, 'F': 414}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before
    bars = axes[0].bar(before.keys(), before.values(), color=CLASS_COLORS, edgecolor='black', linewidth=0.8)
    axes[0].set_title('DS1 Class Distribution (Before Balancing)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Number of Beats', fontsize=11)
    axes[0].set_yscale('log')
    for bar, val in zip(bars, before.values()):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                     f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # After
    bars = axes[1].bar(after.keys(), after.values(), color=CLASS_COLORS, edgecolor='black', linewidth=0.8)
    axes[1].set_title('DS1 Class Distribution (After V-Augmentation)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Number of Beats', fontsize=11)
    axes[1].set_yscale('log')
    for bar, val in zip(bars, after.values()):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                     f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(SUMMARY_DIR, "class_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig2_sample_waveforms():
    """Plot sample raw waveforms for each AAMI class from DS1."""
    X = np.load(os.path.join(DATA_DIR, "DS1_X_raw.npy"))
    y = np.load(os.path.join(DATA_DIR, "DS1_y.npy"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for cls_idx in range(4):
        ax = axes[cls_idx]
        mask = y == cls_idx
        if np.sum(mask) == 0:
            ax.text(0.5, 0.5, 'No samples', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(CLASS_NAMES[cls_idx])
            continue

        # Plot 5 overlaid samples
        indices = np.where(mask)[0]
        np.random.seed(42)
        sample_idx = np.random.choice(indices, min(5, len(indices)), replace=False)

        for i, idx in enumerate(sample_idx):
            alpha = 0.9 if i == 0 else 0.4
            lw = 1.5 if i == 0 else 0.8
            ax.plot(X[idx], color=CLASS_COLORS[cls_idx], alpha=alpha, linewidth=lw)

        ax.set_title(CLASS_NAMES[cls_idx], fontsize=13, fontweight='bold', color=CLASS_COLORS[cls_idx])
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Amplitude (mV)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=90, color='red', linestyle='--', alpha=0.5, label='R-peak')
        if cls_idx == 0:
            ax.legend(fontsize=9)

    plt.suptitle('Sample Raw ECG Waveforms by AAMI Class', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(SUMMARY_DIR, "sample_waveforms.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig3_pipeline_diagram():
    """Create a pipeline flowchart as a figure."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    boxes = [
        (1, 3, 'MIT-BIH\nRaw Records\n(48 patients)', '#3498db'),
        (3.5, 3, 'R-Peak\nSegmentation\n(234 samples)', '#2ecc71'),
        (6, 3, 'AAMI\nLabelling\n(N/S/V/F)', '#e67e22'),
        (8.5, 3, 'De Chazal\nSplit\n(DS1/DS2)', '#e74c3c'),
        (11, 4.2, 'Signal\nAugmentation\n(V → N parity)', '#9b59b6'),
        (11, 1.8, 'Feature\nEngineering\n(~89 features)', '#1abc9c'),
    ]

    for (x, y, text, color) in boxes:
        rect = plt.Rectangle((x-0.9, y-0.7), 1.8, 1.4, linewidth=2,
                              edgecolor=color, facecolor=color, alpha=0.15, 
                              zorder=2, clip_on=False)
        ax.add_patch(rect)
        rect_border = plt.Rectangle((x-0.9, y-0.7), 1.8, 1.4, linewidth=2,
                                     edgecolor=color, facecolor='none', 
                                     zorder=3, clip_on=False)
        ax.add_patch(rect_border)
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                fontweight='bold', color=color, zorder=4)

    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='#555555')
    for (x1, x2, y) in [(1.9, 2.6, 3), (4.4, 5.1, 3), (6.9, 7.6, 3)]:
        ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=arrow_props)

    # Split arrows from De Chazal
    ax.annotate('', xy=(10.1, 4.2), xytext=(9.4, 3.4), arrowprops=arrow_props)
    ax.annotate('', xy=(10.1, 1.8), xytext=(9.4, 2.6), arrowprops=arrow_props)

    ax.set_title('Data Preparation Pipeline Overview', fontsize=15, fontweight='bold', pad=20)

    path = os.path.join(SUMMARY_DIR, "pipeline_diagram.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_class_distribution()
    fig2_sample_waveforms()
    fig3_pipeline_diagram()
    print("All figures generated.")
