import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
results_dir = os.path.join(project_root, "results")
models_dir = os.path.join(project_root, "models_saved")
summary_fig_dir = os.path.abspath(
    os.path.join(project_root, "..", "Summary", "figures")
)
update_fig_dir = os.path.abspath(
    os.path.join(project_root, "..", "project update 01", "figures")
)

os.makedirs(summary_fig_dir, exist_ok=True)
os.makedirs(update_fig_dir, exist_ok=True)

# 1. Plot Hybrid Confusion Matrix
hybrid_res_path = os.path.join(results_dir, "hybrid_lgbm_results.json")
if os.path.exists(hybrid_res_path):
    with open(hybrid_res_path, "r") as f:
        res = json.load(f)
    cm = np.array(res["confusion_matrix"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["N", "S", "V", "F"],
        yticklabels=["N", "S", "V", "F"],
    )
    plt.title("Hybrid ML Ensemble (LightGBM) Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_fig_dir, "cm_hybrid.png"), dpi=300)
    plt.savefig(os.path.join(update_fig_dir, "cm_hybrid.png"), dpi=300)
    plt.close()

# 1.3 Plot DSP Expert Confusion Matrix
dsp_res_path = os.path.join(results_dir, "dsp_expert_results.json")
if os.path.exists(dsp_res_path):
    with open(dsp_res_path, "r") as f:
        dsp_res = json.load(f)
    if "confusion_matrix" in dsp_res:
        cm_dsp = np.array(dsp_res["confusion_matrix"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_dsp,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["N", "S", "V", "F"],
            yticklabels=["N", "S", "V", "F"],
        )
        plt.title("DSP Expert (Rule-based) Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(summary_fig_dir, "cm_expert.png"), dpi=300)
        plt.savefig(os.path.join(update_fig_dir, "cm_expert.png"), dpi=300)
        plt.close()

# 1.5. Plot ResNet Confusion Matrix
dl_res_path = os.path.join(results_dir, "resnet1d_se_results.json")
if os.path.exists(dl_res_path):
    with open(dl_res_path, "r") as f:
        dl_res = json.load(f)
    dl_cm = np.array(dl_res["confusion_matrix"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        dl_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["N", "S", "V", "F"],
        yticklabels=["N", "S", "V", "F"],
    )
    plt.title("ResNet1D-SE (Deep Learning) Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_fig_dir, "cm_resnet.png"), dpi=300)
    plt.savefig(os.path.join(update_fig_dir, "cm_resnet.png"), dpi=300)
    plt.close()

# 2. Plot NSTDB Stress Test Comparison
dsp_nstdb = os.path.join(results_dir, "nstdb_stress_test.json")
dl_nstdb = os.path.join(results_dir, "nstdb_resnet1d_se_stress_test.json")
hybrid_nstdb = os.path.join(results_dir, "nstdb_hybrid_lgbm_stress_test.json")

if (
    os.path.exists(dsp_nstdb)
    and os.path.exists(hybrid_nstdb)
    and os.path.exists(dl_nstdb)
):
    with open(dsp_nstdb, "r") as f:
        dsp_res = json.load(f)
    with open(dl_nstdb, "r") as f:
        dl_res = json.load(f)
    with open(hybrid_nstdb, "r") as f:
        hyb_res = json.load(f)

    snrs = sorted([int(k) for k in dsp_res.keys()], reverse=True)

    dsp_acc = [dsp_res[str(s)]["accuracy"] for s in snrs]
    dl_acc = [dl_res[str(s)]["accuracy"] for s in snrs]
    hyb_acc = [hyb_res[str(s)]["accuracy"] for s in snrs]

    dsp_f2 = [dsp_res[str(s)]["weighted_f2"] for s in snrs]
    dl_f2 = [dl_res[str(s)]["weighted_f2"] for s in snrs]
    hyb_f2 = [hyb_res[str(s)]["weighted_f2"] for s in snrs]

    plt.figure(figsize=(10, 6))
    plt.plot(
        snrs,
        dsp_acc,
        marker="o",
        linestyle=":",
        color="gray",
        label="DSP Expert (Rule-based)",
    )
    plt.plot(
        snrs,
        dl_acc,
        marker="^",
        linestyle="--",
        color="orange",
        label="ResNet1D-SE (Deep Learning)",
    )
    plt.plot(
        snrs,
        hyb_acc,
        marker="s",
        linestyle="-",
        color="green",
        linewidth=2.5,
        label="Hybrid ML Ensemble",
    )

    plt.title("NSTDB Stress Test: Accuracy vs. SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_fig_dir, "nstdb_comparison.png"), dpi=300)
    plt.savefig(os.path.join(update_fig_dir, "nstdb_comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        snrs,
        dsp_f2,
        marker="o",
        linestyle=":",
        color="gray",
        label="DSP Expert (Rule-based)",
    )
    plt.plot(
        snrs,
        dl_f2,
        marker="^",
        linestyle="--",
        color="orange",
        label="ResNet1D-SE (Deep Learning)",
    )
    plt.plot(
        snrs,
        hyb_f2,
        marker="s",
        linestyle="-",
        color="green",
        linewidth=2.5,
        label="Hybrid ML Ensemble",
    )

    plt.title("NSTDB Stress Test: Weighted F2 Score vs. SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Weighted F2 Score")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_fig_dir, "nstdb_f2_comparison.png"), dpi=300)
    plt.savefig(os.path.join(update_fig_dir, "nstdb_f2_comparison.png"), dpi=300)
    plt.close()

# 3. Plot LightGBM Feature Importance
model_path = os.path.join(models_dir, "hybrid_lgbm.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
    importances = model.feature_importances_

    # We have 89 features. We don't have exact names for all 89, so we'll just plot top 20 by index.
    indices = np.argsort(importances)[::-1][:20]

    # Attempt to assign basic names based on index range
    # 0-23: Phase Space (24)
    # 24-31: RR Context (8)
    # 32-43: Regional (12)
    # 44-51: Stat (8)
    # 52-55: Template Corr (4)
    # 56-63: Wavelet (8)
    # 64-67: Autocorr (4)
    # 68: Takens (1)
    # 69-88: TDA (20)
    def get_feature_name(idx):
        if idx < 24:
            return f"PhaseSpace_{idx}"
        elif idx < 32:
            return f"RR_Context_{idx-24}"
        elif idx < 44:
            return f"Regional_{idx-32}"
        elif idx < 52:
            return f"Stats_{idx-44}"
        elif idx < 56:
            return f"Template_{idx-52}"
        elif idx < 64:
            return f"Wavelet_{idx-56}"
        elif idx < 68:
            return f"AutoCorr_{idx-64}"
        elif idx == 68:
            return "Takens_Embedding"
        else:
            return f"TDA_Homology_{idx-69}"

    names = [get_feature_name(i) for i in indices]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances[indices], y=names, palette="viridis")
    plt.title("Top 20 Feature Importances (LightGBM Gain)")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(os.path.join(summary_fig_dir, "feature_importance.png"), dpi=300)
    plt.savefig(os.path.join(update_fig_dir, "feature_importance.png"), dpi=300)
    plt.close()
# 4. Plot Class Imbalance Before/After Balancing
data_dir = os.path.join(project_root, "data")
ds1_y_path = os.path.join(data_dir, "DS1_y.npy")
ds2_y_path = os.path.join(data_dir, "DS2_y.npy")

if os.path.exists(ds1_y_path) and os.path.exists(ds2_y_path):
    y_ds1 = np.load(ds1_y_path)
    y_ds2 = np.load(ds2_y_path)

    class_names = ["N", "S", "V", "F"]

    # DS1 is already balanced (V augmented to match N). We can figure out the
    # original counts: N count stayed the same, S and F stayed the same,
    # V was augmented UP to match N.  So the "before" V count is total - augmented.
    ds1_counts = [np.sum(y_ds1 == c) for c in range(4)]
    ds2_counts = [np.sum(y_ds2 == c) for c in range(4)]

    # The original (pre-augmentation) V count in DS1: since V was augmented to match N,
    # we need to infer it. We know N count = V count after balancing.
    # Original V count is unknown from the balanced file, so we use DS2's V/N ratio
    # as a proxy OR we can just load the raw data and count.
    # Actually, let's just show DS1 (after balancing) vs DS2 (natural distribution)
    # to illustrate the imbalance problem AND what we did about it.

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # DS1 Original (Hardcoded from build_dataset logs before augmentation)
    ds1_orig_counts = [45935, 932, 4173, 417]
    colors_orig = ["#2196F3", "#FF9800", "#F44336", "#9C27B0"]
    
    # Top-Left: DS1 Before
    bars1 = axes[0, 0].bar(class_names, ds1_orig_counts, color=colors_orig, edgecolor="black", linewidth=0.8)
    axes[0, 0].set_title("DS1 (Training Set) — Before Balancing", fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("Number of Beats")
    axes[0, 0].set_yscale("log")
    for bar, count in zip(bars1, ds1_orig_counts):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Top-Right: DS1 After N↔V Balancing
    colors_ds1_after = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    bars2 = axes[0, 1].bar(class_names, ds1_counts, color=colors_ds1_after, edgecolor="black", linewidth=0.8)
    axes[0, 1].set_title("DS1 (Training Set) — After N↔V Balancing", fontsize=13, fontweight="bold")
    axes[0, 1].set_ylabel("Number of Beats")
    axes[0, 1].set_yscale("log")
    for bar, count in zip(bars2, ds1_counts):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Bottom-Left: DS2 Before
    bars3 = axes[1, 0].bar(class_names, ds2_counts, color=colors_orig, edgecolor="black", linewidth=0.8)
    axes[1, 0].set_title("DS2 (Test Set) — Natural Distribution", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("AAMI Class")
    axes[1, 0].set_ylabel("Number of Beats")
    axes[1, 0].set_yscale("log")
    for bar, count in zip(bars3, ds2_counts):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Bottom-Right: DS2 After (Identical)
    bars4 = axes[1, 1].bar(class_names, ds2_counts, color=colors_orig, edgecolor="black", linewidth=0.8)
    axes[1, 1].set_title("DS2 (Test Set) — Unchanged (Strict Isolation)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("AAMI Class")
    axes[1, 1].set_ylabel("Number of Beats")
    axes[1, 1].set_yscale("log")
    for bar, count in zip(bars4, ds2_counts):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    fig.suptitle("Class Imbalance: Training (DS1) vs Testing (DS2)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_fig_dir, "class_imbalance.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(update_fig_dir, "class_imbalance.png"), dpi=300, bbox_inches="tight")
    plt.close()

print("Plots generated successfully!")
