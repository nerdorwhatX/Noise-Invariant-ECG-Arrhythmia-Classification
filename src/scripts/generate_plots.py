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

print("Plots generated successfully!")
