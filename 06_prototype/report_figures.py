"""Generate the figures embedded in Experiment_Output.md from saved results.

This script is intentionally independent of unfinished Faster R-CNN outputs.  It
only visualizes artifacts that already exist in data/manifest.csv and
results/rsa/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures" / "experiment_report"
CONDITIONS = ("A10", "A70", "A90", "B10", "B70", "B90")
LAYERS = ("V1_t1", "V2_t1", "V2_t2", "V4_t1", "V4_t2", "V4_t3", "V4_t4", "IT_t1", "IT_t2")


def correlation_rdm(patterns: np.ndarray) -> np.ndarray:
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    result = 1.0 - np.clip(normalized @ normalized.T, -1, 1)
    np.fill_diagonal(result, 0.0)
    return result


def rank_residual(values: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    ranked_y = pd.Series(values).rank(method="average").to_numpy()
    ranked_x = pd.Series(nuisance).rank(method="average").to_numpy()
    design = np.column_stack((np.ones_like(ranked_x), ranked_x))
    return ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]


def behavioral_accuracy() -> None:
    occlusion = np.array((10, 70, 90))
    accuracy = np.array((.96, .79, .62))
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    ax.plot(occlusion, accuracy * 100, marker="o", markersize=8, linewidth=2.4, color="#b0445b")
    for x, y in zip(occlusion, accuracy):
        ax.annotate(f"{y:.0%}", (x, y * 100), xytext=(0, 9), textcoords="offset points", ha="center")
    ax.set(xlabel="Occlusion (%)", ylabel="Accuracy (%)",
           title="Human behavioral accuracy by occlusion (reported, n=66)")
    ax.set_xticks(occlusion)
    ax.set_ylim(50, 102)
    ax.grid(alpha=.25)
    fig.savefig(OUT / "human_accuracy_by_occlusion.png", dpi=180)
    plt.close(fig)


def subject01_brain_model_rsa() -> None:
    betas = np.load(ROOT / "results" / "cornet_s_openneuro" / "betas" / "sub-01_occipital_betas.npy")
    brain = correlation_rdm(betas)
    upper = np.triu_indices(6, 1)
    brain_vector = brain[upper]
    pixel_vector = np.load(ROOT / "results" / "rsa" / "low_level_pixel_rdm.npy")[upper]
    raw, controlled = [], []
    brain_ranks = pd.Series(brain_vector).rank(method="average").to_numpy()
    for layer in LAYERS:
        model_vector = np.load(ROOT / "results" / "rsa" / f"cornet_s_{layer}_rdm.npy")[upper]
        raw.append(spearmanr(brain_vector, model_vector).statistic)
        controlled.append(np.corrcoef(brain_ranks, rank_residual(model_vector, pixel_vector))[0, 1])
    x = np.arange(len(LAYERS))
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    ax.plot(x, raw, marker="o", linewidth=2, label="Raw Spearman RSA", color="#376795")
    ax.plot(x, controlled, marker="s", linewidth=2, label="Pixel-controlled semipartial RSA", color="#c36b35")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(x, LAYERS, rotation=40, ha="right")
    ax.set(ylabel="Rho", title="Occipital cortex ↔ CORnet-S layer RSA (sub-01, descriptive only)")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(OUT / "sub01_occipital_to_cornet_rsa.png", dpi=180)
    plt.close(fig)
    pd.DataFrame({"layer": LAYERS, "spearman_rho": raw,
                  "semipartial_rho_pixel_controlled": controlled}).to_csv(
        OUT / "sub01_occipital_to_cornet_rsa.csv", index=False)


def subject01_occipital_rdm_occlusion_order() -> None:
    betas = np.load(ROOT / "results" / "cornet_s_openneuro" / "betas" / "sub-01_occipital_betas.npy")
    original = correlation_rdm(betas)
    order = (0, 3, 1, 4, 2, 5)
    labels = ("10%-A", "10%-B", "70%-A", "70%-B", "90%-A", "90%-B")
    matrix = original[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6.3, 5.3), constrained_layout=True)
    image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=matrix.max())
    ax.set_xticks(range(6), labels, rotation=45, ha="right")
    ax.set_yticks(range(6), labels)
    ax.set_title("Occipital RDM ordered by occlusion → aircraft (sub-01)")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if matrix[i, j] > matrix.max() * .48 else "black")
    fig.colorbar(image, ax=ax, label="Correlation distance (1 - Pearson r)")
    fig.savefig(OUT / "sub01_occipital_rdm_occlusion_aircraft_order.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        OUT / "sub01_occipital_rdm_occlusion_aircraft_order.csv")


def foreground_curve() -> None:
    manifest = pd.read_csv(ROOT / "data" / "manifest.csv")
    grouped = manifest.groupby(["aircraft", "occlusion"]).fg_frac.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for aircraft, color in ((1, "#3465a4"), (2, "#d65f5f")):
        part = grouped[grouped.aircraft == aircraft]
        ax.errorbar(part.occlusion, part["mean"] * 100, yerr=part["std"] * 100,
                    marker="o", linewidth=2, capsize=3, color=color, label=f"Aircraft {aircraft}")
    ax.set(xlabel="Occlusion (%)", ylabel="Foreground pixels (%)",
           title="Visible foreground decreases with occlusion")
    ax.set_xticks((0, 10, 70, 90))
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(OUT / "foreground_by_occlusion.png", dpi=180)
    plt.close(fig)


def rdm_layer_curve() -> None:
    summary = pd.read_csv(ROOT / "results" / "rsa" / "cornet_s_rdm_summary.csv")
    summary["layer"] = pd.Categorical(summary.layer, LAYERS, ordered=True)
    summary = summary.sort_values("layer")
    colors = ["#4878a8"] * 3 + ["#6a9f58"] * 4 + ["#c05a65"] * 2
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    bars = ax.bar(summary.layer.astype(str), summary.rdm_mean, color=colors)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in summary.rdm_mean], padding=3, fontsize=8)
    ax.set(xlabel="CORnet-S layer / recurrent step", ylabel="Mean correlation distance",
           title="Mean separation among six condition representations")
    ax.set_ylim(0, max(summary.rdm_mean) * 1.22)
    ax.grid(axis="y", alpha=.25)
    fig.savefig(OUT / "cornet_s_layer_rdm_mean.png", dpi=180)
    plt.close(fig)


def selected_rdms() -> None:
    selected = ("low_level_pixel", "V1_t1", "V4_t4", "IT_t2")
    labels = ("32x32 pixels", "V1 t1", "V4 t4", "IT t2")
    matrices = []
    for name in selected:
        filename = "low_level_pixel_rdm.npy" if name == "low_level_pixel" else f"cornet_s_{name}_rdm.npy"
        matrices.append(np.load(ROOT / "results" / "rsa" / filename))
    vmax = max(matrix.max() for matrix in matrices)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    image = None
    for ax, matrix, label in zip(axes, matrices, labels):
        image = ax.imshow(matrix, vmin=0, vmax=vmax, cmap="magma")
        ax.set_xticks(range(6), CONDITIONS, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(6), CONDITIONS, fontsize=8)
        ax.set_title(label)
    fig.colorbar(image, ax=axes, shrink=.83, label="Correlation distance (1 - Pearson r)")
    fig.suptitle("Condition representational dissimilarity matrices", fontsize=13)
    fig.savefig(OUT / "selected_condition_rdms.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    behavioral_accuracy()
    subject01_brain_model_rsa()
    subject01_occipital_rdm_occlusion_order()
    foreground_curve()
    rdm_layer_curve()
    selected_rdms()
    print(f"Saved report figures to {OUT}")


if __name__ == "__main__":
    main()
