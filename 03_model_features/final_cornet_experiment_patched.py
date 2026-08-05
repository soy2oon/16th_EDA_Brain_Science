"""Leakage-safe frozen CORnet-S aircraft occlusion experiment.

The paired source manifest and preprocessing rules intentionally match the
final ViT experiment.  The model-specific choices are the official
ImageNet-pretrained CORnet-S backbone, final recurrent outputs
V1_t1/V2_t2/V4_t4/IT_t2, and an IT_t2 global-average-pooled linear probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache-cornet"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache-cornet"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torchvision
from PIL import Image
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from final_vit_experiment_patched import (
    PairRecord,
    build_pair_manifest,
    sha256,
    visible_pixel_pair_audit,
)


SEEDS = (42, 142, 242, 342, 442)
OCCLUSION_LEVELS = (10, 70, 90)
RDM_LAYERS = ("V1_t1", "V2_t2", "V4_t4", "IT_t2")
EXPECTED_REPEATS = {"V1": 1, "V2": 2, "V4": 4, "IT": 2}
CONDITION_ORDER = ("A10", "B10", "A70", "B70", "A90", "B90")
HUMAN_DEFAULT = {10: 0.956154, 70: 0.792769, 90: 0.618769}


class PathDataset(Dataset):
    def __init__(self, paths: Sequence[str], transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), index


class BrightnessMatchedTransform:
    """Match the final ViT grayscale/visible-luminance transformation."""

    def __init__(
        self,
        base_transform,
        black_threshold: int = 20,
        target_mean: float = 110.0,
        target_std: float = 45.0,
    ):
        self.base_transform = base_transform
        self.black_threshold = black_threshold
        self.target_mean = target_mean
        self.target_std = target_std

    def __call__(self, image: Image.Image):
        gray = np.asarray(image.convert("L"), dtype=np.float64)
        visible = gray > self.black_threshold
        controlled = np.zeros_like(gray)
        if visible.sum() > 10:
            values = gray[visible]
            controlled[visible] = np.clip(
                (values - values.mean())
                / (values.std() + 1e-6)
                * self.target_std
                + self.target_mean,
                0,
                255,
            )
        output = Image.fromarray(controlled.astype(np.uint8)).convert("RGB")
        return self.base_transform(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final leakage-safe frozen CORnet-S experiment."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/Users/gyuhongcho/Desktop/DSL/EDA_AircraftFiles"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "final_cornet_results",
    )
    parser.add_argument(
        "--human-csv",
        type=Path,
        default=Path(
            "/Users/gyuhongcho/Downloads/"
            "26-2_EDA_Brain_Science-main/results/human_group.csv"
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("color", "brightness_matched"),
        default=["color", "brightness_matched"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def base_transform():
    return transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def load_cornet(device: torch.device):
    from cornet import cornet_s

    wrapped = cornet_s(pretrained=True, map_location="cpu")
    model = wrapped.module if hasattr(wrapped, "module") else wrapped
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("CORnet-S backbone was not fully frozen.")
    return model


def register_endpoint_hooks(model, capture: dict[str, list[torch.Tensor]]):
    handles = []
    for area in ("V1", "V2", "V4", "IT"):
        endpoint = getattr(model, area).output

        def save_output(_module, _inputs, output, name=area):
            capture.setdefault(name, []).append(output.detach())

        handles.append(endpoint.register_forward_hook(save_output))
    return handles


def pooled_endpoint_features(
    model,
    paths: Sequence[str],
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        PathDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks: dict[str, list[np.ndarray]] = {layer: [] for layer in RDM_LAYERS}
    capture: dict[str, list[torch.Tensor]] = {}
    handles = register_endpoint_hooks(model, capture)
    model.eval()
    try:
        with torch.inference_mode():
            for images, _indices in loader:
                capture.clear()
                model(images.to(device, non_blocking=True))
                observed = {area: len(values) for area, values in capture.items()}
                if observed != EXPECTED_REPEATS:
                    raise RuntimeError(
                        f"Unexpected CORnet recurrent outputs: {observed}; "
                        f"expected {EXPECTED_REPEATS}."
                    )
                for area, times in EXPECTED_REPEATS.items():
                    activation = capture[area][times - 1]
                    pooled = activation.mean(dim=(-2, -1)).cpu().numpy()
                    chunks[f"{area}_t{times}"].append(pooled)
    finally:
        for handle in handles:
            handle.remove()
    output = {
        name: np.concatenate(values).astype(np.float64)
        for name, values in chunks.items()
    }
    for name, values in output.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite CORnet features in {name}.")
    return output


def blank_features(model, transform, device: torch.device) -> dict[str, np.ndarray]:
    blank = transform(Image.new("RGB", (224, 224), color=(0, 0, 0)))
    temporary = ROOT / ".blank-cornet.png"
    Image.new("RGB", (224, 224), color=(0, 0, 0)).save(temporary)
    try:
        return pooled_endpoint_features(
            model,
            [str(temporary)],
            transform,
            device,
            batch_size=1,
            num_workers=0,
        )
    finally:
        temporary.unlink(missing_ok=True)


def make_probe(
    c_value: float,
    pca_components: int,
    sample_count: int,
    feature_count: int,
    seed: int,
) -> Pipeline:
    components = min(pca_components, sample_count - 1, feature_count)
    if components < 1:
        raise ValueError("Not enough samples for PCA.")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=components, svd_solver="full")),
            (
                "logistic",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )


def validate_probe(probe: Pipeline) -> None:
    arrays = {
        "scaler.mean": probe.named_steps["scaler"].mean_,
        "scaler.scale": probe.named_steps["scaler"].scale_,
        "pca.components": probe.named_steps["pca"].components_,
        "logistic.coef": probe.named_steps["logistic"].coef_,
        "logistic.intercept": probe.named_steps["logistic"].intercept_,
    }
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite fitted value: {name}")


def stratification(records: Sequence[PairRecord], indices: Iterable[int]):
    return np.array(
        [
            f"{records[index].class_name}_{records[index].occlusion_percent}"
            for index in indices
        ]
    )


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    values = np.clip(probability, 1e-12, 1 - 1e-12)
    return -(
        values * np.log(values) + (1 - values) * np.log(1 - values)
    )


def fixed_test_order(
    records: Sequence[PairRecord], indices: Sequence[int], seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = []
    for level in OCCLUSION_LEVELS:
        block = np.array(
            [
                index
                for index in indices
                if records[index].occlusion_percent == level
            ]
        )
        output.extend(rng.permutation(block).tolist())
    return np.asarray(output, dtype=int)


def metrics_for_subset(
    frame: pd.DataFrame,
    seed: int,
    level: int,
    class_name: str,
) -> dict:
    subset = frame
    if class_name != "overall":
        subset = subset[subset["class_name"] == class_name]
    truth = subset["true_label"].to_numpy()
    raw = subset["raw_prediction"].to_numpy()
    centered = subset["centered_prediction"].to_numpy()
    row = {
        "seed": seed,
        "occlusion_percent": level,
        "class_name": class_name,
        "n": len(subset),
        "raw_accuracy": accuracy_score(truth, raw),
        "raw_aircraft_2_selection_rate": float(np.mean(raw == 1)),
        "raw_confidence": float(subset["raw_confidence"].mean()),
        "raw_entropy": float(subset["raw_entropy"].mean()),
        "centered_accuracy": accuracy_score(truth, centered),
        "centered_aircraft_2_selection_rate": float(np.mean(centered == 1)),
        "centered_confidence": float(subset["centered_confidence"].mean()),
        "centered_entropy": float(subset["centered_entropy"].mean()),
    }
    if class_name == "overall":
        matrix = confusion_matrix(truth, centered, labels=[0, 1])
        row.update(
            {
                "raw_balanced_accuracy": balanced_accuracy_score(truth, raw),
                "centered_balanced_accuracy": balanced_accuracy_score(
                    truth, centered
                ),
                "centered_f1_macro": f1_score(
                    truth, centered, average="macro"
                ),
                "tn": int(matrix[0, 0]),
                "fp": int(matrix[0, 1]),
                "fn": int(matrix[1, 0]),
                "tp": int(matrix[1, 1]),
            }
        )
    else:
        row.update(
            {
                "raw_balanced_accuracy": np.nan,
                "centered_balanced_accuracy": np.nan,
                "centered_f1_macro": np.nan,
                "tn": np.nan,
                "fp": np.nan,
                "fn": np.nan,
                "tp": np.nan,
            }
        )
    return row


def run_probes(
    records: Sequence[PairRecord],
    intact_features: np.ndarray,
    occluded_features: np.ndarray,
    blank_feature: np.ndarray,
    seeds: Sequence[int],
    test_size: float,
    validation_size: float,
    pca_components: int,
    c_grid: Sequence[float],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.array([record.label for record in records])
    all_indices = np.arange(len(records))
    prediction_frames = []
    metric_rows = []
    split_rows = []
    validation_rows = []
    output.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        seed_everything(seed)
        train_indices, test_indices = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratification(records, all_indices),
        )
        fit_indices, validation_indices = train_test_split(
            train_indices,
            test_size=validation_size,
            random_state=seed,
            stratify=stratification(records, train_indices),
        )
        if set(train_indices) & set(test_indices):
            raise RuntimeError("Source leakage between train and test.")

        best_c = None
        best_score = -math.inf
        for c_value in c_grid:
            candidate = make_probe(
                c_value,
                pca_components,
                len(fit_indices),
                intact_features.shape[1],
                seed,
            )
            candidate.fit(intact_features[fit_indices], labels[fit_indices])
            validate_probe(candidate)
            prediction = candidate.predict(
                intact_features[validation_indices]
            )
            score = balanced_accuracy_score(
                labels[validation_indices], prediction
            )
            validation_rows.append(
                {
                    "seed": seed,
                    "c": c_value,
                    "raw_balanced_accuracy": score,
                }
            )
            if score > best_score:
                best_c = float(c_value)
                best_score = float(score)
        assert best_c is not None

        probe = make_probe(
            best_c,
            pca_components,
            len(train_indices),
            intact_features.shape[1],
            seed,
        )
        probe.fit(intact_features[train_indices], labels[train_indices])
        validate_probe(probe)
        joblib.dump(probe, output / f"linear_probe_seed_{seed}.joblib")

        order = fixed_test_order(records, test_indices, seed)
        raw_logits = probe.decision_function(occluded_features[order])
        blank_logit = float(
            probe.decision_function(blank_feature.reshape(1, -1))[0]
        )
        centered_logits = raw_logits - blank_logit
        raw_probability = 1 / (1 + np.exp(-np.clip(raw_logits, -50, 50)))
        centered_probability = 1 / (
            1 + np.exp(-np.clip(centered_logits, -50, 50))
        )
        raw_prediction = (raw_logits >= 0).astype(int)
        centered_prediction = (centered_logits >= 0).astype(int)
        ordered_records = [records[index] for index in order]

        frame = pd.DataFrame(
            {
                "seed": seed,
                "trial_order": np.arange(1, len(order) + 1),
                "source_id": [record.source_id for record in ordered_records],
                "class_name": [
                    record.class_name for record in ordered_records
                ],
                "true_label": labels[order],
                "occlusion_percent": [
                    record.occlusion_percent for record in ordered_records
                ],
                "intact_path": [
                    record.intact_path for record in ordered_records
                ],
                "occluded_path": [
                    record.occluded_path for record in ordered_records
                ],
                "selected_c": best_c,
                "validation_raw_balanced_accuracy": best_score,
                "raw_logit": raw_logits,
                "raw_prediction": raw_prediction,
                "raw_probability_aircraft_2": raw_probability,
                "raw_confidence": np.maximum(
                    raw_probability, 1 - raw_probability
                ),
                "raw_entropy": binary_entropy(raw_probability),
                "blank_logit": blank_logit,
                "centered_logit": centered_logits,
                "centered_prediction": centered_prediction,
                "centered_probability_aircraft_2": centered_probability,
                "centered_confidence": np.maximum(
                    centered_probability, 1 - centered_probability
                ),
                "centered_entropy": binary_entropy(centered_probability),
            }
        )
        frame["raw_correct"] = (
            frame["raw_prediction"] == frame["true_label"]
        ).astype(int)
        frame["centered_correct"] = (
            frame["centered_prediction"] == frame["true_label"]
        ).astype(int)
        prediction_frames.append(frame)

        for level in OCCLUSION_LEVELS:
            level_frame = frame[frame["occlusion_percent"] == level]
            for class_name in ("overall", "aircraft_1", "aircraft_2"):
                metric_rows.append(
                    metrics_for_subset(
                        level_frame, seed, level, class_name
                    )
                )

        train_set = set(train_indices)
        fit_set = set(fit_indices)
        validation_set = set(validation_indices)
        for index, record in enumerate(records):
            split = "test" if index in set(test_indices) else "train"
            role = (
                "test"
                if split == "test"
                else "fit"
                if index in fit_set
                else "validation"
                if index in validation_set
                else "train_refit"
            )
            split_rows.append(
                {
                    "seed": seed,
                    "source_id": record.source_id,
                    "class_name": record.class_name,
                    "occlusion_percent": record.occlusion_percent,
                    "split": split,
                    "selection_role": role,
                    "in_final_refit": index in train_set,
                }
            )

    trials = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    trials.to_csv(output / "trial_predictions.csv", index=False)
    metrics.to_csv(output / "metrics_by_seed_class.csv", index=False)
    pd.DataFrame(split_rows).to_csv(
        output / "source_splits.csv", index=False
    )
    pd.DataFrame(validation_rows).to_csv(
        output / "c_selection.csv", index=False
    )
    summary = (
        metrics.groupby(["occlusion_percent", "class_name"], as_index=False)
        .agg(
            mean_raw_accuracy=("raw_accuracy", "mean"),
            std_raw_accuracy=("raw_accuracy", "std"),
            mean_centered_accuracy=("centered_accuracy", "mean"),
            std_centered_accuracy=("centered_accuracy", "std"),
            mean_centered_balanced_accuracy=(
                "centered_balanced_accuracy",
                "mean",
            ),
            mean_raw_aircraft_2_selection_rate=(
                "raw_aircraft_2_selection_rate",
                "mean",
            ),
            mean_centered_aircraft_2_selection_rate=(
                "centered_aircraft_2_selection_rate",
                "mean",
            ),
            mean_centered_confidence=("centered_confidence", "mean"),
            mean_centered_entropy=("centered_entropy", "mean"),
        )
    )
    summary.to_csv(output / "metrics_summary.csv", index=False)
    return summary, trials


def correlation_rdm(features: np.ndarray) -> np.ndarray:
    values = features.astype(np.float64)
    values -= values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values /= np.clip(norms, 1e-12, None)
    result = 1 - np.clip(values @ values.T, -1, 1)
    np.fill_diagonal(result, 0)
    return result.astype(np.float32)


def save_rdms(
    records: Sequence[PairRecord],
    features: dict[str, np.ndarray],
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    order = sorted(
        range(len(records)),
        key=lambda index: (
            records[index].occlusion_percent,
            records[index].label,
            records[index].source_id,
        ),
    )
    ordered_records = [records[index] for index in order]
    pd.DataFrame(asdict(record) for record in ordered_records).to_csv(
        output / "stimulus_order.csv", index_label="rdm_index"
    )
    condition_rdms = []
    for layer in RDM_LAYERS:
        ordered = features[layer][order]
        matrix = correlation_rdm(ordered)
        np.save(output / f"rdm_{layer}_all.npy", matrix)
        patterns = []
        for condition in CONDITION_ORDER:
            label = 0 if condition[0] == "A" else 1
            level = int(condition[1:])
            indices = [
                i
                for i, record in enumerate(ordered_records)
                if record.label == label
                and record.occlusion_percent == level
            ]
            patterns.append(ordered[indices].mean(axis=0))
        condition_rdm = correlation_rdm(np.stack(patterns))
        condition_rdms.append(condition_rdm)
        np.save(output / f"condition_rdm_{layer}.npy", condition_rdm)
        pd.DataFrame(
            condition_rdm,
            index=CONDITION_ORDER,
            columns=CONDITION_ORDER,
        ).to_csv(output / f"condition_rdm_{layer}.csv")
    figure, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    maximum = max(float(matrix.max()) for matrix in condition_rdms)
    for axis, layer, matrix in zip(axes, RDM_LAYERS, condition_rdms):
        image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=maximum)
        axis.set_title(layer)
        axis.set_xticks(range(6), CONDITION_ORDER, rotation=45)
        axis.set_yticks(range(6), CONDITION_ORDER)
    figure.colorbar(image, ax=axes, shrink=0.8, label="1 - Pearson r")
    figure.suptitle("Frozen CORnet-S condition RDMs")
    figure.savefig(output / "condition_rdms.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def load_human(path: Path) -> dict[int, float]:
    if not path.is_file():
        return HUMAN_DEFAULT.copy()
    frame = pd.read_csv(path)
    if {"occlusion", "mean"} <= set(frame.columns):
        return {
            int(row.occlusion): float(row.mean)
            for row in frame.itertuples()
        }
    return HUMAN_DEFAULT.copy()


def create_final_outputs(
    summaries: dict[str, pd.DataFrame],
    human: dict[int, float],
    output: Path,
) -> pd.DataFrame:
    rows = []
    for level in OCCLUSION_LEVELS:
        row = {
            "occlusion_percent": level,
            "human_accuracy": human[level],
        }
        for condition, summary in summaries.items():
            value = summary[
                (summary["occlusion_percent"] == level)
                & (summary["class_name"] == "overall")
            ].iloc[0]
            row[f"{condition}_centered_mean"] = value[
                "mean_centered_accuracy"
            ]
            row[f"{condition}_centered_sd"] = value[
                "std_centered_accuracy"
            ]
            row[f"{condition}_raw_mean"] = value["mean_raw_accuracy"]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "final_comparison_table.csv", index=False)
    display = table[
        [
            "occlusion_percent",
            "human_accuracy",
            "color_centered_mean",
            "brightness_matched_centered_mean",
        ]
    ].rename(
        columns={
            "occlusion_percent": "Occlusion",
            "human_accuracy": "Human",
            "color_centered_mean": "CORnet-S Color Centered",
            "brightness_matched_centered_mean": (
                "CORnet-S Brightness/Color Controlled Centered"
            ),
        }
    )
    display.to_csv(
        output / "final_comparison_table_display.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    levels = np.array(OCCLUSION_LEVELS)
    axis.plot(
        levels,
        [human[level] for level in levels],
        "-o",
        color="black",
        linewidth=2.6,
        label="Human",
    )
    labels = {
        "color": "CORnet-S color centered",
        "brightness_matched": "CORnet-S brightness/color controlled centered",
    }
    colors = {"color": "#2E5A88", "brightness_matched": "#C45A3C"}
    for condition, summary in summaries.items():
        overall = summary[summary["class_name"] == "overall"].sort_values(
            "occlusion_percent"
        )
        axis.errorbar(
            overall["occlusion_percent"],
            overall["mean_centered_accuracy"],
            yerr=overall["std_centered_accuracy"],
            marker="o",
            linewidth=2.2,
            capsize=4,
            color=colors[condition],
            label=labels[condition],
        )
    axis.axhline(0.5, color="gray", linestyle=":", label="Chance")
    axis.set(
        xticks=levels,
        ylim=(0.3, 1.03),
        xlabel="Occlusion (%)",
        ylabel="Blank-centered forced-choice accuracy",
        title="Human vs frozen CORnet-S",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "final_comparison_graph.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 2.7))
    axis.axis("off")
    rendered = display.copy()
    rendered["Occlusion"] = rendered["Occlusion"].map(lambda x: f"{x}%")
    for column in rendered.columns[1:]:
        rendered[column] = rendered[column].map(lambda x: f"{x:.3f}")
    table_artist = axis.table(
        cellText=rendered.values,
        colLabels=[
            "Occlusion",
            "Human",
            "CORnet-S Color\nCentered",
            "CORnet-S Brightness/Color\nControlled Centered",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9)
    table_artist.scale(1, 1.5)
    figure.tight_layout()
    figure.savefig(output / "final_comparison_table.png", dpi=220)
    plt.close(figure)
    return table


def create_bias_figure(
    summaries: dict[str, pd.DataFrame], output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    colors = {"aircraft_1": "#4F8A5B", "aircraft_2": "#C45A3C"}
    for condition, summary in summaries.items():
        for class_name in ("aircraft_1", "aircraft_2"):
            subset = summary[
                summary["class_name"] == class_name
            ].sort_values("occlusion_percent")
            axes[0].plot(
                subset["occlusion_percent"],
                subset["mean_centered_accuracy"],
                marker="o",
                linestyle="-" if condition == "color" else "--",
                color=colors[class_name],
                label=f"{condition} {class_name}",
            )
        overall = summary[
            summary["class_name"] == "overall"
        ].sort_values("occlusion_percent")
        axes[1].plot(
            overall["occlusion_percent"],
            overall["mean_centered_aircraft_2_selection_rate"],
            marker="o",
            label=condition,
        )
    axes[0].set(
        ylabel="Centered class accuracy",
        xlabel="Occlusion (%)",
        title="Class accuracy",
        ylim=(-0.02, 1.02),
    )
    axes[1].axhline(0.5, color="gray", linestyle=":")
    axes[1].set(
        ylabel="Centered Aircraft 2 selection rate",
        xlabel="Occlusion (%)",
        title="Response bias",
        ylim=(-0.02, 1.02),
    )
    for axis in axes:
        axis.set_xticks(OCCLUSION_LEVELS)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "class_accuracy_and_bias.png", dpi=220)
    plt.close(figure)


def create_raw_centered_figure(
    summaries: dict[str, pd.DataFrame], output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    labels = {
        "color": "Color",
        "brightness_matched": "Brightness/color controlled",
    }
    for axis, (condition, summary) in zip(axes, summaries.items()):
        overall = summary[
            summary["class_name"] == "overall"
        ].sort_values("occlusion_percent")
        axis.errorbar(
            overall["occlusion_percent"],
            overall["mean_raw_accuracy"],
            yerr=overall["std_raw_accuracy"],
            marker="o",
            linewidth=2.2,
            capsize=4,
            label="Raw",
        )
        axis.errorbar(
            overall["occlusion_percent"],
            overall["mean_centered_accuracy"],
            yerr=overall["std_centered_accuracy"],
            marker="o",
            linewidth=2.2,
            capsize=4,
            label="Blank-centered",
        )
        axis.axhline(0.5, color="gray", linestyle=":")
        axis.set(
            xticks=OCCLUSION_LEVELS,
            ylim=(0.3, 1.03),
            xlabel="Occlusion (%)",
            title=labels[condition],
        )
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Forced-choice accuracy")
    figure.suptitle("CORnet-S raw vs blank-centered decisions")
    figure.tight_layout()
    figure.savefig(output / "raw_vs_blank_centered.png", dpi=220)
    plt.close(figure)


def write_run_config(args, output: Path, device: torch.device) -> None:
    checkpoint = (
        Path(os.environ["TORCH_HOME"])
        / "hub"
        / "checkpoints"
        / "cornet_s-1d3f7974.pth"
    )
    config = {
        "seeds": args.seeds,
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "pca_components": args.pca_components,
        "pca_solver": "full",
        "c_grid": args.c_grid,
        "device": str(device),
        "backbone": "official CORnet-S",
        "checkpoint": "cornet_s-1d3f7974.pth",
        "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
        "backbone_frozen": True,
        "classification_feature": "IT_t2 global average pooling (512D)",
        "rdm_layers": list(RDM_LAYERS),
        "probe": "StandardScaler -> PCA30(full) -> L2 logistic(liblinear)",
        "c_selection": "intact validation raw balanced accuracy",
        "primary_decision": "blank-centered logit >= 0",
        "diagnostic_decision": "raw logit >= 0",
        "source_leakage_control": (
            "stratified class x occlusion source-ID split before "
            "intact training / paired occluded testing"
        ),
        "conditions": args.conditions,
        "brightness_matched": {
            "grayscale": True,
            "black_threshold": 20,
            "target_visible_mean": 110,
            "target_visible_std": 45,
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seeds[0])

    records = build_pair_manifest(args.project_root)
    manifest = pd.DataFrame(asdict(record) for record in records)
    manifest.to_csv(output / "paired_source_manifest.csv", index=False)
    audit = visible_pixel_pair_audit(records)
    audit.to_csv(output / "pairing_audit.csv", index=False)
    if len(records) != 300:
        raise RuntimeError(f"Expected 300 paired sources, found {len(records)}.")
    if float(audit["visible_pixel_correlation"].min()) < 0.80:
        raise RuntimeError("At least one paired source failed pixel audit.")
    print(
        f"Data audit passed: {len(records)} paired sources; "
        f"minimum visible correlation="
        f"{audit['visible_pixel_correlation'].min():.3f}"
    )
    if args.audit_only:
        return

    device = select_device(args.device)
    print(f"Loading official frozen CORnet-S on {device}...")
    model = load_cornet(device)
    base = base_transform()
    condition_transforms = {
        "color": base,
        "brightness_matched": BrightnessMatchedTransform(base),
    }
    intact_paths = [record.intact_path for record in records]
    occluded_paths = [record.occluded_path for record in records]
    summaries = {}

    for condition in args.conditions:
        print(f"\n=== {condition} ===")
        condition_dir = output / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        transform = condition_transforms[condition]
        intact = pooled_endpoint_features(
            model,
            intact_paths,
            transform,
            device,
            args.batch_size,
            args.num_workers,
        )
        occluded = pooled_endpoint_features(
            model,
            occluded_paths,
            transform,
            device,
            args.batch_size,
            args.num_workers,
        )
        blank = blank_features(model, transform, device)
        feature_dir = condition_dir / "features"
        feature_dir.mkdir(exist_ok=True)
        for layer in RDM_LAYERS:
            np.save(feature_dir / f"intact_{layer}.npy", intact[layer])
            np.save(feature_dir / f"occluded_{layer}.npy", occluded[layer])
        summary, _trials = run_probes(
            records,
            intact["IT_t2"],
            occluded["IT_t2"],
            blank["IT_t2"],
            args.seeds,
            args.test_size,
            args.validation_size,
            args.pca_components,
            args.c_grid,
            condition_dir,
        )
        summaries[condition] = summary
        save_rdms(records, occluded, condition_dir / "rdm")
        print(
            summary[summary["class_name"] == "overall"][
                [
                    "occlusion_percent",
                    "mean_centered_accuracy",
                    "std_centered_accuracy",
                    "mean_raw_accuracy",
                ]
            ].to_string(index=False)
        )

    human = load_human(args.human_csv)
    final_table = create_final_outputs(summaries, human, output)
    create_bias_figure(summaries, output)
    create_raw_centered_figure(summaries, output)
    write_run_config(args, output, device)
    print("\nFinal comparison:")
    print(final_table.round(4).to_string(index=False))
    print(f"\nComplete: {output}")


if __name__ == "__main__":
    main()
