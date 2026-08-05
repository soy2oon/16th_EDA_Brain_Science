#!/usr/bin/env python3
"""Final frozen-ViT pipeline for the EDA aircraft occlusion experiment.

Primary question
----------------
Can a frozen ImageNet-pretrained ViT representation learned from intact
Aircraft 1/2 images support classification of unseen source images after
10%, 70%, or 90% occlusion?

Core safeguards
---------------
1. Intact/occluded files are paired by source ID.
2. Each seed splits source IDs before training. An intact source used for
   training can never contribute its occluded version to that seed's test set.
3. The ViT backbone is frozen. Only a deterministic linear probe is trained.
4. Raw forced-choice predictions are the primary outcome. Blank-centering is
   saved only as a sensitivity analysis.
5. Scaler, PCA, and C selection use training data only.
6. ViT classification and RDM analyses both use CLS representations.

Default data layout
-------------------
PROJECT_ROOT/
├── original_image/
│   ├── aircraft_1_original/
│   └── aircraft_2_original/
└── aircraft_dataset/
    ├── aircraft_1/
    └── aircraft_2/

The current Aircraft2 naming convention is mapped as follows:
10% index 1..50 -> Complete 1..50
70% index 1..50 -> Complete 51..100
90% index 1..50 -> Complete 101..150
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/eda_vit_matplotlib")

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
from scipy.spatial.distance import pdist, squareform
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
from torchvision.models import ViT_B_16_Weights, vit_b_16


LEVELS = (10, 70, 90)
SEEDS = (42, 142, 242, 342, 442)
RDM_LAYERS = (2, 4, 6, 8, 10, 12)
CLASS_NAMES = ("aircraft_1", "aircraft_2")
CLASS_TO_LABEL = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PairRecord:
    source_id: str
    class_name: str
    label: int
    occlusion_percent: int
    within_level_index: int
    intact_path: str
    occluded_path: str


class PathDataset(Dataset):
    def __init__(self, paths: Sequence[str], transform: Callable[[Image.Image], torch.Tensor]):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), index


class GrayscaleTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.base_transform(image.convert("L").convert("RGB"))


class BrightnessMatchedTransform:
    """Remove chroma and standardize non-black luminance per image.

    This matches the shared pipeline definition: pixels <= 20 remain black,
    while visible pixels are standardized to mean 110 and standard deviation
    45 before clipping to uint8.
    """

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

    def __call__(self, image: Image.Image) -> torch.Tensor:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        foreground = gray > self.black_threshold
        controlled = np.zeros_like(gray)
        if int(foreground.sum()) > 10:
            values = gray[foreground]
            controlled[foreground] = np.clip(
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
        description="Run the final leakage-safe frozen ViT aircraft experiment."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/Users/gyuhongcho/Desktop/DSL/EDA_AircraftFiles"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "final_vit_results",
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
        choices=("color", "grayscale", "brightness_matched"),
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
        default="auto",
    )
    parser.add_argument("--rdm-layers", nargs="+", type=int, default=list(RDM_LAYERS))
    parser.add_argument(
        "--structure-permutations",
        type=int,
        default=0,
        help="Permutation tests per layer for occlusion/class structure RSA; 0 skips.",
    )
    parser.add_argument(
        "--occipital-rdm",
        type=Path,
        default=None,
        help="Optional square brain RDM with exactly the saved stimulus order.",
    )
    parser.add_argument(
        "--dacc-rdm",
        type=Path,
        default=None,
        help="Optional square dACC RDM with exactly the saved stimulus order.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Build and validate the paired manifest, then stop.",
    )
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pair_manifest(project_root: Path) -> list[PairRecord]:
    train_root = project_root / "original_image"
    test_root = project_root / "aircraft_dataset"
    required = [
        train_root / "aircraft_1_original",
        train_root / "aircraft_2_original",
        test_root / "aircraft_1",
        test_root / "aircraft_2",
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Required data folders are missing: {missing}")

    records: list[PairRecord] = []
    errors: list[str] = []

    a1_pattern = re.compile(r"^Aircraft1_(10|70|90)%_(\d+)$", re.IGNORECASE)
    for occluded in image_files(test_root / "aircraft_1"):
        match = a1_pattern.match(occluded.stem)
        if not match:
            errors.append(f"Unrecognized Aircraft1 file: {occluded.name}")
            continue
        level, index = map(int, match.groups())
        # The ViT-design task established that these files are intact even
        # though their retained source names contain "10/70/90%".
        intact = train_root / "aircraft_1_original" / f"{occluded.stem}_original.jpg"
        if not intact.is_file():
            errors.append(f"Missing paired intact file for {occluded.name}: {intact}")
            continue
        records.append(
            PairRecord(
                source_id=f"aircraft_1_{level}_{index:03d}",
                class_name="aircraft_1",
                label=0,
                occlusion_percent=level,
                within_level_index=index,
                intact_path=str(intact.resolve()),
                occluded_path=str(occluded.resolve()),
            )
        )

    a2_pattern = re.compile(r"^Aircraft2_(10|70|90)__?(\d+)$", re.IGNORECASE)
    offsets = {10: 0, 70: 50, 90: 100}
    for occluded in image_files(test_root / "aircraft_2"):
        match = a2_pattern.match(occluded.stem)
        if not match:
            errors.append(f"Unrecognized Aircraft2 file: {occluded.name}")
            continue
        level, index = map(int, match.groups())
        # The original ViT experiment used Complete_1..150. Pixel-level
        # verification confirms these three consecutive 50-image blocks.
        complete_index = offsets[level] + index
        intact = (
            train_root
            / "aircraft_2_original"
            / f"Aircraft2_Complete_{complete_index}.jpg"
        )
        if not intact.is_file():
            errors.append(f"Missing paired intact file for {occluded.name}: {intact}")
            continue
        records.append(
            PairRecord(
                source_id=f"aircraft_2_{level}_{index:03d}",
                class_name="aircraft_2",
                label=1,
                occlusion_percent=level,
                within_level_index=index,
                intact_path=str(intact.resolve()),
                occluded_path=str(occluded.resolve()),
            )
        )

    if errors:
        raise RuntimeError("\n".join(errors))
    records.sort(
        key=lambda record: (
            record.occlusion_percent,
            record.class_name,
            record.within_level_index,
        )
    )
    if len({record.source_id for record in records}) != len(records):
        raise RuntimeError("Duplicate source IDs were found.")

    frame = pd.DataFrame(asdict(record) for record in records)
    counts = frame.groupby(["class_name", "occlusion_percent"]).size()
    expected_index = pd.MultiIndex.from_product(
        [CLASS_NAMES, LEVELS], names=["class_name", "occlusion_percent"]
    )
    counts = counts.reindex(expected_index, fill_value=0)
    if not bool((counts == 50).all()):
        raise RuntimeError(
            "Expected exactly 50 source pairs per class and occlusion level, found:\n"
            f"{counts.to_string()}"
        )
    return records


def visible_pixel_pair_audit(records: Sequence[PairRecord]) -> pd.DataFrame:
    """Check whether visible occluded pixels resemble the paired intact image."""
    rows = []
    for record in records:
        with Image.open(record.occluded_path) as image:
            occluded = np.asarray(image.convert("RGB"), dtype=np.float32)
        with Image.open(record.intact_path) as image:
            intact_image = image.convert("RGB")
            if intact_image.size != (occluded.shape[1], occluded.shape[0]):
                intact_image = intact_image.resize(
                    (occluded.shape[1], occluded.shape[0]), Image.Resampling.BILINEAR
                )
            intact = np.asarray(intact_image, dtype=np.float32)
        visible = occluded.mean(axis=2) > 20
        if int(visible.sum()) < 20:
            correlation = np.nan
            mae = np.nan
        else:
            left = occluded[visible].ravel()
            right = intact[visible].ravel()
            correlation = float(np.corrcoef(left, right)[0, 1])
            mae = float(np.abs(left - right).mean())
        rows.append(
            {
                "source_id": record.source_id,
                "class_name": record.class_name,
                "occlusion_percent": record.occlusion_percent,
                "visible_fraction": float(visible.mean()),
                "visible_pixel_correlation": correlation,
                "visible_pixel_mae": mae,
            }
        )
    return pd.DataFrame(rows)


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
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    # CPU is the reproducible default on Apple machines.
    return torch.device("cpu")


def vit_cls_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    requested_layers: set[int],
) -> dict[str, torch.Tensor]:
    x = model._process_input(images)
    cls = model.class_token.expand(x.shape[0], -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = model.encoder.dropout(x + model.encoder.pos_embedding)
    output: dict[str, torch.Tensor] = {}
    last_layer = len(model.encoder.layers)
    for number, layer in enumerate(model.encoder.layers, start=1):
        x = layer(x)
        if number in requested_layers and number != last_layer:
            output[f"block_{number:02d}"] = x[:, 0]
    x = model.encoder.ln(x)
    if last_layer in requested_layers:
        output[f"block_{last_layer:02d}"] = x[:, 0]
    output["final_cls"] = x[:, 0]
    return output


def extract_features(
    model: torch.nn.Module,
    paths: Sequence[str],
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    rdm_layers: Sequence[int],
    collect_layers: bool,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        PathDataset(paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    requested = set(rdm_layers if collect_layers else [])
    chunks: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            batch = vit_cls_features(
                model, images.to(device, non_blocking=True), requested
            )
            if not collect_layers:
                batch = {"final_cls": batch["final_cls"]}
            for name, values in batch.items():
                chunks.setdefault(name, []).append(values.cpu().numpy())
    output = {
        name: np.concatenate(values).astype(np.float64)
        for name, values in chunks.items()
    }
    for name, values in output.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values found in {name}; rerun on CPU.")
    return output


def make_probe(
    c_value: float,
    pca_components: int,
    sample_count: int,
    feature_count: int,
    seed: int,
) -> Pipeline:
    effective_components = min(
        pca_components, sample_count - 1, feature_count
    )
    if effective_components < 1:
        raise ValueError("Not enough training samples for PCA.")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=effective_components,
                    svd_solver="full",
                ),
            ),
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


def quiet_numeric_call(function):
    """Run a sklearn BLAS operation and reject actual non-finite outputs.

    Apple's Accelerate backend can emit spurious matmul RuntimeWarnings for
    finite PCA operations. Warnings are suppressed only around that call;
    every returned numeric result and learned parameter is checked separately.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*encountered in matmul",
            category=RuntimeWarning,
        )
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return function()


def validate_fitted_probe(probe: Pipeline) -> None:
    learned_arrays = {
        "scaler.mean_": probe.named_steps["scaler"].mean_,
        "scaler.scale_": probe.named_steps["scaler"].scale_,
        "pca.components_": probe.named_steps["pca"].components_,
        "pca.mean_": probe.named_steps["pca"].mean_,
        "logistic.coef_": probe.named_steps["logistic"].coef_,
        "logistic.intercept_": probe.named_steps["logistic"].intercept_,
    }
    for name, values in learned_arrays.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite fitted probe parameter: {name}")


def validate_numeric_output(name: str, values: np.ndarray) -> np.ndarray:
    output = np.asarray(values)
    if not np.isfinite(output).all():
        raise RuntimeError(f"Non-finite numeric output: {name}")
    return output


def stratification(records: Sequence[PairRecord], indices: Iterable[int]) -> np.ndarray:
    return np.array(
        [
            f"{records[index].class_name}_{records[index].occlusion_percent}"
            for index in indices
        ]
    )


def binary_entropy(probability_2: np.ndarray) -> np.ndarray:
    p = np.clip(probability_2, 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


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
    condition_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.array([record.label for record in records])
    all_indices = np.arange(len(records))
    prediction_frames = []
    metric_rows = []
    split_rows = []

    for seed in seeds:
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
            raise RuntimeError("Source leakage: train and test indices overlap.")

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
            quiet_numeric_call(
                lambda: candidate.fit(
                    intact_features[fit_indices], labels[fit_indices]
                )
            )
            validate_fitted_probe(candidate)
            predictions = validate_numeric_output(
                "validation predictions",
                quiet_numeric_call(
                    lambda: candidate.predict(
                        intact_features[validation_indices]
                    )
                ),
            )
            score = balanced_accuracy_score(
                labels[validation_indices], predictions
            )
            if score > best_score:
                best_score = float(score)
                best_c = float(c_value)
        assert best_c is not None

        # Refit every preprocessing step and the selected classifier on all
        # non-test intact sources.
        probe = make_probe(
            best_c,
            pca_components,
            len(train_indices),
            intact_features.shape[1],
            seed,
        )
        quiet_numeric_call(
            lambda: probe.fit(
                intact_features[train_indices], labels[train_indices]
            )
        )
        validate_fitted_probe(probe)

        raw_logits = validate_numeric_output(
            "test logits",
            quiet_numeric_call(
                lambda: probe.decision_function(
                    occluded_features[test_indices]
                )
            ),
        )
        raw_predictions = (raw_logits >= 0.0).astype(int)
        raw_probability_2 = 1.0 / (
            1.0 + np.exp(-np.clip(raw_logits, -50, 50))
        )

        blank_logit = float(
            validate_numeric_output(
                "blank logit",
                quiet_numeric_call(
                    lambda: probe.decision_function(
                        blank_feature.reshape(1, -1)
                    )
                ),
            )[0]
        )
        centered_logits = raw_logits - blank_logit
        centered_predictions = (centered_logits >= 0.0).astype(int)
        centered_probability_2 = 1.0 / (
            1.0 + np.exp(-np.clip(centered_logits, -50, 50))
        )

        test_records = [records[index] for index in test_indices]
        frame = pd.DataFrame(
            {
                "seed": seed,
                "source_id": [record.source_id for record in test_records],
                "class_name": [record.class_name for record in test_records],
                "true_label": labels[test_indices],
                "occlusion_percent": [
                    record.occlusion_percent for record in test_records
                ],
                "intact_path": [record.intact_path for record in test_records],
                "occluded_path": [
                    record.occluded_path for record in test_records
                ],
                "raw_logit": raw_logits,
                "raw_prediction": raw_predictions,
                "raw_probability_aircraft_2": raw_probability_2,
                "raw_confidence": np.maximum(
                    raw_probability_2, 1.0 - raw_probability_2
                ),
                "raw_entropy": binary_entropy(raw_probability_2),
                "blank_logit": blank_logit,
                "centered_logit": centered_logits,
                "centered_prediction": centered_predictions,
                "centered_probability_aircraft_2": centered_probability_2,
                "centered_confidence": np.maximum(
                    centered_probability_2, 1.0 - centered_probability_2
                ),
                "centered_entropy": binary_entropy(centered_probability_2),
                "selected_c": best_c,
                "validation_balanced_accuracy": best_score,
            }
        ).sort_values(["occlusion_percent", "source_id"])
        frame["raw_correct"] = (
            frame["raw_prediction"] == frame["true_label"]
        ).astype(int)
        frame["centered_correct"] = (
            frame["centered_prediction"] == frame["true_label"]
        ).astype(int)
        prediction_frames.append(frame)

        for split_name, indices in (
            ("probe_train", train_indices),
            ("probe_test", test_indices),
        ):
            for index in indices:
                split_rows.append(
                    {
                        "seed": seed,
                        "split": split_name,
                        "source_id": records[index].source_id,
                        "class_name": records[index].class_name,
                        "occlusion_percent": records[index].occlusion_percent,
                    }
                )

        for level in LEVELS:
            level_frame = frame[frame["occlusion_percent"] == level]
            y_true = level_frame["true_label"].to_numpy()
            raw_pred = level_frame["raw_prediction"].to_numpy()
            centered_pred = level_frame["centered_prediction"].to_numpy()
            raw_matrix = confusion_matrix(y_true, raw_pred, labels=[0, 1])
            metric_rows.append(
                {
                    "seed": seed,
                    "occlusion_percent": level,
                    "class_name": "overall",
                    "n": len(level_frame),
                    "raw_accuracy": accuracy_score(y_true, raw_pred),
                    "raw_balanced_accuracy": balanced_accuracy_score(
                        y_true, raw_pred
                    ),
                    "raw_f1_macro": f1_score(
                        y_true, raw_pred, average="macro"
                    ),
                    "raw_aircraft_2_selection_rate": float(
                        np.mean(raw_pred == 1)
                    ),
                    "raw_confidence": float(
                        level_frame["raw_confidence"].mean()
                    ),
                    "raw_entropy": float(level_frame["raw_entropy"].mean()),
                    "centered_accuracy": accuracy_score(
                        y_true, centered_pred
                    ),
                    "centered_aircraft_2_selection_rate": float(
                        np.mean(centered_pred == 1)
                    ),
                    "tn": int(raw_matrix[0, 0]),
                    "fp": int(raw_matrix[0, 1]),
                    "fn": int(raw_matrix[1, 0]),
                    "tp": int(raw_matrix[1, 1]),
                    "selected_c": best_c,
                    "validation_balanced_accuracy": best_score,
                }
            )
            for class_name, label in CLASS_TO_LABEL.items():
                subset = level_frame[level_frame["true_label"] == label]
                metric_rows.append(
                    {
                        "seed": seed,
                        "occlusion_percent": level,
                        "class_name": class_name,
                        "n": len(subset),
                        "raw_accuracy": float(subset["raw_correct"].mean()),
                        "raw_balanced_accuracy": np.nan,
                        "raw_f1_macro": np.nan,
                        "raw_aircraft_2_selection_rate": float(
                            np.mean(subset["raw_prediction"] == 1)
                        ),
                        "raw_confidence": float(
                            subset["raw_confidence"].mean()
                        ),
                        "raw_entropy": float(subset["raw_entropy"].mean()),
                        "centered_accuracy": float(
                            subset["centered_correct"].mean()
                        ),
                        "centered_aircraft_2_selection_rate": float(
                            np.mean(subset["centered_prediction"] == 1)
                        ),
                        "tn": np.nan,
                        "fp": np.nan,
                        "fn": np.nan,
                        "tp": np.nan,
                        "selected_c": best_c,
                        "validation_balanced_accuracy": best_score,
                    }
                )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    splits = pd.DataFrame(split_rows)
    predictions.to_csv(
        condition_dir / "trial_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        condition_dir / "metrics_by_seed_and_class.csv",
        index=False,
        encoding="utf-8-sig",
    )
    splits.to_csv(
        condition_dir / "source_splits.csv",
        index=False,
        encoding="utf-8-sig",
    )

    numeric_columns = [
        column
        for column in metrics.columns
        if column not in {"seed", "occlusion_percent", "class_name"}
    ]
    summary = (
        metrics.groupby(["occlusion_percent", "class_name"])[numeric_columns]
        .agg(["mean", "std", "sem"])
    )
    summary.columns = [
        f"{stat}_{column}" for column, stat in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()
    summary.to_csv(
        condition_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics, summary


def correlation_rdm(features: np.ndarray) -> np.ndarray:
    distances = pdist(features, metric="correlation")
    distances = np.nan_to_num(distances, nan=0.0, posinf=2.0, neginf=0.0)
    return squareform(distances).astype(np.float32)


def condensed_upper(matrix: np.ndarray) -> np.ndarray:
    return squareform(matrix, checks=False)


def permutation_pvalue(
    target_rdm: np.ndarray,
    model_rdm: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    observed = float(
        spearmanr(
            condensed_upper(target_rdm),
            condensed_upper(model_rdm),
        ).statistic
    )
    if permutations <= 0:
        return observed, np.nan
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        order = rng.permutation(model_rdm.shape[0])
        permuted = model_rdm[np.ix_(order, order)]
        candidate = spearmanr(
            condensed_upper(target_rdm),
            condensed_upper(permuted),
        ).statistic
        exceedances += int(candidate >= observed)
    return observed, (exceedances + 1) / (permutations + 1)


def save_rdms_and_rsa(
    records: Sequence[PairRecord],
    features_by_layer: dict[str, np.ndarray],
    condition_dir: Path,
    permutations: int,
    brain_rdms: dict[str, Path],
) -> None:
    rdm_dir = condition_dir / "rdm"
    rdm_dir.mkdir(parents=True, exist_ok=True)
    order = pd.DataFrame(asdict(record) for record in records)
    order.insert(0, "rdm_index", np.arange(len(order)))
    order.to_csv(rdm_dir / "stimulus_order.csv", index=False, encoding="utf-8-sig")

    levels = np.array([record.occlusion_percent for record in records])
    labels = np.array([record.label for record in records])
    occ_structure = np.abs(levels[:, None] - levels[None, :]).astype(np.float32)
    class_structure = (labels[:, None] != labels[None, :]).astype(np.float32)
    structure_rows = []

    loaded_brain_rdms = {}
    for name, path in brain_rdms.items():
        if path is None:
            continue
        brain = np.load(path)
        if brain.shape != (len(records), len(records)):
            raise ValueError(
                f"{name} RDM shape {brain.shape} does not match "
                f"model stimulus count {len(records)}."
            )
        loaded_brain_rdms[name] = brain

    for layer_name, features in features_by_layer.items():
        rdm = correlation_rdm(features)
        np.save(rdm_dir / f"{layer_name}_rdm.npy", rdm)
        occ_r, occ_p = permutation_pvalue(
            occ_structure, rdm, permutations, seed=0
        )
        class_r, class_p = permutation_pvalue(
            class_structure, rdm, permutations, seed=1
        )
        row = {
            "layer": layer_name,
            "occlusion_structure_rsa": occ_r,
            "occlusion_structure_p": occ_p,
            "aircraft_structure_rsa": class_r,
            "aircraft_structure_p": class_p,
        }
        for brain_name, brain_rdm in loaded_brain_rdms.items():
            brain_r, brain_p = permutation_pvalue(
                brain_rdm, rdm, permutations, seed=2
            )
            row[f"{brain_name}_rsa"] = brain_r
            row[f"{brain_name}_p"] = brain_p
        structure_rows.append(row)
    pd.DataFrame(structure_rows).to_csv(
        rdm_dir / "rsa_summary.csv", index=False, encoding="utf-8-sig"
    )


def read_human_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    aliases = {
        "occlusion": "occlusion_percent",
        "mean": "mean_accuracy",
        "sem": "sem_accuracy",
    }
    frame = frame.rename(columns=aliases)
    required = {"occlusion_percent", "mean_accuracy"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Human CSV must contain {sorted(required)}; found {list(frame.columns)}"
        )
    if "sem_accuracy" not in frame:
        frame["sem_accuracy"] = np.nan
    return frame[
        ["occlusion_percent", "mean_accuracy", "sem_accuracy"]
    ].sort_values("occlusion_percent")


def make_final_outputs(
    summaries: dict[str, pd.DataFrame],
    human: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    table = human.rename(
        columns={
            "mean_accuracy": "human_mean_accuracy",
            "sem_accuracy": "human_sem",
        }
    ).copy()
    for condition, summary in summaries.items():
        overall = summary[summary["class_name"] == "overall"][
            [
                "occlusion_percent",
                "mean_raw_accuracy",
                "std_raw_accuracy",
                "sem_raw_accuracy",
                "mean_raw_aircraft_2_selection_rate",
                "mean_raw_confidence",
                "mean_raw_entropy",
                "mean_centered_accuracy",
            ]
        ].rename(
            columns={
                column: f"{condition}_{column}"
                for column in [
                    "mean_raw_accuracy",
                    "std_raw_accuracy",
                    "sem_raw_accuracy",
                    "mean_raw_aircraft_2_selection_rate",
                    "mean_raw_confidence",
                    "mean_raw_entropy",
                    "mean_centered_accuracy",
                ]
            }
        )
        table = table.merge(overall, on="occlusion_percent", how="left")
    table.to_csv(
        output_dir / "final_comparison_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "occlusion_percent",
        "human_mean_accuracy",
        *[
            f"{condition}_mean_raw_accuracy"
            for condition in summaries
        ],
    ]
    table[display_columns].to_csv(
        output_dir / "final_comparison_table_display.csv",
        index=False,
        encoding="utf-8-sig",
    )

    palette = {
        "color": "#4472C4",
        "grayscale": "#70AD47",
        "brightness_matched": "#ED7D31",
    }
    labels = {
        "color": "ViT Color",
        "grayscale": "ViT Grayscale",
        "brightness_matched": "ViT Brightness-matched",
    }
    figure, axis = plt.subplots(figsize=(8.4, 5.7))
    axis.errorbar(
        human["occlusion_percent"],
        human["mean_accuracy"],
        yerr=human["sem_accuracy"],
        color="black",
        marker="o",
        linewidth=2.6,
        capsize=4,
        label="Human (mean ± SEM)",
    )
    for condition, summary in summaries.items():
        overall = summary[summary["class_name"] == "overall"].sort_values(
            "occlusion_percent"
        )
        axis.errorbar(
            overall["occlusion_percent"],
            overall["mean_raw_accuracy"],
            yerr=overall["std_raw_accuracy"],
            color=palette[condition],
            marker="o",
            linewidth=2.2,
            capsize=4,
            label=f"{labels[condition]} (mean ± SD across seeds)",
        )
    axis.axhline(0.5, color="#777777", linestyle=":", label="Chance (0.5)")
    axis.set(
        xlabel="Occlusion (%)",
        ylabel="Forced-choice accuracy",
        title="Final comparison: Human vs frozen ViT",
        xticks=list(LEVELS),
        ylim=(0.30, 1.03),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8.5)
    figure.tight_layout()
    figure.savefig(output_dir / "final_comparison_graph.png", dpi=220)
    plt.close(figure)

    # Presentation-ready rendering of the compact final comparison table.
    compact = table[display_columns].copy()
    compact.columns = [
        "Occlusion",
        "Human",
        *[
            {
                "color": "ViT Color",
                "grayscale": "ViT Grayscale",
                "brightness_matched": "ViT Brightness-matched",
            }[condition]
            for condition in summaries
        ],
    ]
    compact["Occlusion"] = compact["Occlusion"].map(lambda value: f"{value}%")
    for column in compact.columns[1:]:
        compact[column] = compact[column].map(lambda value: f"{value:.3f}")
    figure, axis = plt.subplots(
        figsize=(2.1 * len(compact.columns), 1.25 + 0.58 * len(compact))
    )
    axis.axis("off")
    rendered_table = axis.table(
        cellText=compact.values,
        colLabels=compact.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    rendered_table.auto_set_font_size(False)
    rendered_table.set_fontsize(10)
    rendered_table.scale(1.0, 1.55)
    for (row, _), cell in rendered_table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#D9EAF7")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F7FA")
    axis.set_title("Final Comparison Table", fontsize=14, weight="bold", pad=10)
    figure.tight_layout()
    figure.savefig(output_dir / "final_comparison_table.png", dpi=220)
    plt.close(figure)

    # Diagnostic plot: class-specific accuracy and A2 response bias.
    figure, axes = plt.subplots(
        1, len(summaries), figsize=(5.2 * len(summaries), 4.8), squeeze=False
    )
    for axis, (condition, summary) in zip(axes[0], summaries.items()):
        for class_name, color in (
            ("aircraft_1", "#5B9BD5"),
            ("aircraft_2", "#ED7D31"),
        ):
            subset = summary[summary["class_name"] == class_name].sort_values(
                "occlusion_percent"
            )
            axis.errorbar(
                subset["occlusion_percent"],
                subset["mean_raw_accuracy"],
                yerr=subset["std_raw_accuracy"],
                marker="o",
                capsize=3,
                color=color,
                label=class_name,
            )
        overall = summary[summary["class_name"] == "overall"].sort_values(
            "occlusion_percent"
        )
        axis.plot(
            overall["occlusion_percent"],
            overall["mean_raw_aircraft_2_selection_rate"],
            "--s",
            color="#7030A0",
            label="A2 selection rate",
        )
        axis.axhline(0.5, color="#777777", linestyle=":")
        axis.set(
            title=labels[condition],
            xlabel="Occlusion (%)",
            ylabel="Rate",
            xticks=list(LEVELS),
            ylim=(-0.03, 1.03),
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Class accuracy and response-bias diagnostic")
    figure.tight_layout()
    figure.savefig(output_dir / "class_accuracy_and_bias.png", dpi=220)
    plt.close(figure)
    return table


def make_blank_centered_outputs(
    summaries: dict[str, pd.DataFrame],
    human: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Render the candidate main analysis using blank-centered decisions."""
    table = human.rename(
        columns={
            "mean_accuracy": "human_mean_accuracy",
            "sem_accuracy": "human_sem",
        }
    ).copy()
    for condition, summary in summaries.items():
        overall = summary[summary["class_name"] == "overall"][
            [
                "occlusion_percent",
                "mean_centered_accuracy",
                "std_centered_accuracy",
                "sem_centered_accuracy",
                "mean_centered_aircraft_2_selection_rate",
            ]
        ].rename(
            columns={
                column: f"{condition}_{column}"
                for column in [
                    "mean_centered_accuracy",
                    "std_centered_accuracy",
                    "sem_centered_accuracy",
                    "mean_centered_aircraft_2_selection_rate",
                ]
            }
        )
        table = table.merge(overall, on="occlusion_percent", how="left")
    table.to_csv(
        output_dir / "final_comparison_table_blank_centered.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "occlusion_percent",
        "human_mean_accuracy",
        *[
            f"{condition}_mean_centered_accuracy"
            for condition in summaries
        ],
    ]
    table[display_columns].to_csv(
        output_dir / "final_comparison_table_blank_centered_display.csv",
        index=False,
        encoding="utf-8-sig",
    )

    palette = {
        "color": "#4472C4",
        "grayscale": "#70AD47",
        "brightness_matched": "#ED7D31",
    }
    labels = {
        "color": "ViT Color",
        "grayscale": "ViT Grayscale",
        "brightness_matched": "ViT Brightness-matched",
    }
    figure, axis = plt.subplots(figsize=(8.4, 5.7))
    axis.errorbar(
        human["occlusion_percent"],
        human["mean_accuracy"],
        yerr=human["sem_accuracy"],
        color="black",
        marker="o",
        linewidth=2.6,
        capsize=4,
        label="Human (mean ± SEM)",
    )
    for condition, summary in summaries.items():
        overall = summary[summary["class_name"] == "overall"].sort_values(
            "occlusion_percent"
        )
        axis.errorbar(
            overall["occlusion_percent"],
            overall["mean_centered_accuracy"],
            yerr=overall["std_centered_accuracy"],
            color=palette[condition],
            marker="o",
            linewidth=2.2,
            capsize=4,
            label=f"{labels[condition]} centered (mean ± SD)",
        )
    axis.axhline(0.5, color="#777777", linestyle=":", label="Chance (0.5)")
    axis.set(
        xlabel="Occlusion (%)",
        ylabel="Blank-centered forced-choice accuracy",
        title="Human vs frozen ViT: blank-centered decision",
        xticks=list(LEVELS),
        ylim=(0.30, 1.03),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8.5)
    figure.tight_layout()
    figure.savefig(
        output_dir / "final_comparison_graph_blank_centered.png", dpi=220
    )
    plt.close(figure)

    compact = table[display_columns].copy()
    compact.columns = [
        "Occlusion",
        "Human",
        *[
            {
                "color": "ViT Color centered",
                "grayscale": "ViT Grayscale centered",
                "brightness_matched": "ViT Brightness-matched centered",
            }[condition]
            for condition in summaries
        ],
    ]
    compact["Occlusion"] = compact["Occlusion"].map(lambda value: f"{value}%")
    for column in compact.columns[1:]:
        compact[column] = compact[column].map(lambda value: f"{value:.3f}")
    figure, axis = plt.subplots(
        figsize=(2.35 * len(compact.columns), 1.25 + 0.58 * len(compact))
    )
    axis.axis("off")
    rendered_table = axis.table(
        cellText=compact.values,
        colLabels=compact.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    rendered_table.auto_set_font_size(False)
    rendered_table.set_fontsize(9.5)
    rendered_table.scale(1.0, 1.55)
    for (row, _), cell in rendered_table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#D9EAF7")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F7FA")
    axis.set_title(
        "Final Comparison Table — Blank-centered",
        fontsize=14,
        weight="bold",
        pad=10,
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "final_comparison_table_blank_centered.png", dpi=220
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(summaries), figsize=(5.2 * len(summaries), 4.8), squeeze=False
    )
    for axis, (condition, summary) in zip(axes[0], summaries.items()):
        for class_name, color in (
            ("aircraft_1", "#5B9BD5"),
            ("aircraft_2", "#ED7D31"),
        ):
            subset = summary[summary["class_name"] == class_name].sort_values(
                "occlusion_percent"
            )
            axis.errorbar(
                subset["occlusion_percent"],
                subset["mean_centered_accuracy"],
                yerr=subset["std_centered_accuracy"],
                marker="o",
                capsize=3,
                color=color,
                label=class_name,
            )
        overall = summary[summary["class_name"] == "overall"].sort_values(
            "occlusion_percent"
        )
        axis.plot(
            overall["occlusion_percent"],
            overall["mean_centered_aircraft_2_selection_rate"],
            "--s",
            color="#7030A0",
            label="A2 selection rate",
        )
        axis.axhline(0.5, color="#777777", linestyle=":")
        axis.set(
            title=labels[condition],
            xlabel="Occlusion (%)",
            ylabel="Rate",
            xticks=list(LEVELS),
            ylim=(-0.03, 1.03),
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Blank-centered class accuracy and response bias")
    figure.tight_layout()
    figure.savefig(
        output_dir / "class_accuracy_and_bias_blank_centered.png", dpi=220
    )
    plt.close(figure)
    return table


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if not 0 < args.validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1.")
    if args.pca_components < 1:
        raise ValueError("--pca-components must be positive.")
    if any(layer < 1 or layer > 12 for layer in args.rdm_layers):
        raise ValueError("ViT-B/16 layer numbers must be in 1..12.")

    records = build_pair_manifest(args.project_root.expanduser().resolve())
    manifest = pd.DataFrame(asdict(record) for record in records)
    manifest.to_csv(
        output_dir / "paired_source_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit = visible_pixel_pair_audit(records)
    audit.to_csv(
        output_dir / "pairing_audit.csv", index=False, encoding="utf-8-sig"
    )
    print(
        "Pair audit:",
        audit.groupby(["class_name", "occlusion_percent"])[
            ["visible_fraction", "visible_pixel_correlation"]
        ]
        .mean()
        .round(3)
        .to_string(),
    )
    minimum_pair_correlation = float(
        audit["visible_pixel_correlation"].min()
    )
    if minimum_pair_correlation < 0.80:
        raise RuntimeError(
            "At least one filename-derived original/occluded pair failed "
            f"pixel verification (minimum r={minimum_pair_correlation:.3f})."
        )
    if args.audit_only:
        print(f"Audit complete: {output_dir}")
        return

    seed_everything(args.seeds[0])
    device = select_device(args.device)
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    base_transform = weights.transforms()
    model = vit_b_16(weights=weights).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("ViT backbone freeze check failed.")
    print(f"Device: {device}; weights: ViT_B_16_Weights.IMAGENET1K_V1")

    condition_transforms = {
        "color": base_transform,
        "grayscale": GrayscaleTransform(base_transform),
        "brightness_matched": BrightnessMatchedTransform(base_transform),
    }
    summaries: dict[str, pd.DataFrame] = {}
    intact_paths = [record.intact_path for record in records]
    occluded_paths = [record.occluded_path for record in records]

    for condition in args.conditions:
        print(f"\n=== Condition: {condition} ===")
        condition_dir = output_dir / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        transform = condition_transforms[condition]
        intact_features = extract_features(
            model,
            intact_paths,
            transform,
            device,
            args.batch_size,
            args.num_workers,
            args.rdm_layers,
            collect_layers=True,
        )
        occluded_by_layer = extract_features(
            model,
            occluded_paths,
            transform,
            device,
            args.batch_size,
            args.num_workers,
            args.rdm_layers,
            collect_layers=True,
        )
        feature_dir = condition_dir / "features"
        feature_dir.mkdir(exist_ok=True)
        for _name, _values in intact_features.items():
            np.save(feature_dir / f"intact_{_name}.npy", _values)
        for _name, _values in occluded_by_layer.items():
            np.save(feature_dir / f"occluded_{_name}.npy", _values)
        blank = transform(Image.new("RGB", (224, 224), color=(0, 0, 0))).unsqueeze(0)
        with torch.inference_mode():
            blank_feature = (
                vit_cls_features(model, blank.to(device), set())["final_cls"][0]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        _, summary = run_probes(
            records,
            intact_features["final_cls"],
            occluded_by_layer["final_cls"],
            blank_feature,
            args.seeds,
            args.test_size,
            args.validation_size,
            args.pca_components,
            args.c_grid,
            condition_dir,
        )
        summaries[condition] = summary
        rdm_features = {
            name: values
            for name, values in occluded_by_layer.items()
            if name != "final_cls"
        }
        save_rdms_and_rsa(
            records,
            rdm_features,
            condition_dir,
            args.structure_permutations,
            {
                "occipital": args.occipital_rdm,
                "dacc": args.dacc_rdm,
            },
        )

    human = read_human_results(args.human_csv.expanduser().resolve())
    final_table = make_final_outputs(summaries, human, output_dir)
    make_blank_centered_outputs(summaries, human, output_dir)
    config = {
        "project_root": str(args.project_root.expanduser().resolve()),
        "output_dir": str(output_dir),
        "conditions": args.conditions,
        "levels": list(LEVELS),
        "seeds": args.seeds,
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "pca_components": args.pca_components,
        "c_grid": args.c_grid,
        "rdm_layers_one_based": args.rdm_layers,
        "structure_permutations": args.structure_permutations,
        "device": str(device),
        "backbone": "torchvision vit_b_16",
        "weights": "ViT_B_16_Weights.IMAGENET1K_V1",
        "weights_file_sha256": (
            sha256(Path(torch.hub.get_dir()) / "checkpoints" / "vit_b_16-c867db91.pth")
            if (Path(torch.hub.get_dir()) / "checkpoints" / "vit_b_16-c867db91.pth").is_file()
            else None
        ),
        "backbone_frozen": True,
        "classification_feature": "final LayerNorm CLS",
        "probe": "StandardScaler -> PCA(full, 30) -> L2 logistic regression",
        "primary_decision": "raw logit >= 0",
        "blank_centering": "sensitivity analysis only",
        "source_leakage_control": (
            "stratified source-ID split before intact training / occluded testing"
        ),
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
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nFinal comparison table:")
    print(final_table.round(4).to_string(index=False))
    print(f"\nComplete: {output_dir}")


if __name__ == "__main__":
    main()
