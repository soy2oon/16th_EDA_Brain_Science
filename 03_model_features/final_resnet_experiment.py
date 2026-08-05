"""Leakage-safe frozen ResNet-50 aircraft occlusion experiment.

This pipeline uses the same paired manifest, preprocessing, source-level
splits, probe selection, decisions, metrics, and RDM conventions as the final
ViT and CORnet-S experiments. ResNet-specific choices are ImageNet-1K V2
ResNet-50, fully frozen weights, and global-average-pooled residual features.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache-resnet"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache-resnet"))

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
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from final_cornet_experiment import (
    BrightnessMatchedTransform,
    base_transform,
    create_bias_figure,
    load_human,
    run_probes,
    seed_everything,
    select_device,
)
from final_vit_experiment import (
    PairRecord,
    build_pair_manifest,
    sha256,
    visible_pixel_pair_audit,
)


SEEDS = (42, 142, 242, 342, 442)
OCCLUSION_LEVELS = (10, 70, 90)
RDM_LAYERS = ("layer1", "layer2", "layer3", "layer4", "avgpool")
BEHAVIOR_LAYER = "avgpool"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final leakage-safe frozen ResNet-50 experiment."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/Users/gyuhongcho/Desktop/DSL/EDA_AircraftFiles"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "final_resnet_results",
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


def load_resnet(device: torch.device):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("ResNet-50 backbone was not fully frozen.")
    return model, weights


def pooled_features(
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
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in RDM_LAYERS}
    capture: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def save_output(_module, _inputs, output):
            capture[name] = output.detach()

        return save_output

    handles = [
        getattr(model, name).register_forward_hook(hook(name))
        for name in RDM_LAYERS
    ]
    model.eval()
    try:
        with torch.inference_mode():
            for images, _indices in loader:
                capture.clear()
                model(images.to(device, non_blocking=True))
                if set(capture) != set(RDM_LAYERS):
                    raise RuntimeError(
                        f"Missing ResNet activations: observed {sorted(capture)}."
                    )
                for name in RDM_LAYERS:
                    activation = capture[name]
                    if activation.ndim == 4:
                        activation = activation.mean(dim=(-2, -1))
                    chunks[name].append(
                        activation.flatten(start_dim=1).cpu().numpy()
                    )
    finally:
        for handle in handles:
            handle.remove()

    output = {
        name: np.concatenate(values).astype(np.float64)
        for name, values in chunks.items()
    }
    for name, values in output.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite ResNet features in {name}.")
    return output


def blank_features(model, transform, device: torch.device):
    temporary = ROOT / ".blank-resnet.png"
    Image.new("RGB", (224, 224), color=(0, 0, 0)).save(temporary)
    try:
        return pooled_features(
            model,
            [str(temporary)],
            transform,
            device,
            batch_size=1,
            num_workers=0,
        )
    finally:
        temporary.unlink(missing_ok=True)


def correlation_rdm(features: np.ndarray) -> np.ndarray:
    values = features.astype(np.float64)
    values -= values.mean(axis=1, keepdims=True)
    values /= np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
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
                index
                for index, record in enumerate(ordered_records)
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

    figure, axes = plt.subplots(1, len(RDM_LAYERS), figsize=(19, 3.8))
    maximum = max(float(matrix.max()) for matrix in condition_rdms)
    for axis, layer, matrix in zip(axes, RDM_LAYERS, condition_rdms):
        image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=maximum)
        axis.set_title(layer)
        axis.set_xticks(range(6), CONDITION_ORDER, rotation=45)
        axis.set_yticks(range(6), CONDITION_ORDER)
    figure.colorbar(image, ax=axes, shrink=0.8, label="1 - Pearson r")
    figure.suptitle("Frozen ResNet-50 condition RDMs")
    figure.savefig(output / "condition_rdms.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_final_outputs(
    summaries: dict[str, pd.DataFrame],
    human: dict[int, float],
    output: Path,
) -> pd.DataFrame:
    rows = []
    for level in OCCLUSION_LEVELS:
        row = {"occlusion_percent": level, "human_accuracy": human[level]}
        for condition, summary in summaries.items():
            value = summary[
                (summary["occlusion_percent"] == level)
                & (summary["class_name"] == "overall")
            ].iloc[0]
            row[f"{condition}_centered_mean"] = value["mean_centered_accuracy"]
            row[f"{condition}_centered_sd"] = value["std_centered_accuracy"]
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
            "color_centered_mean": "ResNet-50 Color Centered",
            "brightness_matched_centered_mean": (
                "ResNet-50 Brightness/Color Controlled Centered"
            ),
        }
    )
    display.to_csv(output / "final_comparison_table_display.csv", index=False)

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
        "color": "ResNet-50 color centered",
        "brightness_matched": "ResNet-50 brightness/color controlled centered",
    }
    colors = {"color": "#E08040", "brightness_matched": "#4F8A5B"}
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
        title="Human vs frozen ResNet-50",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "final_comparison_graph.png", dpi=220)
    plt.close(figure)
    return table


def create_raw_centered_figure(
    summaries: dict[str, pd.DataFrame], output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    labels = {
        "color": "Color",
        "brightness_matched": "Brightness/color controlled",
    }
    for axis, (condition, summary) in zip(axes, summaries.items()):
        overall = summary[summary["class_name"] == "overall"].sort_values(
            "occlusion_percent"
        )
        for column, label in (
            ("mean_raw_accuracy", "Raw"),
            ("mean_centered_accuracy", "Blank-centered"),
        ):
            axis.errorbar(
                overall["occlusion_percent"],
                overall[column],
                yerr=overall[
                    "std_raw_accuracy"
                    if label == "Raw"
                    else "std_centered_accuracy"
                ],
                marker="o",
                linewidth=2.2,
                capsize=4,
                label=label,
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
    figure.suptitle("ResNet-50 raw vs blank-centered decisions")
    figure.tight_layout()
    figure.savefig(output / "raw_vs_blank_centered.png", dpi=220)
    plt.close(figure)


def write_run_config(
    args,
    output: Path,
    device: torch.device,
    weights: models.ResNet50_Weights,
) -> None:
    checkpoint = (
        Path(os.environ["TORCH_HOME"])
        / "hub"
        / "checkpoints"
        / Path(weights.url).name
    )
    config = {
        "seeds": args.seeds,
        "test_size": args.test_size,
        "validation_size": args.validation_size,
        "pca_components": args.pca_components,
        "pca_solver": "full",
        "c_grid": args.c_grid,
        "device": str(device),
        "backbone": "torchvision ResNet-50",
        "weights": "ResNet50_Weights.IMAGENET1K_V2",
        "weights_file_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
        "backbone_frozen": True,
        "classification_feature": "avgpool global average pooling (2048D)",
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
    pd.DataFrame(asdict(record) for record in records).to_csv(
        output / "paired_source_manifest.csv", index=False
    )
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
    print(f"Loading frozen ImageNet ResNet-50 on {device}...")
    model, weights = load_resnet(device)
    base = base_transform()
    transforms = {
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
        transform = transforms[condition]
        intact = pooled_features(
            model,
            intact_paths,
            transform,
            device,
            args.batch_size,
            args.num_workers,
        )
        occluded = pooled_features(
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
            intact[BEHAVIOR_LAYER],
            occluded[BEHAVIOR_LAYER],
            blank[BEHAVIOR_LAYER],
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
    if set(human) != set(OCCLUSION_LEVELS):
        human = HUMAN_DEFAULT.copy()
    final_table = create_final_outputs(summaries, human, output)
    create_bias_figure(summaries, output)
    create_raw_centered_figure(summaries, output)
    write_run_config(args, output, device, weights)
    print("\nFinal comparison:")
    print(final_table.round(4).to_string(index=False))
    print(f"\nComplete: {output}")


if __name__ == "__main__":
    main()
