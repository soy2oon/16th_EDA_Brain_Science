"""Extract CORnet-S representations and build condition-level RDMs.

The six conditions are always ordered as A10, A70, A90, B10, B70, B90.
Each RDM uses correlation distance (1 - Pearson r) between condition means.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

import config as C


CONDITIONS = ("A10", "A70", "A90", "B10", "B70", "B90")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def condition_name(row) -> str:
    return f"{row.class_name}{int(row.occlusion)}"


def upper_triangle(rdm: np.ndarray) -> np.ndarray:
    return rdm[np.triu_indices(len(rdm), k=1)]


def correlation_rdm(patterns: np.ndarray) -> np.ndarray:
    """Correlation-distance RDM, robust to constant vectors."""
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    similarity = np.clip(normalized @ normalized.T, -1, 1)
    rdm = 1.0 - similarity
    np.fill_diagonal(rdm, 0.0)
    return rdm


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[(frame.split == "test") & frame.occlusion.isin((10, 70, 90))].copy()
    frame["condition"] = frame.apply(condition_name, axis=1)
    counts = frame.groupby("condition").size().reindex(CONDITIONS)
    if counts.isna().any():
        raise ValueError(f"Missing conditions in manifest: {counts[counts.isna()].index.tolist()}")
    return frame


def make_transform():
    return v2.Compose([
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def iter_batches(paths, transform, batch_size):
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
        yield torch.stack(images)


def low_level_patterns(frame: pd.DataFrame) -> np.ndarray:
    """Condition means of 32x32 grayscale pixels (low-level nuisance model)."""
    rows = []
    for condition in CONDITIONS:
        vectors = []
        for path in frame.loc[frame.condition == condition, "path"]:
            with Image.open(path) as image:
                image = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
                vectors.append(np.asarray(image, dtype=np.float64).reshape(-1) / 255.0)
        rows.append(np.mean(vectors, axis=0))
    return np.stack(rows)


def extract(args) -> None:
    try:
        from cornet import cornet_s
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_manifest(Path(args.manifest))
    device = resolve_device(args.device)
    model = cornet_s(pretrained=not args.random_weights, map_location="cpu")
    model = model.module.to(device).eval()  # official loader wraps DataParallel

    # output hooks fire once per recurrent step: V1x1, V2x2, V4x4, ITx2.
    captures: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles = []
    for area in ("V1", "V2", "V4", "IT"):
        module = getattr(model, area).output
        handles.append(module.register_forward_hook(
            lambda _m, _i, value, area=area: captures[area].append(value.detach().cpu())
        ))

    transform = make_transform()
    sums: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    counts: dict[str, int] = {}
    with torch.inference_mode():
        for condition in CONDITIONS:
            paths = frame.loc[frame.condition == condition, "path"].tolist()
            counts[condition] = len(paths)
            layer_sums: dict[str, np.ndarray] = {}
            seen = 0
            for batch in iter_batches(paths, transform, args.batch_size):
                captures.clear()
                model(batch.to(device))
                for area, time_outputs in captures.items():
                    for time_index, activation in enumerate(time_outputs, start=1):
                        key = f"{area}_t{time_index}"
                        vector = activation.double().sum(dim=0).reshape(-1).numpy()
                        layer_sums[key] = layer_sums.get(key, 0) + vector
                seen += len(batch)
            for key, value in layer_sums.items():
                sums[key][condition] = value / seen

    for handle in handles:
        handle.remove()

    layer_order = [f"{a}_t{t}" for a, n in (("V1", 1), ("V2", 2), ("V4", 4), ("IT", 2)) for t in range(1, n + 1)]
    summary_rows = []
    for layer in layer_order:
        patterns = np.stack([sums[layer][condition] for condition in CONDITIONS])
        rdm = correlation_rdm(patterns)
        np.save(output / f"cornet_s_{layer}_patterns.npy", patterns.astype(np.float32))
        np.save(output / f"cornet_s_{layer}_rdm.npy", rdm)
        pd.DataFrame(rdm, index=CONDITIONS, columns=CONDITIONS).to_csv(output / f"cornet_s_{layer}_rdm.csv")
        summary_rows.append({"layer": layer, "n_features": patterns.shape[1], "rdm_mean": upper_triangle(rdm).mean()})

    nuisance_patterns = low_level_patterns(frame)
    nuisance_rdm = correlation_rdm(nuisance_patterns)
    np.save(output / "low_level_pixel_rdm.npy", nuisance_rdm)
    pd.DataFrame(nuisance_rdm, index=CONDITIONS, columns=CONDITIONS).to_csv(output / "low_level_pixel_rdm.csv")
    pd.DataFrame(summary_rows).to_csv(output / "cornet_s_rdm_summary.csv", index=False)
    metadata = {
        "condition_order": list(CONDITIONS), "condition_counts": counts,
        "distance": "correlation (1 - Pearson r)", "weights": "random" if args.random_weights else "ImageNet",
        "device": str(device), "preprocessing": "resize_shorter_256, center_crop_224, ImageNet normalization",
        "nuisance": "condition-mean 32x32 grayscale pixels",
    }
    (output / "cornet_s_rsa_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Saved CORnet-S RDMs to {output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(C.MANIFEST))
    parser.add_argument("--output", default=str(C.RESULTS_DIR / "rsa"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--random-weights", action="store_true", help="Smoke tests only; do not use for RQ1")
    return parser.parse_args()


if __name__ == "__main__":
    extract(parse_args())
