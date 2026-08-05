"""Extract frozen ImageNet CORnet-S representations for all 600 images."""
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

import config as C


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def transform():
    return v2.Compose([
        v2.Resize(256, antialias=True), v2.CenterCrop(224), v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])


def batches(frame, batch_size):
    tx = transform()
    for start in range(0, len(frame), batch_size):
        part = frame.iloc[start:start + batch_size]
        images = []
        for path in part.path:
            with Image.open(path) as image:
                images.append(tx(image.convert("RGB")))
        yield part, torch.stack(images)


def main(batch_size=8):
    from cornet import cornet_s

    manifest = pd.read_csv(C.DATA / "manifest.csv")
    compute = device()
    model = cornet_s(pretrained=True, map_location="cpu").module.to(compute).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    captures = defaultdict(list)
    handles = []
    for area in ("V1", "V2", "V4", "IT"):
        handles.append(getattr(model, area).output.register_forward_hook(
            lambda _m, _i, value, area=area: captures[area].append(value.detach().cpu())
        ))

    pooled = {layer: [] for layer in C.LAYERS}
    condition_sums = {layer: {} for layer in C.LAYERS}
    condition_counts = defaultdict(int)
    ordered_rows = []
    with torch.inference_mode():
        for part, images in batches(manifest, batch_size):
            captures.clear()
            model(images.to(compute))
            layer_batches = {}
            for area, outputs in captures.items():
                for time, activation in enumerate(outputs, 1):
                    layer = f"{area}_t{time}"
                    layer_batches[layer] = activation.double()
                    pooled[layer].append(activation.mean(dim=(-2, -1)).float().numpy())
            for row_index, row in enumerate(part.itertuples(index=False)):
                if row.condition in C.CONDITIONS:
                    condition_counts[row.condition] += 1
                    for layer, activation in layer_batches.items():
                        vector = activation[row_index].reshape(-1).numpy()
                        condition_sums[layer][row.condition] = condition_sums[layer].get(row.condition, 0) + vector
            ordered_rows.append(part)

    for handle in handles:
        handle.remove()
    feature_rows = pd.concat(ordered_rows, ignore_index=True)
    feature_rows.to_csv(C.FEATURES / "feature_index.csv", index=False)
    for layer in C.LAYERS:
        np.save(C.FEATURES / f"pooled_{layer}.npy", np.concatenate(pooled[layer], axis=0))
        patterns = np.stack([condition_sums[layer][condition] / condition_counts[condition]
                             for condition in C.CONDITIONS]).astype(np.float32)
        np.save(C.FEATURES / f"condition_patterns_{layer}.npy", patterns)
    metadata = {
        "weights": "official ImageNet pretrained, frozen", "device": str(compute),
        "preprocessing": "resize shorter edge 256; center crop 224; ImageNet normalization",
        "pooled_probe_features": "global average over H,W for every layer/time",
        "rsa_features": "flattened spatial activation, averaged across 50 images per condition",
        "condition_order": list(C.CONDITIONS), "condition_counts": dict(condition_counts),
    }
    (C.FEATURES / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Extracted frozen features on {compute}; rows={len(feature_rows)}")


if __name__ == "__main__":
    main()

