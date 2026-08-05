"""Build a portable manifest for intact training/calibration and occluded testing."""
import re
from pathlib import Path

import pandas as pd

import config as C


OCCLUDED = re.compile(r"Aircraft(?P<aircraft>[12])_(?P<occlusion>10|70|90)__(?P<index>\d+)\.jpg")
INTACT = re.compile(r"Aircraft(?P<aircraft>[12])__(?P<index>\d+)\.jpg")


def main():
    rows = []
    for path in sorted(C.STIMULI.rglob("*.jpg")):
        match = INTACT.fullmatch(path.name) if path.parent.name == "intact" else OCCLUDED.fullmatch(path.name)
        if not match:
            continue
        aircraft = int(match.group("aircraft"))
        occlusion = 0 if path.parent.name == "intact" else int(match.group("occlusion"))
        rows.append({
            "image_id": f"A{aircraft}_O{occlusion}_{int(match.group('index')):03d}",
            "path": str(path.resolve()), "aircraft": aircraft,
            "label": aircraft - 1, "class_name": "A" if aircraft == 1 else "B",
            "occlusion": occlusion, "condition": f"{'A' if aircraft == 1 else 'B'}{occlusion}",
            "split": "intact" if occlusion == 0 else "occluded_test",
        })
    frame = pd.DataFrame(rows).sort_values(["split", "aircraft", "occlusion", "image_id"])
    expected = {("intact", 1, 0): 150, ("intact", 2, 0): 150}
    expected.update({("occluded_test", a, o): 50 for a in (1, 2) for o in (10, 70, 90)})
    counts = frame.groupby(["split", "aircraft", "occlusion"]).size().to_dict()
    if counts != expected:
        raise RuntimeError(f"Unexpected stimulus counts: {counts}")
    frame.to_csv(C.DATA / "manifest.csv", index=False)
    print(frame.groupby(["split", "condition"]).size().to_string())
    print(f"Saved {len(frame)} rows to {C.DATA / 'manifest.csv'}")


if __name__ == "__main__":
    main()

