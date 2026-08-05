#!/usr/bin/env python3
"""자극 세트 사전 검증 — 모델 실행 전에 반드시 통과해야 함.

이번 사고(Aircraft2 = Aircraft1 중복본)를 모델 없이, 몇 초 안에 잡아냅니다.
픽셀 수준만 보므로 torch / 가중치 / GPU 가 필요 없습니다.

사용:
    python3 preflight_stimulus_check.py --project-root /path/to/EDA_AircraftFiles

통과 조건:
    1. aircraft_2_original 의 파일이 stimuli_original 로 resolve 되지 않아야 함
       (= 복원본을 쓰고 있어야 함)
    2. intact aircraft_1 과 aircraft_2 가 픽셀 수준 near-duplicate 가 아니어야 함
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THUMB = 64
DUP_CORR = 0.99          # 이 이상이면 near-duplicate 로 간주
DUP_FRAC_FAIL = 0.20     # 교차클래스 near-duplicate 가 이 비율 넘으면 실패
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMG_EXT and not p.name.startswith("."))


def build_manifest(project_root: Path):
    """final_vit_experiment.build_pair_manifest 와 동일한 페어링 규칙."""
    train_root = project_root / "original_image"
    test_root = project_root / "aircraft_dataset"
    required = [train_root / "aircraft_1_original", train_root / "aircraft_2_original",
                test_root / "aircraft_1", test_root / "aircraft_2"]
    missing = [str(p) for p in required if not p.is_dir()]
    if missing:
        sys.exit(f"[FAIL] 필수 폴더 없음: {missing}")

    records = []
    a1 = re.compile(r"^Aircraft1_(10|70|90)%_(\d+)$", re.IGNORECASE)
    for occ in image_files(test_root / "aircraft_1"):
        m = a1.match(occ.stem)
        if not m:
            continue
        lvl, idx = map(int, m.groups())
        intact = train_root / "aircraft_1_original" / f"{occ.stem}_original.jpg"
        if intact.is_file():
            records.append(("aircraft_1", 0, lvl, idx, intact, occ))

    a2 = re.compile(r"^Aircraft2_(10|70|90)__?(\d+)$", re.IGNORECASE)
    off = {10: 0, 70: 50, 90: 100}
    for occ in image_files(test_root / "aircraft_2"):
        m = a2.match(occ.stem)
        if not m:
            continue
        lvl, idx = map(int, m.groups())
        intact = (train_root / "aircraft_2_original"
                  / f"Aircraft2_Complete_{off[lvl] + idx}.jpg")
        if intact.is_file():
            records.append(("aircraft_2", 1, lvl, idx, intact, occ))
    return records


def thumb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L").resize((THUMB, THUMB), Image.BILINEAR)
    v = np.asarray(im, dtype=np.float64).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.project_root.expanduser().resolve()

    print(f"project-root: {root}\n")
    recs = build_manifest(root)
    print(f"페어링된 이미지: {len(recs)} (기대값 300)")
    if len(recs) != 300:
        print("[WARN] 300이 아닙니다. 파일명/개수 확인 필요.")

    fails = []

    # ---- 검사 1: 복원본을 쓰고 있는가 -----------------------------------
    print("\n[검사 1] aircraft_2_original 의 실제 resolve 대상")
    a2 = [r for r in recs if r[0] == "aircraft_2"]
    targets = {}
    for *_, intact, _occ in a2:
        d = str(intact.resolve().parent)
        targets[d] = targets.get(d, 0) + 1
    for d, n in sorted(targets.items(), key=lambda kv: -kv[1]):
        print(f"   {n:>4}개 -> {d}")
    bad = sum(n for d, n in targets.items()
              if "stimuli_original" in d or "stimuli" == Path(d).name)
    if bad:
        print(f"   [FAIL] {bad}개가 원본 OIID stimuli 로 resolve 됩니다.")
        print("          Aircraft2 는 복원본(stimuli_aircraft2_v2)을 가리켜야 합니다.")
        print("          symlink 가 아직 옛 경로를 향해 있을 가능성이 높습니다.")
        fails.append("aircraft_2 가 복원본이 아님")
    else:
        print("   [OK] 원본 stimuli 로 resolve 되지 않습니다.")

    # ---- 검사 2: 픽셀 수준 교차클래스 중복 ------------------------------
    print("\n[검사 2] intact 이미지 픽셀 수준 교차클래스 중복 검사")
    labels = np.array([r[1] for r in recs])
    X = np.stack([thumb(r[4]) for r in recs])
    S = X @ X.T
    np.fill_diagonal(S, -9.0)

    cross = S.copy()
    cross[labels[:, None] == labels[None, :]] = -9.0
    best_cross = cross.max(1)
    same = S.copy()
    same[labels[:, None] != labels[None, :]] = -9.0
    best_same = same.max(1)

    nn_cross_frac = float((S.argmax(1) >= 0).mean() and
                          (labels[S.argmax(1)] != labels).mean())
    dup_frac = float((best_cross > DUP_CORR).mean())

    print(f"   최근접 이웃이 반대 클래스인 비율 : {nn_cross_frac:.1%}  (기대 ~50%)")
    print(f"   교차클래스 상관 > {DUP_CORR} 비율    : {dup_frac:.1%}  (기대 ~0%)")
    print(f"   평균 최고 교차클래스 상관        : {best_cross.mean():.4f}")
    print(f"   평균 최고 동일클래스 상관        : {best_same.mean():.4f}")

    if dup_frac > DUP_FRAC_FAIL:
        print(f"   [FAIL] 두 클래스가 같은 사진입니다. 분류 과제가 성립하지 않습니다.")
        fails.append(f"교차클래스 중복 {dup_frac:.1%}")
    elif nn_cross_frac > 0.75:
        print("   [FAIL] 최근접 이웃이 거의 항상 반대 클래스입니다 (역예측 위험).")
        fails.append("교차클래스 최근접 편중")
    else:
        print("   [OK] 두 클래스가 픽셀 수준에서 구별됩니다.")

    # ---- 결과 ------------------------------------------------------------
    print("\n" + "=" * 62)
    if fails:
        print("PREFLIGHT 실패:")
        for f in fails:
            print(f"  - {f}")
        print("\n모델 실행을 중단하세요. 자극을 먼저 고쳐야 합니다.")
        return 1
    print("PREFLIGHT 통과. 모델 실행을 진행해도 됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
