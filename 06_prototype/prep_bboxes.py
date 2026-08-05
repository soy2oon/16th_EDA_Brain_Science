"""자극 스캔 → GT bbox 자동생성 → manifest.csv 작성 + QC 오버레이.

검은 배경 위 실루엣이므로 GT bbox는 non-black 픽셀의 tight box로 유도한다.
- intact/ (occ=0)  → split=train  (bbox = 온전 실루엣, 신뢰 높음)
- aircraft{N}/ (occ∈{10,70,90}) → split=test
  주의: 고가림 test의 bbox는 '보이는 파편'의 extent다(가림도와 교란) — best_iou 해석 시 유의.
"""
import re
import csv
import numpy as np
from PIL import Image, ImageDraw

import config as C


def foreground_bbox(img: Image.Image):
    """non-black 픽셀의 tight bbox. 전경이 없으면 None."""
    g = np.asarray(img.convert("L"))
    ys, xs = np.where(g > C.FG_THRESH)
    if xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    H, W = g.shape
    p = C.BBOX_PAD
    x0 = max(0, x0 - p); y0 = max(0, y0 - p)
    x1 = min(W - 1, x1 + p); y1 = min(H - 1, y1 + p)
    fg_frac = float((g > C.FG_THRESH).sum()) / g.size
    return [int(x0), int(y0), int(x1), int(y1)], fg_frac


def scan():
    rows = []
    reo = re.compile(C.FNAME_OCCLUDED)
    rei = re.compile(C.FNAME_INTACT)
    for p in sorted(C.STIMULI_DIR.rglob("*.jpg")):
        name = p.name
        is_intact = p.parent.name == "intact"
        m = rei.match(name) if is_intact else reo.match(name)
        if m is None:
            # intact 폴더에 가림 파일이 섞였거나 그 반대 — 반대 규약도 시도
            m = (reo if is_intact else rei).match(name)
            if m is None:
                print(f"[skip] 규약 불일치: {p}")
                continue
        n = int(m.group("n"))
        occ = 0 if is_intact or "occ" not in m.groupdict() or m.groupdict().get("occ") is None else int(m.group("occ"))
        if n not in C.AIRCRAFT_TO_CLASS:
            print(f"[skip] 미정의 항공기 {n}: {p}")
            continue
        res = foreground_bbox(Image.open(p))
        if res is None:
            print(f"[warn] 전경 없음(완전 검정): {p}")
            continue
        bbox, fg_frac = res
        rows.append({
            "path": str(p),
            "aircraft": n,
            "class_id": C.AIRCRAFT_TO_CLASS[n],
            "class_name": C.CLASS_TO_NAME[C.AIRCRAFT_TO_CLASS[n]],
            "occlusion": occ,
            "split": "train" if occ == 0 else "test",
            "x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3],
            "fg_frac": round(fg_frac, 5),
        })
    return rows


def write_manifest(rows):
    cols = ["path", "aircraft", "class_id", "class_name", "occlusion",
            "split", "x0", "y0", "x1", "y1", "fg_frac"]
    with open(C.MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] manifest 작성: {C.MANIFEST}  ({len(rows)} rows)")


def qc_overlays(rows, per_group=2):
    """(aircraft×occlusion) 그룹별 몇 장에 bbox 오버레이 저장 → 육안 검증."""
    seen = {}
    for r in rows:
        key = (r["aircraft"], r["occlusion"])
        seen.setdefault(key, 0)
        if seen[key] >= per_group:
            continue
        seen[key] += 1
        img = Image.open(r["path"]).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([r["x0"], r["y0"], r["x1"], r["y1"]], outline=(0, 255, 0), width=3)
        out = C.QC_DIR / f"A{r['aircraft']}_occ{r['occlusion']}_{seen[key]}.png"
        img.save(out)
    print(f"[ok] QC 오버레이 {sum(seen.values())}장 → {C.QC_DIR}")


def summary(rows):
    from collections import Counter
    cnt = Counter((r["aircraft"], r["occlusion"]) for r in rows)
    ff = {}
    for r in rows:
        ff.setdefault((r["aircraft"], r["occlusion"]), []).append(r["fg_frac"])
    print("\n조건별 (aircraft, occ): 장수 | 평균 전경비율")
    for k in sorted(cnt):
        arr = ff[k]
        print(f"  A{k[0]} occ{k[1]:>2}: {cnt[k]:>3}장 | fg={sum(arr)/len(arr):.3f}")


if __name__ == "__main__":
    rows = scan()
    if not rows:
        raise SystemExit("자극을 찾지 못함 — data/stimuli/ 확인")
    write_manifest(rows)
    qc_overlays(rows)
    summary(rows)
