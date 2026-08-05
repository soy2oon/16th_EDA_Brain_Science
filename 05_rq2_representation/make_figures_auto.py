#!/usr/bin/env python3
"""재실행 결과에서 fig1~fig6 을 자동 생성.

features/ 폴더가 있는 모델을 스스로 찾아서 그림에 포함합니다.
ViT features 가 생기면 인자 변경 없이 자동으로 3모델 그림이 됩니다.

사용:
    python3 make_figures_auto.py --results-root ./final_unified_analysis/raw_results
    python3 make_figures_auto.py --results-root ... --out ./figures

라벨(Aircraft1/2)을 전혀 쓰지 않는 지표만 사용하므로,
자극 라벨 문제와 독립적으로 성립합니다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

LEVELS = (10, 70, 90)
# 모델별 행동 레이어 우선순위 (앞에 있는 것부터 찾음)
BEHAVIOR_LAYER = {
    "resnet": ("avgpool", "layer4"),
    "cornet": ("IT_t2", "V4_t4"),
    "vit": ("final_cls", "block_12", "block12"),
}
PRETTY = {"resnet": "ResNet-50", "cornet": "CORnet-S", "vit": "ViT-B/16"}
CONDS = {"color": "컬러", "brightness_matched": "밝기·흑백 통제"}
PALETTE = ["#1f6fb4", "#7fb3dd", "#c8482a", "#e8a08c", "#2e8b57", "#8fcbaa"]
HUMAN = {10: 0.956154, 70: 0.792769, 90: 0.618769}


def setup_font() -> None:
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/System/Library/Fonts/AppleSDGothicNeo.ttc",
              "/Library/Fonts/AppleGothic.ttf"):
        if Path(p).exists():
            try:
                fm.fontManager.addfont(p)
                name = fm.FontProperties(fname=p).get_name()
                plt.rcParams["font.family"] = name
                print(f"[font] {name}")
                return
            except Exception:
                continue
    print("[font] 한글 폰트 없음 — 라벨이 깨질 수 있습니다.")


def unit(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1, n)


def discover(root: Path):
    """(model, cond, layer, intact, occluded, manifest) 목록 반환."""
    found = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = mdir.name
        man_path = mdir / "paired_source_manifest.csv"
        if not man_path.is_file():
            continue
        man = pd.read_csv(man_path)
        for cond in CONDS:
            fdir = mdir / cond / "features"
            if not fdir.is_dir():
                continue
            names = [p.stem[len("intact_"):] for p in fdir.glob("intact_*.npy")]
            layer = None
            for cand in BEHAVIOR_LAYER.get(model, ()):
                if cand in names:
                    layer = cand
                    break
            if layer is None:
                # 마지막 수단: intact/occluded 쌍이 있는 이름 중 첫 번째
                pairs = [n for n in names if (fdir / f"occluded_{n}.npy").is_file()]
                if not pairs:
                    print(f"[skip] {model}/{cond}: intact/occluded 쌍 없음")
                    continue
                layer = sorted(pairs)[-1]
            occ = fdir / f"occluded_{layer}.npy"
            if not occ.is_file():
                print(f"[skip] {model}/{cond}: occluded_{layer}.npy 없음")
                continue
            I = np.load(fdir / f"intact_{layer}.npy").astype(np.float64)
            O = np.load(occ).astype(np.float64)
            if I.shape != O.shape or len(I) != len(man):
                print(f"[skip] {model}/{cond}: 형태 불일치 "
                      f"intact{I.shape} occluded{O.shape} manifest{len(man)}")
                continue
            found.append((model, cond, layer, I, O, man))
            print(f"[found] {model}/{cond}  layer={layer}  shape={I.shape}")
    return found


def measures(I, O, lvl):
    In, On = unit(I), unit(O)
    S = On @ In.T
    N = S.shape[1]
    rank, cos, pair = {}, {}, {}
    for lv in LEVELS:
        m = np.where(lvl == lv)[0]
        if m.size == 0:
            continue
        s = S[m]
        own = s[np.arange(len(m)), m]
        rank[lv] = ((s > own[:, None]).sum(1) + 1) / N
        cos[lv] = own
        w = On[m] @ On[m].T
        iu = np.triu_indices(len(m), 1)
        pair[lv] = w[iu]
    return rank, cos, pair


def boot(x, fn, rng, B=2000):
    v = np.array([fn(rng.choice(x, len(x), replace=True)) for _ in range(B)])
    return fn(x), np.percentile(v, 2.5), np.percentile(v, 97.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("./figures_auto"))
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()
    setup_font()
    plt.rcParams.update({
        "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    found = discover(args.results_root.expanduser().resolve())
    if not found:
        sys.exit("features/ 를 가진 모델을 찾지 못했습니다. "
                 "ViT 패치를 적용하고 재실행했는지 확인하세요.")

    rng = np.random.default_rng(0)
    data, colors, stats, comp = {}, {}, [], []
    for k, (model, cond, layer, I, O, man) in enumerate(found):
        lvl = man["occlusion_percent"].to_numpy()
        rank, cos, pair = measures(I, O, lvl)
        key = (model, cond)
        data[key] = (rank, cos, pair)
        colors[key] = PALETTE[k % len(PALETTE)]
        lab = f"{PRETTY.get(model, model)} · {CONDS[cond]}"

        for nm, d in (("identification_rank", rank), ("cosine_to_own", cos)):
            u, p = mannwhitneyu(d[10], d[90])
            cliff = abs(2 * u / (len(d[10]) * len(d[90])) - 1)
            allv = np.concatenate([d[lv] for lv in LEVELS])
            alll = np.concatenate([[lv] * len(d[lv]) for lv in LEVELS])
            rho, pr = spearmanr(alll, allv)
            stats.append({"model": PRETTY.get(model, model), "condition": CONDS[cond],
                          "layer": layer, "measure": nm, "cliffs_delta_abs": cliff,
                          "mannwhitney_p": p, "spearman_rho": rho, "spearman_p": pr})

        # 종합 혼란 지수
        parts = []
        for d, flip in ((rank, False), (cos, True), (pair, False)):
            allv = np.concatenate([d[lv] for lv in LEVELS])
            lo, hi = allv.min(), allv.max()
            rng_ = hi - lo if hi > lo else 1.0
            vals = [(np.mean(d[lv]) - lo) / rng_ for lv in LEVELS]
            parts.append([1 - v for v in vals] if flip else vals)
        idx = np.array(parts).mean(0)
        for lv, v in zip(LEVELS, idx):
            comp.append({"model": PRETTY.get(model, model), "condition": CONDS[cond],
                         "occlusion_percent": lv, "confusion_index": v, "label": lab})

    x = np.arange(3)
    XT = ["10%", "70%", "90%"]

    def lineplot(which, title, ylab, note, fname, agg=np.mean, err="boot"):
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        for (model, cond), (rank, cos, pair) in data.items():
            d = {"rank": rank, "cos": cos, "pair": pair}[which]
            mu, lo, hi = [], [], []
            for lv in LEVELS:
                if err == "boot":
                    v, a, b = boot(d[lv], agg, rng)
                else:
                    v = agg(d[lv])
                    s = d[lv].std() / np.sqrt(len(d[lv]))
                    a, b = v - s, v + s
                mu.append(v); lo.append(a); hi.append(b)
            mu = np.array(mu)
            ax.errorbar(x, mu, yerr=[mu - np.array(lo), np.array(hi) - mu],
                        fmt="o-", lw=2.2, ms=7, capsize=4,
                        color=colors[(model, cond)],
                        label=f"{PRETTY.get(model, model)} · {CONDS[cond]}")
        ax.set_xticks(x, XT)
        ax.set_xlabel("가림 수준 (occlusion level)", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_title(title, fontsize=13, pad=14, weight="bold")
        ax.legend(frameon=False, fontsize=8.5)
        ax.text(0, -0.17, note, transform=ax.transAxes, fontsize=8.5,
                color="#555", va="top")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    lineplot("rank", "가림이 심해질수록 모델은 어떤 사진인지 못 알아본다",
             "정규화 식별 순위 (0 = 정확, 0.5 = 무작위)",
             "가려진 이미지를 원본 전체와 비교해 자기 원본의 순위를 매김. 클래스 라벨 미사용.\n"
             "오차막대 = bootstrap 95% CI (2000회).",
             "fig1_identification_rank.png")
    lineplot("cos", "가림이 심해질수록 자기 원본과의 표상 유사도가 무너진다",
             "자기 원본과의 cosine 유사도",
             "각 가려진 이미지와 그 원본 사이의 feature cosine. 오차막대 = bootstrap 95% CI.",
             "fig2_cosine_to_original.png")
    lineplot("pair", "가림이 심해질수록 모든 이미지가 서로 똑같아 보인다",
             "같은 수준 내 이미지 쌍 평균 cosine",
             "각 가림 수준 안 모든 이미지 쌍의 feature cosine 평균. 1에 가까울수록 구별 불가.\n"
             "오차막대 = SEM.",
             "fig3_representational_collapse.png", err="sem")

    ci = pd.DataFrame(comp)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for lab, g in ci.groupby("label", sort=False):
        g = g.sort_values("occlusion_percent")
        key = next(k for k in data if
                   f"{PRETTY.get(k[0], k[0])} · {CONDS[k[1]]}" == lab)
        ax.plot(x, g.confusion_index, "o-", lw=2.4, ms=8,
                color=colors[key], label=lab)
    ax.set_xticks(x, XT); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("가림 수준 (occlusion level)", fontsize=11)
    ax.set_ylabel("혼란 지수 (0 = 명확, 1 = 최대 혼란)", fontsize=11)
    ax.set_title("종합: 가림 수준이 높아질수록 모델 혼란이 커진다",
                 fontsize=13, pad=14, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    n_mono = sum(
        1 for _, g in ci.groupby("label")
        if g.sort_values("occlusion_percent").confusion_index.is_monotonic_increasing)
    ax.text(0, -0.17,
            "식별 순위 · 원본 유사도 · 표상 붕괴 세 지표를 각각 0–1로 정규화해 평균.\n"
            f"단조 증가 {n_mono}/{ci.label.nunique()} 조합.",
            transform=ax.transAxes, fontsize=8.5, color="#555", va="top")
    fig.tight_layout()
    fig.savefig(out / "fig4_confusion_index.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    keys = list(data)
    fig, axes = plt.subplots(1, max(2, len(keys)),
                             figsize=(6.2 * max(2, len(keys)) / 1.0, 5.0),
                             sharey=True, squeeze=False)
    for a, key in zip(axes[0], keys):
        _r, cos, _p = data[key]
        parts = a.violinplot([cos[lv] for lv in LEVELS], positions=x,
                             showmeans=True, widths=0.72)
        for pc in parts["bodies"]:
            pc.set_facecolor(colors[key]); pc.set_alpha(0.55)
        for k in ("cmeans", "cmins", "cmaxes", "cbars"):
            parts[k].set_color("#333")
        a.set_xticks(x, XT)
        a.set_title(f"{PRETTY.get(key[0], key[0])}\n{CONDS[key[1]]}",
                    fontsize=10.5, weight="bold")
        a.set_xlabel("가림 수준", fontsize=10)
    for a in axes[0][len(keys):]:
        a.axis("off")
    axes[0][0].set_ylabel("자기 원본과의 cosine 유사도", fontsize=11)
    fig.suptitle("분포 전체가 이동한다 — 소수 이상치가 아니다",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig5_distribution_shift.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    fig, a1 = plt.subplots(figsize=(7.8, 5.2))
    a1.plot(x, [HUMAN[lv] for lv in LEVELS], "o-", lw=2.6, ms=9, color="#222",
            label="인간 정확도 (n=65)")
    a1.set_ylabel("인간 강제선택 정확도", fontsize=11)
    a1.set_ylim(0.45, 1.02); a1.axhline(0.5, ls=":", c="#888", lw=1)
    a2 = a1.twinx(); a2.spines["top"].set_visible(False); a2.grid(False)
    for model in sorted({k[0] for k in data}):
        g = ci[(ci.model == PRETTY.get(model, model))]
        g = g.groupby("occlusion_percent").confusion_index.mean().reindex(LEVELS)
        a2.plot(x, g.to_numpy(), "s--", lw=2.0, ms=7,
                color=colors[next(k for k in data if k[0] == model)],
                label=f"{PRETTY.get(model, model)} 혼란 지수")
    a2.set_ylabel("모델 혼란 지수 (0–1)", fontsize=11)
    a1.set_xticks(x, XT)
    a1.set_xlabel("가림 수준 (occlusion level)", fontsize=11)
    a1.set_title("인간 성능 저하와 모델 혼란 증가", fontsize=13, pad=14, weight="bold")
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="center left")
    a1.text(0, -0.17,
            "두 축의 단위가 다르므로 기술적 병치이며 통계적 우열 비교가 아님.\n"
            "모델 값은 조건 평균.",
            transform=a1.transAxes, fontsize=8.5, color="#555", va="top")
    fig.tight_layout()
    fig.savefig(out / "fig6_human_reference.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(stats).to_csv(out / "figure_statistics.csv", index=False)
    ci.to_csv(out / "confusion_index_values.csv", index=False)
    print("\n" + pd.DataFrame(stats).round(4).to_string(index=False))
    print(f"\n모델·조건 {len(data)}개, 혼란 지수 단조 증가 "
          f"{n_mono}/{ci.label.nunique()}")
    print(f"출력: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
