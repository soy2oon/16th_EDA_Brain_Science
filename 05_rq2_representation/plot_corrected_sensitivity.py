# -*- coding: utf-8 -*-
"""보정 분류 민감도 그림 (발표 15쪽).

원본 make_unified_figs.py는 그림 4장을 만들지만, 그중 발표에 쓰인 것은
u_corrected_sensitivity.png 한 장뿐이다. 나머지 3장은 CORnet-S 조건 RDM(.npy)을
요구하므로 여기서는 제외했다.

입력 : analyze_corrected_entropy.py 가 만든 corrected_entropy_summary.csv
출력 : u_corrected_sensitivity.png
"""
import argparse, csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for cand in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False); plt.rcParams["font.family"] = cand; break
    except Exception: continue
plt.rcParams["axes.unicode_minus"] = False

ap = argparse.ArgumentParser()
ap.add_argument("--summary", required=True, help="corrected_entropy_summary.csv 경로")
ap.add_argument("--out-dir", required=True, help="그림을 저장할 디렉터리")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

rows = [r for r in csv.DictReader(open(args.summary)) if r["arm"] == "real_labels"]
D = {}  # (model,cond,occ) -> dict
for r in rows:
    D[(r["model"], r["condition"], int(r["occlusion_percent"]))] = {
        "acc": float(r["accuracy_mean"]), "auc": float(r["auc_mean"]),
        "ent": float(r["entropy_mean_nats"]), "ent_sd": float(r["entropy_sd_across_seeds"]),
    }
MODELS = [("vit", "ViT-B/16"), ("resnet", "ResNet-50"), ("cornet", "CORnet-S")]
OCC = [10, 70, 90]
LN2 = np.log(2)

fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0), dpi=200, sharex=True)
fig.suptitle("보정 분류 민감도 (corrected classification sensitivity)", fontsize=15, y=0.98)
C = {"bm_acc": "#1f77b4", "bm_auc": "#ff7f0e", "c_acc": "#2ca02c", "c_auc": "#d62728",
     "ent_bm": "#1f77b4", "ent_c": "#ff7f0e"}
for j, (mk, mname) in enumerate(MODELS):
    ax = axes[0][j]
    acc_bm = [D[(mk, "brightness_matched", o)]["acc"] for o in OCC]
    auc_bm = [D[(mk, "brightness_matched", o)]["auc"] for o in OCC]
    acc_c = [D[(mk, "color", o)]["acc"] for o in OCC]
    auc_c = [D[(mk, "color", o)]["auc"] for o in OCC]
    ax.plot(OCC, acc_bm, "-o", color=C["bm_acc"], label="밝기·흑백 통제 · 정확도")
    ax.plot(OCC, auc_bm, "--s", color=C["bm_auc"], label="밝기·흑백 통제 · AUC")
    ax.plot(OCC, acc_c, "-o", color=C["c_acc"], label="컬러 · 정확도")
    ax.plot(OCC, auc_c, "--s", color=C["c_auc"], label="컬러 · AUC")
    ax.set_title(mname, fontsize=12); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.25)
    if j == 0: ax.set_ylabel("정확도 · AUC\n(accuracy · AUC)", fontsize=11)
    ax2 = axes[1][j]
    ent_bm = [D[(mk, "brightness_matched", o)]["ent"] for o in OCC]
    ent_c = [D[(mk, "color", o)]["ent"] for o in OCC]
    ax2.plot(OCC, ent_bm, "-o", color=C["ent_bm"], label="밝기·흑백 통제")
    ax2.plot(OCC, ent_c, "-o", color=C["ent_c"], label="컬러")
    ax2.axhline(LN2, ls=":", color="gray", lw=1.2)
    ax2.set_ylim(0.1, 0.74); ax2.grid(alpha=0.25)
    ax2.set_xticks(OCC); ax2.set_xticklabels(["10%", "70%", "90%"])
    ax2.set_xlabel("가림 수준 (occlusion level)", fontsize=11)
    if j == 0: ax2.set_ylabel("이진 엔트로피\n(binary entropy, nats)", fontsize=11)
axes[0][2].legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
axes[1][2].legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.text(0.5, 0.005, "점선 = 이진 엔트로피 이론적 최대 ln(2)≈0.693 · real labels · 5 seeds",
         ha="center", fontsize=9, color="#555")
plt.tight_layout(rect=[0, 0.02, 0.99, 0.96])
out = os.path.join(args.out_dir, "u_corrected_sensitivity.png")
fig.savefig(out, bbox_inches="tight"); plt.close(fig)
print("saved", out)
