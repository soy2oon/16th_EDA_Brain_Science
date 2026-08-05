# -*- coding: utf-8 -*-
"""낱개 자극(297) 수준 인간 후두엽 ↔ CORnet·ViT·ResNet RSA (조건평균 X).
슬라이드10 방식과 동일: 297x297 RDM끼리 Spearman."""
import os, json, glob, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
import torch
from PIL import Image as PILImage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from torchvision.models import vit_b_16, ViT_B_16_Weights, resnet50, ResNet50_Weights

for c in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    try: font_manager.findfont(c, fallback_to_default=False); plt.rcParams["font.family"] = c; break
    except Exception: continue
plt.rcParams["axes.unicode_minus"] = False

BASE = "/Users/kimjyeongmin/rcnn-occlusion-experiment 2"
CE = f"{BASE}/CORNET-EXP"
ROI = "/Users/kimjyeongmin/Downloads/roi_occipital_stimfile"
OUT = f"{BASE}/_extra"
dev = "mps" if torch.backends.mps.is_available() else "cpu"

def hkey(l):
    m = re.match(r"Aircraft(\d)_(\d+)pct_(\d+)", l); return (m.group(1), int(m.group(2)), int(m.group(3)))
def mkey(p):
    m = re.match(r"Aircraft(\d)_(\d+)__(\d+)", os.path.basename(p)); return (m.group(1), int(m.group(2)), int(m.group(3)))

# ---- 공통 자극 순서 ----
fi = pd.read_csv(f"{CE}/features/feature_index.csv")
occ = fi[fi.split == "occluded_test"].reset_index(drop=True)
occ_pos = fi.index[fi.split == "occluded_test"].to_numpy()   # pooled 행 위치
model_keys = {mkey(p): i for i, p in enumerate(occ.path)}     # key -> occ 행

subs = sorted(glob.glob(f"{ROI}/sub-*"))
sub_labels = {}
common = set(model_keys)
for sd in subs:
    meta = json.load(open(f"{sd}/roi_meta.json"))
    ks = [hkey(l) for l in meta["conditions"]]
    sub_labels[os.path.basename(sd)] = ks
    common &= set(ks)
order = sorted(common)
M = len(order)
print(f"공통 낱개 자극: {M}개  (기종1={sum(k[0]=='1' for k in order)}, 기종2={sum(k[0]=='2' for k in order)})")
iu = np.triu_indices(M, 1)
utri = lambda X: X[iu]
def sp(a, b):
    r = spearmanr(a, b).correlation
    return r if np.isfinite(r) else np.nan

def rdm_from(feat):  # (M, D) -> upper-tri of 1-corr
    return utri(1 - np.corrcoef(feat.astype(np.float64)))

# ---- 인간 RDM (subject별, 공통순서로 정렬) ----
brains = {}
for sd in subs:
    sid = os.path.basename(sd)
    meta = json.load(open(f"{sd}/roi_meta.json")); b = np.load(f"{sd}/betas_occipital.npy")
    if b.shape[0] != len(meta["conditions"]): continue
    row = {hkey(l): i for i, l in enumerate(meta["conditions"])}
    idx = [row[k] for k in order]
    brains[sid] = rdm_from(b[idx])
sub_ids = list(brains); N = len(sub_ids)
stack = np.array([brains[s] for s in sub_ids]); gm = stack.mean(0)
nc_lo = np.mean([sp(stack[i], (stack.sum(0) - stack[i]) / (N - 1)) for i in range(N)])
nc_hi = np.mean([sp(stack[i], gm) for i in range(N)])
print(f"N={N} | per-image noise ceiling: {nc_lo:.3f}~{nc_hi:.3f}")

# ---- 모델 feature (공통순서) ----
model_idx = [model_keys[k] for k in order]   # occ 행 인덱스 (공통순서)

def reorder(feat_occ):  # feat_occ: (300, D) in occ order
    return feat_occ[model_idx]

# CORnet: pooled_*.npy (600,D) -> occluded -> 공통순서
cor_layers = ["V1_t1", "V2_t1", "V2_t2", "V4_t1", "V4_t2", "V4_t3", "V4_t4", "IT_t1", "IT_t2"]
cor_rdm = {}
for L in cor_layers:
    pooled = np.load(f"{CE}/features/pooled_{L}.npy")[occ_pos]   # (300,D) occ order
    cor_rdm[L] = rdm_from(reorder(pooled))

# ViT / ResNet: 낱개 feature 추출 (occ order) -> 공통순서
@torch.no_grad()
def extract(model, tf, setup):
    store = {}; hs = setup(model, store); per = {}
    paths = list(occ.path)
    for i in range(0, len(paths), 32):
        imgs = torch.stack([tf(PILImage.open(p).convert("RGB")) for p in paths[i:i+32]]).to(dev)
        store.clear(); model(imgs)
        for k, v in store.items(): per.setdefault(k, []).append(v.cpu().numpy())
    for h in hs: h.remove()
    return {k: np.concatenate(v, 0) for k, v in per.items()}
def rn_hooks(m, s):
    def mk(n, pool):
        def f(a, b, o): o = o.detach(); s[n] = o.mean((2, 3)) if pool else torch.flatten(o, 1)
        return f
    return [m.layer1.register_forward_hook(mk("L1", True)), m.layer2.register_forward_hook(mk("L2", True)),
            m.layer3.register_forward_hook(mk("L3", True)), m.layer4.register_forward_hook(mk("L4", True)),
            m.avgpool.register_forward_hook(mk("avgpool", False))]
def vt_hooks(m, s):
    def mk(n):
        def f(a, b, o): s[n] = o.detach()[:, 0]
        return f
    hs = [m.encoder.layers[k].register_forward_hook(mk(f"b{k+1}")) for k in [1, 3, 5, 7, 9, 11]]
    hs.append(m.encoder.ln.register_forward_hook(mk("cls"))); return hs

print("ResNet/ViT 추출 중...")
w = ResNet50_Weights.IMAGENET1K_V2; rf = extract(resnet50(weights=w).eval().to(dev), w.transforms(), rn_hooks)
rn_layers = ["L1", "L2", "L3", "L4", "avgpool"]
rn_rdm = {L: rdm_from(reorder(rf[L])) for L in rn_layers}
w = ViT_B_16_Weights.IMAGENET1K_V1; vf = extract(vit_b_16(weights=w).eval().to(dev), w.transforms(), vt_hooks)
vt_layers = ["b2", "b4", "b6", "b8", "b10", "b12", "cls"]
vt_rdm = {L: rdm_from(reorder(vf[L])) for L in vt_layers}

# ---- RSA: 모델별 '최고계층'(집단평균 기준) 낱개 RSA, 피험자별 ----
def fdr(ps):
    p = np.array(ps); m = len(p); o = np.argsort(p); out = np.empty(m); prev = 1.0
    for r, i in enumerate(reversed(o)):
        k = m - r; prev = min(prev, p[i] * m / k); out[i] = prev
    return out

def pstr(p): return "p<.001" if p < 1e-3 else (f"p={p:.3f}" if p < .05 else "n.s.")

models = [("CORnet-S", cor_rdm, cor_layers, "#2E5BA8"),
          ("ViT-B/16", vt_rdm, vt_layers, "#2CA02C"),
          ("ResNet-50", rn_rdm, rn_layers, "#D62728")]
names, cols, best_lay, per_subj, means, sems, pvals = [], [], [], [], [], [], []
for nm, rd, ly, col in models:
    # 계층별 피험자 RSA 행렬 (nan 안전)
    mat = {L: np.array([sp(brains[s], rd[L]) for s in sub_ids]) for L in ly}
    gmean = {L: np.nanmean(mat[L]) for L in ly}
    valid = [L for L in ly if np.isfinite(gmean[L])]
    Lstar = max(valid, key=lambda L: gmean[L])              # 집단평균 최고계층(유효)
    vals = mat[Lstar]; vals = vals[np.isfinite(vals)]       # nan 피험자 제거
    names.append(nm); cols.append(col); best_lay.append(Lstar); per_subj.append(vals)
    means.append(vals.mean()); sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))
    pvals.append(wilcoxon(vals, alternative="greater")[1])
    print(f"{nm}: 최고계층 {Lstar}, 낱개 RSA {vals.mean():.3f} ± {sems[-1]:.3f} (n={len(vals)}), p={pvals[-1]:.4f}")
pfdr = fdr(pvals)
means, sems = np.array(means), np.array(sems)

# ---- 막대그래프 ----
x = np.arange(3); jr = np.random.default_rng(42)
fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=200)
ax.axhspan(nc_lo, nc_hi, color="#cfd6e6", alpha=0.6, zorder=1, label="noise ceiling (인간끼리)")
ax.bar(x, means, width=0.55, color=cols, alpha=0.9, zorder=3)
for xi in range(3):
    jx = xi + (jr.random(N) - 0.5) * 0.30
    ax.scatter(jx, per_subj[xi], s=14, color="#0d2a55", alpha=0.45, zorder=4,
               edgecolors="white", linewidths=0.3, label="피험자별 ρ (N=25)" if xi == 0 else None)
ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="#12305e", elinewidth=1.8, capsize=6, zorder=5)
ax.axhline(0, color="black", lw=1.0)
for xi in range(3):
    top = max(means[xi] + sems[xi], per_subj[xi].max()) + 0.012
    ax.text(xi, top, pstr(pfdr[xi]), ha="center", va="bottom", fontsize=11,
            color="#B33A1E", fontweight="bold", zorder=6)
    ax.text(xi, -0.028, f"최고계층: {best_lay[xi]}", ha="center", va="top", fontsize=8.5, color="#555")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
ax.set_ylabel("스피어만 상관계수 ρ (Spearman ρ)", fontsize=12)
ax.set_title("인간 후두엽 ↔ 3개 모델 RSA", fontsize=16, pad=12)
ax.set_ylim(-0.05, nc_hi + 0.06)
ax.grid(axis="y", alpha=0.25); ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
ax.text(0.5, -0.145,
        f"낱개 {M}×{M} RDM끼리 Spearman(조건평균 아님) · 모델별 집단평균 최고계층 · "
        f"p = 피험자 25명 ρ의 Wilcoxon(ρ>0)+FDR · 오차막대=SEM · 회색띠=인간끼리 noise ceiling",
        transform=ax.transAxes, ha="center", fontsize=8.3, color="#3a3f48",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f3f5f8", ec="#d5d9e0"))
plt.tight_layout(); fig.savefig(f"{OUT}/perimage_rsa_3models.png", bbox_inches="tight")
print("saved perimage_rsa_3models.png (막대)")
