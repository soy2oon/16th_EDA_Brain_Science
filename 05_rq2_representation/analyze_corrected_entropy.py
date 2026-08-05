#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODELS = {"vit": "final_cls", "resnet": "avgpool", "cornet": "IT_t2"}
CONDITIONS = ("color", "brightness_matched")
LEVELS = (10, 70, 90)
SEEDS = (42, 142, 242, 342, 442)
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def strat(y, level, idx):
    return np.array([f"{y[i]}_{level[i]}" for i in idx])

def fit_basis(I, O, idx):
    basis = np.vstack([I[idx], O[idx]])
    sc = StandardScaler().fit(basis)
    z = sc.transform(basis)
    pc = PCA(min(30, len(idx)-1, z.shape[1]), svd_solver="full").fit(z)
    return sc, pc

def transform(sc, pc, x):
    return pc.transform(sc.transform(x))

def choose_c(Iz, y, fit, val, seed):
    best_c, best_score = None, -np.inf
    for c in C_GRID:
        clf = LogisticRegression(C=c, penalty="l2", solver="liblinear",
                                 max_iter=3000, random_state=seed)
        clf.fit(Iz[fit], y[fit])
        score = balanced_accuracy_score(y[val], clf.predict(Iz[val]))
        if score > best_score:
            best_c, best_score = c, float(score)
    return best_c, best_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.results_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    trial_rows, metric_rows = [], []

    for model, layer in MODELS.items():
        manifest = pd.read_csv(root/model/"paired_source_manifest.csv")
        y_true = manifest.label.to_numpy()
        level = manifest.occlusion_percent.to_numpy()
        all_idx = np.arange(len(y_true))
        for condition in CONDITIONS:
            fdir = root/model/condition/"features"
            I = np.load(fdir/f"intact_{layer}.npy")
            O = np.load(fdir/f"occluded_{layer}.npy")
            for seed in SEEDS:
                train, test = train_test_split(
                    all_idx, test_size=.2, random_state=seed,
                    stratify=strat(y_true, level, all_idx))
                fit, val = train_test_split(
                    train, test_size=.2, random_state=seed,
                    stratify=strat(y_true, level, train))

                # C selection basis sees only fit sources, but both intact and
                # occluded domains. Labels are used only by the intact classifier.
                sc0, pc0 = fit_basis(I, O, fit)
                Iz0 = transform(sc0, pc0, I)
                # Final basis sees all non-test sources from both domains.
                sc, pc = fit_basis(I, O, train)
                Iz, Oz = transform(sc, pc, I), transform(sc, pc, O)

                for arm in ("real_labels", "permuted_labels"):
                    if arm == "real_labels":
                        y = y_true
                    else:
                        y = np.random.default_rng(seed).permutation(y_true)
                    best_c, val_bal = choose_c(Iz0, y, fit, val, seed)
                    clf = LogisticRegression(C=best_c, penalty="l2",
                        solver="liblinear", max_iter=3000, random_state=seed)
                    clf.fit(Iz[train], y[train])

                    offsets = {
                        lv: float(np.median(clf.decision_function(
                            Oz[train[level[train] == lv]])))
                        for lv in LEVELS
                    }
                    logits = clf.decision_function(Oz[test]).astype(float)
                    for lv in LEVELS:
                        logits[level[test] == lv] -= offsets[lv]
                    prob = 1/(1+np.exp(-np.clip(logits, -50, 50)))
                    pred = (logits >= 0).astype(int)
                    ent = entropy(prob)
                    correct = (pred == y[test]).astype(int)

                    for pos, idx in enumerate(test):
                        trial_rows.append({
                            "model": model, "condition": condition, "arm": arm,
                            "seed": seed, "source_id": manifest.iloc[idx].source_id,
                            "true_label": int(y[idx]), "occlusion_percent": int(level[idx]),
                            "logit_level_median_centered": logits[pos],
                            "probability_aircraft2": prob[pos],
                            "binary_entropy_nats": ent[pos],
                            "prediction": pred[pos], "correct": correct[pos],
                            "selected_c": best_c,
                            "intact_validation_balanced_accuracy": val_bal,
                            "level_median_offset": offsets[int(level[idx])],
                        })
                    for lv in LEVELS:
                        mask = level[test] == lv
                        yt, yp, pp = y[test][mask], pred[mask], prob[mask]
                        metric_rows.append({
                            "model": model, "condition": condition, "arm": arm,
                            "seed": seed, "occlusion_percent": lv,
                            "accuracy": accuracy_score(yt, yp),
                            "balanced_accuracy": balanced_accuracy_score(yt, yp),
                            "auc": roc_auc_score(yt, pp),
                            "entropy_mean_nats": ent[mask].mean(),
                            "entropy_sd_trials": ent[mask].std(ddof=1),
                            "aircraft2_selection_rate": np.mean(yp == 1),
                            "selected_c": best_c,
                            "intact_validation_balanced_accuracy": val_bal,
                        })
            print(f"done {model}/{condition}", flush=True)

    trials, metrics = pd.DataFrame(trial_rows), pd.DataFrame(metric_rows)
    trials.to_csv(out/"corrected_entropy_trials.csv", index=False)
    metrics.to_csv(out/"corrected_entropy_by_seed.csv", index=False)
    summary = (metrics.groupby(
        ["model","condition","arm","occlusion_percent"], as_index=False)
        .agg(n_seeds=("seed","nunique"),
             accuracy_mean=("accuracy","mean"), accuracy_sd=("accuracy","std"),
             balanced_accuracy_mean=("balanced_accuracy","mean"),
             auc_mean=("auc","mean"), auc_sd=("auc","std"),
             entropy_mean_nats=("entropy_mean_nats","mean"),
             entropy_sd_across_seeds=("entropy_mean_nats","std"),
             aircraft2_selection_rate_mean=("aircraft2_selection_rate","mean"),
             intact_validation_balanced_accuracy_mean=(
                 "intact_validation_balanced_accuracy","mean")))
    summary.to_csv(out/"corrected_entropy_summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
