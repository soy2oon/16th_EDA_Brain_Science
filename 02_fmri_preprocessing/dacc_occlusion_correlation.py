"""
dACC beta와 occlusion_percent(가림 수준) 상관관계 스크립트
------------------------------------
목적: dacc_entropy_correlation.py에서 모델 entropy vs dACC 상관관계가 거의
      안 나온 것에 대한 진단. 모델 기반 세밀한 entropy 대신, 훨씬 단순한
      "가림 수준(10/70/90%)" 자체가 dACC와 상관이 있는지부터 확인한다.
      이것마저 안 나오면 entropy 측정치 자체보다 더 근본적인 문제(에: dACC
      정의, ROI, GLM)를 의심해야 하고, 이건 나오는데 entropy만 안 나오면
      "가림 수준 같은 큰 단위 차이는 잡지만, 이미지별 미세한 entropy 차이는
      dACC와 무관"이라는 뜻이 된다.

조건명(예: "Aircraft1_70pct_43")에서 occlusion_percent(70)를 바로 파싱하므로
entropy 데이터와의 매칭이 필요 없다 - 피험자당 297개 조건 전부 사용 가능
(entropy 매칭 때는 197개로 줄었던 것과 다름).

상관관계 단위: dacc_entropy_correlation.py와 동일하게 피험자별로 각자
상관계수를 내고, Fisher z-transform 후 one-sample t-test로 그룹 수준 검정.

전제:
- roi_extraction_dacc.py를 먼저 실행해서 roi_dacc_stimfile/sub-XX/betas_dacc.csv
  가 만들어져 있어야 함.

사용법: python dacc_occlusion_correlation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp

from dacc_entropy_correlation import CONDITION_NAME_RE, DACC_ROOT, fisher_group_stats

OUT_CSV = Path(__file__).resolve().parent / "dacc_occlusion_correlation_results.csv"


def condition_name_to_occlusion_percent(name: str) -> int | None:
    m = CONDITION_NAME_RE.match(name)
    if not m:
        return None
    return int(m.group(2))


def load_subject_dacc_means(sub_dir: Path) -> pd.Series:
    df = pd.read_csv(sub_dir / "betas_dacc.csv", index_col=0)
    return df.mean(axis=1)


def main():
    sub_dirs = sorted(d for d in DACC_ROOT.glob("sub-*") if (d / "betas_dacc.csv").exists())
    print(f"대상 피험자 {len(sub_dirs)}명: {[d.name for d in sub_dirs]}")

    rows = []
    for sub_dir in sub_dirs:
        sub_id = sub_dir.name
        dacc_means = load_subject_dacc_means(sub_dir)

        occlusion = dacc_means.index.map(condition_name_to_occlusion_percent)
        unparsed = pd.isna(occlusion)
        if unparsed.any():
            print(f"  [{sub_id}] 조건명 파싱 실패 {unparsed.sum()}개 (건너뜀)")
        x = np.asarray(occlusion[~unparsed], dtype=float)
        y = dacc_means.to_numpy()[~unparsed]

        r, p = pearsonr(x, y)
        rho, sp = spearmanr(x, y)

        rows.append({
            "subject": sub_id,
            "n_conditions": len(x),
            "pearson_r": r,
            "pearson_p": p,
            "spearman_r": rho,
            "spearman_p": sp,
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\n피험자별 상관계수 저장: {OUT_CSV} ({len(results_df)}행)")

    print("\n=== 그룹 통계 (Fisher z-transform 후 one-sample t-test, H0: mean_r=0) ===")
    for col, label in [("pearson_r", "Pearson"), ("spearman_r", "Spearman")]:
        r_values = results_df[col].to_numpy()
        stats = fisher_group_stats(r_values)
        print(
            f"  [{label}] n={stats['n']}, mean_r={stats['mean_r']:.4f}, "
            f"t({stats['df']})={stats['t']:.3f}, p={stats['p']:.4g}"
        )


if __name__ == "__main__":
    main()
