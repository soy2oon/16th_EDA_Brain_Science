"""
dACC beta와 AI 오분류(이진) 상관관계 스크립트
------------------------------------
목적: dacc_entropy_correlation.py(연속값 entropy) 대신, 훨씬 단순한 이진 신호
      "AI가 이 이미지를 틀렸는가(1)/맞았는가(0)"와 dACC를 상관 내본다.
      해석: "AI가 틀리는 이미지일수록 dACC가 더 반응하는가"를 본다.

참고 - 인간 정확도는 못 씀:
- 원래 "AI가 틀리는 조건 = 인간도 틀리는 조건 = dACC가 반응하는 조건"까지
  삼중으로 겹쳐보려 했으나, events.tsv의 key_fix 컬럼이 19,500 trial 전부에서
  stim_lable과 100% 일치함(확인 완료) - 즉 참가자의 실제 응답이 아니라 자극
  라벨을 그대로 복사해둔 메타데이터라서 인간 정확도를 계산할 방법이 없음.
  (key_time은 진짜 반응시간으로 보이니 필요하면 그건 별도로 쓸 수 있음.)
  그래서 이 스크립트는 AI 오분류 vs dACC만 본다.

AI 오분류율 계산:
- corrected_entropy_trials.csv의 "correct"(0/1) 컬럼을 이미지(source_id)별로
  평균 -> 여러 seed가 있으면 오분류 "비율"(예: 2번 중 1번 틀림 = 0.5), 대부분은
  seed 1개뿐이라 결국 0 또는 1인 경우가 많음. ai_error_rate = 1 - mean(correct).
- condition == "color", arm == "real_labels"만 사용 (dacc_entropy_correlation.py와 동일).

나머지 구조(이미지 매칭, 앙상블, 피험자별 상관 -> Fisher z 그룹 통계)는
dacc_entropy_correlation.py와 동일해서 그 안의 함수를 재사용한다.

사용법: python dacc_ai_error_correlation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from dacc_entropy_correlation import (
    ARM,
    DACC_ROOT,
    ENTROPY_CSV,
    MIN_MATCHED_CONDITIONS,
    MODELS,
    STIM_CONDITION,
    fisher_group_stats,
    load_subject_dacc_means,
    source_id_to_condition_name,
)

OUT_CSV = Path(r"D:\OIID") / "dacc_ai_error_correlation_results.csv"
MIN_SEED_COUNT = 1  # dacc_entropy_correlation.py와 동일한 신뢰도 필터 (기본 1 = 필터 없음)


def load_ai_error_by_model(min_seed_count: int = 1) -> dict[str, pd.Series]:
    """모델별로 {fMRI 조건명 -> ai_error_rate(=1-mean(correct))} Series를 반환."""
    df = pd.read_csv(ENTROPY_CSV)
    df = df[(df["condition"] == STIM_CONDITION) & (df["arm"] == ARM)]

    result = {}
    for model in MODELS:
        sub = df[df["model"] == model]
        if sub.empty:
            raise ValueError(f"entropy 데이터에 model={model!r} 행이 없습니다.")
        grouped = sub.groupby("source_id")["correct"]
        seed_counts = grouped.size()
        error_rate = 1.0 - grouped.mean()

        error_rate = error_rate[seed_counts >= min_seed_count]

        condition_names = error_rate.index.map(source_id_to_condition_name)
        unmapped = condition_names.isna().sum()
        if unmapped:
            print(f"  [{model}] source_id 파싱 실패 {unmapped}개 (건너뜀)")
        error_rate = error_rate[~condition_names.isna()]
        error_rate.index = condition_names[~condition_names.isna()]

        result[model] = error_rate

    # 3개 모델이 다 틀린 이미지일수록 "합의된 오분류"로 보고, 모델 간 평균 오분류율을
    # 앙상블 지표로 추가한다 (dacc_entropy_correlation.py의 ensemble과 동일한 논리).
    combined = pd.concat(result.values(), axis=1, keys=result.keys())
    result["ensemble"] = combined.dropna().mean(axis=1)

    return result


def main():
    print("AI 오분류율 데이터 로드 중...")
    ai_error_by_model = load_ai_error_by_model(min_seed_count=MIN_SEED_COUNT)
    for model, s in ai_error_by_model.items():
        print(f"  [{model}] 이미지 {len(s)}개, 오분류율 평균={s.mean():.3f}")

    sub_dirs = sorted(d for d in DACC_ROOT.glob("sub-*") if (d / "betas_dacc.csv").exists())
    print(f"\n대상 피험자 {len(sub_dirs)}명: {[d.name for d in sub_dirs]}")

    rows = []
    for sub_dir in sub_dirs:
        sub_id = sub_dir.name
        dacc_means = load_subject_dacc_means(sub_dir)

        for model in MODELS + ["ensemble"]:
            ai_error = ai_error_by_model[model]
            common = dacc_means.index.intersection(ai_error.index)

            if len(common) < MIN_MATCHED_CONDITIONS:
                print(f"  [{sub_id}/{model}] 매칭된 조건 {len(common)}개 (< {MIN_MATCHED_CONDITIONS}) -> 건너뜀")
                continue

            x = ai_error.loc[common].to_numpy()
            y = dacc_means.loc[common].to_numpy()

            if np.allclose(x, x[0]):
                # 매칭된 조건이 전부 정답/전부 오답이면 분산이 0이라 상관계수 정의 불가
                print(f"  [{sub_id}/{model}] 오분류율 분산 0 (전부 {x[0]}) -> 건너뜀")
                continue

            r, p = pearsonr(x, y)
            rho, sp = spearmanr(x, y)

            rows.append({
                "subject": sub_id,
                "model": model,
                "n_matched": len(common),
                "pearson_r": r,
                "pearson_p": p,
                "spearman_r": rho,
                "spearman_p": sp,
            })

    out_csv = OUT_CSV
    if MIN_SEED_COUNT > 1:
        out_csv = out_csv.with_stem(f"{out_csv.stem}_min_seed{MIN_SEED_COUNT}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_csv, index=False)
    print(f"\n피험자x모델별 상관계수 저장: {out_csv} ({len(results_df)}행)")

    print("\n=== 모델별 그룹 통계 (Fisher z-transform 후 one-sample t-test, H0: mean_r=0) ===")
    for model in MODELS + ["ensemble"]:
        r_values = results_df.loc[results_df["model"] == model, "pearson_r"].to_numpy()
        if len(r_values) == 0:
            print(f"  [{model}] 데이터 없음")
            continue
        stats = fisher_group_stats(r_values)
        print(
            f"  [{model}] n={stats['n']}, mean_r={stats['mean_r']:.4f}, "
            f"t({stats['df']})={stats['t']:.3f}, p={stats['p']:.4g}"
        )
    print(
        f"\n주의: {len(MODELS)}개 개별 모델 + ensemble까지 총 {len(MODELS)+1}번 검정했으므로 "
        f"다중비교 보정(예: Bonferroni, alpha=.05/{len(MODELS)+1}={0.05/(len(MODELS)+1):.4f})을 고려하세요."
    )


if __name__ == "__main__":
    main()
