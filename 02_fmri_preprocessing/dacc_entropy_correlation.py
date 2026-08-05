"""
dACC beta와 모델(entropy) 상관관계 스크립트
------------------------------------
목적: roi_extraction_dacc.py가 뽑은 피험자별 dACC beta(조건 x 복셀)와
      aircraft_occlusion_rerun_20260725의 모델별(cornet/resnet/vit) 이미지별
      binary entropy(불확실성)를 이미지 단위로 매칭해서, "모델이 헷갈려하는
      이미지일수록 dACC 활성이 높아지는가"를 상관관계로 본다.

이미지 매칭 방법 (중요):
- fMRI 조건명(glm_stimfile_condition.py의 sanitize_stim_name 결과)은 원본
  자극 파일명 그대로 파생됨: 예) "Aircraft1_70%_43.jpg" -> "Aircraft1_70pct_43".
  (참고: 파일명의 "70%"는 실제 occlusion 메타데이터 0.75와 다르지만, 모델
  rerun 쪽도 동일하게 파일명 기준 "70"을 쓰므로 매칭엔 문제 없음.)
- 모델 rerun의 source_id는 final_vit_experiment.py에서
  `f"aircraft_{1|2}_{level}_{index:03d}"`로 만들어지는데, 이 level/index가
  원본 파일명에서 정규식으로 뽑아낸 것과 동일한 숫자라 fMRI 조건명과 1:1로
  대응된다: "Aircraft1_70pct_43" <-> "aircraft_1_70_043".
- 이미지 pool은 모델 쪽이 300개(아ircraft 2종 x 3단계 x 50장), fMRI 쪽은
  피험자당 297개(반복 trial 3개를 첫 trial만 조건으로 써서 3개 적음)라
  매칭 후 보통 297개가 남고 나머지는 그냥 버려짐 - 정상.

entropy 데이터에서 쓰는 하위집합:
- condition == "color"  (fMRI 자극이 컬러 이미지였으므로 brightness_matched
  대조군은 제외)
- arm == "real_labels"  (label-permutation 대조군 제외)
- 여러 seed에 걸친 binary_entropy_nats를 평균해서 이미지당 하나의 값으로 사용.

상관관계 단위: 피험자마다 각자 dACC(이미지별 복셀 평균) vs entropy로 상관계수를
하나씩 내고(모델별로 따로), 24명의 상관계수 분포에 대해 Fisher z-transform 후
one-sample t-test로 그룹 수준에서 0과 다른지 검정한다 (표준 RSA류 그룹 통계 방식).

전제:
- roi_extraction_dacc.py를 먼저 실행해서 roi_dacc_stimfile/sub-XX/betas_dacc.csv
  가 만들어져 있어야 함.
- aircraft_occlusion_rerun_20260725 폴더의 corrected_entropy_trials.csv가 있어야 함.

사용법: python dacc_entropy_correlation.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp

# ==== 여기만 수정하세요 ====
ROOT = Path(r"D:\OIID")
DACC_ROOT = ROOT / "roi_dacc_stimfile"
ENTROPY_CSV = (
    ROOT
    / "aircraft_occlusion_rerun_20260725"
    / "aircraft_occlusion_rerun_20260725"
    / "results"
    / "rerun_analysis"
    / "corrected_entropy_trials.csv"
)
OUT_CSV = ROOT / "dacc_entropy_correlation_results.csv"  # main()에서 MIN_SEED_COUNT>1이면 파일명에 접미사 추가

MODELS = ["cornet", "resnet", "vit"]
STIM_CONDITION = "color"  # fMRI 자극과 매칭되는 쪽 (brightness_matched 대조군 제외)
ARM = "real_labels"  # label-permutation 대조군 제외
MIN_MATCHED_CONDITIONS = 10  # 이보다 매칭된 조건 수가 적으면 그 피험자x모델은 건너뜀
MIN_SEED_COUNT = 1  # 이 값 이상 seed에서 평가된 이미지만 사용 (신뢰도 필터, load_entropy_by_model 설명 참고)
# ==========================

CONDITION_NAME_RE = re.compile(r"^Aircraft([12])_(10|70|90)pct_(\d+)$")
SOURCE_ID_RE = re.compile(r"^aircraft_([12])_(10|70|90)_(\d+)$")


def condition_name_to_source_id(name: str) -> str | None:
    m = CONDITION_NAME_RE.match(name)
    if not m:
        return None
    aircraft_num, level, index = m.groups()
    return f"aircraft_{aircraft_num}_{level}_{int(index):03d}"


def source_id_to_condition_name(source_id: str) -> str | None:
    m = SOURCE_ID_RE.match(source_id)
    if not m:
        return None
    aircraft_num, level, index = m.groups()
    return f"Aircraft{aircraft_num}_{level}pct_{int(index)}"


def load_entropy_by_model(min_seed_count: int = 1) -> dict[str, pd.Series]:
    """
    모델별로 {fMRI 조건명 -> seed 평균 binary_entropy_nats} Series를 반환.

    min_seed_count: 이 값 이상의 seed에서 실제로 평가된 이미지만 남긴다(기본 1 =
    필터 없음). 모델 rerun이 seed마다 300개 중 60개만 랜덤 평가하는 방식이라
    대부분 이미지가 seed 1개짜리 값이라 노이즈가 큼 - 이 값을 높이면(예: 3)
    여러 seed에 걸쳐 안정적으로 추정된 이미지만 남겨 신뢰도를 높일 수 있는
    대신 표본 수(조건 수)가 크게 줄어든다.
    """
    df = pd.read_csv(ENTROPY_CSV)
    df = df[(df["condition"] == STIM_CONDITION) & (df["arm"] == ARM)]

    result = {}
    for model in MODELS:
        sub = df[df["model"] == model]
        if sub.empty:
            raise ValueError(f"entropy 데이터에 model={model!r} 행이 없습니다.")
        grouped = sub.groupby("source_id")["binary_entropy_nats"]
        seed_counts = grouped.size()
        by_source = grouped.mean()

        by_source = by_source[seed_counts >= min_seed_count]

        condition_names = by_source.index.map(source_id_to_condition_name)
        unmapped = condition_names.isna().sum()
        if unmapped:
            print(f"  [{model}] source_id 파싱 실패 {unmapped}개 (건너뜀)")
        by_source = by_source[~condition_names.isna()]
        by_source.index = condition_names[~condition_names.isna()]

        result[model] = by_source

    # 3개 모델 각각의 entropy 추정치는 독립적인 노이즈를 갖고 있을 수 있으므로,
    # 모델 간 평균("앙상블 혼란도")을 내면 노이즈가 상쇄되어 이미지 고유의
    # 난이도 신호가 더 뚜렷해질 수 있다. 세 모델이 정확히 같은 source_id
    # 집합을 공유하므로(같은 seed로 뽑은 같은 테스트 서브셋) 단순 평균으로 충분.
    combined = pd.concat(result.values(), axis=1, keys=result.keys())
    result["ensemble"] = combined.dropna().mean(axis=1)

    return result


def load_subject_dacc_means(sub_dir: Path) -> pd.Series:
    """betas_dacc.csv(조건 x 복셀)에서 조건별 복셀 평균 -> 이미지당 스칼라 1개."""
    df = pd.read_csv(sub_dir / "betas_dacc.csv", index_col=0)
    return df.mean(axis=1)


def fisher_group_stats(r_values: np.ndarray) -> dict:
    z = np.arctanh(np.clip(r_values, -0.999999, 0.999999))
    t_stat, p_val = ttest_1samp(z, 0.0)
    return {
        "n": len(r_values),
        "mean_r": float(np.tanh(z.mean())),
        "t": float(t_stat),
        "df": len(r_values) - 1,
        "p": float(p_val),
    }


def main():
    print("entropy 데이터 로드 중...")
    entropy_by_model = load_entropy_by_model(min_seed_count=MIN_SEED_COUNT)
    for model, s in entropy_by_model.items():
        print(f"  [{model}] 이미지 {len(s)}개, entropy 평균={s.mean():.3f}")

    sub_dirs = sorted(d for d in DACC_ROOT.glob("sub-*") if (d / "betas_dacc.csv").exists())
    print(f"\n대상 피험자 {len(sub_dirs)}명: {[d.name for d in sub_dirs]}")

    rows = []
    for sub_dir in sub_dirs:
        sub_id = sub_dir.name
        dacc_means = load_subject_dacc_means(sub_dir)

        for model in MODELS + ["ensemble"]:
            entropy = entropy_by_model[model]
            common = dacc_means.index.intersection(entropy.index)

            if len(common) < MIN_MATCHED_CONDITIONS:
                print(f"  [{sub_id}/{model}] 매칭된 조건 {len(common)}개 (< {MIN_MATCHED_CONDITIONS}) -> 건너뜀")
                continue

            x = entropy.loc[common].to_numpy()
            y = dacc_means.loc[common].to_numpy()

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
        f"다중비교 보정(예: Bonferroni, alpha=.05/{len(MODELS)+1}={0.05/(len(MODELS)+1):.4f})을 고려하세요. "
        "ensemble은 개별 모델과 독립적인 가설이 아니라 사전에 계획한 보조 분석이므로, "
        "주 검정은 ensemble 하나로 보고 개별 모델 3개는 탐색적으로 보는 것도 방법입니다."
    )


if __name__ == "__main__":
    main()
