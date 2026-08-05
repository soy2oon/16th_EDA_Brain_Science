"""
OIID stim_file 단위 First-level GLM 스크립트
------------------------------------
목적: 6조건(Aircraft1/2 x 10/75/90%)으로 뭉뚱그리는 대신, run-01+run-02를
      합쳐서 나오는 고유 stim_file 297개를 각각 별도 조건으로 취급해
      조건별 beta(effect size) map을 추출한다.

전제 (glm_6condition.py와 동일한 데이터/버그 검증을 그대로 재사용):
- 전처리 완료된 BOLD: derivatives/pre-processed_data/space-MNI/sub-XXX/.../*_desc-preproc_bold.nii.gz
- events.tsv: 원본 sub-XXX/ses-01/func/*_events.tsv
- TR = 2.0s
- SUBJECT_NUMS도 glm_6condition.py에서 그대로 가져옴 (BOLD 중복/손상,
  events.tsv 오염으로 이미 제외된 피험자와 동일하게 제외됨)

조건 정의:
- run-01(고정 149개) + run-02(고정 148개)는 정상 피험자라면 서로 겹치지 않지만,
  각 run 내부에 stim_file이 2번씩 나오는 고정 반복 trial이 있음
  (run-01: Aircraft1_10%_1.jpg, run-02: Aircraft1_90%_1.jpg, Aircraft1_70%_1.jpg).
  이런 반복은 "첫 번째로 나온 trial"만 조건으로 쓰고 두 번째 이후는 제외
  (rest와 동일하게 trial_type=None -> 암묵적 baseline으로 흡수).
- 결과적으로 피험자당 조건 수 = 297개 (294개는 trial 1개, 3개는 원래 2개 있었지만
  1개만 사용).

사용법: python glm_stimfile_condition.py
"""

# glm_6condition.py를 numpy/nilearn보다 먼저 import해야 그 안에서 설정하는
# BLAS 스레드 제한(OMP/OPENBLAS/MKL_NUM_THREADS)과 TEMP 경로 재지정이
# 적용된 채로 numpy가 로드됨.
import glm_6condition as G

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # tqdm 없으면 그냥 진행바 없이 진행

OUT_ROOT = G.BIDS_ROOT / "glm_stimfile_betas"

# glm_6condition.py처럼 여기서 실행 대상을 직접 조절. 기본값은 G.SUBJECT_NUMS 전체지만,
# 시간 실측 등을 위해 예: SUBJECT_NUMS = [2] 처럼 줄여서 돌려볼 수 있음.
SUBJECT_NUMS = [19]


def sanitize_stim_name(stim_file: str) -> str:
    """'Aircraft1_10%_2.jpg' -> 'Aircraft1_10pct_2' (파일명/컬럼명으로 안전하게)"""
    return stim_file.replace(".jpg", "").replace("%", "pct")


def build_deduped_events(sub_id: str) -> tuple[list[pd.DataFrame], list[str]]:
    """
    run-01, run-02 events.tsv를 순서대로 훑으면서 stim_file마다 처음 나온
    trial만 조건으로 남기고, 이미 나왔던 stim_file(반복 trial)은 rest와
    동일하게 trial_type=None으로 만들어 제외한다.

    반환: (run별 events DataFrame 리스트, 이 피험자에서 실제로 쓰인 조건명 리스트)
    """
    seen_stim = set()
    events_by_run = []

    for run in range(1, G.N_RUNS + 1):
        df = pd.read_csv(G.events_path(sub_id, run), sep="\t")

        trial_types = []
        for stim in df["stim_file"]:
            if pd.isna(stim) or stim == "rest.jpg":
                trial_types.append(None)
                continue
            if stim in seen_stim:
                trial_types.append(None)  # 2번째 이후 반복 -> 제외
            else:
                seen_stim.add(stim)
                trial_types.append(sanitize_stim_name(stim))

        df["trial_type"] = trial_types
        events = df.dropna(subset=["trial_type"])[["onset", "duration", "trial_type"]].reset_index(drop=True)
        events_by_run.append(events)

    condition_names = sorted(seen_stim)
    condition_names = [sanitize_stim_name(s) for s in condition_names]
    return events_by_run, condition_names


def run_glm_for_subject(sub_num: int) -> tuple[dict, list[str]]:
    sub_id = G.folder_id(sub_num)
    f_id = G.file_id(sub_num)

    # ---- 1. 파일 존재 확인 ----
    bold_files = {}
    for run in range(1, G.N_RUNS + 1):
        bold_file = G.find_bold_file(sub_id, f_id, run)
        if not bold_file.exists():
            raise FileNotFoundError(f"bold 파일 없음: {bold_file}")
        if not G.events_path(sub_id, run).exists():
            raise FileNotFoundError(f"events.tsv 없음: run-{run:02d}")
        bold_files[run] = bold_file

    # ---- 2. 뇌 마스크 ----
    print("  뇌 마스크 계산 중 (MNI152 템플릿 기반, nearest-neighbor resample)...")
    t_mask = time.time()
    mask_img = G.get_brain_mask(bold_files[1], sub_id)
    print(f"  마스크 준비 완료 ({time.time()-t_mask:.1f}초)")

    # ---- 3. stim_file 단위로 중복 제거한 events + confounds ----
    events_list, condition_names = build_deduped_events(sub_id)
    print(f"  이 피험자 조건 수: {len(condition_names)}개 (중복 반복 trial은 첫 trial만 사용)")

    bold_imgs, confounds_list = [], []
    for run in range(1, G.N_RUNS + 1):
        t1 = time.time()
        print(f"  [run-{run}] data-driven confounds 계산 중 (뇌 마스크 내 복셀만 사용)...")
        bold_file = bold_files[run]
        bold_imgs.append(str(bold_file))
        confounds_list.append(
            G.get_data_driven_confounds(str(bold_file), G.N_HIGH_VARIANCE_CONFOUNDS, mask_img=mask_img)
        )
        print(f"  [run-{run}] confounds 준비 완료 ({time.time()-t1:.1f}초)")

    print(f"  GLM fitting 시작 (가장 오래 걸리는 단계, n_jobs={G.N_JOBS}로 AR1 bin 병렬화)...")
    t_fit = time.time()
    glm = FirstLevelModel(
        t_r=G.TR,
        smoothing_fwhm=G.SMOOTHING_FWHM,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=0.01,
        standardize=False,
        mask_img=mask_img,
        noise_model="ar1",
        minimize_memory=True,
        # 6조건용 캐시(NILEARN_CACHE_DIR)는 design matrix가 완전히 달라서(6개 vs 297개
        # 조건) 어차피 캐시 히트가 안 남. 재사용도 안 되고 별도로 새로 쌓을 필요도 없어서
        # 캐싱 자체를 끔 (뇌 마스크 캐시는 G.get_brain_mask를 통해 그대로 재사용됨).
        memory=None,
        n_jobs=G.N_JOBS,
        verbose=1,
    )
    glm = glm.fit(bold_imgs, events=events_list, confounds=confounds_list)
    print(f"  GLM fitting 완료 ({time.time()-t_fit:.1f}초)")

    beta_maps = {}
    t_contrast = time.time()
    for cond in tqdm(condition_names, desc="  조건별 beta map 계산"):
        contrast_vector = G.build_contrast_vector(glm, cond)
        beta_maps[cond] = glm.compute_contrast(contrast_vector, output_type="effect_size")
    print(f"  contrast 계산 완료 ({time.time()-t_contrast:.1f}초)")

    return beta_maps, condition_names


def main():
    OUT_ROOT.mkdir(exist_ok=True)

    for sub_num in SUBJECT_NUMS:
        sub_id = G.folder_id(sub_num)
        print(f"\n=== {sub_id} stim_file 단위 GLM 시작 ===")
        t_start = time.time()

        try:
            beta_maps, condition_names = run_glm_for_subject(sub_num)
        except (FileNotFoundError, EOFError, OSError, ValueError) as e:
            print(f"  건너뜀 ({type(e).__name__}): {e}")
            continue

        sub_out = OUT_ROOT / sub_id
        sub_out.mkdir(parents=True, exist_ok=True)

        for cond, beta_map in beta_maps.items():
            nib.save(beta_map, sub_out / f"{cond}_beta.nii.gz")

        with open(sub_out / "conditions.json", "w", encoding="utf-8") as f:
            json.dump({"subject": sub_id, "n_conditions": len(condition_names), "conditions": condition_names},
                       f, ensure_ascii=False, indent=2)

        print(f"  완료: {sub_out} (조건 {len(condition_names)}개, 총 소요시간: {time.time()-t_start:.1f}초)")


if __name__ == "__main__":
    main()
