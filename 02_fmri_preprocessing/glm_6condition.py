"""
OIID 6조건 First-level GLM 스크립트
------------------------------------
목적: 각 피험자마다 6개 조건(Aircraft1/2 x 10/75/90%)에 대한
      beta(effect size) map을 추출 -> 이후 occipital ROI RSA에 사용

전제:
- 전처리 완료된 BOLD: derivatives/pre-processed_data/space-MNI/sub-XXX/.../*_desc-preproc_bold.nii.gz
- events.tsv: 원본 sub-XXX/ses-01/func/*_events.tsv (onset, duration, stim_file, stim_lable, levelOfOcclusion, key_time, key_fix)
- TR = 2.0s (원 논문 명시)

사용법: python glm_6condition.py
경로만 본인 환경에 맞게 수정하세요.
"""

import os
# n_jobs>1로 AR(1) GLM fitting을 병렬화할 것이므로, numpy/scipy가 쓰는 BLAS
# 스레드 풀을 프로세스당 1개로 제한해야 함. 안 그러면 워커 프로세스가 각자
# 내부적으로도 멀티스레드 BLAS를 돌려서 (n_jobs x BLAS 스레드 수)만큼 코어를
# 과다구독(oversubscribe)하게 되어 오히려 더 느려질 수 있음.
# numpy/nilearn을 import하기 "전에" 설정해야 적용됨.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# C: 드라이브 여유공간이 거의 없어서(확인됨, 1GB대), joblib(loky)가 병렬 워커
# 사이에 큰 배열(run_glm의 AR1 bin 결과 등)을 주고받을 때 쓰는 memmap 임시파일이
# 기본값인 C:의 시스템 TEMP로 가다가 디스크가 가득 차서 GLM fitting 도중
# "No space left on device"로 죽는 문제가 있었음. 여유공간이 넉넉한 D: 드라이브로
# TEMP/TMP를 돌려서 방지. numpy/nilearn/joblib을 import하기 "전에" 설정해야 함.
os.makedirs(r"D:\OIID\glm_6condition_work\tmp", exist_ok=True)
os.environ["TEMP"] = r"D:\OIID\glm_6condition_work\tmp"
os.environ["TMP"] = r"D:\OIID\glm_6condition_work\tmp"
os.environ["JOBLIB_TEMP_FOLDER"] = r"D:\OIID\glm_6condition_work\tmp"

from pathlib import Path
import time
import pandas as pd
import numpy as np
from nilearn import datasets, image
from nilearn.glm.first_level import FirstLevelModel
import nibabel as nib
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # tqdm 없으면 그냥 진행바 없이 진행

# ==== 여기만 수정하세요 ====
BIDS_ROOT = Path(r"D:\OIID")
PREPROC_ROOT = BIDS_ROOT / "derivatives" / "pre-processed_data" / "space-MNI"
TR = 2.0
# desc-preproc BOLD를 직접 확인한 결과 스무딩이 적용 안 된 상태로 확인됨
# (이전에는 SPM12 전처리 단계에서 이미 스무딩됐을 것으로 추정했었으나 틀렸음).
# 따라서 이중 스무딩 걱정 없이 6mm를 그대로 적용.
SMOOTHING_FWHM = 6.0
N_RUNS = 2  # run-1, run-2

# confounds 파일이 이 데이터셋엔 없는 것으로 확인됨 (OIID Scientific Data 논문 Data Records 참조)
# -> nilearn의 data-driven high-variance confounds로 대체
N_HIGH_VARIANCE_CONFOUNDS = 5

# AR(1) noise model 피팅(nilearn 내부적으로 복셀을 최대 100개 bin으로 나눠 bin별
# GLS 회귀)은 기본값(n_jobs=1)이면 전부 싱글코어 순차 처리됨 -> 결과(beta 값)는
# 완전히 동일하게 유지한 채 bin 단위 병렬화만 적용해 fitting 시간을 단축.
# OS/다른 프로세스용 여유를 위해 논리 코어 수보다 2개 적게 사용.
N_JOBS = max(1, (os.cpu_count() or 2) - 2)

# nilearn 계산 캐시(같은 입력이면 재계산 건너뜀) + 뇌 마스크 캐시 위치
WORK_DIR = BIDS_ROOT / "glm_6condition_work"
NILEARN_CACHE_DIR = WORK_DIR / "nilearn_cache"
MASK_CACHE_DIR = WORK_DIR / "masks"
# ==========================

def folder_id(sub_num: int) -> str:
    """폴더명 형식: sub-01 (2자리) - events.tsv 등 원본 경로에 쓰임"""
    return f"sub-{sub_num:02d}"

def file_id(sub_num: int) -> str:
    """파일명 형식: sub-001 (3자리) - 전처리된 bold nii.gz 파일명에 쓰임"""
    return f"sub-{sub_num:03d}"

# OpenNeuro ds005226 v1.0.8 derivatives 자체의 버그 확인됨 (S3 원본과 checksum 대조로 검증):
# - sub-10, sub-11: 전처리된 image task BOLD 파일 자체가 없음
# - sub-12, sub-13: 전처리된 image task BOLD 파일이 업로드 시점부터 잘려있음(gzip stream 손상)
# - sub-31~41, sub-44~65: 전처리된 BOLD가 sub-01~35의 파일과 byte-for-byte 동일(30명 간격 중복
#   업로드 버그). 이벤트(events.tsv)는 진짜 해당 피험자 것이라 그대로 GLM 돌리면 에러 없이
#   조용히 "엉뚱한 사람 뇌영상 + 다른 사람 이벤트"로 결과가 나옴 -> 반드시 제외.
# - sub-42, sub-43: 위 중복 패턴에서 벗어나 있고 gzip도 정상이라 고유 데이터로 보임 -> 포함.
#
# events.tsv 자체 오염(파일 크기 대조 + run 순서 전수비교로 추가 확인, 2026-07-23):
# - sub-01, sub-33: run-02 이벤트가 실제 run-02 시퀀스가 아니라 run-01 시퀀스로 중복됨.
# - sub-22: run-01, run-02 둘 다 run-01이 아니라 sub-02의 run-02 시퀀스로 중복됨(자기 고유
#   데이터가 없음).
# - sub-07, sub-17, sub-37, sub-47: BOLD 파일 크기가 서로 완전히 동일한 4중 중복 클러스터라
#   단순 30명 오프셋 짝이 아님. 어느 쪽이 진짜 원본인지 미확인이라 sub-07, sub-17 둘 다 보류.
# 위 사유로 sub-01, sub-07, sub-17, sub-22, sub-33도 14~30 범위 밖에서 추가로 제외.
SUBJECT_NUMS = [16]

CONDITIONS = [
    ("Aircraft1", 0, 0.1), ("Aircraft1", 0, 0.75), ("Aircraft1", 0, 0.9),
    ("Aircraft2", 1, 0.1), ("Aircraft2", 1, 0.75), ("Aircraft2", 1, 0.9),
]

def make_trial_type(row):
    """stim_lable + levelOfOcclusion을 조합해 6조건 라벨 생성. rest는 None(암묵적 baseline)."""
    if pd.isna(row["stim_lable"]) or row["stim_lable"] not in [0, 1]:
        return None
    aircraft = "Aircraft1" if row["stim_lable"] == 0 else "Aircraft2"
    occ_pct = int(round(row["levelOfOcclusion"] * 100))
    return f"{aircraft}_{occ_pct}"

def events_path(sub_id: str, run: int) -> Path:
    """events.tsv 경로: sub-XXX/ses-01/func/sub-XXX_ses-01_task-image_run-NN_events.tsv"""
    return BIDS_ROOT / sub_id / "ses-01" / "func" / f"{sub_id}_ses-01_task-image_run-{run:02d}_events.tsv"

def load_events_for_run(sub_id: str, run: int) -> pd.DataFrame:
    fpath = events_path(sub_id, run)

    df = pd.read_csv(fpath, sep="\t")
    df["levelOfOcclusion"] = pd.to_numeric(df["levelOfOcclusion"], errors="coerce")
    df["stim_lable"] = pd.to_numeric(df["stim_lable"], errors="coerce")
    df["trial_type"] = df.apply(make_trial_type, axis=1)

    events = df.dropna(subset=["trial_type"])[["onset", "duration", "trial_type"]].reset_index(drop=True)
    return events

def find_bold_file(sub_id: str, f_id: str, run: int) -> Path:
    """bold 파일 경로: derivatives/pre-processed_data/space-MNI/sub-XX/sub-XXX_task-image_run-N_..._bold.nii.gz"""
    return PREPROC_ROOT / sub_id / f"{f_id}_task-image_run-{run}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"

def get_data_driven_confounds(bold_path: str, n_confounds: int = 5, mask_img=None) -> pd.DataFrame:
    """
    별도 confounds 파일이 없을 때, BOLD 이미지 자체에서 nilearn의
    high_variance_confounds로 데이터 기반 noise 성분을 추출 (CompCor와 유사한 원리).
    별도 motion parameter 파일 없이도 GLM에 nuisance regressor로 넣을 수 있음.
    mask_img를 넘기면 뇌 영역 복셀만 대상으로 SVD를 계산해서 배경 복셀까지 포함하는
    경우보다 훨씬 빠름.
    """
    from nilearn.image import high_variance_confounds
    confounds = high_variance_confounds(bold_path, n_confounds=n_confounds, mask_img=mask_img)
    return pd.DataFrame(confounds, columns=[f"hv_conf_{i}" for i in range(confounds.shape[1])])

def get_brain_mask(reference_bold_path: Path, sub_id: str) -> nib.Nifti1Image:
    """
    표준 MNI152 뇌 마스크를 BOLD grid에 nearest-neighbor로 resample해서 사용.

    이전에 compute_brain_mask(mask_type="whole-brain")를 썼을 때, 내부적으로
    템플릿 마스크를 continuous(trilinear) 보간으로 BOLD grid에 resample했는데,
    두 그리드가 정확히 voxel-aligned가 아니라서 경계 복셀 값이 0.5 근처에서
    들쭉날쭉해지고 그 결과 thresholding 후 niivue 시상면에서 체크무늬 패턴으로
    보이는 마스크 경계가 생겼음(확인됨). nearest-neighbor 보간 + BOLD 자체의
    affine/header를 그대로 사용하면 이 문제가 사라짐.
    """
    MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MASK_CACHE_DIR / f"{sub_id}_space-bold_brain-mask.nii.gz"

    if cache_path.exists() and cache_path.stat().st_mtime >= reference_bold_path.stat().st_mtime:
        return nib.load(str(cache_path))

    ref_3d = image.index_img(str(reference_bold_path), 0)
    mni_mask = datasets.load_mni152_brain_mask(resolution=2)
    resampled = image.resample_to_img(mni_mask, ref_3d, interpolation="nearest")

    mask_data = (resampled.get_fdata() > 0.5).astype(np.uint8)
    if mask_data.sum() == 0:
        raise ValueError("MNI mask resampling 결과가 비어 있습니다.")

    # ref_3d의 affine/header를 그대로 물려받아 BOLD와 완전히 동일한 grid를 보장.
    header = ref_3d.header.copy()
    header.set_data_dtype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, ref_3d.affine, header)
    nib.save(mask_img, str(cache_path))
    return mask_img

def build_contrast_vector(glm: FirstLevelModel, condition_name: str) -> list:
    """
    문자열 contrast("Aircraft1_70") 대신 run별 디자인 매트릭스를 직접 참조하는
    벡터를 만든다. 조건이 특정 run의 디자인 매트릭스에 없으면(그 run엔 해당 조건
    trial이 없었던 경우) 그 run은 전부 0벡터로 채워 기여를 0으로 만든다.
    (문자열 contrast는 모든 run에 그 열이 존재해야만 동작하므로 안전하지 않음)
    """
    vectors = []
    for design_matrix in glm.design_matrices_:
        vec = pd.Series(0.0, index=design_matrix.columns)
        if condition_name in design_matrix.columns:
            vec[condition_name] = 1.0
        vectors.append(vec.values)
    return vectors

def run_glm_for_subject(sub_num: int):
    sub_id = folder_id(sub_num)  # events.tsv 경로용 (2자리)
    f_id = file_id(sub_num)      # bold 파일명용 (3자리)

    # ---- 1. 파일 존재 확인 ----
    bold_files = {}
    for run in range(1, N_RUNS + 1):
        bold_file = find_bold_file(sub_id, f_id, run)
        if not bold_file.exists():
            raise FileNotFoundError(f"bold 파일 없음: {bold_file}")
        if not events_path(sub_id, run).exists():
            raise FileNotFoundError(f"events.tsv 없음: run-{run:02d}")
        bold_files[run] = bold_file

    # ---- 2. 뇌 마스크를 먼저 계산 (data-driven confounds 계산에도 재사용해서 속도 개선) ----
    # 데이터가 이미 space-MNI152NLin6Asym으로 정규화되어 있으므로,
    # intensity 기반 EPI 마스킹(compute_multi_epi_mask) 대신 표준 MNI152 템플릿을
    # bold affine에 맞춰 resample하는 방식을 사용. 배경 복셀도 0이 아닌(SPM12 정규화로
    # 인한 노이즈 floor) 이 데이터에서 intensity 기반 방식은 후두엽 등 실제 뇌 영역을
    # 마스크에서 잘라내는 문제가 있었음.
    print("  뇌 마스크 계산 중 (MNI152 템플릿 기반, nearest-neighbor resample)...")
    t_mask = time.time()
    mask_img = get_brain_mask(bold_files[1], sub_id)
    print(f"  마스크 준비 완료 ({time.time()-t_mask:.1f}초)")

    # ---- 3. run별 events + confounds 로드 (confounds는 위 마스크로 제한 -> 속도 개선) ----
    bold_imgs, events_list, confounds_list = [], [], []
    for run in range(1, N_RUNS + 1):
        t0 = time.time()
        print(f"  [run-{run}] 파일 확인 중...")

        bold_file = bold_files[run]
        bold_imgs.append(str(bold_file))
        events_list.append(load_events_for_run(sub_id, run))
        print(f"  [run-{run}] events 로드 완료 ({time.time()-t0:.1f}초)")

        t1 = time.time()
        print(f"  [run-{run}] data-driven confounds 계산 중 (뇌 마스크 내 복셀만 사용)...")
        confounds_list.append(
            get_data_driven_confounds(str(bold_file), N_HIGH_VARIANCE_CONFOUNDS, mask_img=mask_img)
        )
        print(f"  [run-{run}] confounds 준비 완료 ({time.time()-t1:.1f}초)")

    print(f"  GLM fitting 시작 (가장 오래 걸리는 단계, n_jobs={N_JOBS}로 AR1 bin 병렬화)...")
    t_fit = time.time()
    glm = FirstLevelModel(
        t_r=TR,
        smoothing_fwhm=SMOOTHING_FWHM,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=0.01,
        standardize=False,
        mask_img=mask_img,
        noise_model="ar1",
        minimize_memory=True,  # 불필요한 중간 결과 저장을 줄여 속도/메모리 개선 (beta 값에는 영향 없음)
        memory=str(NILEARN_CACHE_DIR),  # 캐시 폴더: 동일 입력이면 재계산 건너뜀
        memory_level=1,
        n_jobs=N_JOBS,  # AR1 bin별 GLS fitting을 병렬화 (결과값은 동일, 속도만 개선)
        verbose=1,  # nilearn 자체 진행 로그 출력
    )
    glm = glm.fit(bold_imgs, events=events_list, confounds=confounds_list)
    print(f"  GLM fitting 완료 ({time.time()-t_fit:.1f}초)")

    condition_names = [f"{a}_{int(round(o*100))}" for a, _, o in CONDITIONS]
    beta_maps = {}
    t_contrast = time.time()
    for cond in tqdm(condition_names, desc="  조건별 beta map 계산"):
        contrast_vector = build_contrast_vector(glm, cond)
        beta_map = glm.compute_contrast(contrast_vector, output_type="effect_size")
        beta_maps[cond] = beta_map
    print(f"  contrast 계산 완료 ({time.time()-t_contrast:.1f}초)")

    return beta_maps

def main():
    out_dir = BIDS_ROOT / "glm_6condition_betas"
    out_dir.mkdir(exist_ok=True)

    for sub_num in SUBJECT_NUMS:
        sub_id = folder_id(sub_num)
        print(f"=== {sub_id} GLM 시작 ===")
        t_start = time.time()
        try:
            beta_maps = run_glm_for_subject(sub_num)
        except (FileNotFoundError, EOFError, OSError, ValueError) as e:
            # ds005226 v1.0.8 derivatives에서 파일 누락/손상(gzip 잘림) 등이 실제로 발견됨
            # -> 한 피험자에서 죽어도 밤새 돌리는 나머지 피험자 처리는 계속 진행되도록 함.
            print(f"  건너뜀 ({type(e).__name__}): {e}")
            continue

        sub_out = out_dir / sub_id
        sub_out.mkdir(exist_ok=True)
        for cond, img in beta_maps.items():
            # 원본 BOLD가 int16이라 img가 그 dtype/헤더를 물려받는 경우가 있음.
            # effect_size(float)를 그대로 저장하면 int16으로 양자화되어 배경이 정확히
            # 0이 아니게 되고 뇌 안쪽 값도 정밀도가 뭉개짐 -> float32로 명시 저장.
            float_img = nib.Nifti1Image(img.get_fdata(dtype=np.float32), img.affine)
            nib.save(float_img, sub_out / f"{cond}_beta.nii.gz")
        print(f"  완료: {sub_out} (총 소요시간: {time.time()-t_start:.1f}초)")

if __name__ == "__main__":
    main()
