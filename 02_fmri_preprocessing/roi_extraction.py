"""
OIID 후두엽(occipital) ROI 추출 스크립트 (stim_file 297조건 버전)
------------------------------------
목적: glm_stimfile_condition.py가 만든 stim_file 단위 beta map(.nii.gz)에서
      Harvard-Oxford 아틀라스 기반 후두엽 ROI 복셀만 골라내어
      RSA(representational similarity analysis)에 바로 쓸 수 있는
      (조건 x 복셀) 행렬로 저장한다.

전제:
- glm_stimfile_condition.py를 먼저 실행해서
  glm_stimfile_betas/sub-XX/{condition}_beta.nii.gz, conditions.json 이
  만들어져 있어야 함.

6조건(glm_6condition.py) 버전과 달라진 점:
- 조건 목록이 전역 상수(예전의 G.CONDITIONS 6개)가 아니라 피험자마다 다를 수
  있음. glm_stimfile_condition.py는 반복 trial(같은 stim_file이 2번 나오는
  경우 첫 trial만 사용)을 피험자 본인의 events.tsv를 훑어서 그때그때 결정하기
  때문에, "정상" 피험자는 보통 297개가 나오지만 이론상 피험자별로 개수/이름이
  달라질 수 있음. 그래서 이 스크립트는 CONDITION_NAMES를 하나의 전역 리스트로
  고정하지 않고, 각 피험자 폴더의 conditions.json에 실제로 기록된 조건 목록을
  그대로 사용한다.
- discover_subjects()도 전역 조건 리스트 존재 여부가 아니라, 피험자 자신의
  conditions.json에 적힌 조건들의 beta 파일이 전부 있는지로 판단한다.
- 주의: 피험자마다 조건 집합(개수/이름)이 다를 수 있으므로, 여러 피험자를
  묶어 RSA를 돌리거나 조건 순서를 맞춰 비교하려면 이후 단계에서 공통 조건만
  추리는 작업이 별도로 필요하다. 이 스크립트는 피험자별 추출까지만 담당.

사용법: python roi_extraction.py
경로/라벨만 본인 환경에 맞게 수정하세요.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, image
from nilearn.maskers import NiftiMasker

import glm_6condition as G
import glm_stimfile_condition as GS

# ==== 여기만 수정하세요 ====
BETAS_ROOT = GS.OUT_ROOT  # glm_stimfile_betas
OUT_ROOT = G.BIDS_ROOT / "roi_occipital_stimfile"

# 정상 피험자 기준 참고값(glm_stimfile_condition.py 독스트링 참고). 강제 조건이
# 아니라 다르면 그냥 로그만 남기고 그대로 진행한다(피험자별로 실제 조건 목록을
# conditions.json에서 그대로 읽어 쓰므로 개수가 달라도 동작은 함).
EXPECTED_N_CONDITIONS = 297
# ==========================


def load_subject_conditions(sub_dir: Path) -> list[str]:
    meta_path = sub_dir / "conditions.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"conditions.json 없음: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta["conditions"]


def discover_subjects() -> dict[int, list[str]]:
    """
    glm_stimfile_betas 아래 실제로 conditions.json에 적힌 조건 beta 파일이
    전부 존재하는 sub-XX 폴더만 스캔해서 {sub_num: condition_names}로 반환한다.

    6조건 버전(discover_subject_nums)과 달리 조건 목록 자체가 피험자마다 다를
    수 있어서, 하나의 전역 CONDITION_NAMES로는 검증할 수 없다 -> 피험자 본인의
    conditions.json을 기준으로 검증한다.
    """
    result = {}
    for sub_dir in sorted(BETAS_ROOT.glob("sub-*")):
        if not sub_dir.is_dir():
            continue

        try:
            condition_names = load_subject_conditions(sub_dir)
        except FileNotFoundError:
            print(f"  [스캔] {sub_dir.name}: conditions.json 없어 제외")
            continue

        has_all = all(
            (sub_dir / f"{cond}_beta.nii.gz").exists() for cond in condition_names
        )
        if not has_all:
            print(f"  [스캔] {sub_dir.name}: conditions.json에 기록된 beta 중 일부가 없어 제외")
            continue

        if len(condition_names) != EXPECTED_N_CONDITIONS:
            print(
                f"  [스캔] {sub_dir.name}: 조건 수 {len(condition_names)}개 "
                f"(참고값 {EXPECTED_N_CONDITIONS}개와 다름 - 그대로 진행)"
            )

        sub_num = int(sub_dir.name.split("-")[1])
        result[sub_num] = condition_names
    return result


SUBJECTS = discover_subjects()

# hoohdini/26-2_EDA_Brain_Science (rsa_pipeline/config.py) 참고: 후두엽 8개 영역
OCCIPITAL_LABELS = [
    "Intracalcarine Cortex",
    "Cuneal Cortex",
    "Lingual Gyrus",
    "Occipital Fusiform Gyrus",
    "Supracalcarine Cortex",
    "Occipital Pole",
    "Lateral Occipital Cortex, superior division",
    "Lateral Occipital Cortex, inferior division",
]
# ==========================


def build_occipital_mask(reference_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Harvard-Oxford 피질 아틀라스(cort-maxprob-thr25-2mm)에서 후두엽 라벨만 골라
    reference_img(beta map) grid에 맞춘 이진 마스크를 만든다.

    nearest-neighbor로 resample하는 이유: glm_6condition.py의 뇌 마스크에서 확인된
    문제와 동일한 원인(continuous 보간은 경계 복셀 값을 0.5 근처에서 들쭉날쭉하게
    만들어 니뷰에서 체크무늬로 보이는 마스크 경계를 만듦)을 여기서도 피하기 위함.
    이후 reference_img의 affine/header를 그대로 물려받아 beta map과 완전히
    동일한 grid를 보장한다.
    """
    print("  [마스크] Harvard-Oxford 아틀라스 로드 중 (최초 실행 시 다운로드로 시간이 걸릴 수 있음)...")
    ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img, labels = ho.maps, ho.labels

    wanted_idx = [i for i, name in enumerate(labels) if name in OCCIPITAL_LABELS]
    found = [labels[i] for i in wanted_idx]
    missing = set(OCCIPITAL_LABELS) - set(found)
    if missing:
        raise RuntimeError(
            f"아틀라스에서 찾지 못한 라벨: {missing}\n실제 라벨 목록: {labels}"
        )
    print(f"  [마스크] 선택된 후두엽 영역 {len(found)}개: {found}")

    atlas_data = np.asarray(atlas_img.dataobj)
    mask_data = np.isin(atlas_data, wanted_idx).astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, atlas_img.affine, atlas_img.header)

    resampled = image.resample_to_img(mask_img, reference_img, interpolation="nearest")
    mask_arr = (np.asarray(resampled.dataobj) > 0.5).astype(np.uint8)

    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    mask_final = nib.Nifti1Image(mask_arr, reference_img.affine, header)

    n_vox = int(mask_arr.sum())
    if n_vox == 0:
        raise RuntimeError("후두엽 마스크가 비어 있습니다 (0 복셀). 라벨/그리드를 확인하세요.")
    print(f"  [마스크] 후두엽 복셀 수: {n_vox}")
    return mask_final


def load_subject_beta_paths(sub_id: str, condition_names: list[str]) -> dict[str, Path]:
    sub_beta_dir = BETAS_ROOT / sub_id
    beta_paths = {}
    for cond in condition_names:
        p = sub_beta_dir / f"{cond}_beta.nii.gz"
        if not p.exists():
            raise FileNotFoundError(
                f"{sub_id}: beta map 없음: {p}\n"
                "glm_stimfile_condition.py를 먼저 실행했는지 확인하세요."
            )
        beta_paths[cond] = p
    return beta_paths


def extract_subject_roi(sub_num: int, condition_names: list[str]) -> tuple[np.ndarray, nib.Nifti1Image]:
    sub_id = G.folder_id(sub_num)
    beta_paths = load_subject_beta_paths(sub_id, condition_names)

    reference_img = nib.load(str(beta_paths[condition_names[0]]))
    mask_img = build_occipital_mask(reference_img)

    masker = NiftiMasker(mask_img=mask_img, standardize=False)
    masker.fit()

    n_vox = int(np.asarray(mask_img.dataobj).sum())
    betas = np.full((len(condition_names), n_vox), np.nan, dtype=np.float32)

    for i, cond in enumerate(condition_names):
        img = nib.load(str(beta_paths[cond]))
        betas[i, :] = masker.transform(img).ravel()

    if not np.isfinite(betas).all():
        raise ValueError(f"{sub_id}: 추출된 ROI beta에 NaN/Inf가 있습니다.")

    return betas, mask_img


def is_already_done(sub_out: Path, condition_names: list[str]) -> bool:
    """
    이전 실행에서 이미 정상적으로 끝난 피험자는 건너뛴다(297조건 x 24명이라
    한 번 돌리는 데 오래 걸리는데, 중간에 끊기면 처음부터 24명을 다시 계산하는
    낭비를 막기 위함). roi_meta.json이 있고 그 안의 조건 목록이 지금 계산하려는
    condition_names와 완전히 같을 때만 "완료"로 인정한다 - npy/csv/mask 파일
    중 하나라도 없거나(예: sub-26처럼 csv 쓰다 죽은 경우) meta의 조건 목록이
    다르면 다시 계산한다.
    """
    meta_path = sub_out / "roi_meta.json"
    if not meta_path.exists():
        return False
    required = ["betas_occipital.npy", "betas_occipital.csv", "occipital_mask.nii.gz"]
    if not all((sub_out / name).exists() for name in required):
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return meta.get("conditions") == condition_names


def main():
    OUT_ROOT.mkdir(exist_ok=True)
    print(f"대상 피험자 {len(SUBJECTS)}명: {[G.folder_id(n) for n in SUBJECTS]}")

    for sub_num, condition_names in SUBJECTS.items():
        sub_id = G.folder_id(sub_num)
        sub_out = OUT_ROOT / sub_id

        if is_already_done(sub_out, condition_names):
            print(f"\n=== {sub_id}: 이미 완료된 결과 있음 -> 건너뜀 ===")
            continue

        print(f"\n=== {sub_id} occipital ROI 추출 (조건 {len(condition_names)}개) ===")

        try:
            betas, mask_img = extract_subject_roi(sub_num, condition_names)
        except FileNotFoundError as e:
            print(f"  건너뜀: {e}")
            continue

        sub_out.mkdir(parents=True, exist_ok=True)

        np.save(sub_out / "betas_occipital.npy", betas)
        pd.DataFrame(betas, index=condition_names).to_csv(sub_out / "betas_occipital.csv")
        nib.save(mask_img, sub_out / "occipital_mask.nii.gz")

        meta = {
            "subject": sub_id,
            "conditions": condition_names,
            "n_conditions": len(condition_names),
            "n_voxels": int(betas.shape[1]),
            "occipital_labels": OCCIPITAL_LABELS,
            "note": "행=조건(이 피험자의 conditions.json 순서, 피험자마다 다를 수 있음), 열=후두엽 복셀. occipital_mask.nii.gz로 niivue에서 ROI 위치 확인 가능.",
        }
        with open(sub_out / "roi_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"  완료: {sub_out} (betas shape={betas.shape})")


if __name__ == "__main__":
    main()
