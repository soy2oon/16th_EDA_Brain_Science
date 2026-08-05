"""
OIID dACC(dorsal anterior cingulate cortex) ROI 추출 스크립트 (stim_file 297조건 버전)
------------------------------------
목적: glm_stimfile_condition.py가 만든 stim_file 단위 beta map(.nii.gz)에서
      dACC 복셀만 골라내어 RSA에 바로 쓸 수 있는 (조건 x 복셀) 행렬로 저장한다.
      roi_extraction.py(occipital)와 거의 동일한 구조이고, ROI 정의 부분만 다르다.

전제:
- glm_stimfile_condition.py를 먼저 실행해서
  glm_stimfile_betas/sub-XX/{condition}_beta.nii.gz, conditions.json 이
  만들어져 있어야 함.

dACC 마스크를 만드는 방법(중요, roi_extraction.py와 다른 부분):
- Harvard-Oxford 피질 아틀라스(cort-maxprob-thr25-2mm)에는 "Cingulate Gyrus,
  anterior division"이라는 라벨 하나만 있고, 이 안에 dACC(인지 통제/갈등 모니터링/
  통증 등과 관련된 dorsal 부분)와 rostral/subgenual ACC(정서/보상 관련 ventral
  부분)가 구분 없이 섞여 있음 - dorsal/ventral 세부 라벨이 아예 없음.
- 그래서 "anterior division" 라벨 마스크를 만든 뒤, MNI 좌표계의 z축(상하) 기준으로
  Z_THRESHOLD_MM 이상인 복셀만 남겨서 dorsal 부분만 대략적으로 잘라낸다.
  이건 엄밀한 해부학적 경계가 아니라 문헌에서 흔히 쓰는 근사 기준(Shackman et al.,
  2011 dACC 메타분석 좌표대 참고)이라, 필요하면 Z_THRESHOLD_MM 값을 조정하거나
  더 세밀한 아틀라스(Schaefer, Glasser HCP-MMP1 등)로 바꿔야 함.

사용법: python roi_extraction_dacc.py
경로/라벨/Z_THRESHOLD_MM만 본인 환경에 맞게 수정하세요.
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
OUT_ROOT = G.BIDS_ROOT / "roi_dacc_stimfile"

# 정상 피험자 기준 참고값(glm_stimfile_condition.py 독스트링 참고). 강제 조건이
# 아니라 다르면 그냥 로그만 남기고 그대로 진행한다.
EXPECTED_N_CONDITIONS = 297

# Harvard-Oxford에는 anterior/posterior division만 있고 dorsal/ventral 구분이
# 없어서, 이 라벨 마스크에 MNI z좌표 기준을 추가로 적용해 dorsal(dACC)만 남긴다.
DACC_LABELS = ["Cingulate Gyrus, anterior division"]
Z_THRESHOLD_MM = 20  # 참고: Shackman et al. 2011 dACC 메타분석 좌표대 근사치. 조정 가능.
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
    (roi_extraction.py의 discover_subjects()와 동일한 로직)
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


def build_dacc_mask(reference_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Harvard-Oxford "Cingulate Gyrus, anterior division" 라벨 마스크에서
    MNI z좌표 >= Z_THRESHOLD_MM인 복셀만 남겨 dACC(dorsal 부분) 근사 마스크를
    만들고, reference_img(beta map) grid에 맞춘다.

    nearest-neighbor로 resample하는 이유: roi_extraction.py의 후두엽 마스크와
    동일(continuous 보간의 경계 체크무늬 문제 회피).
    """
    print("  [마스크] Harvard-Oxford 아틀라스 로드 중 (최초 실행 시 다운로드로 시간이 걸릴 수 있음)...")
    ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img, labels = ho.maps, ho.labels

    wanted_idx = [i for i, name in enumerate(labels) if name in DACC_LABELS]
    found = [labels[i] for i in wanted_idx]
    missing = set(DACC_LABELS) - set(found)
    if missing:
        raise RuntimeError(
            f"아틀라스에서 찾지 못한 라벨: {missing}\n실제 라벨 목록: {labels}"
        )
    print(f"  [마스크] 선택된 라벨 {len(found)}개: {found} (z>={Z_THRESHOLD_MM}mm만 dACC로 사용)")

    atlas_data = np.asarray(atlas_img.dataobj)
    label_mask = np.isin(atlas_data, wanted_idx)

    ijk = np.indices(atlas_data.shape).reshape(3, -1).T
    world = nib.affines.apply_affine(atlas_img.affine, ijk)
    z_coords = world[:, 2].reshape(atlas_data.shape)

    dorsal_mask = (label_mask & (z_coords >= Z_THRESHOLD_MM)).astype(np.uint8)
    mask_img = nib.Nifti1Image(dorsal_mask, atlas_img.affine, atlas_img.header)

    resampled = image.resample_to_img(mask_img, reference_img, interpolation="nearest")
    mask_arr = (np.asarray(resampled.dataobj) > 0.5).astype(np.uint8)

    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    mask_final = nib.Nifti1Image(mask_arr, reference_img.affine, header)

    n_vox = int(mask_arr.sum())
    if n_vox == 0:
        raise RuntimeError("dACC 마스크가 비어 있습니다 (0 복셀). Z_THRESHOLD_MM/라벨/그리드를 확인하세요.")
    print(f"  [마스크] dACC 복셀 수: {n_vox}")
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
    mask_img = build_dacc_mask(reference_img)

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
    """roi_extraction.py와 동일한 재개(resume) 로직."""
    meta_path = sub_out / "roi_meta.json"
    if not meta_path.exists():
        return False
    required = ["betas_dacc.npy", "betas_dacc.csv", "dacc_mask.nii.gz"]
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

        print(f"\n=== {sub_id} dACC ROI 추출 (조건 {len(condition_names)}개) ===")

        try:
            betas, mask_img = extract_subject_roi(sub_num, condition_names)
        except FileNotFoundError as e:
            print(f"  건너뜀: {e}")
            continue

        sub_out.mkdir(parents=True, exist_ok=True)

        np.save(sub_out / "betas_dacc.npy", betas)
        pd.DataFrame(betas, index=condition_names).to_csv(sub_out / "betas_dacc.csv")
        nib.save(mask_img, sub_out / "dacc_mask.nii.gz")

        meta = {
            "subject": sub_id,
            "conditions": condition_names,
            "n_conditions": len(condition_names),
            "n_voxels": int(betas.shape[1]),
            "dacc_labels": DACC_LABELS,
            "z_threshold_mm": Z_THRESHOLD_MM,
            "note": (
                "행=조건(이 피험자의 conditions.json 순서, 피험자마다 다를 수 있음), "
                "열=dACC 복셀. Harvard-Oxford 'Cingulate Gyrus, anterior division' 라벨 중 "
                f"MNI z>={Z_THRESHOLD_MM}mm만 남긴 근사 dACC 마스크 (dorsal/ventral 세부 라벨이 "
                "없는 아틀라스 한계로 인한 근사치). dacc_mask.nii.gz로 niivue에서 위치 확인 가능."
            ),
        }
        with open(sub_out / "roi_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"  완료: {sub_out} (betas shape={betas.shape})")


if __name__ == "__main__":
    main()
