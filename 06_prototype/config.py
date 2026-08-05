"""중앙 설정 — 경로, 파일명 규약, 클래스/가림 레벨 정의.

가림(occlusion) 인지과정 실험: Faster R-CNN 2단계(RPN/ROI)를
인간 인지 2단계(시각처리/의미분류)에 대응시켜 측정한다.
"""
from pathlib import Path

# ---- 경로 ----
ROOT = Path(__file__).resolve().parent.parent
STIMULI_DIR = ROOT / "data" / "stimuli"   # aircraft1/, aircraft2/, intact/
QC_DIR = ROOT / "data" / "qc"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MANIFEST = ROOT / "data" / "manifest.csv"
for _d in (QC_DIR, CKPT_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- 클래스 매핑 (torchvision detection: 0 = background) ----
# 항공기 1 = 표적 A = class_id 1, 항공기 2 = 표적 B = class_id 2
AIRCRAFT_TO_CLASS = {1: 1, 2: 2}
CLASS_TO_NAME = {1: "A", 2: "B"}
NUM_CLASSES = 3  # background + A + B

# ---- 가림 레벨 ----
OCCLUSION_LEVELS = [10, 70, 90]   # 테스트 조건 (%). 0 = intact(학습 전용)
HUMAN_ACCURACY = {10: 0.96, 70: 0.79, 90: 0.62}   # 논문 Table (n=66)
HUMAN_RT = {10: 1.23, 70: 1.65, 90: 1.89}          # 초

# ---- 파일명 규약 ----
# 가림:  Aircraft{N}_{occ}__{idx}.jpg   예) Aircraft2_70__13.jpg
# 무가림: Aircraft{N}__{idx}.jpg (intact/ 폴더)  예) Aircraft2__13.jpg
FNAME_OCCLUDED = r"Aircraft(?P<n>\d+)_(?P<occ>\d+)__(?P<idx>\d+)\.jpg"
FNAME_INTACT = r"Aircraft(?P<n>\d+)__(?P<idx>\d+)\.jpg"

# ---- 전처리 / bbox ----
IMG_SIZE = (535, 678)          # (H, W) — 원본 규격
FG_THRESH = 15                 # 0..255, 이보다 밝으면 전경(non-black)으로 간주
BBOX_PAD = 4                   # bbox 여유 픽셀

# ---- 추론 파라미터 ----
DEVICE = "auto"                # cuda > mps > cpu 순서로 자동 선택
OBJECTNESS_TOPK = 100          # RPN 제안 상위 K개로 stage-1 지표 계산
OBJECTNESS_THRESH = 0.5        # n_proposals 카운트 임계
DETECTION_SCORE_THRESH = 0.05  # 최종 탐지 채택 하한(고가림에서도 신호 확보 위해 낮게)
SEEDS = [0, 1, 2]              # 학습 시드 (≥3)
