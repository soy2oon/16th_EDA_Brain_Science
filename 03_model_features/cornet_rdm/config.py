from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT.parent
STIMULI = SOURCE_ROOT / "data" / "stimuli"
SOURCE_BETAS = SOURCE_ROOT / "results" / "cornet_s_openneuro" / "betas"
DATA = ROOT / "data"
FEATURES = ROOT / "features"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
for directory in (DATA, FEATURES, RESULTS, FIGURES, MODELS):
    directory.mkdir(parents=True, exist_ok=True)

CONDITIONS = ("A10", "A70", "A90", "B10", "B70", "B90")
LAYERS = ("V1_t1", "V2_t1", "V2_t2", "V4_t1", "V4_t2", "V4_t3", "V4_t4", "IT_t1", "IT_t2")
REGION_ENDPOINTS = ("V1_t1", "V2_t2", "V4_t4", "IT_t2")
HUMAN_ACCURACY = {10: .96, 70: .79, 90: .62}
HUMAN_RT = {10: 1.23, 70: 1.65, 90: 1.89}
SEEDS = tuple(range(10))
TRAIN_FRACTION = .80
C_GRID = (.01, .1, 1.0, 10.0)
