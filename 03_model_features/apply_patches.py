#!/usr/bin/env python3
"""세 final 스크립트에 재실행용 패치를 적용합니다.

원본은 건드리지 않고 *_patched.py 를 새로 만듭니다.
적용 후 sha256 을 출력하니 provenance 에 기록하세요.

사용:
    python3 apply_patches.py --scripts-dir /Users/kimjyeongmin/Downloads
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# (파일, 설명, 찾을 문자열, 바꿀 문자열)
PATCHES = [
    (
        "final_vit_experiment.py",
        "ViT feature 배열 저장 (fig1/fig2 재현에 필수)",
        '        blank = transform(Image.new("RGB", (224, 224), color=(0, 0, 0))).unsqueeze(0)',
        '        feature_dir = condition_dir / "features"\n'
        '        feature_dir.mkdir(exist_ok=True)\n'
        '        np.save(feature_dir / "intact_final_cls.npy", intact_features)\n'
        '        for _name, _values in occluded_by_layer.items():\n'
        '            np.save(feature_dir / f"occluded_{_name}.npy", _values)\n'
        '        blank = transform(Image.new("RGB", (224, 224), color=(0, 0, 0))).unsqueeze(0)',
    ),
    (
        "final_vit_experiment.py",
        "ViT intact 도 블록별로 수집 (층별 그림용)",
        "            args.rdm_layers,\n            collect_layers=False,\n        )[\"final_cls\"]",
        "            args.rdm_layers,\n            collect_layers=True,\n        )",
    ),
    (
        "final_vit_experiment.py",
        "위 변경에 맞춰 probe 입력을 final_cls 로 지정",
        "        _, summary = run_probes(\n            records,\n            intact_features,",
        "        _, summary = run_probes(\n            records,\n            intact_features[\"final_cls\"],",
    ),
    (
        "final_vit_experiment.py",
        "intact feature 저장을 dict 형태에 맞춤",
        '        np.save(feature_dir / "intact_final_cls.npy", intact_features)',
        '        for _name, _values in intact_features.items():\n'
        '            np.save(feature_dir / f"intact_{_name}.npy", _values)',
    ),
    (
        "final_resnet_experiment.py",
        "conv1 추가 (회의에서 정한 6개 탭 지점 복원)",
        'RDM_LAYERS = ("layer1", "layer2", "layer3", "layer4", "avgpool")',
        'RDM_LAYERS = ("conv1", "layer1", "layer2", "layer3", "layer4", "avgpool")',
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts-dir", type=Path, required=True)
    args = ap.parse_args()
    d = args.scripts_dir.expanduser().resolve()

    texts: dict[str, str] = {}
    for fname in {p[0] for p in PATCHES}:
        src = d / fname
        if not src.is_file():
            sys.exit(f"[FAIL] 없음: {src}")
        texts[fname] = src.read_text(encoding="utf-8")

    for fname, desc, old, new in PATCHES:
        t = texts[fname]
        n = t.count(old)
        if n == 0:
            sys.exit(f"[FAIL] {fname}: 앵커를 못 찾음 ({desc})\n  찾던 문자열:\n{old}")
        if n > 1:
            sys.exit(f"[FAIL] {fname}: 앵커가 {n}번 나타남, 모호함 ({desc})")
        texts[fname] = t.replace(old, new, 1)
        print(f"[OK] {fname}: {desc}")

    print()
    for fname, t in texts.items():
        out = d / fname.replace(".py", "_patched.py")
        out.write_text(t, encoding="utf-8")
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        print(f"작성: {out}")
        print(f"  sha256 {h}")

    # cornet 은 코드 변경 없음 — 이름만 맞춰 복사해 import 가 깨지지 않게 함
    cornet = d / "final_cornet_experiment.py"
    if cornet.is_file():
        tgt = d / "final_cornet_experiment_patched.py"
        tgt.write_text(cornet.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"작성: {tgt}  (변경 없음, import 호환용)")

    print("\n주의: *_patched.py 는 서로를 import 해야 합니다.")
    print("아래 명령으로 import 문을 함께 고치세요:")
    print("  cd", d)
    print("  sed -i '' 's/from final_cornet_experiment import/"
          "from final_cornet_experiment_patched import/' final_resnet_experiment_patched.py")
    print("  sed -i '' 's/from final_vit_experiment import/"
          "from final_vit_experiment_patched import/' final_resnet_experiment_patched.py")
    print("  sed -i '' 's/from final_vit_experiment import/"
          "from final_vit_experiment_patched import/' final_cornet_experiment_patched.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
