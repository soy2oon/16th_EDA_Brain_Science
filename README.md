# HUMAN & C.V. : 가림 수준에 따른 인간과 AI의 객체 인식 표상 유사도 연구

## 참여자
- **팀장** : 배소윤 (15기)
- **팀원** : 김정민 (15기), 박성하 (15기), 이준희 (16기), 조규홍 (16기)

---

## EDA 프로젝트 자료 소개

### Dataset
- [An fMRI Dataset on Occluded Image Interpretation for Human Amodal Completion Research (Li et al., 2025, *Scientific Data*)](https://doi.org/10.1038/s41597-025-05414-w) : 참가자 64명이 항공기(ISAR) 2기종 이미지를 10%(기준)·70%·90% 세 단계의 검은 사각형 가림(occlusion) 조건에서 관찰할 때의 fMRI 반응을 담은 공개 데이터셋. 자극 수는 2기종 × 3가림수준 × 50장 = 총 300장.

### 발표자료 및 코드
- **EDA 최종 코드** : [GitHub Repository](https://github.com/soy2oon/16th_EDA_Brain_Science)
  - `01_stimuli` : 실험 자극(가림 조건별 항공기 이미지) 관련 코드
  - `roi_occipital`, `roi_extraction.py`, `roi_extraction_dacc.py` : 후두엽(occipital) 및 dACC ROI 추출
  - `glm_6condition.py`, `glm_stimfile_condition.py` : GLM fitting을 통한 beta map 추출
  - `03_model_features` : CORnet-S / ViT-B/16 / ResNet-50 특징(feature) 추출
  - `04_rq1_brain_model_rsa` : RQ1 - 인간 후두엽 RDM과 비전 모델 RDM 간 RSA 분석
  - `05_rq2_representation` : RQ2 - 템플릿 매칭·선형 프로브, cosine similarity, 혼란도(엔트로피) 분석
  - `06_prototype` : 프로토타입 관련 코드
  - `dacc_entropy_correlation.py`, `dacc_occlusion_correlation.py`, `dacc_ai_error_correlation.py` : dACC 관련 상관 분석

---

## EDA 프로젝트 요약

### 프로젝트 주제 및 목적
인간의 뇌는 가림(occlusion) 현상에서 감각 자극을 분석하는 **상향식(Bottom-up) 처리**와 기억·기대를 바탕으로 가려진 부분을 추론하는 **하향식(Top-down) 처리**를 상호작용시켜 가려진 물체 전체를 지각한다. 반면 CNN·ViT와 같은 비전 모델은 대체로 순전파(Feed-forward) 위주의 구조로 작동한다.

본 프로젝트는 다음 두 가지 연구 질문(RQ)을 다룬다.
- **RQ1** : 인간과 비전 모델(CORnet-S, ViT-B/16, ResNet-50)의 시감각 프로세스 표상 유사도를, 기존 연구처럼 가림이 없는 조건이 아니라 **가림 현상에서도** RSA(Representational Similarity Analysis)로 비교했을 때 여전히 성립하는가?
- **RQ2** : RQ1에서 확인된 인간-모델 간 표상 유사도 격차가 어디에서 비롯되는지, 모델의 불확실성 신호(엔트로피·확신도) 분석과 밝기(brightness) 통제 조건을 통해 규명한다.

### 데이터 전처리
- **표준 전처리** : 원 논문(Li et al., 2023)이 SPM12로 수행한 결과물을 그대로 사용. Slice timing 보정, 모션 보정(realignment), MNI152 표준공간 정규화 수행.
- **뇌 마스킹** : 정규화·보간 과정에서 발생하는 배경 노이즈로 인해 후두엽 일부가 배경으로 오판되는 문제를 확인하여, 표준 MNI152 템플릿 기반 마스크(`compute_brain_mask`)를 적용.
- **노이즈 처리** : 모션 파라미터 파일이 공개되지 않아 BOLD 신호 자체에서 데이터 기반 고분산 성분을 추출(`high_variance_confounds`)해 대체. RSA(패턴 비교)가 목적이므로 2mm 수준의 가벼운 스무딩을 적용.
- **GLM fitting → beta map 추출** : `y = β* × (Explained variance) + (Error/Residuals)` 형태의 GLM을 적합하여 조건별 beta map을 산출 ([`glm_6condition.py`](https://github.com/soy2oon/16th_EDA_Brain_Science/blob/main/glm_6condition.py), [`glm_stimfile_condition.py`](https://github.com/soy2oon/16th_EDA_Brain_Science/blob/main/glm_stimfile_condition.py)).

### 분석 방법 및 결과

**RQ1 : 인간-비전 모델 표상 유사도 (RSA)**
- 비전 모델 3종 채용 : ① **CORnet-S** – V1·V2·V4·IT 4개 계층으로 영장류 시각피질 구조를 모사한 재귀적(recurrent) 구조 모델, ② **ViT-B/16** – 이미지 패치 간 self-attention으로 관계를 계산하는 Vision Transformer, ③ **ResNet-50** – 잔차 학습(residual learning)·스킵 연결 기반 CNN.
- 세 모델 모두 ImageNet 사전학습 가중치를 고정한 채 특징만 추출(레이어별)하고, 각 레이어의 RDM을 계산해 인간 후두엽 RDM과 Spearman 상관(RSA)으로 비교 ([`03_model_features`](https://github.com/soy2oon/16th_EDA_Brain_Science/tree/main/03_model_features), [`04_rq1_brain_model_rsa`](https://github.com/soy2oon/16th_EDA_Brain_Science/tree/main/04_rq1_brain_model_rsa)).
- **결과** : 인간 후두엽 RDM은 비교적 연속적인 패턴을 보이는 반면, 세 비전 모델의 RDM은 가림 수준(10/70/90%)에 따른 블록 구조가 뚜렷하게 나타남. 인간-모델 간 RSA 상관계수는 전 레이어에 걸쳐 낮게 유지(약 0.1 내외)되어, **비전 모델과 인간 후두엽 표상은 유사도가 떨어짐**을 확인.

**RQ2 : 표상 격차의 원인 분석**
- **템플릿 매칭 vs 선형 프로브** : 템플릿 매칭은 원본 두 이미지의 특징벡터를 템플릿으로 삼아 테스트 이미지와의 코사인 유사도로 분류, 선형 프로브는 10% 가림 이미지 특징을 PCA(30차원)로 축소 후 로지스틱 회귀로 학습(5-fold 교차검증)하여 70%·90%에 그대로 적용 ([`05_rq2_representation`](https://github.com/soy2oon/16th_EDA_Brain_Science/tree/main/05_rq2_representation)).
- **Cosine similarity 분석** : 가림 수준이 높아질수록 모든 모델에서 원본 이미지와의 자기 유사도는 감소하고, 서로 다른 이미지 간 평균 유사도는 증가 → 가림이 심할수록 모델의 표현 공간(representational space)이 압축되어 서로 다른 객체를 구별하지 못하는 경향을 보임.
- **혼란도(엔트로피) 분석** : brightness-matched 조건에서 ViT-B/16은 90% 가림 시 정확도가 우연 수준(약 0.51)까지 폭락 → 색·밝기 아티팩트에 의존했음을 시사. ResNet-50·CORnet-S는 90% 가림에서도 정확도는 유지되지만 이진 엔트로피는 뚜렷이 증가 → "정답은 맞히지만 점점 확신을 잃는" 확신도-정확도 해리(dissociation) 발견. 이는 인간의 dACC(dorsal anterior cingulate cortex)처럼 헷갈림을 감지해 능동적으로 개입하는 감시 체계가 비전 모델에는 없음을 시사 ([`dacc_entropy_correlation.py`](https://github.com/soy2oon/16th_EDA_Brain_Science/blob/main/dacc_entropy_correlation.py) 등).
- **선행 문헌 검토** : Zhu et al.(2019)의 compositional model, Geirhos et al.(2019)의 texture bias·Stylized-ImageNet 재학습, Jang & Tong(2023)의 발달 커리큘럼 훈련 실험을 종합하면, CNN은 원래 극단적 가림에 취약하고, 사람과 다른 국소 텍스처 기반 판단 원리를 사용하며, 훈련 방식(데이터·커리큘럼)만 조정해서는 인간과의 격차가 근본적으로 좁혀지지 않음을 확인.

### 결론
1. **RQ1 – 표상 유사도는 부분적으로만 유지된다** : 기존 CNN-시각피질 대응 연구의 위계적 대응 관계는 가림 상황에서도 어느 정도 재현되지만, RDM을 직접 시각화하면 비전 모델은 가림 수준에 따라 뚜렷한 블록 구조로 항공기 A/B를 구분하는 반면 인간 후두엽에는 이러한 구조가 거의 존재하지 않는다. 즉 인간과 비전 모델은 가림 조건에서 정체성을 조직하는 원리 자체가 다르다.
2. **RQ2 – 인간과 비전 모델이 다른 원리로 판단하는 근거** : 비전 모델은 가림이 심해질수록 객체 정체성을 구분할 수 있는 표현 공간 자체가 붕괴되며, 학습 방식(데이터·커리큘럼 조정)만으로는 인간의 하향식(Top-down) 추론 능력이 생기지 않는다.
3. **종합** : CNN/ViT는 특정 조건에서 인간과 비슷한 위계적 표상을 보이지만, 가림이라는 조건에서는 표상이 붕괴되고 저차원 단서에 의존하며 불확실성을 능동적으로 감시하는 기전이 없다. 이는 학습 데이터나 훈련 방식 조정만으로는 해결되지 않는 **아키텍처 자체의 근본적 차이**를 시사한다.

### 아쉬운 점
- **OIID 데이터셋의 한계** : Stimuli 이미지 셋이 온전하지 않아 절반가량을 AI 생성 이미지로 복구해야 했고, 최대한 복구했음에도 인간이 실제로 본 이미지와 모델이 입력받은 이미지를 완벽히 1:1로 매칭하기는 불가능했다. 또한 이미지 자체가 일반적인 AI 모델 학습 이미지 분포와 결이 많이 달라(분포 밖, out-of-distribution), 가림 수준별 fMRI 뇌 반응을 확인할 수 있는 데이터셋이 OIID 외에는 없었다는 근본적 제약도 있었다.
- **비전 모델에 대한 이해의 한계** : 인간은 가려진 부분을 능동적으로 유추해 채우지만, 분류 과제만 입력받은 비전 모델은 데이터 증강을 하더라도 "가림을 무시하고 분류하라"는 과제만 수행하므로 정확도가 낮아질 수밖에 없었다. 또한 검은 픽셀이 가림임을 명시적으로 알려주지 않으면 모델은 이를 이미지의 일부로 인식해 버리며, 최소한의 미세조정된 헤드가 필요하다는 한계도 확인했다.

### 추가로 하면 좋을 분석 방법
- 가림에 특화된 **compositional(부품 기반) 모델**이나 Stylized-ImageNet과 같은 텍스처 편향 교정 학습을 적용한 모델을 동일한 RSA·엔트로피 파이프라인으로 재평가.
- 인간의 dACC처럼 불확실성을 능동적으로 감시·개입하는 모듈을 비전 모델에 추가한 뒤, 가림 조건에서의 표상 붕괴가 완화되는지 검증.
- 항공기 외 다른 카테고리(얼굴, 동물, 사물 등)의 가림 fMRI 데이터로 확장하여 결과의 일반화 가능성을 검증.
