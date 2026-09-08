---
notion-url: https://app.notion.com/p/Object-detection_part1-c4977fa64f7f838f999681442f88f1ed
title: Object detection_part1
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Object-detection_part1-c4977fa64f7f838f999681442f88f1ed
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

1. **Object Detection이란?**

	- 딥러닝 기반 이미지 인식 과업은 크게 세 가지로 구분됩니다.

		- **객체 분류(Classification):** 이미지 내 객체가 어떤 클래스에 속하는지 결정

		- **객체 위치 예측(Localization):** 객체가 이미지 내 어느 위치에 있는지 bounding box로 표시

		- **객체 인식(Detection):** 한 프레임 내 여러 객체의 클래스와 위치를 동시에 예측

1. **1-Stage vs 2-Stage Detector**

	- **1-Stage Detector:** 단일 단계로 처리하여 이미지 전체를 한 번에 처리, 객체 위치와 클래스를 동시 예측. 대표 모델: YOLO, SSD

	- **2-Stage Detector:**

		- Stage 1 – 영역 제안(Region Proposal): 관심 영역 후보 생성

		- Stage 2 – 분류 및 위치 보정: 후보 영역에 대해 객체 분류 및 박스 정밀 보정

		- 대표 모델: R-CNN, Fast R-CNN, Faster R-CNN

1. **Object Detection이 어려운 이유**

	- 분류와 회귀를 동시에 진행: 여러 물체를 분류함과 동시에 위치를 탐지해야 함

	- 다양한 스케일과 모양: 객체마다 크기·형태가 다르고, 거리·각도에 따라 달라짐

	- 명확하지 않은 이미지

	- 배경 잡음 및 복잡성: 복잡한 배경 속 객체 구분의 어려움

	- 객체 간 겹침(Occlusion): 가려진 객체의 예측 어려움

	- 전체 이미지에서 객체가 차지하는 비중이 낮음 (배경이 대부분)

	- 실시간 처리 요구: 속도와 정확도의 균형 필요

	- 데이터셋 부족: annotation 제작이 상대적으로 어려움

1. **객체 인식 Dataset**

	- 데이터셋의 구성 요소

		- **이미지:** 탐지 대상 객체를 포함한 실제 이미지 파일

		- **어노테이션(Annotation):** 객체의 클래스와 위치 정보를 나타내는 메타데이터

 | Dataset | 카테고리 수 | 포맷 | 
 | ---- | ---- | ---- | 
 | **MS COCO 2017** | 80개 | JSON | 
 | **Open Image V7** | 600개 | CSV | 
 | **Pascal VOC 2012** | 20개 | XML | 

1. **성능평가지표 – IoU (Intersection over Union)**

	- 예측된 바운딩 박스와 실제 바운딩 박스의 **교집합과 합집합의 비율**로 정의

1. **성능평가지표 – Precision / Recall / Accuracy**

	- **Precision(정밀도):** 내가 맞았다고 한 것 중에 진짜 맞은 것

	- **Recall(재현율):** 진짜 맞은 것 중에 내가 맞다고 한 것

	- **Accuracy:** 전체 중에 내가 맞힌 개수

1. **Object Detection에서의 TP, FP, FN**

	- Class 정보도 맞추고 위치 정보도 맞춰야 → **TP(True Positive)**

1. **정밀도-재현율 트레이드오프**

	- 분류기는 각 샘플의 점수를 계산해 threshold보다 크면 양성, 작으면 음성으로 분류

	- **threshold를 낮게:** 재현율↑ 목표 → FN 감소 시도 → 모두 Positive로 예측 → FP 증가 → 정밀도 저하

	- **threshold를 높게:** 정밀도↑ 목표 → FP 감소 시도 → 확실한 것만 Positive → FN 증가 → 재현율 저하

	- 하나의 지표만 참고하면 극단적 수치 조작이 가능하므로, 두 지표를 적절히 조합하여 종합 성능을 평가해야 함

1. **Average Precision (AP)**

	- **Precision-Recall Curve:** threshold 변화에 따른 정밀도-재현율 관계를 시각화한 그래프

	- 곡선 아래 면적(AUC)이 클수록 좋은 성능을 의미

	- **AP:** 단일 클래스에 대해 PR 곡선의 AUC를 계산한 값

1. **mean Average Precision (mAP) & F1-Score**

	- **mAP:** 여러 클래스에 대한 AP의 평균값

	- **F1-Score:** Precision과 Recall의 조화 평균, 두 지표 간 균형을 측정

1. **IoU Threshold별 mAP**

	- **MS COCO:** IoU 0.5~0.95 (0.05 간격)로 AP를 측정 후 평균 → mAP 산출, 크기 유형(대/중/소)별 mAP도 측정

	- **Pascal VOC:** IoU 0.5 이상을 예측 성공 기준으로 설정 (COCO보다 단순)

[//]: # (heading_4 is not supported)

1. **Object Localization**

	- 기존 이미지 분류 모델의 대표 구조를 활용하여 물체 localization 문제를 어떻게 풀 수 있는지에서 출발

1. **Bounding Box 회귀를 바로 적용할 때 발생하는 문제**

	- **검색 공간의 방대함:** 모든 가능한 위치·크기를 고려해야 하는 매우 큰 검색 공간

	- **초기 추정치 부족:** 바운딩 박스 회귀는 보정 역할이므로 근사값(초기 후보 영역) 없이는 불안정

	- **불균형한 학습 데이터:** 배경 비율이 압도적으로 높아 네트워크가 배경에 초점을 맞출 위험

	- **학습의 복잡성 증가:** 다양한 스케일·위치·비율을 동시에 고려해야 하므로 학습이 어려워짐

1. **Region Proposal**

	- 이미지 내 객체가 있을 법한 영역들을 선별하여 후보로 제시

		- **연산 비용 감소:** 후보 영역에 대해서만 세밀 처리 → 계산량 절감

		- **정확도 향상:** 후속 단계에서 분류·위치 보정 집중 수행 가능

1. **R-CNN (Regions with CNN features)**

	- 이미지 내 객체의 위치와 종류를 동시 예측, 2013년 발표되어 딥러닝 기반 객체 검출의 기초를 마련

	- **접근법:** Region Proposal 생성(Selective Search) → CNN(VGGNet 등) 특징 추출 → SVM 분류 + 회귀기로 클래스·박스 보정

	- 논문: *"Rich feature hierarchies for accurate object detection and semantic segmentation"* (arXiv:1311.2524)

1. **Region Proposal – Selective Search**

	- 과정: 초기 이미지 → 초과 분할(Superpixels) → 특징 추출 → 유사도 기반 병합 → 후보 영역 생성

 | 구분 | 내용 | 
 | ---- | ---- | 
 | **장점** | 다양한 후보 영역 생성(색상·질감·크기·위치 조합) / 비지도 방식으로 별도 학습 불필요 / 단순하고 직관적인 접근 | 
 | **단점** | 높은 연산 비용·느린 속도(실시간 부적합) / 중복 영역 → NMS 등 후처리 필요 / RPN 등 학습 기반 방법 대비 낮은 효율성 | 

1. **R-CNN 동작 단계**

	1. **Region Proposal:** Selective Search로 이미지당 약 2,000개 후보 영역 생성

	1. **Warping:** 후보 영역을 227×227 크기로 고정 (Crop & Resize)

	1. **CNN Feature Extract:** ImageNet 사전학습 CNN을 Detection용 데이터셋으로 fine-tuning

	1. **이미지 분류 with SVM:** 클래스별 SVM Classifier 학습 (CNN 분류기보다 mAP가 4%가량 높아 SVM 채택)

	1. **Bounding Box Regression:** 선형 회귀로 박스 위치 보정

	1. **NMS:** 중복 검출 제거

1. **R-CNN Bounding Box Regression**

	- SS Proposal과 Ground Truth 간의 변환 관계를 회귀 모델로 학습

	- 좌표는 폭·높이로 정규화(스케일 독립성), 크기는 log 비율로 변환("두 배·절반" 같은 곱셈 차이를 덧셈 오차로 표현)

	- 모델이 예측한 보정값(dx, dy, dw, dh)을 앵커/후보 박스에 적용해 최종 박스 산출

	- R-CNN(후보 상자 classify만) vs R-CNN BB(classify + 회귀로 좌표 보정)

1. **NMS (Non-Max Suppression)**

	- 각 Bounding box는 confidence score를 가짐 → threshold 이하 박스 제거 (예: <0.5 제거)

	- 남은 박스를 confidence 기준 내림차순 정렬

	- 최상위 박스 기준으로 다른 박스와 IoU 계산, threshold 이상이면 제거 → 반복 수행

	- confidence threshold가 높을수록, IoU threshold가 낮을수록 더 많은 박스가 제거됨

1. **R-CNN 장단점**

 | 구분 | 내용 | 
 | ---- | ---- | 
 | **장점** | 높은 정확도(HoG+SVM 대비 월등) / 사전 학습된 CNN으로 적은 데이터셋에서도 강력한 특징 추출 | 
 | **단점** | 매우 느린 속도(CPU 기준 한 장당 약 50초) / 2,000개 후보 각각 CNN 입력 → 연산량 과다 / 특징 디스크 저장 필요 → 저장 공간 부담 / Selective Search·CNN·SVM·BB Regression이 독립 학습되어 End-to-End 불가능 | 

1. **Fast R-CNN**

	- R-CNN의 주요 병목을 줄이려는 대표 모델. 이미지 전체에 대해 한 번의 CNN 순전파 후, 공유 특징 맵에서 여러 ROI 특징을 추출해 위치·클래스 동시 예측

	- **접근법:** 단일 단계 end-to-end 학습 / 공유 합성곱 특징으로 계산 중복 제거 / ROI Pooling으로 고정 크기 특징 벡터 생성

	- 속도: R-CNN 대비 학습 9배, 테스트 최대 213배 향상

	- 논문: *"Fast R-CNN"* (arXiv:1504.08083)

1. **ROI Pooling (Region of Interest Pooling)**

	- 각 ROI를 고정 크기 그리드로 나눈 후 각 셀에서 max pooling → 고정 크기 출력 특징 맵 생성 (예: 5×7 → 2×2)

	- 다양한 크기의 ROI를 동일한 크기 벡터로 변환해 FC 층에 전달 가능

1. **Fast R-CNN Loss 함수**

	- **분류 손실(L_cls):** 정답 클래스에 대한 소프트맥스 손실

	- **회귀 손실(L_loc):** Smooth L1 Loss 기반 바운딩 박스 회귀 손실 (오차가 작을 때 L2, 클 때 L1으로 동작)

1. **Fast R-CNN 장단점**

 | 구분 | 내용 | 
 | ---- | ---- | 
 | **장점** | End-to-End 학습으로 파이프라인 단순화 / 공유 합성곱 특징으로 속도 대폭 향상 / ROI 풀링으로 일관된 입력 제공 / 별도 캐싱 불필요로 저장 공간 효율적 | 
 | **단점** | Region Proposal 품질에 성능 의존 / 특징 맵 해상도 제한으로 작은 객체 검출 어려움 / 깊은 네트워크 사용 시 메모리 부담 / 단일 스케일 학습으로 극단적 크기 변화에 취약 | 

1. **Faster R-CNN**

	- 기존 R-CNN·Fast R-CNN은 CPU 기반 Selective Search로 인한 병목 존재

	- **CNN 기반 RPN(Region Proposal Network)**을 도입해 후보 영역 생성을 네트워크 내에서 처리

	- **Faster R-CNN = RPN + Fast R-CNN**

	- 논문: *"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"* (arXiv:1506.01497)

 | 블록 | 역할 | 출력 채널 | 
 | ---- | ---- | ---- | 
 | 3×3 Conv | 지역 특징 집약 | 512 | 
 | 1×1 Conv (cls) | Anchor별 객체성 점수 2×k | h×w×2k (k=9 → ×18) | 
 | 1×1 Conv (reg) | Anchor별 δ-오프셋 4×k | h×w×4k (×36) | 

1. **Region Proposal Network (RPN)**

	- **목표:** 사전 정의된 anchor box 기준으로 후보 영역 생성, 실제 객체 경계에 맞도록 회귀 조정 + objectness 분류

	- **핵심 아이디어:** feature map 위에서 슬라이딩 윈도우 방식으로 각 grid cell마다 9개의 anchor 배정

		- **분류(cls):** 객체(positive)/배경(negative) 2-class softmax

		- **회귀(reg):** 앵커 오프셋(tx, ty, tw, th) 예측

	- **Anchor Box 구성:** 3가지 스케일 × 3가지 종횡비(예: 128/256/512px, 1:1/1:2/2:1) → 8×8 feature map 기준 576개 앵커

	- **앵커 처리:** 경계를 넘는 앵커는 학습에서 제외 (1000×600 이미지 기준 약 20,000개 중 약 6,000개만 학습에 사용), 테스트 시 이미지 경계로 클리핑

	- **앵커 Labeling:**

		- Positive: GT와 IoU 최고이거나 IoU ≥ 0.7

		- Negative: 모든 GT와 IoU < 0.3

		- Ignored: 위 기준 외 → 손실 계산 제외

1. **RPN Loss 및 미니배치 구성**

	- 이미지당 앵커 수가 많아(예: 576개) 음성 비율이 지나치게 높아지는 것을 방지하기 위해 이미지별 256개 앵커를 무작위 샘플링

	- 양성:음성 비율 최대 1:1 유지 (양성 128개 미만 시 음성으로 채움)

1. **RPN 최종 Proposal 생성 및 후속 처리**

	- 회귀 오프셋을 앵커 박스에 적용해 후보 영역 산출

	- NMS로 낮은 객체성 점수 및 중복 영역 제거

	- 상위 N개(예: 300개) proposal을 Fast R-CNN 검출기로 전달

	- 테스트 시 경계를 넘는 proposal은 이미지 경계로 클리핑

	- **RPN 동작 요약:** Feature Map 추출(Backbone) → Anchor 생성 및 Labeling → 3×3 슬라이딩 윈도우 → 1×1 conv 분류/회귀 분기 → Loss 계산 → 최종 Proposal 생성/후처리

	- Faster R-CNN 최종 출력: 이진 분류(obj Y/N) + BB regression(X, Y, W, H)

[//]: # (heading_4 is not supported)

- Object Detection 개요: [https://dacon.io/forum/405806](https://dacon.io/forum/405806), [https://wikidocs.net/167508](https://wikidocs.net/167508)

- 1-stage/2-stage 구조 비교: [https://www.researchgate.net/figure/Basic-deep-learning-based-one-stage-vs-two-stage-object-detection-model-architectures_fig2_358362847](https://www.researchgate.net/figure/Basic-deep-learning-based-one-stage-vs-two-stage-object-detection-model-architectures_fig2_358362847)

- COCO Dataset: [https://cocodataset.org/#home](https://cocodataset.org/#home)

- Open Images: [https://storage.googleapis.com/openimages/web/factsfigures_v7.html](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)

- Pascal VOC 2012: [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

- IoU: [https://oniss.tistory.com/36](https://oniss.tistory.com/36)

- Precision/Recall/mAP: [https://herbwood.tistory.com/2](https://herbwood.tistory.com/2)

- Precision-Recall Trade-off: [https://datascience-george.medium.com/the-precision-recall-trade-off-aa295faba140](https://datascience-george.medium.com/the-precision-recall-trade-off-aa295faba140)

- mAP 계산: [https://github.com/rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)

- Object Localization/Detection: [https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/object_localization_and_detection](https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/object_localization_and_detection)

- Region Proposal 개요: [https://velog.velcdn.com/images/qtly_u/post/c8552acf-3671-4bed-a307-d4fdde78b71f/image.webp](https://velog.velcdn.com/images/qtly_u/post/c8552acf-3671-4bed-a307-d4fdde78b71f/image.webp)

- R-CNN 논문: [https://arxiv.org/pdf/1311.2524](https://arxiv.org/pdf/1311.2524)

- R-CNN 정리: [https://wikidocs.net/148633](https://wikidocs.net/148633)

- Selective Search: [https://developer-lionhong.tistory.com/31](https://developer-lionhong.tistory.com/31), [http://www.huppelen.nl/publications/selectiveSearchDraft.pdf](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)

- Bounding Box Regression: [https://better-tomorrow.tistory.com/entry/Bounding-box-regression](https://better-tomorrow.tistory.com/entry/Bounding-box-regression)

- NMS: [https://wikidocs.net/142645](https://wikidocs.net/142645)

- Fast R-CNN 논문: [https://arxiv.org/pdf/1504.08083](https://arxiv.org/pdf/1504.08083)

- Fast R-CNN 개념: [https://seongkyun.github.io/papers/2019/01/06/Object_detection/](https://seongkyun.github.io/papers/2019/01/06/Object_detection/)

- ROI Pooling: [https://herbwood.tistory.com/8](https://herbwood.tistory.com/8)

- Faster R-CNN 논문: [https://arxiv.org/pdf/1506.01497](https://arxiv.org/pdf/1506.01497)

- Faster R-CNN 개념: [https://viso.ai/deep-learning/faster-r-cnn-2/](https://viso.ai/deep-learning/faster-r-cnn-2/)

- RPN/Anchor Box: [https://herbwood.tistory.com/10](https://herbwood.tistory.com/10)

- Faster R-CNN 정리(nttuan8): [https://nttuan8.com/bai-11-object-detection-voi-faster-r-cnn/](https://nttuan8.com/bai-11-object-detection-voi-faster-r-cnn/)

