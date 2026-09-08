---
notion-url: https://app.notion.com/p/Object-detection_part2-a5e77fa64f7f820ba88901e32ac15a2d
title: Object detection_part2
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Object-detection_part2-a5e77fa64f7f820ba88901e32ac15a2d
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

1. **YOLO v1 (You Only Look Once)**

	- 기존 방식이 여러 영역을 분리해 개별적으로 분류·예측하는 반면, YOLO는 전체 이미지를 한 번만 보고 위치와 클래스를 동시에 예측

	- **접근법:** 단일 회귀 문제로 재정의 / End-to-End 학습 / 기본 모델 45FPS, Fast YOLO 155FPS의 실시간 성능

	- 논문: *"You Only Look Once: Unified, Real-Time Object Detection"* (arXiv:1506.02640)

	- **기존 방식의 한계:** Region Proposal 기반 검출(R-CNN 등)은 여러 단계 파이프라인으로 속도가 느리고 시스템 복잡도가 높음

	- **YOLO 접근:** 이미지를 S×S 그리드로 분할, 각 셀이 해당 영역의 객체를 예측 / 단일 신경망이 좌표·신뢰도·클래스 확률을 동시 예측 / 전체 이미지를 보므로 전역 문맥 반영

1. **YOLO v1 구조**

	- **입력:** 448×448로 리사이즈된 이미지

	- **출력:** S×S 그리드, 각 셀이 B개의 박스와 5개 값(x, y, w, h, 신뢰도) + C개 클래스 확률 예측

		- 예: PASCAL VOC(S=7, B=2, C=20) → 최종 출력 7×7×30 텐서

	- **모델 구조:** 약 24개 컨볼루션 계층(DarkNet) + 2개 완전연결 계층(최종 출력 1470) / ImageNet 사전학습 후 전이학습

	- **손실 함수:** 전체 손실 = Localization Loss + Confidence Loss + Classification Loss

	- **NMS:** 클래스별 개별 수행, Object Confidence와 IoU 임계값으로 결과 필터링

1. **YOLO v1 장단점**

 | 구분 | 내용 | 
 | ---- | ---- | 
 | **장점** | 빠른 처리 속도, 단순한 단일 네트워크 구조 / 전역 문맥 반영 / 뛰어난 도메인 일반화 능력 | 
 | **한계** | 단일 회귀 방식으로 위치 예측 정밀도 부족 / 그리드 셀당 예측 가능 객체 수 제한 / 작은 객체에서 상대적으로 큰 오차 | 

1. **SSD (Single Shot MultiBox Detector)**

	- 객체 위치와 카테고리를 한 번의 네트워크 전파로 동시 예측, 다중 스케일 피처 맵과 기본 박스 전략으로 실시간+고정확도 달성

	- **주요 특징:** 후보 영역 생성 단계 없음(빠른 추론) / 다중 스케일 예측(여러 해상도 피처 맵 활용) / Default Boxes(피처 맵 셀마다 다양한 크기·비율의 박스 배치)

	- 논문: *"SSD: Single Shot MultiBox Detector"* (arXiv:1512.02325)

	- **핵심 아이디어:** 단일 단계 처리로 연산량·시간 절감 / 초기 계층(고해상도)은 작은 객체, 후반 계층(추상화)은 큰 객체 특징 포착 / 기본 박스마다 클래스 점수(존재 확률)와 위치 오프셋(조정값)을 동시 예측

1. **SSD – Default Boxes & Feature Map별 스케일**

	- 여러 해상도의 피처 맵(conv4_3, conv7 등)마다 기본 박스를 두고 오프셋·클래스 점수 예측

	- 피처 맵 레벨별로 수용 필드 크기가 달라, 앵커 박스도 특정 스케일 객체에만 책임지도록 재조정 (예: 개는 4×4 상위 레벨, 고양이는 8×8 하위 레벨에서 검출)

	- 작은 객체는 해상도가 높은 초반 계층, 큰 객체는 해상도가 낮은 후반 계층에서 검출

	- 최종 출력 채널 수 = k × (C + 4) (k: 기본 박스 수, C: 클래스 수, 4: 좌표)

1. **SSD Loss**

	- GT 박스와 기본 박스 매칭으로 양성/음성 샘플 구성

	- **Localization Loss:** Smooth L1 Loss로 중심 좌표·폭·높이 회귀

	- **Confidence Loss:** Softmax Loss로 다중 클래스 분류 손실 최소화

1. **SSD 장단점**

 | 구분 | 내용 | 
 | ---- | ---- | 
 | **장점** | 후보 영역 생성 단계 없어 실시간 검출 가능 / End-to-End 구조로 구현·유지보수 용이 / 다중 스케일 처리로 다양한 크기 객체 검출 | 
 | **단점** | 상위 계층 피처 맵의 정보 부족으로 작은 객체 검출 성능 저하 (데이터 증강·박스 타일링 개선 필요) | 

1. **YOLO v2**

	- 실시간 감지에서 속도와 정확도를 동시에 향상하는 것이 목표

	- **주요 특징:** 배치 정규화 / 고해상도 분류기(224→448px fine-tune) / 앵커 박스 + 차원 클러스터링(k-means) / 직접 위치 예측(그리드 셀 기준 상대 좌표) / 멀티 스케일 학습 / Darknet-19 백본

	- 논문: *"YOLO9000: Better, Faster, Stronger"* (arXiv:1612.08242)

1. **YOLO v2 개선 단계별 성능 (VOC mAP)**

 | 개선 단계 | 내용 | mAP | 
 | ---- | ---- | ---- | 
 | 기준(YOLOv1) | Darknet-19 이전, FC 출력 | 63.4 | 
 | + Batch Norm | 모든 conv 뒤 BN 삽입, 학습 안정·과적합↓ | 65.8 (+2.4) | 
 | + Hi-res Classifier | 224→448px fine-tune | 69.5 | 
 | + Fully Convolutional | FC 제거, 전층 conv | 69.2 | 
 | + Anchor Boxes | k-means anchor, recall↑ | 69.6 | 
 | + New Network (Darknet-19) | 경량 19-layer convnet | 74.4 | 
 | + Dimension Priors | k-means 최적 anchor 5개 | 75.4 | 
 | + Direct Location Prediction | σ(tx,ty)+셀오프셋, 수렴↑ | 76.8 | 
 | + Passthrough Layer | 26×26→13×13 concat, 작은 물체 recall↑ | 78.6 | 
 | + Multi-Scale Training | 320~608px 랜덤 | 78.6 (속도 유연) | 


```plain text
- **차원 클러스터(Dimension Clusters):** 앵커 박스 크기를 수작업 대신 k-평균 군집화로 자동 산출
- **Fine-Grained Features:** 13×13 특징 맵의 한계를 보완하기 위해 26×26 저층 특징을 가져오는 passthrough layer 추가
- **Darknet-19:** 19개 conv + 5개 pooling 계층, ImageNet Top-1 72.9%/Top-5 91.2%
- **동작 원리:** 이미지 분할(그리드) → 앵커 기반 박스 예측 → NMS로 최종 감지
- PASCAL VOC2012 테스트 기준 YOLOv2는 Faster R-CNN(ResNet)·SSD512와 대등한 성능이면서 2~10배 빠름
```

1. **YOLO v2 장단점**

 | 구분 | 내용 | 
 | ---- | ---- | 
 | **장점** | 높은 FPS의 실시간 처리 / 배치정규화·앵커박스 등으로 높은 mAP / Darknet-19 경량 네트워크 / 멀티 스케일로 유연한 입력 크기 지원 | 
 | **한계** | 앵커 박스 도입으로 mAP 소폭 감소 가능(재현율과 trade-off) / 여러 기법 결합으로 학습 과정 복잡, 하이퍼파라미터 조정 중요 | 

1. **YOLO 9000**

	- YOLOv2 기반, 감지·분류 데이터를 동시 학습해 9000개 이상의 객체 범주를 실시간 감지

	- **핵심 아이디어:** 공동 학습(Joint Training) – 감지 이미지엔 전체 손실, 분류 이미지엔 분류 손실만 적용 / 계층적 분류(WordTree) – WordNet 기반 계층 구조로 서로 다른 데이터셋 라벨 통합

	- **해결 문제:** 분류 데이터(ImageNet, 세부 품종)와 감지 데이터(COCO, 상위 범주)의 클래스 계층 불일치 → 단일 softmax의 상호 배타성 가정 문제

	- **WordTree:** "물리적 객체"→"동물"→"개"→각 품종으로 이어지는 트리, 조건부 확률의 곱으로 최종 확률 산출

1. **YOLO 9000 장단점**

 | 구분 | 내용 | 
 | ---- | ---- | 
 | **장점** | 9000개 이상 클래스로 확장된 인식 범위 / YOLOv2 구조 유지로 실시간 처리 가능 / WordTree로 불확실성 완화 | 
 | **단점** | 감지·분류 손실 간 균형 문제 / 계층적 확률 계산의 복잡성 / 라벨 통합 과정에서의 모호성으로 성능 일부 저하 가능 | 

1. **YOLOv3**

	- YOLO 시리즈 3세대, 기본 검출 성능에 집중하며 다중 라벨 분류 방식 유지

	- 다중 스케일 예측(3가지 해상도), Darknet-53 백본(더 깊고 잔차 연결 도입)

	- **Backbone/Neck/Head 역할**

		- **Backbone:** 기본 시각 특징 추출 (ResNet, Darknet 등)

		- **Neck:** 여러 수준 특징 결합 (FPN, PANet 등) → 다중 스케일 검출에 효과적

		- **Head:** 최종 바운딩 박스·객체 존재 여부·클래스 확률 예측

1. **FPN (Feature Pyramid Network)**

	- **문제제기:** 다양한 크기의 객체 인식은 핵심 과제이나, 기존 방식은 속도 저하·메모리 과다 사용 문제 존재

	- **핵심 아이디어:** 상위 계층(의미 정보 강함)을 하향식(top-down)으로 전파하며 하위 계층(고해상도)과 lateral connection으로 결합 → 모든 스케일에서 강한 의미 정보를 갖는 특징 피라미드 생성

	- **구조:**

		- **Bottom-Up Pathway:** 여러 계층을 거치며 해상도↓, 시맨틱 정보↑

		- **Top-Down Pathway:** 상위 계층에서 업샘플링하며 해상도 복원

		- **Lateral Connections:** 1×1 conv로 차원을 맞춘 후 특징 결합

	- 업샘플링은 보통 2배 Nearest Neighbor 방식(계산량 적고 구현 용이)

	- **적용 사례:** Faster R-CNN + FPN(작은 객체 검출 향상), RetinaNet(FPN 백본으로 단일 단계에서도 고정확도)

1. **YOLOv3 네트워크 구조**

	- **Backbone – Darknet-53:** 잔차 연결로 기울기 소실 방지

	- **Detection Head:** 3개의 서로 다른 스케일에서 바운딩 박스·클래스 예측

	- **앵커 박스:** 총 9개(스케일당 3개), k-means로 산출 후 큰 순서대로 스케일에 배분

	- **예측 방식:** 객체 존재 여부는 로지스틱 회귀 / 클래스 예측은 softmax 대신 독립 로지스틱 분류기 사용(다중 라벨 예측 가능 – 예: "사람"과 "운동선수" 동시 할당)

1. **YOLOv1 / v2 / v3 비교**

 | 특징 | YOLOv1 | YOLOv2 | YOLOv3 | 
 | ---- | ---- | ---- | ---- | 
 | 출시년도 | 2016 | 2016/2017 (YOLO9000 포함) | 2018 | 
 | Backbone | 커스텀 네트워크 | Darknet-19 | Darknet-53 | 
 | 예측 방식 | 그리드 셀 직접 회귀 | 앵커 박스 + 차원 클러스터링, 단일 스케일 | 다중 스케일(3종) + 독립 로지스틱 분류기 | 
 | 주요 개선점 | 실시간 처리, 단순 구조 | 앵커 박스·배치정규화, YOLO9000 클래스 확장 | 다중 스케일로 작은 객체 개선, 깊은 네트워크, 모듈화(Backbone/Neck/Head) | 
 | Grid당 Anchor 수 | 2개 | 5개 | 3개 × 3스케일 = 9개 | 
 | Anchor 결정법 | - | K-Means | K-Means | 

1. **YOLO 시리즈 계보**

	- **Joseph Redmon(창시자):** YOLO 시리즈의 초석, 윤리적 고민으로 2020년 컴퓨터 비전 연구 은퇴

	- **YOLOv4:** Alexey Bochkovskiy 주도, Bag of Freebies & Bag of Specials로 성능 강화

	- **YOLOv5:** Ultralytics의 PyTorch 기반, 다양한 사이즈 모델 제공

	- **YOLOv6~v11:** 각기 다른 팀이 성능·효율성·확장성 개선, NMS-free 훈련 등 최신 기법 도입

[//]: # (heading_4 is not supported)

- YOLO v1 논문: [https://arxiv.org/pdf/1506.02640](https://arxiv.org/pdf/1506.02640)

- YOLO v1 구조: [https://medium.com/@saptarshimt/yolo-v1-pascal-voc-simplistic-pytorch-implementation-from-scratch-961fa36f4d4d](https://medium.com/@saptarshimt/yolo-v1-pascal-voc-simplistic-pytorch-implementation-from-scratch-961fa36f4d4d)

- SSD 논문: [https://arxiv.org/pdf/1512.02325](https://arxiv.org/pdf/1512.02325)

- SSD 개념/스케일: [https://lilianweng.github.io/posts/2018-12-27-object-recognition-part-4/](https://lilianweng.github.io/posts/2018-12-27-object-recognition-part-4/)

- SSD 출력 채널: [https://wikidocs.net/142280](https://wikidocs.net/142280)

- YOLO v2 논문: [https://arxiv.org/pdf/1612.08242](https://arxiv.org/pdf/1612.08242)

- YOLO v2 앵커/구조: [https://wikidocs.net/167664](https://wikidocs.net/167664)

- YOLO 9000 해설: [https://ffighting.net/deep-learning-paper-review/object-detection/yolo-v2/](https://ffighting.net/deep-learning-paper-review/object-detection/yolo-v2/)

- FPN 개념: [https://herbwood.tistory.com/18](https://herbwood.tistory.com/18)

- YOLOv3 구조: [https://wikidocs.net/163583](https://wikidocs.net/163583), [https://wikidocs.net/163607](https://wikidocs.net/163607)

- YOLOv3 로지스틱 분류: [https://web.stanford.edu/~nanbhas/blog/sigmoid-softmax/](https://web.stanford.edu/~nanbhas/blog/sigmoid-softmax/)

- YOLO 시리즈 계보: [https://devocean.sk.com/blog/techBoardDetail.do?ID=166976](https://devocean.sk.com/blog/techBoardDetail.do?ID=166976)

