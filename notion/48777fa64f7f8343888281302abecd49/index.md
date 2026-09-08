---
notion-url: https://app.notion.com/p/Computer-Vision-History-and-Milestones-48777fa64f7f8343888281302abecd49
title: Computer-Vision-History-and-Milestones
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Computer-Vision-History-and-Milestones-48777fa64f7f8343888281302abecd49
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

1. **생체 시각 메커니즘과 기하학적 형태 검출의 시작**

	- **Hubel & Wiesel의 시각 처리 메커니즘 연구 (1959):** 영장류 및 포유류의 대뇌 시각 피질을 연구하여, 고양이의 시각 뉴런이 특정 방향의 경계(Oriented edge)에 선택적으로 반응함을 규명했습니다. 이는 시각 처리가 단순한 기본 구조(Edge)에서 시작하여 점차 복잡한 형태로 통합된다는 딥러닝 계층 구조의 생물학적 기틀을 제시했습니다.

	- **Hough Transform 알고리즘 (1962):** 픽셀 좌표 공간을 매개변수 공간(Parameter space)으로 변환하여 직선, 원 등의 기하학적 형태를 효과적으로 검출하는 기법입니다. 현재까지도 자율주행의 차선 검출 및 도로 구조 인식의 핵심 전처리 알고리즘으로 널리 사용됩니다.

1. **Marr의 "Vision" 이론 (1982)**

	- **3단계 시각 처리 수준 분리:**

		- **저수준(Low-Level):** 이미지 원본에서 밝기 변화를 바탕으로 경계(Edge)를 검출하고 기본 텍스처를 분석합니다.

		- **중간수준(Mid-Level):** 경계선을 연결하고 유사 픽셀을 그룹화하여 객체의 표면 구조와 영역을 분할(Segmentation)합니다.

		- **고수준(High-Level):** 장면 전체의 맥락을 파악하고 개별 객체의 정체를 최종 인식(Recognition)합니다.

	- **3단계 시각 표현 모델의 진화:**

		- **Primal Sketch:** 경계(Edges), 막대(Bars), 끝점(Ends), 가상 선(Virtual lines), 곡선(Curves) 등 2차원 평면의 국소적 특징을 표현합니다.

		- **2.5D Sketch:** 관찰자 시점을 기준으로 표면의 방향(Surfaces), 깊이(Depth), 불연속점, 겹침 층(Layer) 정보를 복원합니다.

		- **3D Model:** 관찰자 시점에 종속되지 않고 객체의 중심축 및 입체 형상 단위(Volumetric primitives)로 구조화된 최종 3차원 입체 모델을 구축합니다.

[//]: # (heading_4 is not supported)

1. **초기 세그멘테이션 기법과 Normalized Cut (1997)**

	- **당시 접근의 한계와 의의:** 복잡한 객체 인식이 연산 한계로 인해 난항을 겪자, 픽셀 간의 유사도 그래프를 구성하여 배경과 사물을 분할하는 이미지 세그멘테이션(Normalized Cut) 연구가 선행되었습니다. 고차원적 의미 맥락 없이도 인접 픽셀의 수치적 관계만으로 유의미한 영역 분할이 가능함을 입증했습니다.

1. **강건한 수동 특징 기술자(Feature Descriptors)의 정립**

	- **SIFT (Scale-Invariant Feature Transform, 1990년대 후반):** 이미지의 크기 변화(Scale), 회전, 조명 변화, 카메라 시점 왜곡에도 불변하는 핵심 특징점(Keypoints)을 추출하여 객체 매칭 및 추적에 획기적인 안정성을 제공했습니다.

	- **HOG (Histogram of Oriented Gradients, 2000년대):** 이미지 내 국소 영역의 에지 크기와 기울기 방향 분포를 히스토그램으로 수치화하여 보행자 및 얼굴의 외형 윤곽 패턴을 성공적으로 캡처했습니다. 주로 SVM(Support Vector Machine)과 결합하여 실시간 검출 시스템을 구축했습니다.

[//]: # (heading_4 is not supported)

1. **ImageNet 프로젝트와 딥러닝 벤치마크**

	- **ImageNet (2009):** 약 1,400만 장 이상의 대규모 이미지와 1,000개 클래스로 구성된 표준 데이터셋을 구축하여, 컴퓨터 비전 연구가 알고리즘 중심에서 대규모 데이터 기반 패러다임으로 전환되는 발판을 마련했습니다.

	- **ILSVRC 대회 주요 모델 계보:**

		- **AlexNet (2012):** GPU 병렬 연산, ReLU 활성화 함수, Dropout 정규화, Data Augmentation을 적극 도입하여 ILSVRC에서 압도적인 우승을 차지하며 딥러닝 시대를 열었습니다.

		- **VGGNet (2014):** 작은 3\times3 합성곱 필터를 깊게 쌓는 균일하고 단순한 아키텍처를 통해 네트워크 깊이(Depth)가 표현력에 미치는 결정적 영향을 규명했습니다.

		- **ResNet (2015):** Residual Block(Skip Connection, 잔차 연결)을 도입하여 심층 신경망 학습의 고질적 난제였던 기울기 소실(Vanishing Gradient) 문제를 근본적으로 해결했습니다.

1. **Vision Transformer(ViT) 및 멀티모달·자기지도학습으로의 확장**

	- **Vision Transformer (ViT, 2020):** 이미지를 패치(Patch) 단위로 분할하여 자연어 처리의 트랜스포머 아키텍처에 직접 입력함으로써, CNN 중심 구조를 탈피하고 대규모 데이터셋 기반 사전 학습의 우수성을 입증했습니다.

	- **멀티모달 AI 및 자기지도학습(Self-Supervised Learning):** 텍스트와 이미지를 공동 임베딩 공간에 매핑하는 대조 학습 모델(CLIP)과 이미지 생성 모델(DALL-E), 레이블 없는 방대한 원본 데이터를 활용하는 SimCLR, MAE 등이 현대 비전 연구의 주축을 이루고 있습니다.

1. **역사와 현재의 연결점 (Core Takeaways)**

	- Hough Transform은 최신 딥러닝 파이프라인의 고속 기하학적 전처리 기법으로 계승되었습니다.

	- Marr의 다단계 비전 처리 이론은 CNN의 계층적 특징 추출(Low-level Edge \rightarrow High-level Semantic Feature) 개념으로 발전했습니다.

	- SIFT/HOG의 불변성 탐색 원리는 현대 딥러닝의 데이터 증강 전략과 초기 특징 표현 방식의 이론적 토대가 되었습니다.

	- ImageNet 체계는 현대 인공지능 전반의 데이터 중심 접근(Data-Centric AI)과 표준 평가 지표의 기반이 되었습니다.

[//]: # (heading_4 is not supported)

- [SimCLR Official GitHub Repository](https://github.com/google-research/simclr)

- [Masked Autoencoders (MAE) Official GitHub Repository](https://github.com/facebookresearch/mae)

- [ImageNet Official Benchmark Database](https://www.image-net.org/)

- [AlexNet Paper - NIPS 2012](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

- [Deep Residual Learning for Image Recognition (ResNet) Paper](https://arxiv.org/abs/1512.03385)

