---
notion-url: https://app.notion.com/p/Deep-CNN-Architectures-and-XAI-6fe77fa64f7f83c4b93181ad5c3bed98
title: Deep-CNN-Architectures-and-XAI
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Deep-CNN-Architectures-and-XAI-6fe77fa64f7f83c4b93181ad5c3bed98
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

1. **LeNet-5 (1998) - 현대 CNN의 시초**

	- **역사적 의의:** 손글씨 숫자(MNIST) 인식에서 98% 정확도를 달성하며 전통 머신러닝 대비 딥러닝의 우수성을 입증한 최초의 실용적 CNN 모델입니다.

	- **핵심 구조:** `Conv → Pool → Conv → Pool → FC`의 정형화된 골격을 구축하였으며, 계층 간 선택적 연결(C3-S2 구조)을 통해 파라미터 수를 억제했습니다.

1. **AlexNet (2012) - 딥러닝 혁명의 기폭제**

	- **ILSVRC 2012 성과:** 1,000개 클래스/수백만 장의 ImageNet 대회에서 기존 전통 머신러닝(SVM, Random Forest)의 오류율(26%)을 **16% 수준**으로 대폭 낮추며 압도적으로 우승했습니다.

	- **주요 기술적 혁신:**

		- **ReLU 활성화 함수 대규모 최초 적용:** Sigmoid/Tanh의 포화 현상을 없애고 연산 속도 향상 및 기울기 소실을 완화했습니다.

		- **Multiple GPU(CUDA) 병렬 학습:** 3GB 메모리의 GTX 580 GPU 2대를 활용하여 상·하단 네트워크 간 교차 통신 구조를 설계했습니다.

		- **과적합 방지 기법:** Dropout(0.5), 데이터 증강(Random Crop, Horizontal Flip, RGB 색조 변환), L2 Weight Decay(5\times10^{-4})를 도입했습니다.

	- **AlexNet 레이어 구성 명세:**

 | Layer | Input Size | Filter / Kernel Spec | Output Size | 설명 | 
 | ---- | ---- | ---- | ---- | ---- | 
 | **CONV1** | 224\times224\times3 | 96개 (11\times11, stride=4, pad=2) | 55\times55\times96 | 초기 공간 특징 추출 | 
 | **POOL1** | 55\times55\times96 | 3\times3, stride=2 (Overlapping) | 27\times27\times96 | 다운샘플링 | 
 | **NORM1** | 27\times27\times96 | LRN (Local Response Normalization) | 27\times27\times96 | 측면 억제 모방 | 
 | **CONV2** | 27\times27\times96 | 256개 (5\times5, stride=1, pad=2) | 27\times27\times256 | GPU 채널 분할 연산 | 
 | **POOL2 / NORM2** | 27\times27\times256 | 3\times3, stride=2 / LRN 적용 | 13\times13\times256 | 해상도 축소 | 
 | **CONV3** | 13\times13\times256 | 384개 (3\times3, stride=1, pad=1) | 13\times13\times384 | GPU 간 전역 연결 | 
 | **CONV4** | 13\times13\times384 | 384개 (3\times3, stride=1, pad=1) | 13\times13\times384 | 특징 고도화 | 
 | **CONV5** | 13\times13\times384 | 256개 (3\times3, stride=1, pad=1) | 13\times13\times256 | 최종 특징 맵 추출 | 
 | **POOL3** | 13\times13\times256 | 3\times3, stride=2 | 6\times6\times256 | 공간 압축 (6\times6\times256=9,216) | 
 | **FC6 / FC7** | 9,216 / 4,096 | Fully Connected + Dropout(0.5) | 4,096 / 4,096 | 고차원 특징 결합 | 
 | **FC8** | 4,096 | Fully Connected + Softmax | 1,000 | 최종 클래스 확률 예측 | 

1. **VGGNet (2014) - 단순성과 3\times3 소형 필터 중첩의 미학**

	- **핵심 철학:** 네트워크 깊이(Depth)가 시각 표현력의 핵심임을 규명하고, 모든 합성곱 층의 필터 크기를 **3\times3 (stride=1, pad=1)**로 통일했습니다.

	- **3\times3 필터 중첩의 2대 이점:**

		- **동일한 유효 수용장(Receptive Field) & 파라미터 절감:** 3\times3 Conv를 2번 쌓으면 5\times5 1번과 동일한 시야각을 가지며, 3번 쌓으면 7\times7 1번과 동일한 시야각을 가집니다. 이때 7\times7 단일 계층(7^2\times C^2 = 49C^2) 대비 3\times3 3개 계층(3\times(3^2\times C^2) = 27C^2)은 **파라미터 수를 약 45% 절감**합니다.

		- **비선형성 증가:** 각 3\times3 계층마다 ReLU가 추가되어 함수 근사 표현력이 비약적으로 상승합니다.

	- **구조적 설계 문답:**

		- *왜 마지막에 FCN(FC Layers)을 거치는가?* \rightarrow 3D 특징 맵(7\times7\times512=25,088)을 1D로 Flatten하여 전역 위치/채널 정보를 결합하고 선형 분류를 수행하기 위함입니다.

		- *왜 모든 Conv 계층마다 Max-Pooling을 하지 않는가?* \rightarrow 해상도가 너무 급격히 줄어들면 미세한 로컬 텍스처 정보가 손실되므로 `Conv-Conv-Pool` 패턴으로 계산량과 정보 보존의 균형을 유지합니다.

	- **깊은 네트워크의 장단점 요약:**

 | 구분 | 장점 (Pros) | 단점 및 한계 (Cons) | 
 | ---- | ---- | ---- | 
 | **표현력** | 계층적 비선형성을 통한 복잡한 추상 함수 근사 | 과적합 위험 증가, 대규모 정규화 필수 | 
 | **Receptive Field** | 계층 누적으로 전역 맥락(Context) 파악 용이 | 과도할 경우 그래디언트 소실 발생 | 
 | **파라미터 효율** | 소형 필터 중첩으로 대형 단일 필터 대비 파라미터 대폭 절감 | 전체 파라미터(138M) 및 FLOPs(15G)가 여전히 거대하여 추론 속도 저하 | 
 | **최적화 난이도** | 16층 \rightarrow 19층 확장 시 정확도 향상 입증 | 단순 적층 방식은 20층 이상에서 최적화 한계 봉착 | 

[//]: # (heading_4 is not supported)

1. **GoogLeNet / Inception Net (2014) - 멀티스케일 병렬 처리와 1\times1 병목 계층**

	- **개요 스펙:** 총 22층 구조, AlexNet 파라미터의 1/12 수준인 **약 6.8M 파라미터**, FC Layer를 제거하고 **Global Average Pooling (GAP)**을 도입하여 Top-5 오류율 6.7%로 우승했습니다.

	- **Inception Module 원리:** 1\times1, 3\times3, 5\times5 합성곱 및 3\times3 Max Pooling을 병렬로 동시 수행 후 채널 방향 결합(Concatenation)하여 다양한 크기의 특징을 동시 포착합니다.

	- **1\times1 Convolution 병목 계층(Bottleneck)의 연산량 절감 효과:**

		- *입력 조건:* 28\times28\times256 특징 맵 기준

		- *병목 계층 적용 전:* 총 연산량 \approx 856.00\text{M ops}

		- *병목 계층(1\times1 Conv 축소) 적용 후:* 총 연산량 \approx 273.16\text{M ops} (**연산량 약 68% 절감**)

	- **보조 분류기 (Auxiliary Classifier):** 학습 도중 중간 층(Inception 4a, 4d 뒤)에서 분기하여 손실을 계산함으로써 역전파 시 그래디언트를 주입, 기울기 소실을 방지합니다.\text{Total Loss} = \text{Main Loss} + 0.3 \times (\text{Aux1 Loss} + \text{Aux2 Loss})

1. **ResNet (2015) - 잔차 학습(Residual Learning)과 스킵 연결(Skip Connection)**

	- **핵심 질문:** "신경망은 단순히 레이어를 더 깊게 쌓기만 하면 계속 성능이 향상되는가?"

	- **Degradation Problem (퇴보 문제):** 망이 깊어질수록 오버피팅이 아니라 학습 자체가 되지 않아 Train/Test Error가 모두 치솟는 최적화 난제(Optimization Difficulty)가 발생합니다.

	- **잔차 학습(Residual Learning)의 수식적 원리:**

		- 이상적인 타겟 매핑을 H(x)라 할 때, 레이어가 직접 H(x)를 학습하는 대신 잔차 F(x) = H(x) - x를 학습하도록 유도합니다.

		- 최종 출력: y = F(x) + x (만약 H(x) = x가 최적이라면 가중치 F(x) \rightarrow 0으로 수렴하면 되므로 학습이 극도로 쉬워집니다.)

	- **ResNet이 성공하는 4대 핵심 메커니즘:**

		- **1. 항등 매핑(Identity Mapping) 초기화:** 가중치가 0에 가까워도 입력 x가 그대로 보존되어 신호 손실이 없습니다.

		- **2. 직통 경사 흐름 (\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + 1):** 역전파 시 +1 항이 항상 존재하여 그래디언트가 소실되지 않고 초기 층까지 직접 전달됩니다.

		- **3. 부드러운 손실 평면 (Smoother Loss Landscape):** 파라미터 변화에 따른 손실 곡면이 볼록(Convex)에 가깝게 완만해져 최적화가 안정화됩니다.

		- **4. 얕은 경로들의 앙상블 효과:** 스킵 연결의 조합으로 수많은 길이의 서브 네트워크 경로가 형성되어 앙상블처럼 작동합니다.

	- **심층 Bottleneck 블록 구조 (ResNet-50 / 101 / 152):**

		- `1×1 Conv` (채널 축소) \rightarrow `3×3 Conv` (특징 추출) \rightarrow `1×1 Conv` (채널 4배 복원)

		- 차원 불일치 시 1\times1 Projection Shortcut(stride=2)을 통해 공간 및 채널 차원을 보정합니다.

1. **CNN 모델 발전 계보 (Timeline)**

	- `LeNet (1998)` \rightarrow `AlexNet (2012)` \rightarrow `VGGNet / GoogLeNet (2014)` \rightarrow `ResNet (2015)` \rightarrow `DenseNet / SENet (2016~2017)` \rightarrow `EfficientNet (2019)`로 이어지며 파라미터 효율성과 깊이의 한계를 지속적으로 돌파해 왔습니다.

[//]: # (heading_4 is not supported)

1. **XAI(eXplainable AI)의 필요성과 분류**

	- **도입 목적:** 딥러닝 모델의 "블랙박스(Black-box)" 특성을 해소하여, 의료·자율주행 등 고위험 의사결정 분야에서 모델의 판단 근거를 시각적으로 검증하고 신뢰성을 확보합니다.

	- **사후 해석 (Post-hoc Interpretation):** 학습 완료된 모델의 파라미터나 그래디언트를 분석하여 예측 이유를 사후에 시각화하는 방식입니다. (예: CAM, Grad-CAM)

1. **CAM (Class Activation Mapping)**

	- **원리:** 모델의 마지막 합성곱 층 뒤에 **Global Average Pooling (GAP)**을 배치하여 각 채널별 공간 평균값 S_k를 구하고, 클래스 c로 향하는 완전연결 계층의 가중치 w_k^c와 선형 결합하여 히트맵을 생성합니다.L_{\text{CAM}}^c(x, y) = \sum_k w_k^c f_k(x, y)

	- **한계점:** 반드시 마지막 계층이 `Conv → GAP → FC` 구조여야 하므로, 전통적인 FCN 구조(VGG 등)를 가진 사전 학습 모델에 바로 적용할 수 없습니다.

1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**

	- **원리:** GAP 구조 제약을 극복하기 위해 **역전파를 통해 전달되는 대상 클래스 점수(y^c)에 대한 특징 맵(A^k)의 그래디언트**를 채널별로 전역 평균 풀링하여 가중치 \alpha_k^c를 계산합니다.\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i, j}^k}

	- **최종 히트맵 생성:** 계산된 가중치와 특징 맵을 선형 결합한 후, 해당 클래스 예측에 긍정적인 영향을 준 영역만 남기기 위해 **ReLU**를 적용합니다.L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)

	- **장점:** 마지막 Conv 층만 존재한다면 ResNet, VGG, Object Detection, VQA 등 모델 구조에 상관없이 범용적으로 적용할 수 있습니다.

1. **CAM vs Grad-CAM 핵심 비교표**

 | 비교 항목 | CAM (Class Activation Mapping) | Grad-CAM (Gradient-weighted CAM) | 
 | ---- | ---- | ---- | 
 | **구조적 의존성** | 마지막 층에 **GAP 계층이 반드시 포함**되어야 함 | **모델 구조에 제약 없음** (임의의 CNN 아키텍처 지원) | 
 | **가중치 산출 방식** | GAP 뒤 FC 계층에 **학습된 가중치(w_k^c)** 사용 | 역전파로 계산된 **그래디언트 공간 평균(\alpha_k^c)** 사용 | 
 | **적용 범위** | 제한적 (GAP 기반 모델로 재학습 필요) | 매우 범용적 (VGG, ResNet, 캡셔닝, VLM 등 즉시 적용) | 
 | **연산 복잡도** | 단순 순전파 기반 선형 결합 | 역전파 그래디언트 계산을 위한 추가 1회 연산 필요 | 

[//]: # (heading_4 is not supported)

- [AlexNet: ImageNet Classification with Deep CNNs (NIPS 2012 Paper)](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet Paper)](https://arxiv.org/abs/1409.1556)

- [Going Deeper with Convolutions (GoogLeNet Inception Paper)](https://arxiv.org/abs/1409.4842)

- [Deep Residual Learning for Image Recognition (ResNet Paper)](https://arxiv.org/abs/1512.03385)

- [Learning Deep Features for Discriminative Localization (CAM Original Paper)](https://arxiv.org/abs/1512.04150)

- [Grad-CAM: Visual Explanations from Deep Networks (ICCV 2017 Paper)](https://arxiv.org/abs/1610.02391)

- [PyTorch Grad-CAM Official Open-Source Library](https://github.com/jacobgil/pytorch-grad-cam)

