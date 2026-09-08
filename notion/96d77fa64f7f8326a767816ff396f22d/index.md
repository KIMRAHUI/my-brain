---
notion-url: https://app.notion.com/p/Image-Classification-CNN-and-Transfer-Learning-96d77fa64f7f8326a767816ff396f22d
title: Image-Classification-CNN-and-Transfer-Learning
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Image-Classification-CNN-and-Transfer-Learning-96d77fa64f7f8326a767816ff396f22d
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

1. **합성곱 신경망(CNN, Convolutional Neural Network)의 정의**

	- **개념 및 구조적 특징:** 생체 시각 피질의 작동 방식에서 영감을 얻은 인공 신경망으로, 2차원/3차원 이미지 데이터의 **공간적 지역성(Spatial Locality)**과 인접 픽셀 간 상관관계를 보존하며 특징을 추출하는 "합성곱(Convolution)" 연산을 핵심으로 사용합니다.

	- **적용 도메인:** 이미지 분류, 객체 탐지(Detection), 세그멘테이션(Segmentation) 등 시각 분야는 물론 음성 인식 및 자연어 처리(NLP) 등 순차/공간 데이터 전반에 활용됩니다.

1. **합성곱 계층(Convolutional Layer)의 수치 연산과 특성 맵(Feature Map)**

	- **커널(Kernel / Filter) 슬라이딩 연산:** 작은 크기(N \times N, 예: 3\times3)의 가중치 행렬인 필터가 입력 이미지 위를 일정한 보폭(Stride)으로 이동하며, 겹치는 영역의 원소별 곱(Element-wise Multiplication)을 수행하고 이를 합산한 뒤 편향(Bias)을 더해 단일 출력값을 생성합니다.

		- *단일 필터 연산 예시:* (118\times0 + 53\times1 + 118\times1 + 53\times1 + 220\times0 + 53\times1 + 72\times1 + 237\times0 + 194\times1) = 543

		- *다중 필터 적용 예시:* 여러 필터의 연산 결과에 편향을 더해 최종 출력을 구성합니다. (예: 474 + 257 + 543 + 3(\text{Bias}) = 1277)

	- **특성 맵(Feature Map)의 정의:** 필터가 전체 이미지를 스캔하여 생성한 2차원 출력 행렬로, 외곽선(Edge), 모서리, 텍스처 등의 공간적 특징 정보를 포함합니다. 이때 **커널의 가중치 값**이 어떤 시각적 패턴을 검출할지를 결정합니다.

1. **풀링 계층(Pooling Layer)의 다운샘플링 역할**

	- **공간 크기 축소:** 합성곱 계층 뒤에 위치하여 특성 맵의 가로·세로 해상도를 축소(Max Pooling, Average Pooling)함으로써 학습 파라미터 수와 연산량을 절감합니다.

	- **위치 불변성(Translation Invariance) 확보:** 대상 객체가 이미지 내에서 미세하게 이동하거나 회전하더라도 핵심 특징 정보를 안정적으로 유지하도록 돕습니다.

[//]: # (heading_4 is not supported)

1. **초기 CNN과 딥러닝 붐의 태동 (1998~2014)**

	- **LeNet (1998):** 손글씨 숫자 인식(MNIST)을 위해 제안된 모델로, `Conv → Pool → Conv → Pool → FC` 구조를 확립하며 현대식 CNN의 기본 골격을 완성했습니다.

	- **AlexNet (2012):** ILSVRC 2012에서 압도적인 성적으로 우승하며 딥러닝 혁명을 촉발한 모델로, **ReLU 활성화 함수**, **Dropout 정규화**, **GPU 병렬 연산**을 적극 도입하여 심층망 학습의 실용성을 증명했습니다.

	- **VGGNet (2014):** 큰 필터 대신 3\times3 소형 필터만을 중첩하여 깊은 네트워크(16~19층)를 구축하는 규칙적이고 단순한 구조를 통해, 네트워크 깊이(Depth)의 중요성을 입증하고 전이학습의 표준 백본으로 장기간 활용되었습니다.

1. **구조적 다변화와 극단적 심층화 (2014~2016)**

	- **Inception / GoogLeNet (2014):** 한 층 내에서 1\times1, 3\times3, 5\times5 합성곱과 풀링을 병렬로 수행하는 **Inception 모듈**을 도입하고, 1\times1 Conv를 통한 차원 축소로 연산량을 대폭 절감했습니다.

	- **ResNet (2015):** 망이 깊어질수록 학습이 저하되는 문제를 **Residual Connection (Skip Connection / 잔차 연결)**으로 해결하여, 152층 이상의 극심층 신경망에서도 기울기 소실 없이 안정적으로 학습할 수 있는 표준 구조를 정립했습니다.

	- **DenseNet (2016):** 모든 층의 출력을 이후의 모든 층에 직접 연결하는 **Dense Connectivity** 구조를 통해 특성 재사용(Feature Reuse)을 극대화하고 파라미터 효율성을 크게 끌어올렸습니다.

1. **채널 어텐션과 모바일 경량화 및 최적화 (2017~2019)**

	- **SENet (2017):** Squeeze(전역 풀링)와 Excitation(FC) 과정을 통해 채널별 중요도를 동적으로 계산하고 가중치를 재조정하는 **채널 어텐션(Channel Attention)** 메커니즘을 도입했습니다.

	- **MobileNet (2017~):** 일반 Convolution을 공간 연산과 채널 연산으로 분리한 **Depthwise Separable Convolution**으로 대체하여, 성능 손실을 최소화하면서 연산량과 모델 크기를 대폭 줄여 모바일/엣지 환경에 최적화했습니다.

	- **EfficientNet (2019):** 네트워크의 깊이(Depth), 너비(Width), 해상도(Resolution)를 균형 있게 확장하는 **복합 스케일링(Compound Scaling)** 기법을 제안하여, 적은 연산량으로 최고 수준의 정확도를 달성했습니다.

[//]: # (heading_4 is not supported)

1. **처음부터 학습(Train from Scratch) vs 전이학습(Transfer Learning)**

	- **프로그래밍 상속 관점의 비유:** 객체 지향 프로그래밍에서 클래스 상속이 메서드(틀)만 물려받고 내부 데이터는 새로 채우는 것이라면, **전이학습은 클래스의 구조(틀)뿐만 아니라 대규모 데이터로 이미 훈련된 지식(가중치 데이터)까지 온전히 물려받아 시작하는 개념**입니다.

	- **학습 방식 비교:**

		- **처음부터 학습 (Train from Scratch):** 무작위 초기화(Random Initialization) 상태에서 보유 데이터만으로 학습합니다. 특수 도메인에 맞춘 자유로운 설계가 가능하지만 방대한 데이터와 연산 비용이 필요하며 과적합(Overfitting) 위험이 높습니다.

		- **전이학습 (Transfer Learning):** ImageNet 등 대규모 데이터셋으로 사전 훈련된(Pre-trained) 모델의 가중치를 가져와 새로운 태스크에 맞게 재활용(Fine-Tuning)합니다. 적은 데이터와 짧은 시간으로도 우수한 성능을 도출합니다.

1. **전이학습의 2대 전략 비교**

 | 구분 | Feature Extraction (특징 추출) | Fine-tuning (미세 조정) | 
 | ---- | ---- | ---- | 
 | **가중치 동결 범위** | 사전 학습된 백본(Conv 계층) 전체를 **동결(Freeze)** | 사전 학습된 백본의 **일부 또는 전체 가중치 동결 해제** | 
 | **학습 대상** | 새로 교체한 마지막 **분류기(FC Layer)만 학습** | 새로운 분류기 + 동결 해제된 상위 Conv 계층(또는 전체) 학습 | 
 | **장점** | 연산 속도가 매우 빠르고 과적합 위험이 최소화됨 | 타겟 데이터셋의 세부 도메인 특징에 맞춰 모델 최적화 가능 | 
 | **단점** | 기존 사전 학습된 특징의 표현력 한계에 종속됨 | 학습 시간이 늘어나며 데이터가 적을 경우 과적합 위험 존재 | 
 | **적합한 상황** | 타겟 데이터의 양이 적고 원본 데이터와 유사할 때 | 타겟 데이터의 양이 충분하고 세밀한 도메인 적응이 필요할 때 | 

1. **전이학습의 성공 조건 (Andrew Ng의 3대 원칙)**

	- **원칙 1:** 원본 작업(Task A)과 타겟 작업(Task B)이 **동일한 입력 형태(x, 예: 이미지 텐서)**를 공유해야 합니다.

	- **원칙 2:** 원본 작업(Task A)의 사전 학습 데이터셋 규모가 타겟 작업(Task B)의 데이터셋보다 **훨씬 더 방대**해야 합니다.

	- **원칙 3:** Task A에서 학습된 **낮은 수준의 시각적 특징(Low-level features: 에지, 질감, 모서리 등)**이 Task B를 학습하는 데 실질적인 도움을 주어야 합니다.

1. **전이학습 4단계 실행 프로세스 및 역전파(Backpropagation) 원리**

	- **1단계 - 모델 선택 및 로드:** VGGNet, ResNet, DenseNet 등 대규모 Source Dataset(ImageNet)으로 사전 학습된 검증된 백본 모델을 로드합니다.

	- **2단계 - 분류기 수정 (Classifier Modification):** 기존 모델의 최종 출력 계층(예: ImageNet 1,000개 클래스)을 잘라내고, 타겟 태스크의 클래스 수(예: 특정 객체 4종)에 맞춘 새로운 분류기(New Classifier)로 교체합니다.

	- **3단계 - 가중치 동결(Freezing) 및 역전파 흐름 제어:**

		- **가중치 동결:** 이미지의 기본 시각 특징을 추출하는 앞단 Convolution 계층의 가중치를 동결(`requires_grad = False`)하여 학습 시 업데이트를 차단합니다.

		- **역전파 작동:** 타겟 데이터 입력 후 발생한 손실(Loss)을 역전파할 때, Feature Extraction 방식에서는 동결된 Conv 계층은 건너뛰고 새롭게 추가된 분류기 영역에만 미분값을 전달해 해당 가중치만 업데이트합니다. 반면 Fine-tuning 방식은 상위 계층의 동결을 풀어 전체 또는 일부 계층에 미세하게 역전파를 흘려보냅니다.

	- **4단계 - 모델 학습 및 평가:** 타겟 데이터셋으로 순전파/역전파를 수행하며 Validation 및 Test 셋을 통해 일반화 성능을 최종 평가합니다.

[//]: # (heading_4 is not supported)

- [PyTorch Torchvision Models & Pretrained Weights Documentation](https://pytorch.org/vision/stable/models.html)

- [CS231n Convolutional Neural Networks - Transfer Learning Notes](https://cs231n.github.io/transfer-learning/)

- [Deep Residual Learning for Image Recognition (ResNet Paper)](https://arxiv.org/abs/1512.03385)

- [EfficientNet: Rethinking Model Scaling for CNNs (ICML 2019 Paper)](https://arxiv.org/abs/1905.11946)

- [Andrew Ng - When to use Transfer Learning (YouTube Lecture)](https://www.youtube.com/watch?v=yofjFQddwHE)

- [LG CNS Insight Blog - 전이학습(Transfer Learning) 개념과 원리](https://blog.lgcns.com/1563)

