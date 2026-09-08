---
notion-url: https://app.notion.com/p/Data-Normalization-and-Batch-Normalization-e2677fa64f7f82c09c5801fb7adcb457
title: Data-Normalization-and-Batch-Normalization
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Data-Normalization-and-Batch-Normalization-e2677fa64f7f82c09c5801fb7adcb457
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

1. **데이터 정규화의 개념과 화이트 밸런스 비유**

	- **개념 정의:** 입력 이미지의 각 픽셀 값을 일정 범위로 변환하여 모델 학습을 안정화하고 일반화 성능을 향상시키는 기법입니다. 일반적으로 각 채널별 픽셀 값에서 평균(\mu)을 빼고 표준편차(\sigma)로 나누어 표준정규분포를 따르도록 변환합니다.

	- **카메라 화이트 밸런스 비유:** 카메라 촬영 시 화이트 밸런스를 맞추면 장소나 조명 환경이 달라도 피사체 고유의 색상이 틀어지지 않듯이, 입력 이미지 픽셀의 밝기 분포를 일정하게 정규화해야 모델이 조명 편향에 흔들리지 않고 핵심 시각 특징을 정확하게 추출할 수 있습니다.

1. **데이터 정규화가 필요한 핵심 이유**

	- **학습 안정성 및 수렴 가속화:** 특성(Feature) 간 스케일 불균형을 해소하여 손실 평면(Loss Surface)을 구형(Spherical)에 가깝게 만듭니다. 이에 따라 경사 하강법(Gradient Descent) 시 진동을 줄이고 전역 최소점을 향한 수렴 속도를 대폭 높입니다.

	- **기울기 폭발 및 소실(Gradient Exploding/Vanishing) 방지:** 스케일이 지나치게 큰 입력값이 전달될 때 활성화 값이 급격히 발산하거나 포화 영역(Saturation Region)에 빠지는 문제를 사전에 차단합니다.

	- **채널 간(R/G/B) 기여 균형 유지:** 특정 색상 채널의 수치가 지배적으로 작용하지 않도록 모든 채널의 스케일을 동등하게 정렬하여 공평한 가중치 업데이트가 이루어지도록 보장합니다.

1. **수식 및 PyTorch 전처리 코드**

	- **정규화 수식:**

			- **PyTorch **`**torchvision.transforms**`** 파이프라인 구현:**

		
```python
from torchvision import transforms

# ImageNet 통계 기반 정규화 파이프라인
transform = transforms.Compose([
    transforms.ToTensor(),  # PIL Image/ndarray (0~255) -> Tensor (0.0~1.0)
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # RGB 채널별 평균
        std=[0.229, 0.224, 0.225]    # RGB 채널별 표준편차
    )
])
```

[//]: # (heading_4 is not supported)

1. **배치 정규화의 정의와 내부 공변량 변화(ICS)**

	- **개념 정의:** 입력 데이터뿐만 아니라 모델 내부의 **각 은닉 레이어 출력(Activation)**을 미니배치(Mini-batch) 단위로 정규화하여 학습 속도와 모델 안정성을 극대화하는 기법입니다.

	- **내부 공변량 변화 (Internal Covariate Shift, ICS):**

		- **발생 원인:** 심층 신경망을 학습하는 도중 앞쪽 레이어의 가중치가 업데이트되면, 그 영향으로 뒤쪽 레이어가 전달받는 입력 데이터의 분포가 지속적으로 흔들리고 왜곡되는 현상입니다.

		- **해결 메커니즘:** 각 레이어의 출력 분포를 미니배치 단위로 평균 0, 분산 1로 고정하여 레이어 간의 동적 분포 변동을 억제하고 학습의 독립성을 부여합니다.

1. **배치 정규화 도입의 3대 이점**

	- **학습 안정성 향상:** 각 층의 입력 분포를 일정하게 유지함으로써 ICS를 완화하고 신경망 전반의 역전파 흐름을 견고하게 유지합니다.

	- **더 큰 학습률(Learning Rate) 적용 가능:** 기울기 폭발 및 소실 문제가 크게 줄어들므로 공격적으로 큰 학습률을 설정할 수 있어 전체 학습 시간을 단축합니다.

	- **가중치 초기화(Weight Initialization) 의존성 감소:** 초기 가중치 설정이 다소 최적화되지 않은 상태여도 정규화 계층이 출력 범위를 자동 보정하므로 다양한 초기화 기법을 유연하게 적용할 수 있습니다.

[//]: # (heading_4 is not supported)

1. **배치 정규화의 3단계 연산 프로세스**

	- **Step 1 - 미니배치 평균(\mu_B) 및 분산(\sigma_B^2) 산출:** 크기 m인 미니배치 내 각 뉴런 출력에 대해 통계량을 계산합니다.\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2

	- **Step 2 - 표준화 출력(\hat{x}_i) 계산:** 분모가 0이 되는 수치적 불안정성을 방지하기 위해 미세 상수 \epsilon을 추가하여 정규화합니다.\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}

	- **Step 3 - 학습 가능한 파라미터를 통한 스케일 및 이동 재조정:** \gamma(스케일)와 \beta(이동)를 적용하여 최종 정규화 출력 y_i를 산출합니다.y_i = \gamma \hat{x}_i + \beta

1. **학습 가능한 파라미터(\gamma, \beta)가 필수적인 이유 (표현력 복원)**

	- **표현력 저하(Representation Bottleneck) 문제:** 단순 표준화만 수행하면 모든 은닉층 뉴런의 출력이 평균 0, 표준편차 1로 획일화됩니다. 이 경우 활성화 함수(예: Sigmoid)의 선형 구간에만 갇히거나 Conv 레이어가 생성한 풍부한 비선형 특징과 다양성(색감·패턴의 dynamic range)이 상실됩니다.

	- **표현력 복원 메커니즘:** 모델이 역전파를 통해 \gamma(Scale)와 \beta(Shift)를 학습하도록 하여, 해당 층이 필요로 하는 최적의 평균과 분산 상태로 유연하게 복귀할 수 있는 여지를 제공합니다.

	- **\gamma, \beta 적용 시 ICS 제어 메커니즘:**

 | 단계 | 연산 및 역할 | ICS (내부 공변량 변화)에 미치는 영향 | 
 | ---- | ---- | ---- | 
 | **① 표준화** | \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} | 배치마다 불규칙하게 달라지는 평균과 분산을 강제로 제거하여 **ICS의 근원을 차단**합니다. | 
 | **② \gamma \cdot \beta 적용** | y_i = \gamma \hat{x}_i + \beta | 동일한 채널은 모든 미니배치에서 **동일하게 학습된 \gamma, \beta 값을 공유**하므로 배치 간 무작위 분포 변동이 재발하지 않습니다. (\gamma, \beta는 모든 샘플에 동일한 밝기·대비를 부여하는 일관된 필터 역할을 수행합니다.) | 

[//]: # (heading_4 is not supported)

- [PyTorch nn.BatchNorm2d Official Documentation](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)

- [Torchvision Transforms.Normalize Documentation](https://pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html)

- [Batch Normalization: Accelerating Deep Network Training (Original Paper, ICML 2015)](https://arxiv.org/abs/1502.03167)

- [How Does Batch Normalization Help Optimization? (NeurIPS 2018 Paper)](https://arxiv.org/abs/1805.11604)

- [CS231n Convolutional Neural Networks - Normalization Notes](https://cs231n.github.io/neural-networks-2/#batchnorm)

