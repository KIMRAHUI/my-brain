---
notion-url: https://app.notion.com/p/Deep-Learning-Hyperparameters-PyTorch-Tensors-3c277fa64f7f80529756dd3d3691a857
title: Deep Learning Hyperparameters & PyTorch Tensors
date: '2026-08-20 09:42:00.000'
from_notion: https://app.notion.com/p/Deep-Learning-Hyperparameters-PyTorch-Tensors-3c277fa64f7f80529756dd3d3691a857
author: rahui kim
last_edited_time: '2026-08-20 09:42:00.000'
---
[//]: # (heading_4 is not supported)

- **개념**: 입력 데이터(X)가 신경망의 여러 은닉층(Hidden Layers)을 거치면서 가중치(W)와 편향(b)에 의해 계산되고, 활성화 함수(Activation Function)를 지나 **최종 예측값(\hat{y})에 도달하는 과정입니다.

- **동작 흐름**:

	1. 입력층 \rightarrow 은닉층: 입력 데이터에 가중치를 곱하고 편향을 더함 (z = WX + b)

	1. 활성화 함수 적용: 비선형성을 부여하여 복잡한 패턴을 학습 (a = \sigma(z))

	1. 손실(Loss) 계산: 모델의 예측값(\hat{y})과 실제 정답(y)을 손실 함수(Loss Function)에 전달하여 오차(Error) 수치화

- **목적**: 현재 모델의 가중치 상태로 입력 데이터에 대한 예측 및 오차를 측정하는 것

---

[//]: # (heading_4 is not supported)

- **개념**: 순전파에서 계산된 손실(오차)을 신경망의 반대 방향(출력층 \rightarrow 입력층)으로 전파하면서, 각 가중치가 오차에 얼마나 기여했는지(기울기/Gradient)를 계산하는 과정입니다.

- **동작 흐름**:

	1. **오차 측정**: 출력층에서 발생한 오차 계산

	1. **미분과 체인룰(Chain Rule)**: 미분의 연쇄 법칙을 이용하여 출력층부터 입력층 방향으로 각 가중치에 대한 손실 함수의 편미분값(\frac{\partial Loss}{\partial W})을 산출

	1. **가중치 갱신**: 경사 하강법(Gradient Descent)을 통해 오차를 줄이는 방향으로 가중치 업데이트 (W_{new} = W - lr \times \frac{\partial Loss}{\partial W})

- **목적**: 계산된 기울기(Gradient)를 바탕으로 오차를 최소화하는 최적의 가중치를 찾아가는 것

---

[//]: # (heading_4 is not supported)

 | 분류 | 용어 (Term) | 상세 설명 | 
 | ---- | ---- | ---- | 
 | **학습 기본 설정** | `epochs` (에포크) | 전체 학습 데이터셋이 신경망을 순전파와 역전파로 완전히 통과하여 학습되는 **전체 횟수**입니다. | 
 |  | `batch_size` (배치 크기) | 메모리 한계와 효율적인 가중치 업데이트를 위해 한 번에 처리하는 **데이터 묶음의 개수**입니다. | 
 |  | `iterations` (이터레이션) | 1 에포크를 완수하기 위해 배치를 처리하는 **반복 횟수**입니다. (\text{전체 데이터 수} \div \text{배치 크기}) | 
 | **최적화 및 제어** | `lr` / `learning_rate` (학습률) | 역전파 시 기울기(Gradient) 방향으로 가중치를 업데이트할 때 이동할 **보폭의 크기**를 조절합니다. | 
 |  | `optimizer` (옵티마이저) | 손실 함수의 최솟값을 효율적으로 찾기 위해 가중치 업데이트 방식을 결정하는 알고리즘입니다. (예: `SGD`, `Adam`) | 
 |  | `loss` / `criterion` (손실 함수) | 순전파 결과인 모델의 예측값과 실제 정답 사이의 오차를 수치화하는 함수입니다. (예: `MSELoss`, `CrossEntropyLoss`) | 
 | **기타 관리** | `weight_decay` (가중치 감소) | 모델이 훈련 데이터에 과적합(Overfitting)되지 않도록 가중치가 너무 커지는 것을 억제하는 **L2 규제(L2 Regularization)** 기법입니다. | 
 |  | `step_size` / `scheduler` (학습률 스케줄러) | 학습 진행 상황에 따라 학습률을 동적으로 줄여 최적의 손실 지점에 안정적으로 도달하도록 돕습니다. | 

---

[//]: # (heading_4 is not supported)

 | 분류 | 용어 / 코드 (Term) | 설명 및 의미 | 
 | ---- | ---- | ---- | 
 | **NumPy 연산** | `**.T**`** (Transpose, 전치)** | 행렬의 행과 열을 맞바꿉니다. 행렬 곱연산 시 차원 크기(Shape) 맞춤 규칙을 위해 필수적으로 쓰입니다. | 
 | **NumPy 연산** | `**.dot()**`** (행렬 곱셈)** | 두 행렬의 내적(Dot Product)을 계산합니다. 순전파 시 입력과 가중치의 결합(W \cdot X), 역전파 시 기울기 전파에 쓰입니다. | 
 | **딥러닝 함수** | `**sigmoid_derivative**`** (시그모이드 미분)** | 시그모이드 활성화 함수의 도함수입니다. 역전파 시 체인룰에 의해 출력 오차 기울기와 곱해져 이전 층으로 전달됩니다. | 
 | **가중치 갱신** | `**hidden_layer_activations.T.dot(d_output) * lr**` | 은닉층 출력 전치값(`.T`)과 출력층 오차 기울기(`d_output`)를 행렬곱하여 최종 가중치 오차 기울기를 구하고, 학습률(`lr`)을 곱해 가중치를 갱신하는 공식입니다. | 

---

[//]: # (heading_4 is not supported)

 | 분류 | 속성 (Attribute) / 메서드 | 설명 | 
 | ---- | ---- | ---- | 
 | **배열 기본 정보** | `.shape` | 텐서의 차원 크기를 반환합니다. (예: `torch.Size([2, 3])`) | 
 |  | `.dtype` | 텐서 내 요소의 데이터 타입을 반환합니다. (예: `torch.float32`, `torch.int64`) | 
 |  | `.device` | 텐서 연산이 수행되는 장치를 반환합니다. (`cpu` 또는 `cuda`) | 
 | **형태 변환** | `.reshape(*shape)` / `.view(*shape)` | 메모리 구조 및 데이터를 유지하며 텐서의 **차원(Shape)**을 변환합니다. | 
 |  | `.squeeze()` | 텐서의 차원 중 크기가 1인 차원을 모두 제거하여 차원을 축소합니다. | 
 |  | `.unsqueeze(dim)` | 지정한 위치(`dim`)에 크기가 1인 차원을 새롭게 추가합니다. | 
 | **타입 및 값 변환** | `.item()` | 단 하나의 요소를 가진 스칼라 텐서의 값을 **파이썬 기본 숫자 타입**으로 추출합니다. | 
 |  | `.numpy()` | PyTorch 텐서를 NumPy `ndarray`로 변환합니다. (CPU 메모리에 위치한 텐서만 가능) | 
 |  | `.tolist()` | 텐서의 모든 요소를 파이썬 리스트(List) 구조로 변환합니다. | 
 | **기울기 (Autograd)** | `.requires_grad` | 해당 텐서에 대한 연산 그래프 추적 및 **자동 미분(Gradient 계산)** 여부를 지정합니다 (`True`/`False`). | 
 |  | `.grad` | 역전파(`backward()`) 수행 시 해당 텐서에 대해 계산된 **기울기(Gradient) 값**이 저장되는 속성입니다. | 
 |  | `.detach()` | 연산 그래프 추적에서 분리된 새로운 텐서를 생성하여 기울기 계산을 중단시킵니다. | 
 | **디바이스 제어** | `.to(device)` | 텐서를 지정한 연산 장치(CPU 또는 GPU 디바이스)로 복사/이동합니다. | 
 |  | `.cpu()` | GPU에 위치한 텐서를 CPU 메모리로 이동합니다. | 
 |  | `.cuda()` | 텐서를 GPU(CUDA) 메모리로 이동하여 가속 연산을 준비합니다. | 

---

[//]: # (heading_4 is not supported)

- [PyTorch Official Documentation - Tensor Basics](https://pytorch.org/docs/stable/tensors.html)

- [PyTorch Official Tutorial - Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

- [PyTorch Official Guide - Automatic Differentiation with ](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)`[torch.autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)`

- [CS231n: Convolutional Neural Networks for Visual Recognition - Backpropagation Guide](https://cs231n.github.io/optimization-2/)

