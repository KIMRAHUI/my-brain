---
notion-url: https://app.notion.com/p/Torch-nn-Basics-3c277fa64f7f809f80bed3ba26a10a94
title: Torch-nn-Basics
date: '2026-08-20 09:33:00.000'
from_notion: https://app.notion.com/p/Torch-nn-Basics-3c277fa64f7f809f80bed3ba26a10a94
author: rahui kim
last_edited_time: '2026-08-20 09:39:00.000'
---
[//]: # (heading_4 is not supported)


```python
import torch
import torch.nn as nn  # neural network
```

[//]: # (heading_4 is not supported)

1. **레이어(Layer):** Fully Connected(`Linear`), Convolution, LSTM 등

1. **활성화 함수(Activation Function):** ReLU, Sigmoid, Softmax 등

1. **손실 함수(Loss Function):** MSE, CrossEntropy 등

[//]: # (heading_4 is not supported)


```python
class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = torch.randn(input_dim, output_dim) * 0.01
        self.bias = torch.zeros(output_dim)

    def forward(self, x):
        return x @ self.weights + self.bias

layer = FullyConnectedLayer(8, 4)

x = torch.randn(2, 8)
y = layer.forward(x)
print(y)
```

- 가중치(weight)와 편향(bias)을 직접 정의합니다.

- 입력 데이터를 받아 연산을 수행하는 `forward` 메서드를 직접 구현합니다.

- 이렇게 직접 만들면 가중치 초기화, 그래디언트 관리 등을 전부 수동으로 해야 해서 번거롭습니다.

[//]: # (heading_4 is not supported)


```python
layer = nn.Linear(in_features=8, out_features=4)

x = torch.randn(2, 8)
y = layer(x)
print(y)
```

- `nn.Linear`로 간단히 구현 가능합니다.

- 입력 차원(`in_features`)과 출력 차원(`out_features`)만 설정하면 됩니다.

- 텐서를 레이어 객체에 직접 입력하면(`layer(x)`) 자동으로 `forward` 연산이 수행되어 결과가 계산됩니다.

- 가중치 초기화, `requires_grad` 설정 등을 PyTorch가 알아서 처리해줍니다.

[//]: # (heading_4 is not supported)


```python
relu = nn.ReLU()
softmax = nn.Softmax(dim=1)
```

- 자주 사용하는 활성화 함수도 `nn`에서 객체로 제공합니다.

- `**ReLU**`**:** 0 이상의 값은 그대로 출력하고, 0 미만인 값은 0으로 바꿔서 출력합니다.

- `**Softmax**`**:** 출력 값들을 확률 분포(합이 1)로 변환합니다. `dim` 인자로 어떤 차원을 기준으로 정규화할지 지정합니다.


```python
x = torch.randn(1, 4)
y = relu(x)
print(f'Input: {x}')
print(f'Output: {y}')

x = torch.randn(1, 4)
y = softmax(x)
print(f'Input: {x}')
print(f'Output: {y}')  # 4개 값의 합이 1이 됨
```

[//]: # (heading_4 is not supported)


```python
loss_mse = nn.MSELoss()
loss_ce = nn.CrossEntropyLoss()
```

- 딥러닝에서 자주 쓰는 손실 함수도 `nn` 모듈에서 객체 형태로 바로 사용할 수 있습니다.

- 자세한 손실 함수 종류와 사용법은 [Training-Loop-and-Optimizer](https://app.notion.com/p/Training-Loop-and-Optimizer.md) 노트에서 다룹니다.

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
tensor_0d = torch.tensor(5)                      # 스칼라
tensor_1d = torch.tensor([1, 2, 3])               # 벡터
tensor_2d = torch.tensor([1, 2], [3, 4](./1, 2], [3, 4.md))         # 행렬
tensor_3d = torch.tensor([[1], [2](./[1], [2.md), [3], [4](./3], [4.md)]) # 3차원 텐서

print(f'0차원 텐서 차원: {tensor_0d.ndim}, size: {tensor_0d.size()}')
print(f'1차원 텐서 차원: {tensor_1d.ndim}, size: {tensor_1d.size()}')
print(f'2차원 텐서 차원: {tensor_2d.ndim}, size: {tensor_2d.size()}')
print(f'3차원 텐서 차원: {tensor_3d.ndim}, size: {tensor_3d.size()}')
```

- 이미지 데이터는 (채널, 높이, 너비)로 구성된 3차원 데이터입니다.

- PyTorch에서 실제로 사용하는 이미지 텐서는 (배치, 채널, 높이, 너비)로 구성된 4차원 텐서, 즉 **BCHW**(Batch, Channels, Height, Width) 형태입니다.

- 예: 크기 `(3, 224, 224)`의 RGB 이미지 한 장 → 배치로 32장을 묶으면 `(32, 3, 224, 224)` 텐서가 됩니다.

[//]: # (heading_4 is not supported)


```python
layer = nn.Linear(in_features=8, out_features=4)

x = torch.randn(1, 8)  # 입력 텐서 (배치 크기 1, 입력 차원 8)
y = layer(x)            # 출력 텐서 (배치 크기 1, 출력 차원 4)

print(f'입력 텐서 차원: {x.ndim}, size: {x.size()}')  # 2, [1, 8]
print(f'출력 텐서 차원: {y.ndim}, size: {y.size()}')  # 2, [1, 4]
```

- **텐서의 차원(ndim):** 입력 `x`와 출력 `y`는 모두 2차원 텐서(`ndim=2`)입니다. 이는 그대로입니다.

- **레이어의 입출력 차원:** 입력 텐서의 두 번째 차원 크기는 8(`in_features`), 출력 텐서의 두 번째 차원 크기는 4(`out_features`)입니다. 즉, "레이어의 차원"이 바뀐 것은 텐서의 축(ndim) 개수가 아니라, 배치를 제외한 마지막 축의 **피처 개수**입니다.

[//]: # (heading_4 is not supported)

 | 구분 | 의미 | 확인 방법 | 
 | ---- | ---- | ---- | 
 | 텐서의 차원 | 다차원 배열의 축(axis) 개수 | `.ndim` 속성 | 
 | 레이어의 입출력 차원 | 처리하는 텐서의 피처(feature) 개수 | `in_features`, `out_features` | 

[//]: # (heading_4 is not supported)

- "차원을 늘린다"는 말이 텐서의 `ndim`을 늘리는 건지, `nn.Linear`의 `out_features`를 늘리는 건지 헷갈려서 shape 에러가 자주 발생합니다. 에러 메시지를 볼 때 항상 두 개념을 구분해서 생각하는 것이 중요합니다.

- `nn.Softmax(dim=1)`에서 `dim`을 잘못 지정하면(예: `dim=0`) 배치 방향으로 정규화가 되어 원하는 결과가 나오지 않습니다. 보통 (배치, 클래스) 형태의 2차원 텐서라면 `dim=1`(클래스 방향)을 지정합니다.

---

[//]: # (heading_4 is not supported)

- [PyTorch-Tensor-Data-Modeling-MOC](https://app.notion.com/p/PyTorch-Tensor-Data-Modeling-MOC.md)

- [nn-Module-Model-Building](https://app.notion.com/p/nn-Module-Model-Building.md)

- [Training-Loop-and-Optimizer](https://app.notion.com/p/Training-Loop-and-Optimizer.md)

- [Autograd-Gradient-Computation](https://app.notion.com/p/Autograd-Gradient-Computation.md)

