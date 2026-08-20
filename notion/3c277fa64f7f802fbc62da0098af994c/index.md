---
notion-url: https://app.notion.com/p/Autograd-Gradient-Computation-3c277fa64f7f802fbc62da0098af994c
title: Autograd-Gradient-Computation
date: '2026-08-20 09:32:00.000'
from_notion: https://app.notion.com/p/Autograd-Gradient-Computation-3c277fa64f7f802fbc62da0098af994c
author: rahui kim
last_edited_time: '2026-08-20 09:33:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
# 생성 시 설정
x = torch.tensor([2.0], requires_grad=True)
print(x.requires_grad)  # True

# 이미 만든 텐서의 속성을 나중에 바꾸기
x = torch.tensor([2.0])
x.requires_grad_()        # True로 변경 (밑줄 in-place 메서드)
x.requires_grad_(False)   # 다시 False로 변경
```

- 기본값은 `requires_grad=False`입니다.

- 딥러닝 모델의 학습 가능한 파라미터(가중치, 편향)에는 보통 `requires_grad=True`가 자동으로 설정되어 있습니다.

[//]: # (heading_4 is not supported)


```python
import torch
from torch.autograd import grad

x1 = torch.tensor(2, requires_grad=True, dtype=torch.float16)
x2 = torch.tensor(3, requires_grad=True, dtype=torch.float16)
x3 = torch.tensor(1, requires_grad=True, dtype=torch.float16)
x4 = torch.tensor(4, requires_grad=True, dtype=torch.float16)

z1 = x1 * x2
z2 = x3 * x4
f = z1 + z2   # f = (x1 * x2) + (x3 * x4)

df_dx = grad(outputs=f, inputs=[x1, x2, x3, x4])
print(df_dx)
# (tensor(3.), tensor(2.), tensor(4.), tensor(1.))
```

- `torch.autograd.grad`는 함수 `f`의 결과(outputs)에 대해 지정된 입력 변수들(inputs) 각각의 기울기를 반환합니다.

- 반환값은 **튜플** 형태이며, 기울기 값을 직접 확인할 수 있습니다.

- **장점:** 특정 입력 변수만 선택적으로 기울기를 계산할 수 있습니다.

[//]: # (heading_4 is not supported)


```python
import torch
import torch.optim as optim

x1 = torch.tensor(2, requires_grad=True, dtype=torch.float16)
x2 = torch.tensor(3, requires_grad=True, dtype=torch.float16)
x3 = torch.tensor(1, requires_grad=True, dtype=torch.float16)
x4 = torch.tensor(4, requires_grad=True, dtype=torch.float16)

z1 = x1 * x2
z2 = x3 * x4
f = z1 + z2

f.backward()  # 모든 requires_grad=True 변수의 기울기 자동 계산

print(x1.grad)  # tensor(3., dtype=torch.float16)
print(x2.grad)  # tensor(2., dtype=torch.float16)
```

- `.backward()`는 `requires_grad=True`로 설정된 모든 변수의 기울기를 자동으로 계산하고, 각 변수의 `**.grad**` 속성에 저장합니다.

- **장점:** 코드가 간결하며, 다수의 변수에 대해 빠르게 기울기를 계산할 수 있습니다.

[//]: # (heading_4 is not supported)

 | 항목 | `torch.autograd.grad` | `.backward()` | 
 | ---- | ---- | ---- | 
 | 기울기 계산 방식 | 지정한 입력 변수의 기울기만 반환 | 모든 `requires_grad=True` 변수의 기울기 저장 | 
 | 결과 저장 위치 | 반환값(튜플)로 제공 | 각 텐서의 `.grad` 속성에 저장 | 
 | 사용 편의성 | 필요한 변수만 선택적으로 계산 가능 | 간단한 호출로 모든 변수 계산 가능 | 
 | 주요 사용 사례 | 특정 변수 기울기만 필요할 때 | 대부분의 딥러닝 모델 학습 과정 | 

[//]: # (heading_4 is not supported)


```python
opt = optim.SGD(params=[x1, x2, x3, x4], lr=0.001)

for i in range(3):
    z1 = x1 * x2
    z2 = x3 * x4
    f = z1 + z2
    f.backward()  # 기존 기울기에 계속 누적됨
    print(f"Iteration {i+1}: x1.grad = {x1.grad}")
```


```python
for i in range(3):
    opt.zero_grad()  # 기울기 초기화
    z1 = x1 * x2
    z2 = x3 * x4
    f = z1 + z2
    f.backward()
    print(f"Iteration {i+1}: x1.grad = {x1.grad}")
```

- **초기화 없음:** 기울기가 계속 누적되어 잘못된 업데이트로 이어질 수 있음.

- **초기화 있음:** 매 반복마다 정확한 기울기를 계산하고 학습을 안정적으로 수행 가능.

- 그래서 실제 학습 루프에서는 항상 `optimizer.zero_grad()`를 `backward()` 전후에 호출합니다.

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
x = torch.tensor([2.0], requires_grad=True)

# 연산 기록이 활성화된 경우
y1 = x ** 3
print(y1)  # tensor([8.], grad_fn=<PowBackward0>)

# torch.no_grad()로 연산 기록 비활성화
with torch.no_grad():
    y2 = x ** 3
print(y2)  # tensor([8.])  ← grad_fn이 없음

# 블록을 벗어나면 연산 기록이 자동으로 다시 활성화됨
```

- `torch.no_grad()`는 `with` 블록 내부에서만 작동하며, 블록이 끝나면 자동으로 연산 기록이 재활성화됩니다.

- 주로 검증(validation) 또는 추론(inference) 과정에서 사용됩니다.

[//]: # (heading_4 is not supported)

 | 특징 | `requires_grad=False` | `torch.no_grad()` | 
 | ---- | ---- | ---- | 
 | 사용 목적 | 영구적으로 gradient 계산 비활성화 | 일시적으로 gradient 계산 비활성화 | 
 | 적용 범위 | 특정 텐서 하나 | `with` 블록 내 모든 연산 | 
 | 주요 사용 사례 | 모델의 일부 파라미터를 고정(freeze)할 때 | 평가(evaluation) 및 추론(inference) | 
 | 재활성화 필요 여부 | 텐서 속성을 직접 다시 바꿔야 함 | 블록을 벗어나면 자동으로 재활성화 | 

[//]: # (heading_4 is not supported)

- `backward()`를 호출하기 전에 `optimizer.zero_grad()`를 빼먹으면 기울기가 이전 배치 값과 누적되어 학습이 이상하게 진행됩니다.

- 검증/테스트 루프에서 `torch.no_grad()`를 빼먹으면 불필요하게 메모리를 많이 사용하고 속도도 느려집니다.

- `requires_grad=False`인 텐서에 `.backward()`를 호출하려 하면 `grad_fn`이 없어서 에러가 납니다. 손실(loss) 텐서는 반드시 `grad_fn`을 가지고 있어야(즉, 연산 기록이 살아있어야) `.backward()`가 가능합니다.

---

[//]: # (heading_4 is not supported)

- [PyTorch-Tensor-Data-Modeling-MOC](https://app.notion.com/p/PyTorch-Tensor-Data-Modeling-MOC.md)

- [Tensor-Attributes-Cheatsheet](https://app.notion.com/p/Tensor-Attributes-Cheatsheet.md)

- [Training-Loop-and-Optimizer](https://app.notion.com/p/Training-Loop-and-Optimizer.md)

- [nn-Module-Model-Building](https://app.notion.com/p/nn-Module-Model-Building.md)

