---
notion-url: https://app.notion.com/p/Tensor-Operations-Broadcasting-40c77fa64f7f83fc8aee811a3056c678
title: Tensor-Operations-Broadcasting
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Tensor-Operations-Broadcasting-40c77fa64f7f83fc8aee811a3056c678
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
import torch

a = torch.tensor([1, 2], [3, 4](./1, 2], [3, 4.md))
b = torch.tensor([10, 20], [30, 40](./10, 20], [30, 40.md))

print(a + b)
# tensor([11, 22], [33, 44](./11, 22], [33, 44.md))

print(torch.add(a, b))   # a + b와 완전히 동일
# tensor([11, 22], [33, 44](./11, 22], [33, 44.md))

print(a * b)
# tensor([10, 40], [90, 160](./10, 40], [90, 160.md))   ← 행렬곱이 아니라 같은 위치끼리의 단순 곱셈!
```

- `a + b`와 `torch.add(a, b)`는 결과가 완전히 동일합니다. 연산자(`+`, , , `/`)는 각각 대응하는 함수(`torch.add`, `torch.sub`, `torch.mul`, `torch.div`)의 축약형이라고 보면 됩니다.

- **주의:** `a * b`는 "행렬 곱셈"이 아니라 같은 위치의 원소끼리 곱하는 것입니다. 수학 시간에 배우는 행렬 곱셈과 헷갈리기 쉬운 부분이니 꼭 구분해야 합니다.

[//]: # (heading_4 is not supported)


```python
a = torch.rand(2, 3)   # 2행 3열
b = torch.rand(3, 4)   # 3행 4열

result = torch.matmul(a, b)   # 또는 a @ b
print(result.shape)   # torch.Size([2, 4])
```

- `torch.matmul(a, b)`와 `a @ b`는 완전히 동일한 결과를 냅니다. `@` 연산자가 더 간결해서 실무 코드에서 자주 쓰입니다.

- **조건:** 앞 행렬의 열 개수와 뒤 행렬의 행 개수가 반드시 일치해야 합니다. 위 예시에서는 `a`의 열(3)과 `b`의 행(3)이 같으므로 곱셈이 가능하고, 결과 shape는 `(a의 행, b의 열)` = `(2, 4)`가 됩니다.

- 이 조건이 맞지 않으면 아래와 같은 에러가 발생합니다.


```python
a = torch.rand(2, 3)
b = torch.rand(2, 4)   # a의 열(3) ≠ b의 행(2) → 에러

# result = torch.matmul(a, b)
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x4)
```

- 이 에러가 나면 [Tensor-Shape-Interpretation](https://app.notion.com/p/Tensor-Shape-Interpretation.md)에서 다룬 "행/열 해석"을 다시 떠올려서, 두 텐서의 shape를 각각 출력해보고 어느 쪽을 전치(transpose, `.T`)해야 하는지 확인하는 것이 첫 번째 디버깅 순서입니다.

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

1. 두 차원의 크기가 완전히 같거나

1. 둘 중 하나의 크기가 1인 경우

[//]: # (heading_4 is not supported)


```python
a = torch.tensor([1, 2, 3])   # shape: (3,)
b = 10                         # 스칼라

print(a + b)
# tensor([11, 12, 13])
```

- 스칼라 `10`이 `[10, 10, 10]`처럼 확장되어 각 원소에 더해진 것과 동일한 효과를 냅니다.

[//]: # (heading_4 is not supported)


```python
a = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])   # shape: (2, 3)
b = torch.tensor([10, 20, 30])   # shape: (3,)

print(a + b)
# tensor([[11, 22, 33],
#         [14, 25, 36]])
```

- 뒤쪽 차원부터 비교: `a`의 마지막 차원 `3`과 `b`의 마지막 차원 `3`이 같으므로 브로드캐스팅이 가능합니다.

- `b`가 마치 `[10, 20, 30], [10, 20, 30](./10, 20, 30], [10, 20, 30.md)`처럼 행 방향으로 복제된 것처럼 동작하여 `a`의 각 행에 더해집니다.

[//]: # (heading_4 is not supported)


```python
a = torch.rand(2, 3)   # shape: (2, 3)
b = torch.rand(2, 4)   # shape: (2, 4)

# result = a + b
# RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1
```

- 마지막 차원끼리 비교했을 때 `3`과 `4`는 서로 다르고, 둘 다 1도 아니므로 규칙 1, 2를 모두 만족하지 못해 에러가 발생합니다.

[//]: # (heading_4 is not supported)

 | 비교하는 두 텐서의 (뒤쪽부터) 차원 | 브로드캐스팅 가능 여부 | 이유 | 
 | ---- | ---- | ---- | 
 | `3` vs `3` | 가능 | 크기가 완전히 같음 | 
 | `3` vs `1` | 가능 | 한쪽이 1 | 
 | `1` vs `5` | 가능 | 한쪽이 1 | 
 | `3` vs `4` | 불가능 | 둘 다 다르고 1도 아님 | 

[//]: # (heading_4 is not supported)

- `a * b`(요소별 곱셈)와 `torch.matmul(a, b)`(행렬 곱셈)를 혼동해서 원하는 결과가 안 나오는 경우가 매우 흔합니다. "값끼리 곱하고 싶은가, 선형대수적 행렬 곱셈을 하고 싶은가"를 항상 먼저 명확히 하세요.

- 브로드캐스팅은 "차원의 개수(ndim)"가 달라도 작동합니다. 이때는 부족한 앞쪽 차원에 자동으로 `1`이 채워진다고 생각하면 됩니다. 예를 들어 `(2, 3)`과 `(3,)`을 비교할 때, `(3,)`은 내부적으로 `(1, 3)`처럼 취급되어 뒤쪽부터 비교가 이루어집니다.

- 행렬 곱셈에서 shape가 안 맞으면 `.T`(전치, transpose)로 행과 열을 바꿔서 해결하는 경우가 많은데, 이때 결과의 의미가 원래 의도와 달라질 수 있으니 단순히 에러를 없애기 위해 `.T`를 남용하지 않도록 주의해야 합니다.

---

## 관련 노트

- [PyTorch-Tensor-Data-Modeling-MOC](https://app.notion.com/p/PyTorch-Tensor-Data-Modeling-MOC.md)

- [Tensor-Creation-Functions](https://app.notion.com/p/Tensor-Creation-Functions.md)

- [Tensor-Shape-Interpretation](https://app.notion.com/p/Tensor-Shape-Interpretation.md)

- [Dtype-Conversion-NumPy-Interop](https://app.notion.com/p/Dtype-Conversion-NumPy-Interop.md)

