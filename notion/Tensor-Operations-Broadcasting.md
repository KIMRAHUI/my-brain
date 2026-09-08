# Tensor-Operations-Broadcasting

텐서 연산 종류

요소별 연산 (Element-wise Operation)

같은 위치에 있는 원소끼리 하나씩 짝지어 연산하는 방식입니다. 두 텐서의 shape가 완전히 같을 때 가장 직관적으로 이해할 수 있습니다.

import torch

a = torch.tensor([1, 2], [3, 4](./1, 2], [3, 4.md))
b = torch.tensor([10, 20], [30, 40](./10, 20], [30, 40.md))

print(a + b)
# tensor([11, 22], [33, 44](./11, 22], [33, 44.md))

print(torch.add(a, b))   # a + b와 완전히 동일
# tensor([11, 22], [33, 44](./11, 22], [33, 44.md))

print(a * b)
# tensor([10, 40], [90, 160](./10, 40], [90, 160.md))   ← 행렬곱이 아니라 같은 위치끼리의 단순 곱셈!

a + b와 torch.add(a, b)는 결과가 완전히 동일합니다. 연산자(+, , , /)는 각각 대응하는 함수(torch.add, torch.sub, torch.mul, torch.div)의 축약형이라고 보면 됩니다.

주의: a * b는 "행렬 곱셈"이 아니라 같은 위치의 원소끼리 곱하는 것입니다. 수학 시간에 배우는 행렬 곱셈과 헷갈리기 쉬운 부분이니 꼭 구분해야 합니다.

행렬 곱셈 (Matrix Multiplication)

a = torch.rand(2, 3)   # 2행 3열
b = torch.rand(3, 4)   # 3행 4열

result = torch.matmul(a, b)   # 또는 a @ b
print(result.shape)   # torch.Size([2, 4])

torch.matmul(a, b)와 a @ b는 완전히 동일한 결과를 냅니다. @ 연산자가 더 간결해서 실무 코드에서 자주 쓰입니다.

조건: 앞 행렬의 열 개수와 뒤 행렬의 행 개수가 반드시 일치해야 합니다. 위 예시에서는 a의 열(3)과 b의 행(3)이 같으므로 곱셈이 가능하고, 결과 shape는 (a의 행, b의 열) = (2, 4)가 됩니다.

이 조건이 맞지 않으면 아래와 같은 에러가 발생합니다.

a = torch.rand(2, 3)
b = torch.rand(2, 4)   # a의 열(3) ≠ b의 행(2) → 에러

# result = torch.matmul(a, b)
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x4)

이 에러가 나면 Tensor-Shape-Interpretation에서 다룬 "행/열 해석"을 다시 떠올려서, 두 텐서의 shape를 각각 출력해보고 어느 쪽을 전치(transpose, .T)해야 하는지 확인하는 것이 첫 번째 디버깅 순서입니다.

브로드캐스팅(Broadcasting) 원리

왜 필요한가?

딥러닝에서는 shape가 완전히 같지 않은 텐서끼리 연산해야 하는 경우가 매우 흔합니다. 예를 들어 (100, 3) 크기의 데이터 100개에서 각각 평균값 (3,)을 빼주는 정규화(normalization) 작업이 대표적입니다. 매번 shape를 강제로 맞춰서 복사하는 건 비효율적이므로, PyTorch(그리고 NumPy)는 작은 쪽 텐서의 shape를 자동으로 "확장" 해서 연산이 가능하게 해주는 브로드캐스팅 기능을 제공합니다.

작동 규칙

크기가 다른 두 텐서 간의 연산이 필요할 때, 뒤쪽(오른쪽) 차원에서부터 차례대로 비교했을 때 다음 두 조건 중 하나를 만족하면 자동으로 크기를 늘려 연산을 수행합니다.

두 차원의 크기가 완전히 같거나

둘 중 하나의 크기가 1인 경우

두 조건 중 어느 것도 만족하지 않으면 브로드캐스팅이 불가능하며 에러가 발생합니다.

예시 1: 벡터 + 스칼라

a = torch.tensor([1, 2, 3])   # shape: (3,)
b = 10                         # 스칼라

print(a + b)
# tensor([11, 12, 13])

스칼라 10이 [10, 10, 10]처럼 확장되어 각 원소에 더해진 것과 동일한 효과를 냅니다.

예시 2: 행렬 + 벡터

a = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])   # shape: (2, 3)
b = torch.tensor([10, 20, 30])   # shape: (3,)

print(a + b)
# tensor([[11, 22, 33],
#         [14, 25, 36]])

뒤쪽 차원부터 비교: a의 마지막 차원 3과 b의 마지막 차원 3이 같으므로 브로드캐스팅이 가능합니다.

b가 마치 [10, 20, 30], [10, 20, 30](./10, 20, 30], [10, 20, 30.md)처럼 행 방향으로 복제된 것처럼 동작하여 a의 각 행에 더해집니다.

예시 3: 브로드캐스팅이 불가능한 경우

a = torch.rand(2, 3)   # shape: (2, 3)
b = torch.rand(2, 4)   # shape: (2, 4)

# result = a + b
# RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1

마지막 차원끼리 비교했을 때 3과 4는 서로 다르고, 둘 다 1도 아니므로 규칙 1, 2를 모두 만족하지 못해 에러가 발생합니다.

브로드캐스팅 규칙 체크리스트

자주 하는 실수

a * b(요소별 곱셈)와 torch.matmul(a, b)(행렬 곱셈)를 혼동해서 원하는 결과가 안 나오는 경우가 매우 흔합니다. "값끼리 곱하고 싶은가, 선형대수적 행렬 곱셈을 하고 싶은가"를 항상 먼저 명확히 하세요.

브로드캐스팅은 "차원의 개수(ndim)"가 달라도 작동합니다. 이때는 부족한 앞쪽 차원에 자동으로 1이 채워진다고 생각하면 됩니다. 예를 들어 (2, 3)과 (3,)을 비교할 때, (3,)은 내부적으로 (1, 3)처럼 취급되어 뒤쪽부터 비교가 이루어집니다.

행렬 곱셈에서 shape가 안 맞으면 .T(전치, transpose)로 행과 열을 바꿔서 해결하는 경우가 많은데, 이때 결과의 의미가 원래 의도와 달라질 수 있으니 단순히 에러를 없애기 위해 .T를 남용하지 않도록 주의해야 합니다.

관련 노트

PyTorch-Tensor-Data-Modeling-MOC

Tensor-Creation-Functions

Tensor-Shape-Interpretation

Dtype-Conversion-NumPy-Interop

