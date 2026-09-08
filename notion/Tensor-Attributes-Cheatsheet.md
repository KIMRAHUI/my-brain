# Tensor-Attributes-Cheatsheet

텐서 디버깅 5가지
  에러가 나거나 상태를 확인할 때 가장 먼저 print()로 찍어봐야 할 5가지 속성입니다.
  1. tensor.shape (크기)
  2. tensor.dtype (데이터 타입)
  3. tensor.device (장치 위치: CPU/GPU)
  4. tensor.requires_grad (역전파 여부)
  5. tensor.ndim (차원 수)

import torch

t = torch.rand(2, 3, requires_grad=True)
print("shape:", t.shape)
print("size():", t.size())
print("ndim:", t.ndim)
print("dtype:", t.dtype)
print("device:", t.device)
print("requires_grad:", t.requires_grad)

출력 예시:

shape: torch.Size([2, 3])
size(): torch.Size([2, 3])
ndim: 2
dtype: torch.float32
device: cpu
requires_grad: True

속성별 상세 설명

.shape / .size() — 형태 확인

.shape는 속성(property), .size()는 메서드(method)라서 괄호 유무만 다르고 반환값(torch.Size 객체)은 완전히 같습니다.

torch.Size는 사실상 튜플(tuple)이라서 t.shape[0]처럼 인덱싱해서 특정 축의 크기만 뽑아낼 수도 있습니다.

print(t.shape[0])  # 2 (행의 개수)
print(t.shape[1])  # 3 (열의 개수)

.ndim — 차원 수

스칼라(0차원), 벡터(1차원), 행렬(2차원), 그 이상(3차원 이상)을 구분할 때 씁니다.

scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3])
matrix = torch.rand(2, 3)

print(scalar.ndim)  # 0
print(vector.ndim)  # 1
print(matrix.ndim)  # 2

.dtype — 데이터 타입

기본적으로 torch.rand(), torch.randn() 등으로 만든 텐서는 torch.float32가 기본값입니다.

정수 텐서는 보통 torch.int64(=torch.long)가 기본값입니다.

딥러닝 모델 학습 시 두 텐서의 dtype이 다르면 연산 중 에러가 나므로 항상 확인하는 습관이 중요합니다. (자세한 내용은 Dtype-Conversion-NumPy-Interop 참고)

.device — 연산 장치

텐서가 cpu에 있는지 cuda:0(GPU 0번)에 있는지를 알려줍니다.

서로 다른 device에 있는 텐서끼리 연산하면 에러가 발생합니다. (자세한 내용은 Device-Management-CPU-GPU 참고)

.requires_grad — 자동 미분 추적 여부

True로 설정하면 이 텐서에 대해 이루어지는 모든 연산이 기록되어, 나중에 .backward()를 호출했을 때 기울기(gradient)를 자동으로 계산할 수 있습니다.

주로 모델의 학습 가능한 파라미터(가중치, 편향)에 requires_grad=True를 설정합니다. 입력 데이터나 정답 라벨은 보통 False(기본값)로 둡니다.

w = torch.randn(3, requires_grad=True)
y = (w ** 2).sum()
y.backward()
print(w.grad)  # w에 대한 미분값(기울기)이 출력됨

자주 하는 실수

.shape를 함수처럼 t.shape()로 호출하면 에러가 납니다. .shape는 속성이므로 괄호 없이 써야 합니다. 괄호가 필요한 것은 .size()입니다.

requires_grad=True인 텐서를 바로 .numpy()로 변환하려 하면 에러가 납니다. .detach().numpy()처럼 grad 추적을 끊어줘야 합니다.

관련 노트

PyTorch-Tensor-Data-Modeling-MOC

Tensor-Shape-Interpretation

Device-Management-CPU-GPU

Dtype-Conversion-NumPy-Interop

