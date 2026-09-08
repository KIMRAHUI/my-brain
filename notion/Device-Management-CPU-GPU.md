# Device-Management-CPU-GPU

데이터 타입 변환 및 NumPy 연동

타입 변환 방법

권장 방식: .to()

import torch

t = torch.tensor([1, 2, 3])       # 기본 dtype: torch.int64
t_float = t.to(torch.float32)
print(t_float.dtype)              # torch.float32

.to()는 dtype 변환과 device 이동을 동시에 처리할 수 있는 만능 메서드입니다.

t_gpu_float = t.to(dtype=torch.float32, device="cuda")

이렇게 dtype과 device를 한 번에 지정하면 코드가 간결해지고, 어떤 상태로 바뀌는지 명시적으로 드러나서 가독성도 좋습니다.

간편 메서드

dtype 하나만 빠르게 바꾸고 싶을 때는 아래처럼 축약된 메서드를 사용할 수도 있습니다.

t.float()   # torch.float32로 변환
t.int()     # torch.int32로 변환
t.long()    # torch.int64로 변환 (정수 인덱싱, 라벨 등에 자주 사용)
t.double()  # torch.float64로 변환

주의 사항: 타입 불일치 에러

행렬 곱셈(torch.matmul)이나 다른 연산 시, 연산하려는 두 텐서의 데이터 타입(dtype)이 다르면 RuntimeError가 발생합니다. 예를 들어 하나는 float32, 다른 하나는 int64이면 곧바로 연산이 실패합니다.

a = torch.rand(2, 3)                    # float32
b = torch.tensor([1, 2, 3], [4, 5, 6](./1, 2, 3], [4, 5, 6.md))  # int64

# result = torch.matmul(a, b)  → RuntimeError 발생 가능

b = b.to(a.dtype)   # b를 a와 같은 float32로 맞춤
result = torch.matmul(a, b.T)  # 정상 작동

Device-Management-CPU-GPU에서 다룬 "디바이스 불일치"와 원리가 비슷합니다. device는 물리적 위치를 맞추는 것, dtype은 데이터 타입을 맞추는 것이라고 구분해서 기억하면 됩니다.

PyTorch Tensor ⇄ NumPy 배열 변환

딥러닝 파이프라인에서는 데이터 전처리(pandas, NumPy 기반)와 모델 연산(PyTorch 텐서 기반)을 오가는 경우가 매우 흔합니다.

tensor.numpy() — 메모리 공유(Shallow)

t = torch.ones(3)
arr = t.numpy()

print(arr)        # [1. 1. 1.]

t.add_(1)          # 텐서 값을 in-place로 변경 (t += 1과 유사)
print(arr)         # [2. 2. 2.]  ← NumPy 배열도 같이 바뀜!

tensor.numpy()는 텐서와 메모리를 공유합니다. 즉 실제로 데이터를 복사하지 않고 "같은 메모리를 다른 방식으로 보는 창구"만 하나 더 만드는 것입니다. 그래서 원본 텐서가 바뀌면 NumPy 배열도 함께 바뀝니다.

제약 조건: 이 방식은 CPU 텐서에서만 작동합니다. GPU 텐서는 GPU 메모리(VRAM)에 있고 NumPy는 CPU 메모리만 다룰 수 있기 때문에, GPU 텐서는 반드시 .to("cpu").numpy()를 거쳐야 변환이 가능합니다. (이 경우는 메모리가 공유되지 않고 새로 복사됩니다.)

t_gpu = torch.ones(3).to("cuda")
arr = t_gpu.to("cpu").numpy()   # 정상 작동
# arr = t_gpu.numpy()            → 에러! (GPU 텐서는 바로 변환 불가)

np.array(tensor) — 데이터 복사(Deep Copy)

import numpy as np

t = torch.ones(3)
arr = np.array(t)

t.add_(1)
print(arr)  # [1. 1. 1.]  ← 원본이 바뀌어도 영향 없음 (독립적인 복사본)

np.array(tensor)는 데이터를 실제로 복사(Copy)하여 원본과 완전히 독립된 새로운 NumPy 배열을 생성합니다.

원본 텐서와 연결이 끊기기 때문에, 이후 텐서 값이 바뀌어도 NumPy 배열에는 영향을 주지 않습니다.

tensor.numpy() vs np.array(tensor) 비교

반대로 NumPy → Tensor 변환

문서에는 명시되어 있지 않지만, 반대 방향(NumPy → Tensor) 변환도 실무에서 자주 필요하므로 함께 기억해두면 좋습니다.

arr = np.array([1, 2, 3])
t_from_np = torch.from_numpy(arr)   # 메모리 공유 (numpy()의 반대 개념)
t_copy = torch.tensor(arr)           # 데이터 복사 (np.array(tensor)의 반대 개념)

torch.from_numpy()는 tensor.numpy()와 대칭되는 개념으로 메모리를 공유하고, torch.tensor()는 np.array(tensor)와 대칭되는 개념으로 데이터를 복사합니다.

자주 하는 실수

requires_grad=True인 텐서를 바로 .numpy()로 변환하려 하면 에러가 납니다. .detach().numpy()로 grad 추적을 먼저 끊어줘야 합니다.

GPU 텐서에서 .cpu()를 빼먹고 바로 .numpy()를 호출해서 에러가 나는 경우가 매우 흔합니다. 순서는 항상 GPU → CPU → NumPy입니다.

tensor.numpy()로 메모리를 공유해놓고 나중에 "왜 배열 값이 자꾸 바뀌지?"라며 당황하는 경우가 있습니다. 독립적인 사본이 필요하면 처음부터 np.array(tensor)를 쓰는 것이 안전합니다.

관련 노트

PyTorch-Tensor-Data-Modeling-MOC

Tensor-Attributes-Cheatsheet

Device-Management-CPU-GPU

Tensor-Operations-Broadcasting

