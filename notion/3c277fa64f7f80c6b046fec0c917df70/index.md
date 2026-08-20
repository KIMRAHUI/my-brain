---
notion-url: https://app.notion.com/p/PyTorch-Tensor-Manipulation-and-Operations-Deep-Dive-3c277fa64f7f80c6b046fec0c917df70
title: PyTorch Tensor Manipulation and Operations Deep Dive
date: '2026-08-20 09:33:00.000'
from_notion: https://app.notion.com/p/PyTorch-Tensor-Manipulation-and-Operations-Deep-Dive-3c277fa64f7f80c6b046fec0c917df70
author: rahui kim
last_edited_time: '2026-08-20 09:36:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- **순서 기억하기:** 우리말 "행렬"을 떠올리세요. 행(세로) 먼저, 열(나중)입니다.

- **코드와 출력 매칭 팁 (**`**torch.rand(2, 3)**`** 예시):**

	- **앞의 숫자 **`**2**`** (행 / 세로):** 바깥쪽 대괄호 안에 큰 묶음(줄)이 총 몇 개인지 의미합니다.

	- **뒤의 숫자 **`**3**`** (열 / 가로):** 각 줄마다 요소(숫자)가 몇 개씩 들어있는지 의미합니다.

 | 구분 | 영어 명칭 | 방향 | 텐서 모양 `(x, y)`에서의 위치 | 확인하는 법 | 
 | ---- | ---- | ---- | ---- | ---- | 
 | 행 | Row | 세로 (위 ↕ 아래) | 앞쪽 (`x`) | 바깥쪽 대괄호 안의 줄 개수 | 
 | 열 | Column | 가로 (좌 ↔ 우) | 뒤쪽 (`y`) | 각 줄 안에 들어있는 요소 개수 | 

---

[//]: # (heading_4 is not supported)

 | 분류 | 속성 / 메서드 | 설명 및 역할 | 
 | ---- | ---- | ---- | 
 | 기본 정보 | `.shape` / `.size()` | 텐서의 전체 형태(크기)를 반환합니다. (예: `torch.Size([2, 3])`) | 
 | 차원 수 | `.ndim` | 텐서의 차원 수(Dimensions)를 정수로 반환합니다. | 
 | 데이터 타입 | `.dtype` | 텐서에 저장된 데이터 타입을 확인합니다. (예: `torch.float32`, `torch.int64`) | 
 | 연산 장치 | `.device` | 텐서가 할당된 장치(CPU 또는 GPU)를 확인합니다. (예: `cpu`, `cuda:0`) | 
 | 미분 추적 | `.requires_grad` | 자동 미분(Autograd) 기록 여부를 불리언(`True`/`False`)으로 반환합니다. | 

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- **CPU:** 복잡한 제어 로직, 데이터 전처리, 모델 객체 생성 초기 단계 등에 주로 사용됩니다.

- **GPU:** 대규모 병렬 연산(행렬 곱셈, 역전파 등)을 초고속으로 처리하여 딥러닝 학습 시간을 극적으로 단축시킵니다.

[//]: # (heading_4 is not supported)

- **장치 가용성 확인:** `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` 형태로 현재 환경에서 GPU 사용 가능 여부를 동적으로 체크합니다.

- **텐서를 GPU로 보내기:** `tensor_gpu = tensor.to("cuda")` 또는 `tensor.to(device)`

- **텐서를 CPU로 가져오기:** `tensor_cpu = tensor_gpu.to("cpu")` (GPU 텐서는 바로 NumPy로 변환할 수 없으므로, 반드시 `.to("cpu")`를 거쳐야 `numpy()` 변환이 가능합니다.)

[//]: # (heading_4 is not supported)


```python
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```


```python
# 예시: GPU에 있는 텐서와 CPU에 있는 텐서를 더할 때
a = torch.tensor([1.0, 2.0]).to("cuda")
b = torch.tensor([3.0, 4.0]).to("cpu")  # 또는 기본값 cpu

# b를 cuda로 일치시키거나, a를 cpu로 내려야 에러가 안 남
b = b.to(a.device)
result = a + b
```

---

[//]: # (heading_4 is not supported)

### 1) 랜덤한 값으로 만들기

- `**torch.rand()**`** (균등 분포):** 0과 1 사이의 균등 분포를 따르는 난수 텐서를 생성합니다.

	- 범위 확장 팁: `min_val + (max_val - min_val) * torch.rand(2, 3)`

- `**torch.randn()**`** (정규 분포):** 평균 0, 표준편차 1인 표준 정규 분포를 따르는 난수를 생성합니다. 주로 가중치 초기화(Weight Initialization)에 필수적으로 사용됩니다.

[//]: # (heading_4 is not supported)

- `**torch.zeros()**`**:** 모든 값이 0인 텐서를 생성합니다.

- `**torch.ones()**`**:** 모든 값이 1인 텐서를 생성합니다.

- `**torch.full((shape), value)**`**:** 사용자가 지정한 상수로 텐서를 채웁니다.

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- **권장 방식 (**`**.to()**`**):** `tensor.to(torch.float32)` 또는 `tensor.to(dtype=torch.float32, device="cuda")`

- **간편 메서드:** `.float()`, `.int()`, `.long()`, `.double()` 등

- **주의 사항:** 행렬 곱셈(`torch.matmul`) 시 연산하려는 두 텐서의 데이터 타입(`dtype`)이 다르면 `RuntimeError`가 발생하므로 반드시 `.to()`로 타입을 일치시켜야 합니다.

[//]: # (heading_4 is not supported)

- `**tensor.numpy()**`**:** 텐서와 메모리를 공유합니다. 원본 텐서가 바뀌면 NumPy 배열도 함께 바뀝니다. (단, CPU 텐서에서만 작동하며, GPU 텐서는 반드시 `.to("cpu").numpy()`를 거쳐야 함)

- `**np.array(tensor)**`**:** 데이터를 복사(Copy)하여 독립된 새로운 NumPy 배열을 생성합니다.

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- **요소별 연산 (Element-wise):** `a + b`, `torch.add(a, b)`, `a * b` (같은 위치의 원소끼리 연산)

- **행렬 곱셈 (Matrix Multiplication):** `torch.matmul(a, b)` 또는 `a @ b` (앞 행렬의 열 개수와 뒤 행렬의 행 개수가 일치해야 함)

[//]: # (heading_4 is not supported)

1. 두 차원의 크기가 완전히 같거나

1. 둘 중 하나의 크기가 1인 경우

---

## 7. 공식 문서 및 참고 링크

- [PyTorch Official Documentation - torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html)

- [PyTorch Official Documentation - CUDA semantics (Device Management)](https://docs.pytorch.org/docs/stable/notes/cuda.html)

- [NumPy Official Documentation - Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

