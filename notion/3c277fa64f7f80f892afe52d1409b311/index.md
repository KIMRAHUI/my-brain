---
notion-url: https://app.notion.com/p/numpy_core_functions_and_operations-3c277fa64f7f80f892afe52d1409b311
title: numpy_core_functions_and_operations
date: '2026-08-20 08:29:00.000'
from_notion: https://app.notion.com/p/numpy_core_functions_and_operations-3c277fa64f7f80f892afe52d1409b311
author: rahui kim
last_edited_time: '2026-08-20 08:31:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- **정의:** Numeric + Python의 합성어로, 고성능 수치 연산을 담당하는 배열 엔진이자 ML/DL 전처리의 기반 라이브러리입니다.

- **핵심 특징:**

	- **C 언어 기반 고성능 엔진:** 파이썬의 표준 `list`와 달리 연속된 메모리 공간에 데이터를 저장하며, 내부가 C 언어로 작성되어 반복문(loop) 없이 대량의 수치 연산을 매우 빠르게 처리합니다.

	- **C-Order / Fortran-Order 연속성:** 차원 배열 저장 방식을 지원하여 메모리 접근 효율을 극대화합니다.

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- `**shape**`**:** 배열의 차원 크기와 형태를 튜플 형태로 반환 (예: `(3, 4)` → 3행 4열)

- `**dtype**`**:** 배열 내부 원소들의 데이터 타입 반환 (예: `int32`, `float64`)

- `**ndim**`**:** 배열의 차원 수 반환 (1차원: 1, 2차원: 2, 3차원: 3)

[//]: # (heading_4 is not supported)


```python
import numpy as np

# 1) 리스트로부터 배열 생성
arr = np.array([1, 2, 3], [4, 5, 6](./1, 2, 3], [4, 5, 6.md))
print("=== 배열 및 속성 확인 ===")
print("배열 출력:\n", arr)
print("Shape (형태):", arr.shape)
print("Data Type (타입):", arr.dtype)
print("Dimension (차원):", arr.ndim)

# 2) 수열 기반 배열 생성
print("\n=== 수열 및 균등 분할 생성 ===")
print("arange(0, 10, 2):", np.arange(0, 10, 2))       # 0부터 10 미만까지 2씩 증가
print("linspace(0, 1, 5):", np.linspace(0, 1, 5))    # 0부터 1까지 균등하게 5개 분할

# 3) 특수 배열 생성
print("\n=== 특수 배열 (zeros, ones, eye) ===")
print("zeros((2, 3)):\n", np.zeros((2, 3)))           # 0으로 채워진 2x3 행렬
print("ones((2, 2)):\n", np.ones((2, 2)))             # 1로 채워진 2x2 행렬
print("eye(3):\n", np.eye(3))                         # 3x3 단위 행렬
```

---

[//]: # (heading_4 is not supported)

- `**reshape(shape)**`**:** 배열의 전체 원소 개수를 유지하면서 차원과 구조 변경 (`1`을 사용하면 해당 차원 크기 자동 계산)

- `**astype(dtype)**`**:** 배열 원소의 데이터 타입을 변환 (예: 문자열 → 실수)

- `**tolist()**`**:** NumPy `ndarray` 객체를 일반 파이썬 `list` 객체로 변환

[//]: # (heading_4 is not supported)


```python
# 1) reshape 및 -1 자동 계산
a = np.arange(12)
a_2d = a.reshape(4, 3)     # 12개 원소를 4행 3열로 변환
a_auto = a.reshape(2, -1)   # 행을 2로 지정하고 열은 자동으로 6으로 계산

print("=== Reshape (4x3) ===\n", a_2d)
print("=== Reshape (2x-1) ===\n", a_auto)

# 2) astype 데이터 타입 변환
str_arr = np.array(['1.5', '2.8', '3.1'])
float_arr = str_arr.astype(np.float64)
print("\n타입 변환 (str -> float):", float_arr.dtype, float_arr)

# 3) tolist 파이썬 리스트 변환
py_list = float_arr.tolist()
print("파이썬 리스트 변환:", type(py_list), py_list)
```

---

[//]: # (heading_4 is not supported)

- `**np.random.seed(seed)**`**:** 난수 생성 알고리즘의 초기 시드값을 고정하여 코드 실행 시 마다 동일한 결과를 재현

- `**np.random.rand(d0, d1, ...)**`**:** [0, 1) 구간의 균등 분포(Uniform Distribution) 실수 난수 생성

- `**np.random.randn(d0, d1, ...)**`**:** 평균 0, 표준편차 1인 표준정규분포(Normal Distribution) 실수 난수 생성

- `**np.random.randint(low, high, size)**`**:** 지정한 범위 내 정수 난수 생성

[//]: # (heading_4 is not supported)


```python
# 난수 고정 (재현성 확보)
np.random.seed(42)

print("=== 균등 분포 실수 난수 (2x3) ===")
print(np.random.rand(2, 3))

print("\n=== 지정 범위 정수 난수 (1~45 로또 번호 6개) ===")
print(np.random.randint(1, 46, size=6))
```

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- `**dot()**`** / **`**@**`**:** 행렬 곱셈(내적 연산) 수행

- `**T**`** / **`**transpose()**`**:** 행과 열을 뒤바꾸는 전치 연산

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
A = np.array([1, 2], [3, 4](./1, 2], [3, 4.md))
B = np.array([5, 6], [7, 8](./5, 6], [7, 8.md))

# 1) 요소별 연산 vs 내적 연산
print("=== 요소별 곱셈 (A * B) ===\n", A * B)
print("=== 행렬 내적 (A @ B) ===\n", np.dot(A, B))

# 2) 전치 행렬
print("\n=== A의 전치행렬 (A.T) ===\n", A.T)

# 3) 축(axis) 지정 통계 연산
arr_stat = np.array([1, 2, 3], [4, 5, 6](./1, 2, 3], [4, 5, 6.md))
print("\n=== 통계 연산 (Axis 활용) ===")
print("전체 합계:", arr_stat.sum())
print("열 방향 합계 (axis=0):", arr_stat.sum(axis=0)) # 결과: [5, 7, 9]
print("행 방향 평균 (axis=1):", arr_stat.mean(axis=1)) # 결과: [2., 5.]
```

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```plain text
[1, 2, 3]  +  [10]  ──(브로드캐스팅)──>  [1, 2, 3] + [10, 10, 10] = [11, 12, 13]
```

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- `**np.linalg.inv(A)**`**:** 정방행렬 A의 역행렬(A^{-1}) 계산

- `**np.linalg.solve(A, b)**`**:** 선형 시스템 연립방정식(Ax = b)의 해 구하기

- `**np.linalg.det(A)**`**:** 행렬식(\vert{}A\vert{}) 계산

[//]: # (heading_4 is not supported)


```python
# 1) 브로드캐스팅 예시
matrix = np.array([1, 2, 3], [4, 5, 6](./1, 2, 3], [4, 5, 6.md))
vector = np.array([10, 20, 30])
print("=== 브로드캐스팅 연산 (Matrix + Vector) ===\n", matrix + vector)

# 2) 마스킹 연산 (조건 검색 및 값 변경)
data = np.array([5, 12, 3, 18, 7, 22])
mask = data < 10
print("\n마스크 배열 (data < 10):", mask)
print("조건을 만족하는 원소만 추출:", data[mask])

# 마스킹으로 특정 값 바로 변경
data[data < 10] = 100
print("마스킹 값 변경 후 데이터:", data)

# 3) 선형대수 모듈 (np.linalg)
A_mat = np.array([2, 1], [1, 3](./2, 1], [1, 3.md))
b_vec = np.array([5, 10])

inv_A = np.linalg.inv(A_mat)          # 역행렬
det_A = np.linalg.det(A_mat)          # 행렬식
sol = np.linalg.solve(A_mat, b_vec)   # Ax = b 해 구하기

print("\n=== np.linalg 선형대수 연산 ===")
print("행렬식 (det):", det_A)
print("역행렬 (inv):\n", inv_A)
print("연립방정식 해 (solve):", sol)
```

---

[//]: # (heading_4 is not supported)

- [NumPy Official Documentation & User Guide](https://numpy.org/doc/stable/user/index.html)

- [NumPy Absolute Beginners Guide](https://numpy.org/doc/stable/user/absolute_beginners.html)

- [NumPy Linear Algebra (numpy.linalg) Reference](https://numpy.org/doc/stable/reference/routines.linalg.html)

