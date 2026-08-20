---
notion-url: https://app.notion.com/p/data_manipulation_and_aggregation-3c277fa64f7f809bb487e4718defd4c0
title: data_manipulation_and_aggregation
date: '2026-08-20 08:30:00.000'
from_notion: https://app.notion.com/p/data_manipulation_and_aggregation-3c277fa64f7f809bb487e4718defd4c0
author: rahui kim
last_edited_time: '2026-08-20 08:31:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- `**loc[행_라벨, 열_라벨]**`** (Label-based Indexing):**

	- 명시적인 **인덱스 이름(Label)**이나 불리언 조건식(True/False)을 기준으로 데이터를 선택합니다.

	- 슬라이싱(`0:2`) 적용 시 **마지막 범위(2)가 포함**됩니다.

- `**iloc[행_위치, 열_위치]**`** (Integer Position-based Indexing):**

	- 컴퓨터가 인식하는 **0부터 시작하는 정수 위치 번호**를 기준으로 선택합니다.

	- 슬라이싱(`0:2`) 적용 시 파이썬 기본 규칙과 동일하게 **마지막 범위(2)가 제외**됩니다.

- **불리언 인덱싱 (Boolean Indexing):**

	- 조건식을 활용해 `True`에 해당하는 행만 필터링합니다.

	- 복합 조건 적용 시 비트 연산자 `&`(AND), `|`(OR)를 사용하며, 각 조건식은 반드시 괄호 `()`로 감싸야 합니다.

[//]: # (heading_4 is not supported)


```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Department': ['HR', 'IT', 'IT', 'HR', 'Finance'],
    'Salary': [4000, 5500, 6000, 4500, 5000],
    'Experience': [2, 5, 8, 3, 4]
}
df = pd.DataFrame(data)

# 1) loc 사용: 라벨 이름 기반 조회 (0부터 2번 인덱스 행까지 포함)
print("=== loc 조회 (라벨 기반) ===")
print(df.loc[0:2, ['Name', 'Salary']])

# 2) iloc 사용: 정수 위치 기반 조회 (0부터 1번 행/열까지 선택)
print("\n=== iloc 조회 (정수 위치 기반) ===")
print(df.iloc[0:2, 0:2])

# 3) 불리언 인덱싱 (복합 조건: IT 부서이면서 연봉 5500 이상)
condition = (df['Department'] == 'IT') & (df['Salary'] >= 5500)
print("\n=== 복합 조건 필터링 결과 ===")
print(df[condition])
```

---

[//]: # (heading_4 is not supported)

- `**df.isnull()**`** / **`**df.isna()**`**:** 결측치 위치를 `True`로 반환합니다. (`.sum()`과 조합하여 컬럼별 결측치 개수 파악)

- `**df.dropna(axis=0, how='any')**`**:** 결측치가 포함된 행(`axis=0`) 또는 열(`axis=1`)을 제거합니다.

- `**df.fillna(value)**`**:** 결측치를 특정 값(0, 평균값, 중앙값, 최빈값 등)으로 채웁니다.

[//]: # (heading_4 is not supported)


```python
import numpy as np

# 결측치가 포함된 샘플 데이터 생성
df_na = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [np.nan, 10, 20, 30],
    'C': [100, 200, 300, 400]
})

# 1) 컬럼별 결측치 개수 확인
print("=== 컬럼별 결측치 개수 ===")
print(df_na.isnull().sum())

# 2) 결측치 대체 (A 컬럼은 평균값, B 컬럼은 중앙값으로 대체)
df_filled = df_na.copy()
df_filled['A'] = df_filled['A'].fillna(df_filled['A'].mean())
df_filled['B'] = df_filled['B'].fillna(df_filled['B'].median())

print("\n=== 결측치 대체 후 데이터 ===")
print(df_filled)
```

---

[//]: # (heading_4 is not supported)

- **Split-Apply-Combine 메커니즘:**

	1. **Split:** 특정 기준 컬럼 값에 따라 데이터를 여러 그룹으로 분할합니다.

	1. **Apply:** 각 그룹별로 통계 함수(`mean`, `sum`, `count` 등)를 적용합니다.

	1. **Combine:** 연산 결과를 하나의 데이터 구조로 결합합니다.

- `**df.groupby('기준컬럼')**`**:** 그룹화 기준 컬럼 지정

- `**.agg()**`** 메서드:** 여러 컬럼에 대해 서로 다른 집계 함수를 동시에 적용할 수 있는 다중 집계 함수

[//]: # (heading_4 is not supported)


```python
# 부서(Department) 기준 그룹화 및 컬럼별 다중 집계 적용
dept_summary = df.groupby('Department').agg({
    'Salary': ['mean', 'sum'],  # Salary 컬럼: 평균 및 합계
    'Experience': 'max'         # Experience 컬럼: 최댓값
})

print("=== 부서별 집계 요약 (GroupBy + agg) ===")
print(dept_summary)
```

---

[//]: # (heading_4 is not supported)

- [Pandas Official Guide - Indexing and Selecting Data](https://pandas.pydata.org/docs/user_guide/indexing.html)

- [Pandas Official Guide - Working with Missing Data](https://pandas.pydata.org/docs/user_guide/missing_data.html)

- [Pandas Official Guide - Group By: split-apply-combine](https://pandas.pydata.org/docs/user_guide/groupby.html)

