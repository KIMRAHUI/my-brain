# data_manipulation_and_aggregation

개요

Pandas DataFrame에서 원하는 위치나 조건에 맞춰 데이터를 추출하는 인덱싱 기법(loc, iloc, 불리언 필터링), 비어 있는 데이터를 탐색하고 정제하는 결측치 처리, 그리고 범주별 통계량을 계산하는 그룹화(GroupBy) 및 집계 연산을 상세히 학습합니다.

데이터 선택 및 필터링 (Indexing & Selection)

데이터프레임에서 특정 행(Row)과 열(Column)을 선택하거나 원하는 조건에 맞는 데이터만 걸러내는 방법입니다.

loc[행_라벨, 열_라벨] (Label-based Indexing):

iloc[행_위치, 열_위치] (Integer Position-based Indexing):

불리언 인덱싱 (Boolean Indexing):

인덱싱 및 필터링 코드 예시

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

결측치 처리 (Handling Missing Values)

실제 수집된 데이터셋에는 누락된 값(NaN, None)이 자주 포함되어 있어 모델링이나 통계 분석 전 정제가 필수적입니다.

df.isnull() / df.isna(): 결측치 위치를 True로 반환합니다. (.sum()과 조합하여 컬럼별 결측치 개수 파악)

df.dropna(axis=0, how='any'): 결측치가 포함된 행(axis=0) 또는 열(axis=1)을 제거합니다.

df.fillna(value): 결측치를 특정 값(0, 평균값, 중앙값, 최빈값 등)으로 채웁니다.

결측치 정제 코드 예시

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

그룹화 및 집계 연산 (GroupBy & Aggregation)

데이터를 특정 범주형 컬럼의 고유값 기준으로 그룹화하여 대표 통계량을 산출하는 SQL의 GROUP BY 연산과 동일한 기능입니다.

Split-Apply-Combine 메커니즘:

df.groupby('기준컬럼'): 그룹화 기준 컬럼 지정

.agg() 메서드: 여러 컬럼에 대해 서로 다른 집계 함수를 동시에 적용할 수 있는 다중 집계 함수

GroupBy 집계 연산 코드 예시

# 부서(Department) 기준 그룹화 및 컬럼별 다중 집계 적용
dept_summary = df.groupby('Department').agg({
    'Salary': ['mean', 'sum'],  # Salary 컬럼: 평균 및 합계
    'Experience': 'max'         # Experience 컬럼: 최댓값
})

print("=== 부서별 집계 요약 (GroupBy + agg) ===")
print(dept_summary)

공식 문서 및 참고 링크

Pandas Official Guide - Indexing and Selecting Data

Pandas Official Guide - Working with Missing Data

Pandas Official Guide - Group By: split-apply-combine

