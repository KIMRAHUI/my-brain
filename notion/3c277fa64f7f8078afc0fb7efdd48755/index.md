---
notion-url: https://app.notion.com/p/pandas_series_and_dataframe-3c277fa64f7f8078afc0fb7efdd48755
title: pandas_series_and_dataframe
date: '2026-08-20 08:29:00.000'
from_notion: https://app.notion.com/p/pandas_series_and_dataframe-3c277fa64f7f8078afc0fb7efdd48755
author: rahui kim
last_edited_time: '2026-08-20 08:31:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- **정의:** 1차원 배열 형태의 데이터 구조로, 실제 데이터 값의 나열인 값(Values)과 각 값에 대응하는 고유 이름인 인덱스(Index)로 구성됩니다. (엑셀의 '열(Column)' 하나에 해당)

- **특징:**

	- 동일한 데이터 타입(Homogeneous)의 데이터만 담을 수 있습니다.

	- 위치 기반 정수 인덱스와 명시적 레이블 인덱스를 모두 지원합니다.

[//]: # (heading_4 is not supported)


```python
import pandas as pd

# 1) 기본 Series 생성 (인덱스 미지정 시 0부터 시작하는 정수 인덱스 자동 부여)
s_default = pd.Series([10, 20, 30, 40])
print("=== 기본 Series ===")
print(s_default)

# 2) 커스텀 인덱스 지정 Series 생성
s_custom = pd.Series([100, 200, 300], index=['a', 'b', 'c'])
print("\n=== 커스텀 인덱스 Series ===")
print(s_custom)

# 3) 값 및 인덱스 조회
print("\n'b' 인덱스의 값:", s_custom['b'])  # 출력: 200
print("Series의 값(Values):", s_custom.values)
print("Series의 인덱스(Index):", s_custom.index)
```

---

[//]: # (heading_4 is not supported)

- **정의:** 행(Row)과 열(Column)로 이루어진 2차원 표(Table) 형태의 데이터 구조입니다. (엑셀 시트, SQL 테이블, R의 DataFrame과 동일한 개념)

- **특징:**

	- 서로 다른 데이터 타입(정수, 문자열, 실수, 불리언 등)을 열마다 다르게 가질 수 있습니다.

	- 각 열(Column)은 하나의 독립된 `Series` 객체로 이루어져 있습니다.

[//]: # (heading_4 is not supported)


```python
# 1) 딕셔너리(Dictionary)를 활용한 DataFrame 생성
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'Score': [85.5, 90.0, 78.5, 92.0],
    'Passed': [True, True, False, True]
}

df = pd.DataFrame(data)
print("=== DataFrame 전체 출력 ===")
print(df)

# 2) 특정 열(Series) 추출
print("\n=== Name 열 추출 ===")
print(df['Name'])
```

---

[//]: # (heading_4 is not supported)

- `**df.head(n)**`**:** 상위 n개 행을 조회합니다. (기본값 5개)

- `**df.tail(n)**`**:** 하위 n개 행을 조회합니다. (기본값 5개)

- `**df.shape**`**:** 데이터프레임의 행과 열 개수를 튜플 `(행, 열)` 형태로 반환합니다.

- `**df.info()**`**:** 전체 행/열 개수, 각 컬럼별 데이터 타입(`dtype`), 결측치를 제외한 유효 데이터 수(Non-Null Count), 메모리 사용량을 종합적으로 출력합니다.

- `**df.describe()**`**:** 수치형(Numeric) 컬럼들의 기초 통계량(개수, 평균, 표준편차, 최솟값, 25%/50%/75% 사분위수, 최댓값)을 요약합니다. (범주형 컬럼은 `include='all'` 옵션 적용 필요)

[//]: # (heading_4 is not supported)


```python
# 1) 데이터 구조 파악
print("=== 데이터 차원 (Shape) ===")
print(df.shape)  # 출력: (4, 4)

# 2) 상위 데이터 확인
print("\n=== 상위 2개 데이터 (head) ===")
print(df.head(2))

# 3) 데이터 요약 정보 확인
print("\n=== 데이터 요약 정보 (info) ===")
df.info()

# 4) 수치형 데이터 기초 통계량 확인
print("\n=== 기초 통계량 (describe) ===")
print(df.describe())
```

---

[//]: # (heading_4 is not supported)

- [Pandas Official User Guide - Data Structures Intro](https://pandas.pydata.org/docs/user_guide/dsintro.html)

- [Pandas API Reference - ](https://pandas.pydata.org/docs/reference/api/pandas.Series.html)`[pandas.Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html)`

- [Pandas API Reference - ](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)`[pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)`

