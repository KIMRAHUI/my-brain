---
notion-url: https://app.notion.com/p/data_preprocessing_and_eda-9af77fa64f7f8310af9701315d49b174
title: data_preprocessing_and_eda
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/data_preprocessing_and_eda-9af77fa64f7f8310af9701315d49b174
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- **정확성 (Accuracy):** 실제 현실 세계의 사실/수치와 데이터 값이 일치하는 정도 (예: 실제 연봉과 DB 기록 연봉의 일치 여부)

- **완전성 (Completeness):** 필수적으로 존재해야 하는 데이터 항목에 누락(결측치)이 없는 정도

- **적시성 (Timeliness):** 데이터가 필요한 시점에 지연 없이 수집되고 최신 상태를 유지하는 정도

- **일관성 (Consistency):** 여러 시스템이나 테이블 간에 동일한 데이터가 서로 모순되지 않고 일치하는 정도 (예: 회원 테이블과 주문 테이블의 주소 일치)

- **유효성 (Validity):** 데이터가 정의된 형식, 범위, 규칙을 준수하는 정도 (예: 이메일 양식 `@` 포함 여부, 나이 0~150 범주)

- **고유성 (Uniqueness):** 동일한 관찰 대상이 중복 기록되지 않고 식별 가능한 정도 (예: 중복 회원가입 제거)

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- **정의:** 측정 장비의 오류, 통신 장애 등으로 인해 데이터에 입력된 무작위적 정적 오류 또는 왜곡

- **해결법:** 이동 평균(Moving Average), 평활화(Smoothing), 필터링 기법 적용

[//]: # (heading_4 is not supported)

- **정의:** 데이터 수집 과정에서 값이 누락된 상태 (`NaN`, `Null`)

- **처리 기법:**

	- **삭제 (Deletion):** 전체 행 삭제(Listwise) 또는 해당 컬럼 삭제 (결측 비율이 50% 이상일 때 사용)

	- **단순 대입 (Simple Imputation):** 평균(Mean), 중앙값(Median), 최빈값(Mode) 대입

	- **고급 대입 (Advanced Imputation):** K-NN 기반 대입, 회귀 모델 기반 대입, MICE(Multivariate Imputation)

- **정의:** 대다수의 데이터 범주에서 크게 벗어난 극단적인 값 (수집 오류일 수도 있고, 실제 희귀 이벤트일 수도 있음)

- **탐지 방법:**

	1. **IQR (Interquartile Range) 방식:**$ \text{IQR} = Q3 - Q1 $
$ \text{정상 범위: } [Q1 - 1.5 \times \text{IQR}, \ Q3 + 1.5 \times \text{IQR}] $

	1. **Z-Score 방식:** 데이터가 표준정규분포를 따른다고 가정할 때 $ \vert{}Z\vert{} > 3 $ 이상인 데이터 탐지

- **처리 기법:** 제거, 상/하한값으로 캡핑(Capping/Winsorization), 로그 변환(Log Transformation)

---

[//]: # (heading_4 is not supported)

- **데이터 병합 (Merge/Join):** 여러 데이터 원천(테이블)을 공통 키(Key) 기준으로 조인

- **파생 변수 생성 (Feature Engineering):** 기존 변수들을 조합하여 분석 목적에 맞는 새로운 속성 추출 (예: '생년월일' → '현재 나이')

- **재구조화 (Pivoting/Melt):** 데이터의 행과 열 구조를 뒤바꾸어 분석하기 쉬운 형태로 변환

- **리샘플링 (Resampling):** 시계열 데이터의 주기를 변경(일별 → 월별)하거나 데이터 불균형을 해소하기 위한 오버/언더샘플링

---

[//]: # (heading_4 is not supported)

- **정의:** 통계학자 존 튜키(John W. Tukey)가 제안한 접근법으로, 가설을 세우기에 앞서 기초 통계량과 시각화를 통해 데이터 자체의 구조, 분포, 패턴, 이상점 등을 다각도로 탐색하는 과정입니다.

- **주요 도구:** 박스플롯(Boxplot), 히스토그램, 산점도(Scatter Plot), 상관관계 히트맵

- **EDA 수행 시 주의사항:**

	- **시각화의 함정:** 잘못된 축 범위 설정이나 편향된 차트 선택으로 인해 데이터의 본질이 왜곡될 수 있습니다.

	- **생존자 편향 (Survivorship Bias):** 살아남은(수집된) 데이터만 가지고 분석하여 전체 집단을 오해하는 오류를 경계해야 합니다.

---

[//]: # (heading_4 is not supported)


```python
import pandas as pd
import numpy as np

# 1. 샘플 데이터셋 생성 (결측치 및 이상치 포함)
df = pd.DataFrame({
    'Age': [22, 25, 29, np.nan, 32, 35, 120], # 120: 이상치, np.nan: 결측치
    'Income': [3000, 3200, 3500, 4000, 4200, 4500, 50000] # 50000: 이상치
})

# 2. 결측치 처리 (중앙값 대체)
df['Age'] = df['Age'].fillna(df['Age'].median())

# 3. IQR 방식을 활용한 이상치 탐지 및 정제 함수
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # 정상 범위 내의 데이터만 추출
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df_cleaned = remove_outliers_iqr(df, 'Income')

print("=== 정제 완료된 데이터프레임 ===")
print(df_cleaned)
```

---

[//]: # (heading_4 is not supported)

- [Scikit-Learn Official Guide - Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)

- [Pandas Official Guide - Working with Missing Data](https://pandas.pydata.org/docs/user_guide/missing_data.html)

- [NIST Engineering Statistics Handbook - Exploratory Data Analysis (EDA)](https://www.itl.nist.gov/div898/handbook/eda/eda.htm)

