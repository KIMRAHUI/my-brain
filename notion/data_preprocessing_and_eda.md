# data_preprocessing_and_eda

개요

데이터 분석 프로젝트에서 가장 많은 시간이 소요되는 데이터 품질 관리, 전처리 프로세스(결측치·이상치·잡음 처리), 데이터 랭글링, 그리고 존 튜키(John Tukey)가 제안한 탐색적 데이터 분석(EDA) 방법론을 체계적으로 정리합니다.

데이터 품질 관리 6대 요소 (Data Quality Dimensions)

신뢰할 수 있는 데이터 분석 결과를 얻기 위해 검증해야 하는 6가지 품질 평가 기준입니다.

정확성 (Accuracy): 실제 현실 세계의 사실/수치와 데이터 값이 일치하는 정도 (예: 실제 연봉과 DB 기록 연봉의 일치 여부)

완전성 (Completeness): 필수적으로 존재해야 하는 데이터 항목에 누락(결측치)이 없는 정도

적시성 (Timeliness): 데이터가 필요한 시점에 지연 없이 수집되고 최신 상태를 유지하는 정도

일관성 (Consistency): 여러 시스템이나 테이블 간에 동일한 데이터가 서로 모순되지 않고 일치하는 정도 (예: 회원 테이블과 주문 테이블의 주소 일치)

유효성 (Validity): 데이터가 정의된 형식, 범위, 규칙을 준수하는 정도 (예: 이메일 양식 @ 포함 여부, 나이 0~150 범주)

고유성 (Uniqueness): 동일한 관찰 대상이 중복 기록되지 않고 식별 가능한 정도 (예: 중복 회원가입 제거)

데이터 품질 개선 및 정제 프로세스

① 잡음 (Noise)

정의: 측정 장비의 오류, 통신 장애 등으로 인해 데이터에 입력된 무작위적 정적 오류 또는 왜곡

해결법: 이동 평균(Moving Average), 평활화(Smoothing), 필터링 기법 적용

② 결측치 (Missing Value)

정의: 데이터 수집 과정에서 값이 누락된 상태 (NaN, Null)

처리 기법:

#####③ 이상치 (Outlier)

정의: 대다수의 데이터 범주에서 크게 벗어난 극단적인 값 (수집 오류일 수도 있고, 실제 희귀 이벤트일 수도 있음)

탐지 방법:

처리 기법: 제거, 상/하한값으로 캡핑(Capping/Winsorization), 로그 변환(Log Transformation)

데이터 랭글링과 매니퓰레이션 (Data Wrangling & Manipulation)

분석이나 머신러닝 모델링에 적합한 형태로 원천 데이터를 변환, 구조화, 정제하는 전체 프로세스입니다.

데이터 병합 (Merge/Join): 여러 데이터 원천(테이블)을 공통 키(Key) 기준으로 조인

파생 변수 생성 (Feature Engineering): 기존 변수들을 조합하여 분석 목적에 맞는 새로운 속성 추출 (예: '생년월일' → '현재 나이')

재구조화 (Pivoting/Melt): 데이터의 행과 열 구조를 뒤바꾸어 분석하기 쉬운 형태로 변환

리샘플링 (Resampling): 시계열 데이터의 주기를 변경(일별 → 월별)하거나 데이터 불균형을 해소하기 위한 오버/언더샘플링

탐색적 데이터 분석 (EDA, Exploratory Data Analysis)

정의: 통계학자 존 튜키(John W. Tukey)가 제안한 접근법으로, 가설을 세우기에 앞서 기초 통계량과 시각화를 통해 데이터 자체의 구조, 분포, 패턴, 이상점 등을 다각도로 탐색하는 과정입니다.

주요 도구: 박스플롯(Boxplot), 히스토그램, 산점도(Scatter Plot), 상관관계 히트맵

EDA 수행 시 주의사항:

Python 코드로 보는 전처리 및 EDA 실전 예시

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

공식 문서 및 참고 링크

Scikit-Learn Official Guide - Preprocessing Data

Pandas Official Guide - Working with Missing Data

NIST Engineering Statistics Handbook - Exploratory Data Analysis (EDA)

