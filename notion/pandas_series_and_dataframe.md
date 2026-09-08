# pandas_series_and_dataframe

개요

파이썬 데이터 분석의 표준이자 가장 강력한 라이브러리인 Pandas의 2대 핵심 자료구조인 Series와 DataFrame의 개념, 생성 방법, 그리고 데이터 구조 파악을 위한 기초 통계 메서드를 상세히 학습합니다.

Series (시리즈)

정의: 1차원 배열 형태의 데이터 구조로, 실제 데이터 값의 나열인 값(Values)과 각 값에 대응하는 고유 이름인 인덱스(Index)로 구성됩니다. (엑셀의 '열(Column)' 하나에 해당)

특징:

Series 생성 및 활용 코드 예시

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

DataFrame (데이터프레임)

정의: 행(Row)과 열(Column)로 이루어진 2차원 표(Table) 형태의 데이터 구조입니다. (엑셀 시트, SQL 테이블, R의 DataFrame과 동일한 개념)

특징:

DataFrame 생성 및 활용 코드 예시

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

DataFrame 구조 확인 및 기초 통계 메서드

외부 데이터(CSV, Excel 등)를 불러오거나 새로운 DataFrame을 생성한 직후, 전체적인 데이터의 구조와 품질을 검증하기 위한 필수 탐색 메서드입니다.

df.head(n): 상위 n개 행을 조회합니다. (기본값 5개)

df.tail(n): 하위 n개 행을 조회합니다. (기본값 5개)

df.shape: 데이터프레임의 행과 열 개수를 튜플 (행, 열) 형태로 반환합니다.

df.info(): 전체 행/열 개수, 각 컬럼별 데이터 타입(dtype), 결측치를 제외한 유효 데이터 수(Non-Null Count), 메모리 사용량을 종합적으로 출력합니다.

df.describe(): 수치형(Numeric) 컬럼들의 기초 통계량(개수, 평균, 표준편차, 최솟값, 25%/50%/75% 사분위수, 최댓값)을 요약합니다. (범주형 컬럼은 include='all' 옵션 적용 필요)

구조 및 통계 탐색 실행 예시

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

공식 문서 및 참고 링크

Pandas Official User Guide - Data Structures Intro

Pandas API Reference - pandas.Series

Pandas API Reference - pandas.DataFrame

