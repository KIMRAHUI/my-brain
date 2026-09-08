# EDA_1

호텔 예약 취소 EDA 및 예측 분석

분석 목표

호텔 예약 데이터를 분석하여 예약 취소 및 노쇼(No-Show)의 주요 원인을 파악하고, 이를 바탕으로 예약 취소율을 줄일 수 있는 구체적인 방안을 제안합니다.

분석 흐름

환경 설정: 분석 라이브러리 임포트 및 데이터 정제 준비

데이터 확인 및 정제: 데이터 로드, 중복 제거, 결측치 처리

데이터 분석: 수치형 및 범주형 변수와 취소 여부(is_canceled) 간의 상관관계 분석

주요 요인 심층 분석: lead_time, market_segment, 주차 공간, 특별 요청 등 핵심 변수 심층 탐색

결론 및 제안: 분석 결과 요약 및 예약 취소율 감소 전략 제시

1. 환경 설정

1.1. 라이브러리 임포트 및 기본 설정

# 데이터 분석 및 시각화에 필요한 라이브러리를 임포트합니다.
import numpy as np  # 숫자 연산을 위한 라이브러리
import pandas as pd  # 데이터프레임(DataFrame)을 효과적으로 다루기 위한 라이브러리
import seaborn as sns  # Matplotlib을 기반으로 한 고급 통계 시각화 라이브러리
import matplotlib.pyplot as plt  # 그래프 및 차트를 그리기 위한 라이브러리

# Seaborn의 'whitegrid' 스타일을 설정하여 그래프 배경에 격자를 추가합니다.
sns.set_style('whitegrid')

# Colab 환경에서 한글 폰트 설정
import sys
if 'google.colab' in sys.modules:
    !sudo apt-get install -y fonts-nanum  # 나눔 폰트 설치
    !sudo fc-cache -fv  # 폰트 캐시 새로 고침
    !rm ~/.cache/matplotlib -rf  # matplotlib 캐시 삭제
    print(">> 나눔 폰트 설치 완료. Colab의 경우 '런타임 다시 시작'을 해주세요.")

    plt.rc('font', family='NanumBarunGothic')  # matplotlib 폰트 설정
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

    import warnings
    warnings.filterwarnings('ignore') # 경고 메시지 무시

2. 데이터 확인 및 정제

2.1. 데이터 불러오기

import os

try:
    csv_path = 'hotel_data_modified.csv'

    if not os.path.isfile(csv_path):
        from google.colab import files
        uploaded = files.upload()
        csv_path = next(iter(uploaded))

    df = pd.read_csv(csv_path)

except (ModuleNotFoundError, StopIteration):
    print("file이 존재하지 않습니다. ")

print(f"\n>> 원본 데이터 로드 완료. 형태: {df.shape}")
df.head()
df.columns

2.2. 데이터 컬럼 설명

2.3. 데이터 정제 (중복 및 결측치 처리)

# 중복 데이터 제거
initial_rows = len(df)
df.drop_duplicates(inplace=True)
print(f">> 중복된 행 {initial_rows - len(df)}개를 제거했습니다.")

# 결측치 확인
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

# 결측치 처리 (수치형: 0, 범주형: 최빈값)
for miss in list(missing_values.index):
    if pd.api.types.is_numeric_dtype(df[miss]):
        df[miss].fillna(0, inplace=True)
    else:
        df[miss].fillna(df[miss].mode()[0], inplace=True)

print(">> 결측치 처리를 완료했습니다.")

3. 데이터 분석

3.1. 예약 취소 현황 개요

plt.figure(figsize=(7, 4))
ax = sns.countplot(x='is_canceled', data=df, palette='pastel')
plt.title('Overall Cancellation Status', fontsize=10)
plt.xlabel('Booking Canceled\n(0: No, 1: Yes)')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Canceled', 'Canceled'])

total = len(df['is_canceled'])
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')

plt.show()

분석 결과: 전체 예약 중 약 **27.7%**가 취소되었습니다.

3.2. 예약 취소와의 상관관계 분석

수치형 데이터 상관관계

corr_matrix = df.corr(numeric_only=True)
corr_with_cancel = corr_matrix['is_canceled'].sort_values(ascending=False, key=lambda x : x.abs())
print(corr_with_cancel)

# 상관관계 히트맵 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
plt.show()

주요 포인트: is_canceled를 제외하면 **lead_time**이 예약 취소와 가장 높은 상관관계를 보였으며, required_car_parking_spaces, total_of_special_requests 등이 뒤를 이었습니다.

범주형 데이터 비교 분석 (날짜 생성 및 시각화)

Python



month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' +
                                df['arrival_date_month'].map(month_mapping).astype(str) + '-' +
                                df['arrival_date_day_of_month'].astype(str))
df['arrival_day_of_week'] = df['arrival_date'].dt.day_name()

4. 주요 요인 심층 분석

4.1. Lead Time과 예약 취소

plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='lead_time', hue='is_canceled', fill=True, common_norm=False, palette='crest')
plt.title('Lead Time Distribution by Cancellation Status', fontsize=14)
plt.xlabel('Lead Time (days)')
plt.ylabel('Density')
plt.show()

해석: lead_time(예약 시점부터 도착일까지의 기간)이 길수록 취소될 확률이 높습니다. 너무 일찍 예약한 고객일수록 계획 변경 가능성이 큽니다.

4.2. Market Segment별 취소 현황

plt.figure(figsize=(14, 7))
ax = sns.countplot(x='market_segment', hue='is_canceled', data=df, palette='viridis', order=df['market_segment'].value_counts().index)
plt.title('Cancellation Status by Market Segment', fontsize=14)
plt.xticks(rotation=45)
plt.show()

해석: Online TA (온라인 여행사) 경로가 예약 건수와 취소 건수 모두에서 가장 높았으며, Groups(단체) 세그먼트도 취소 비율이 높았습니다.

4.3. 주차 공간 및 특별 요청사항과 취소율

# 주차 공간 필요 여부에 따른 취소율
cancel_rate_by_parking = df.groupby('required_car_parking_spaces')['is_canceled'].mean() * 100

# 특별 요청 수에 따른 취소율
cancel_rate_by_requests = df.groupby('total_of_special_requests')['is_canceled'].mean() * 100

해석:

5. 결론 및 제안

주요 분석 요약

lead_time이 길수록 취소율이 높음: 사전 예약 기간이 길수록 계획 변경으로 취소 확률 증가

Online TA 및 Groups의 높은 취소율: 온라인 여행사 및 단체 예약에서 취소 경향이 뚜렷함

주차 공간 및 특별 요청이 있는 경우 취소율이 낮음: 고객의 적극적인 요구사항은 예약 유지의 긍정적 지표임

예약 취소율 감소를 위한 전략 제안

제안 1: lead_time이 긴 예약 선제적 관리

제안 2: 보증금(Deposit) 정책 차별화

제안 3: No-Show 고객 이력 관리

제안 4: 사전 요구사항 유도

