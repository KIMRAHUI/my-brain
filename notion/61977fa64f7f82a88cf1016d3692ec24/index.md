---
notion-url: https://app.notion.com/p/EDA_1-61977fa64f7f82a88cf1016d3692ec24
title: EDA_1
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/EDA_1-61977fa64f7f82a88cf1016d3692ec24
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
# 호텔 예약 취소 EDA 및 예측 분석

## 분석 목표

## 분석 흐름

1. 환경 설정: 분석 라이브러리 임포트 및 데이터 정제 준비

1. 데이터 확인 및 정제: 데이터 로드, 중복 제거, 결측치 처리

1. 데이터 분석: 수치형 및 범주형 변수와 취소 여부(is_canceled) 간의 상관관계 분석

1. 주요 요인 심층 분석: lead_time, market_segment, 주차 공간, 특별 요청 등 핵심 변수 심층 탐색

1. 결론 및 제안: 분석 결과 요약 및 예약 취소율 감소 전략 제시

# 1. 환경 설정

## 1.1. 라이브러리 임포트 및 기본 설정


```python
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
```

# 2. 데이터 확인 및 정제

## 2.1. 데이터 불러오기


```python
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
```

## 2.2. 데이터 컬럼 설명

 | **컬럼명** | **설명** | **상세 정보** | 
 | ---- | ---- | ---- | 
 | `hotel` | 호텔 유형 (Resort Hotel 또는 City Hotel) | 항목: Resort Hotel, City Hotel | 
 | `is_canceled` | 예약 취소 여부 (0: 취소 안 됨, 1: 취소됨) | 최소값: 0.00, 최대값: 1.00, 중간값: 0.00 | 
 | `lead_time` | 예약 시점부터 도착 예정일까지의 일수 | 최소값: 0.00, 최대값: 737.00, 중간값: 69.00 | 
 | `arrival_date_year` | 도착 예정 연도 | 최소값: 2015.00, 최대값: 2017.00, 중간값: 2016.00 | 
 | `arrival_date_month` | 도착 예정 월 | 항목: July, August, September, October, November, ... (12개 항목) | 
 | `arrival_date_week_number` | 도착 예정 주차 | 최소값: 1.00, 최대값: 53.00, 중간값: 28.00 | 
 | `arrival_date_day_of_month` | 도착 예정 일 | 최소값: 1.00, 최대값: 31.00, 중간값: 16.00 | 
 | `stays_in_weekend_nights` | 예약된 주말 밤(토, 일) 숙박 일수 | 최소값: 0.00, 최대값: 19.00, 중간값: 1.00 | 
 | `stays_in_week_nights` | 예약된 주중 밤(월~금) 숙박 일수 | 최소값: 0.00, 최대값: 50.00, 중간값: 2.00 | 
 | `adults` | 성인 수 | 최소값: 0.00, 최대값: 55.00, 중간값: 2.00 | 
 | `children` | 어린이 수 | 최소값: 0.00, 최대값: 10.00, 중간값: 0.00 | 
 | `babies` | 유아 수 | 최소값: 0.00, 최대값: 10.00, 중간값: 0.00 | 
 | `meal` | 예약된 식사 유형 (BB, FB, HB, SC 등) | 항목: BB, FB, HB, SC, Undefined | 
 | `country` | 고객의 출신 국가 | 항목 (상위 5개): GBR, PRT, IRL, FRA, ROU, ... (178개 항목) | 
 | `market_segment` | 시장 세그먼트 (예 예약 경로) | 항목: Direct, Corporate, Online TA, Offline TA/TO, Aviation, ... | 
 | `distribution_channel` | 예약 유통 채널 | 항목: Direct, Corporate, TA/TO, Undefined, GDS | 
 | `is_repeated_guest` | 재방문 고객 여부 (0: 아님, 1: 재방문) | 최소값: 0.00, 최대값: 1.00, 중간값: 0.00 | 
 | `previous_cancellations` | 이전 취소 건수 | 최소값: 0.00, 최대값: 26.00, 중간값: 0.00 | 
 | `previous_bookings_not_canceled` | 이전 취소되지 않은 예약 건수 | 최소값: 0.00, 최대값: 72.00, 중간값: 0.00 | 
 | `reserved_room_type` | 예약된 객실 유형 | 항목: C, A, D, E, G, ... (9개 항목) | 
 | `assigned_room_type` | 배정된 객실 유형 | 항목: C, A, D, E, G, ... (12개 항목) | 
 | `booking_changes` | 예약 변경 횟수 | 최소값: 0.00, 최대값: 21.00, 중간값: 0.00 | 
 | `agent` | 예약 대리인 ID | 최소값: 0.00, 최대값: 535.00, 중간값: 14.00 | 
 | `company` | 예약 회사 ID | 최소값: 0.00, 최대값: 543.00, 중간값: 0.00 | 
 | `days_in_waiting_list` | 대기 목록에 있었던 일수 | 최소값: 0.00, 최대값: 391.00, 중간값: 0.00 | 
 | `required_car_parking_spaces` | 필요한 주차 공간 수 | 최소값: 0.00, 최대값: 8.00, 중간값: 0.00 | 
 | `total_of_special_requests` | 총 특별 요청 수 | 최소값: 0.00, 최대값: 5.00, 중간값: 0.00 | 
 | `reservation_status` | 예약 상태 (Canceled, Check-Out, No-Show) | 항목: Check-Out, Canceled, No-Show | 
 | `reservation_status_date` | 예약 상태 변경일 | 항목 (상위 5개): 2015-07-01, 2015-07-02, 2015-07-03, ... (926개 항목) | 

## 2.3. 데이터 정제 (중복 및 결측치 처리)


```plain text
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
```

# 3. 데이터 분석

## 3.1. 예약 취소 현황 개요


```python
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
```

> **분석 결과**: 전체 예약 중 약 **27.7%**가 취소되었습니다.

## 3.2. 예약 취소와의 상관관계 분석

### 수치형 데이터 상관관계


```python
corr_matrix = df.corr(numeric_only=True)
corr_with_cancel = corr_matrix['is_canceled'].sort_values(ascending=False, key=lambda x : x.abs())
print(corr_with_cancel)

# 상관관계 히트맵 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
plt.show()
```

> **주요 포인트**: `is_canceled`를 제외하면 **`lead_time`**이 예약 취소와 가장 높은 상관관계를 보였으며, `required_car_parking_spaces`, `total_of_special_requests` 등이 뒤를 이었습니다.

### 범주형 데이터 비교 분석 (날짜 생성 및 시각화)

# 


```plain text
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' +
                                df['arrival_date_month'].map(month_mapping).astype(str) + '-' +
                                df['arrival_date_day_of_month'].astype(str))
df['arrival_day_of_week'] = df['arrival_date'].dt.day_name()
```

# 4. 주요 요인 심층 분석

## 4.1. Lead Time과 예약 취소


```python
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='lead_time', hue='is_canceled', fill=True, common_norm=False, palette='crest')
plt.title('Lead Time Distribution by Cancellation Status', fontsize=14)
plt.xlabel('Lead Time (days)')
plt.ylabel('Density')
plt.show()
```

> **해석**: `lead_time`(예약 시점부터 도착일까지의 기간)이 길수록 취소될 확률이 높습니다. 너무 일찍 예약한 고객일수록 계획 변경 가능성이 큽니다.

## 4.2. Market Segment별 취소 현황


```python
plt.figure(figsize=(14, 7))
ax = sns.countplot(x='market_segment', hue='is_canceled', data=df, palette='viridis', order=df['market_segment'].value_counts().index)
plt.title('Cancellation Status by Market Segment', fontsize=14)
plt.xticks(rotation=45)
plt.show()
```

> **해석**: `**Online TA**`** (온라인 여행사)** 경로가 예약 건수와 취소 건수 모두에서 가장 높았으며, `**Groups**`(단체) 세그먼트도 취소 비율이 높았습니다.

## 4.3. 주차 공간 및 특별 요청사항과 취소율


```python
# 주차 공간 필요 여부에 따른 취소율
cancel_rate_by_parking = df.groupby('required_car_parking_spaces')['is_canceled'].mean() * 100

# 특별 요청 수에 따른 취소율
cancel_rate_by_requests = df.groupby('total_of_special_requests')['is_canceled'].mean() * 100
```

> **해석**:

	- 주차 공간이 1개 이상 필요한 예약은 취소율이 **0%**에 가까울 정도로 방문 의사가 확고합니다.

	- 특별 요청 수가 많을(`total_of_special_requests`가 높을)수록 취소율이 낮아지는 경향을 보입니다.

# 5. 결론 및 제안

### 주요 분석 요약

1. `**lead_time**`**이 길수록 취소율이 높음**: 사전 예약 기간이 길수록 계획 변경으로 취소 확률 증가

1. `**Online TA**`** 및 **`**Groups**`**의 높은 취소율**: 온라인 여행사 및 단체 예약에서 취소 경향이 뚜렷함

1. **주차 공간 및 특별 요청이 있는 경우 취소율이 낮음**: 고객의 적극적인 요구사항은 예약 유지의 긍정적 지표임

### 예약 취소율 감소를 위한 전략 제안

- **제안 1: **`**lead_time**`**이 긴 예약 선제적 관리**

	- 기준일(예: 100일) 이상 앞서 예약한 고객에게 숙박일 2주 전 리마인드 이메일 및 확인 메시지 발송

- **제안 2: 보증금(Deposit) 정책 차별화**

	- 취소율이 높은 `Online TA` 및 `Groups`에 대해 `Non-Refund`(환불 불가) 조건 또는 소액 보증금 제도 도입

- **제안 3: **`**No-Show**`** 고객 이력 관리**

	- 노쇼 고객 DB를 구축하여 재예약 시 보증금 청구 또는 특별 관리

- **제안 4: 사전 요구사항 유도**

	- 예약 단계에서 주차 요구나 특별 요청 항목을 적극적으로 입력하도록 유도하여 고객 관여도 및 확신 증대

