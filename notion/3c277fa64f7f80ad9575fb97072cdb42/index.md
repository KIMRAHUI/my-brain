---
notion-url: https://app.notion.com/p/EDA_2-3c277fa64f7f80ad9575fb97072cdb42
title: EDA_2
date: '2026-08-20 09:14:00.000'
from_notion: https://app.notion.com/p/EDA_2-3c277fa64f7f80ad9575fb97072cdb42
author: rahui kim
last_edited_time: '2026-08-20 09:28:00.000'
---
# 자전거 대여 수요 예측 및 EDA 분석 보고서

## 1. 미션 배경 및 목표

- **목표**: 자전거 대여 패턴을 분석하여 배치 및 운영 전략을 최적화하고, 대여 수요를 정확히 예측하여 시스템 효율성과 사용자 만족도를 증가시킵니다.

- **최종 목표**: RMSLE (Root Mean Squared Logarithmic Error)를 최대한 낮추는 정확한 수요 예측 모델을 개발합니다.

### RMSLE란?

- **정의**: 예측값과 실제값에 각각 로그를 취해 변환한 후 그 차이의 제곱 평균에 제곱근을 구한 값입니다. 큰 스케일과 작은 스케일의 값을 동등하게 다루며, 과소평가(실제보다 적게 예측) 시 더 큰 페널티를 부과합니다.

- **공식**:

	- **장점**:

	1. **안정성**: 0 또는 0에 가까운 값에 1을 더해 로그 변환을 하므로 무한대 발산을 방지합니다.

	1. **과소평가 방지**: 재고 부족 등 과소평가 리스크가 큰 비즈니스 환경에 적합합니다.

# 2. 데이터 이해 및 전처리

## 2.1 데이터 형태 및 기본 탐색

- **데이터 구성**: 2011~2012년 2년 동안의 자전거 공유 대여 시스템 데이터(`train.csv`, `test.csv`)

- **종속 변수**: `count` (casual과 registered의 합)


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from scipy.stats import shapiro, ttest_ind
import scipy.stats as stats
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from I.display import display

# 데이터 로드
df_train = pd.read_csv("train (1).csv")
df_test = pd.read_csv("test (1).csv")

# 데이터 결합 (모델링 전 준비)
data = pd.concat([df_train, df_test], axis=0)
data.reset_index(inplace=True, drop=True)
```

## 2.2 이상치 확인 및 정제

- **습도(humidity)**: 상대습도가 0인 경우는 지상에서 거의 보기 힘든 비정상 데이터이므로 제거합니다.

- **풍속(windspeed)**: 상계값을 초과하는 이상치는 중앙값으로 대체합니다.


```python
# 습도 0인 행 제거
data = data[data['humidity'] != 0]

# 풍속 이상치 처리 (상계값 기준 초과 시 중앙값 대체)
Q1 = data['windspeed'].quantile(0.25)
Q3 = data['windspeed'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

median_value = data[data['windspeed'] <= upper_bound]['windspeed'].median()
data.loc[data['windspeed'] > upper_bound, 'windspeed'] = median_value
```

# 3. 기본 EDA (탐색적 데이터 분석)

## 3.1 시간 요인별 대여 패턴

- **시간대별(24H)**: 출근 시간(8시) 및 퇴근 시간(17~18시)에 활발한 대여가 일어납니다. 근무일은 전체 대여 추이와 유사하며, 비근무일은 오후까지 상승하다 저녁에 하락하는 완만한 형태를 보입니다.

- **월별**: 5~10월이 피크 시기이며 200,000 전후의 대여 건수를 보입니다.

- **연도별**: 2011년에 비해 2012년의 대여 댓수가 훨씬 많습니다(40만대 이상 차이).


```python
# datetime 파생 변수 생성
df_train['datetime'] = pd.to_datetime(df_train['datetime'])
df_train['year'] = df_train['datetime'].dt.year
df_train['month'] = df_train['datetime'].dt.month
df_train['week'] = df_train['datetime'].dt.isocalendar().week
df_train['weekday'] = df_train['datetime'].dt.dayofweek
df_train['day'] = df_train['datetime'].dt.day
df_train['hour'] = df_train['datetime'].dt.hour
```

## 3.2 날씨 및 기후 요소별 패턴

- **온도/체감온도**: 15~29도 부근에서 대여 건수 및 댓수가 최댓값을 기록합니다.

- **습도**: 습도가 높아질수록 변동폭이 커지며, 80% 중반부터 감소합니다.

- **풍속**: 10m/s에서 20m/s 구간에서 급격한 감소세를 보입니다.

- **날씨 종류**: '맑음'이 압도적이며, 폭우/폭설(날씨 4)은 데이터상 거의 존재하지 않습니다.

# 4. 심화 EDA 및 사용자 행동 분석

## 4.1 계절 전이에 따른 대여 대수

- 여름 시작월인 **6월**에 자전거를 매우 많이 타는 것으로 나타납니다.

- 가을→겨울 전환 시점(12월)이 9월이나 3월 전환보다 더 많은 대여수를 보이는 독특한 특징이 있습니다.

## 4.2 날씨 전이에 따른 대여 대수

- `맑음 ↔ 구름낌/안개` 전이는 신속한 장소 이동 및 기분 전환 목적으로 대여량이 매우 많습니다.

- 2011년과 2012년 데이터를 드릴다운하여 비교했을 때, 2012년이 전반적인 날씨 변화 전이 구간에서 대여량이 크게 증가했습니다.

## 4.3 상관관계 및 회귀 분석

- 습도와 풍속은 대여량과 약한 음의 관계, 온도/체감온도는 약한 양의 관계를 보입니다.

- 체감온도-온도 차이에 따른 다항 회귀를 수행해본 결과, 0~5 사이 값에 밀집되어 있으며 대여량과의 직접적인 선형/다항 상관관계는 뚜렷하지 않습니다. 반면 습도는 뚜렷한 음의 관계를 확인했습니다.

# 5. 모델링 및 성능 최적화

## 5.1 RMSLE 평가 함수 설정


```python
def rmsle(y, y_, convertExp=True):
    if convertExp:
        y = np.exp(y)
        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))
```

## 5.2 데이터 분할 및 인코딩


```python
# 카테고리형 변수 변환
categoricalFeatureNames = ["season", "holiday", "workingday", "weather", "weekday", "month", "year", "hour"]
for var in categoricalFeatureNames:
    data[var] = data[var].astype("category")

# 독립/종속 변수 분리
dropFeatures = ['casual', "count", "datetime", "date", "registered"]
dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])
dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])

yLabels = dataTrain["count"]
dataTrain = dataTrain.drop(dropFeatures, axis=1)
dataTest = dataTest.drop(dropFeatures, axis=1)

# 원-핫 인코딩
dataTrain_encoded = pd.get_dummies(dataTrain, drop_first=True)
yLabelsLog = np.log1p(yLabels)
```

## 5.3 회귀 모델 학습 (Linear, Ridge, Lasso)


```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# 1. 선형 회귀
lModel = LinearRegression()
lModel.fit(X=dataTrain_encoded, y=yLabelsLog)
preds = lModel.predict(X=dataTrain_encoded)
print("RMSLE Value For Linear Regression: ", rmsle(np.exp(yLabelsLog), np.exp(preds), False))

# 2. 릿지 회귀 (Ridge)
ridge_m_ = Ridge()
ridge_params_ = {'max_iter': [3000], 'alpha': [0.1, 1, 5, 10, 20]}
grid_ridge_m = GridSearchCV(ridge_m_, ridge_params_, scoring=metrics.make_scorer(rmsle, greater_is_better=False), cv=5)
grid_ridge_m.fit(dataTrain_encoded, yLabelsLog)
preds_ridge = grid_ridge_m.predict(dataTrain_encoded)
print("RMSLE Value For Ridge Regression:", rmsle(np.exp(yLabelsLog), np.exp(preds_ridge), False))

# 3. 라쏘 회귀 (Lasso)
lasso_m_ = Lasso()
alpha = 1 / np.array([0.1, 1, 5, 10, 20])
lasso_params_ = {'max_iter': [3000], 'alpha': alpha}
grid_lasso_m = GridSearchCV(lasso_m_, lasso_params_, scoring=metrics.make_scorer(rmsle, greater_is_better=False), cv=5)
grid_lasso_m.fit(dataTrain_encoded, yLabelsLog)
preds_lasso = grid_lasso_m.predict(dataTrain_encoded)
print("RMSLE Value For Lasso Regression:", rmsle(np.exp(yLabelsLog), np.exp(preds_lasso), False))
```

## 5.4 트리 기반 모델 학습 (Random Forest, Gradient Boosting)


```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 4. 랜덤 포레스트
rfModel = RandomForestRegressor(n_estimators=100, random_state=42)
rfModel.fit(dataTrain_encoded, yLabelsLog)
preds_rf = rfModel.predict(dataTrain_encoded)
print("RMSLE Value For Random Forest:", rmsle(np.exp(yLabelsLog), np.exp(preds_rf), False))

# 5. 그래디언트 부스팅
gbm = GradientBoostingRegressor(n_estimators=1000, alpha=0.01, random_state=42)
gbm.fit(dataTrain_encoded, yLabelsLog)
preds_gbm = gbm.predict(dataTrain_encoded)
print("RMSLE Value For Gradient Boost:", rmsle(np.exp(yLabelsLog), np.exp(preds_gbm), False))
```

## 5.5 테스트 데이터 예측 및 시각화


```python
# 테스트 데이터 인코딩 및 열 정렬 맞추기
dataTest_encoded = pd.get_dummies(dataTest, drop_first=True)
dataTest_encoded = dataTest_encoded.reindex(columns=dataTrain_encoded.columns, fill_value=0)

# 최종 예측
predsTest = rfModel.predict(dataTest_encoded)

# 실제값 vs 예측값 분포 비교 시각화
fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(12, 5)
sns.histplot(yLabels, ax=ax1, bins=50, kde=True)
ax1.set(title="Distribution of Actual Counts")

sns.histplot(np.exp(predsTest), ax=ax2, bins=50, kde=True)
ax2.set(title="Distribution of Predicted Counts")
plt.show()
```

<br/>

