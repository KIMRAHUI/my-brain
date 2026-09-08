---
notion-url: https://app.notion.com/p/Ensemble_Learning-b9277fa64f7f825d9a310176e498d944
title: Ensemble_Learning
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Ensemble_Learning-b9277fa64f7f825d9a310176e498d944
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)


```plain text
               ┌── Voting   : 서로 다른 유형의 알고리즘 모델들을 결합하여 투표
               ├── Bagging  : 부트스트랩 샘플링 기반 병렬 학습 (Random Forest)
앙상블 (Ensemble) ┼── Boosting : 순차 학습하며 오차/잔차에 집중 (AdaBoost, GBM, XGB, LGBM)
               └── Stacking : 여러 모델의 예측 결과를 입력으로 사용하는 메타 모델 학습
```

- **보팅 (Voting):** 서로 다른 알고리즘 모델들의 예측을 종합합니다.

	- **Hard Voting:** 다수결 원칙에 따라 가장 많은 모델이 선택한 클래스로 결정합니다.

	- **Soft Voting:** 각 모델이 예측한 클래스별 확률의 평균을 구해 가장 확률이 높은 클래스로 결정합니다.

- **배깅 (Bagging):** **Bootstrap Aggregating**의 약자로, 원본 데이터셋에서 중복을 허용하여(복원 추출) 수집한 복수의 데이터셋으로 약한 학습기들을 독립적으로 학습시킨 뒤 결과를 집계(평균 또는 다수결)합니다.

- **부스팅 (Boosting):** 약한 학습기들을 순차적으로 학습시키며, 이전 모델이 틀린 오답에 가중치를 부여하거나 잔차(Residual)를 학습하여 가중 합산합니다.

---

[//]: # (heading_4 is not supported)

- **지니 불순도 (Gini Impurity):** 데이터셋 안에 서로 다른 클래스가 얼마나 섞여 있는지를 측정합니다. 0일 때 가장 순수합니다.$ G(S) = 1 - \sum_{i=1}^{c} p_i^2 $

- **엔트로피 (Entropy):** 데이터셋의 불확실성/무질서도를 측정합니다.$ \text{Entropy}(A) = -\sum_{k=1}^{m} p_k \log_2(p_k) $

- **정보 이득 (Information Gain):** 분할 전 부모 노드의 불순도에서 분할 후 자식 노드들의 가중 평균 불순도를 뺀 값으로, 이 정보 이득이 최대가 되는 특성과 분기 기준을 탐색합니다.

---

[//]: # (heading_4 is not supported)

1. **Bootstrap 샘플링:** 원본 데이터 N개로부터 중복을 허용하여 N개의 샘플 데이터를 무작위 추출합니다.

1. **Feature 무작위 선택:** 전체 입력 특성(Feature) M개 중 중복 없이 무작위로 d개의 특성만 선택하여 트리를 수행함으로써 트리 간의 상관관계를 낮추고 일반화 성능을 높입니다.

1. **Aggregation:** 생성된 K개의 트리의 예측 결과(회귀: 평균, 분류: 최빈값/다수결)를 종합합니다.

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- **원리:** 약한 학습기(깊이가 1인 **Decision Stump**)를 사용하며, 이전 스텀프가 잘못 예측한 데이터의 가중치를 높이고 맞춘 데이터의 가중치를 낮춰 다음 스텀프가 오답에 집중하도록 학습시킵니다.

- **최종 예측:** 각 스텀프의 성능(오차율 기반)에 따라 비중을 다르게 부여하여 가중 합산합니다.

[//]: # (heading_4 is not supported)

- **원리:** AdaBoost처럼 오답 가중치를 조정하는 대신, 실제값과 현재 모델 예측값의 차이인 잔차(Residual / Negative Gradient)를 다음 트리가 학습하도록 경사하강법(Gradient Descent)을 적용합니다.

- **단점:** 순차적 계산으로 인한 과적합(Overfitting) 위험이 있고, 대용량 데이터 학습 속도가 느립니다.

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- **손실 함수 2차 근사 (2nd Order Taylor Expansion):** 손실 함수를 1차 미분값(Gradient)과 2차 미분값(Hessian)으로 근사하여 최적화 속도를 끌어올립니다.

- **규제(Regularization) 내장:** 손실 함수에 리프 노드 수 및 가중치에 대한 L1/L2 규제항을 추가하여 과적합을 정밀하게 제어합니다.

- **Sparsity-aware Split Finding:** One-Hot Encoding 등으로 발생한 희소 데이터(Sparse Data)나 결측값(Missing Value)을 처리할 때 기본 방향(Default Direction)을 자동 학습하여 연산 속도를 대폭 향상시킵니다.

[//]: # (heading_4 is not supported)

- **Leaf-wise 트리 성장 (리프 중심 분할):** 기존의 Level-wise(수평적 균형 성장) 대신, 손실 감소(Gain)가 가장 큰 리프 노드를 지속적으로 우선 분할하여 적은 연산으로 높은 예측 성능을 달성합니다.

- **GOSS (Gradient-based One-Side Sampling):** 기울기(Gradient)가 큰 데이터(오차가 큰 데이터)는 100% 유지하고, 기울기가 작은 데이터는 일부만 무작위 샘플링한 뒤 가중치를 조정하여 전체 분포 왜곡 없이 학습 속도를 극대화합니다.

- **EFB (Exclusive Feature Bundling):** 동시에 0이 아닌 값을 가질 확률이 거의 없는 상호 배타적인 희소 특성들을 하나로 묶어(Bundle) 특성 차원 수를 획기적으로 줄입니다.

---

[//]: # (heading_4 is not supported)


```python
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 샘플 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Random Forest (Bagging)
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
rf.fit(X_train, y_train)

# 2. AdaBoost
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train, y_train)

# 3. GBM
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbm.fit(X_train, y_train)

# 4. XGBoost
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)

# 5. LightGBM (Leaf-wise 성장)
lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42)
lgbm.fit(X_train, y_train)
```

---

[//]: # (heading_4 is not supported)

- [Scikit-learn Ensemble Methods Guide](https://scikit-learn.org/stable/modules/ensemble.html)

- [XGBoost Official Documentation](https://xgboost.readthedocs.io/)

- [LightGBM Official Documentation](https://lightgbm.readthedocs.io/)

