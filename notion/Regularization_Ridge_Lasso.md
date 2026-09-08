# Regularization_Ridge_Lasso

개요

머신러닝 모델의 복잡도를 제어하여 과적합(Overfitting)을 방지하고, 새로운 데이터에 대한 일반화 성능을 극대화하기 위해 손실 함수에 패널티(Penalty)를 부여하는 규제(Regularization) 기법을 다룹니다. L2 규제 기반의 Ridge, L1 규제 기반의 Lasso, 그리고 두 방식을 결합한 ElasticNet의 수식과 특성을 비교 정리합니다.

규제(Regularization)의 개념과 목적

목적: 학습 데이터에 모델이 과도하게 적응하여 복잡해지는 과적합(Overfitting)을 억제하고, 오차 범위를 줄여 일반화 성능(Generalization)을 향상시키는 것입니다.

원리: 원본 손실 함수(Cost Function)에 가중치(\theta) 크기에 비례하는 규제항(Penalty term)을 추가합니다. 모델 학습 과정에서 가중치 값이 지나치게 커지는 것에 패널티를 부여함으로써, 특정 특성(Feature)에 과도하게 의존하는 경향을 제어합니다.

Norm 특강 (벡터의 크기 측정 방식)

규제항 수식을 이해하기 위한 기본 벡터 크기 측정 지표입니다.

L2-Norm (유클리드 거리, Euclidean Distance):

L1-Norm (맨해튼 거리, Manhattan Distance):

릿지(Ridge)와 라쏘(Lasso), 에라스틱넷(ElasticNet) 회귀 수식 비교

① Ridge Regression (선형 회귀 + L2 규제)

손실 함수 (Cost Function):
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n} \theta_j^2

특징 및 효과:

② Lasso Regression (선형 회귀 + L1 규제)

손실 함수 (Cost Function):
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n} \vert{}\theta_j\vert{}

특징 및 효과:

③ ElasticNet (L1 + L2 혼합 규제)

손실 함수 (Cost Function):
J(\theta) = \text{MSE} + r \cdot \lambda \sum_{j=1}^{n} \vert{}\theta_j\vert{} + \frac{1 - r}{2} \cdot \lambda \sum_{j=1}^{n} \theta_j^2

특징 및 효과:

Python 코드로 보는 Ridge, Lasso, ElasticNet 비교

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 로드 및 전처리
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 규제 모델 적용 시 스케일링(Standardization) 필수
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Ridge (L2 규제)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# 2. Lasso (L1 규제 - 자동 변수 선택)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# 3. ElasticNet (L1 + L2 규제 혼합)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)

print("=== 모델별 가중치(Coefficients) 비교 ===")
print("Ridge 가중치 :", ridge.coef_)
print("Lasso 가중치 :", lasso.coef_)       # 0으로 변한 가중치 확인 가능
print("ElasticNet 가중치:", elastic.coef_)

공식 문서 및 참고 링크

Medium - Regularization Techniques: Lasso and Ridge

Scikit-learn Official Guide - Generalized Linear Models (Ridge, Lasso, ElasticNet)

Scikit-learn API Reference - sklearn.linear_model.ElasticNet

