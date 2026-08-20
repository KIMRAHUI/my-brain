---
notion-url: https://app.notion.com/p/Regularization_Ridge_Lasso-3c277fa64f7f808ea456cbc4095fd11a
title: Regularization_Ridge_Lasso
date: '2026-08-20 09:05:00.000'
from_notion: https://app.notion.com/p/Regularization_Ridge_Lasso-3c277fa64f7f808ea456cbc4095fd11a
author: rahui kim
last_edited_time: '2026-08-20 09:29:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- **목적:** 학습 데이터에 모델이 과도하게 적응하여 복잡해지는 과적합(Overfitting)을 억제하고, 오차 범위를 줄여 일반화 성능(Generalization)을 향상시키는 것입니다.

- **원리:** 원본 손실 함수(Cost Function)에 가중치(\theta) 크기에 비례하는 규제항(Penalty term)을 추가합니다. 모델 학습 과정에서 가중치 값이 지나치게 커지는 것에 패널티를 부여함으로써, 특정 특성(Feature)에 과도하게 의존하는 경향을 제어합니다.

---

[//]: # (heading_4 is not supported)

- **L2-Norm (유클리드 거리, Euclidean Distance):**

	- 각 벡터 원소의 제곱의 합에 제곱근을 씌운 값입니다.

	- $ ||x||_2 = sqrt(sum(x_i^2)) $

- **L1-Norm (맨해튼 거리, Manhattan Distance):**

	- 각 벡터 원소의 절댓값들의 합입니다.

	- $ ||x||_1 = sum(|x_i|) $

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- **손실 함수 (Cost Function):
**$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n} \theta_j^2 $

- **특징 및 효과:**

	- 가중치들의 크기를 축소시켜 **0에 가깝게 부드럽게(Shrinkage)** 만듭니다.

	- 가중치를 완전히 0으로 만들지는 않으므로 모든 특성을 계속 유지합니다.

	- 특성 간 **다중공선성(Multicollinearity)** 문제가 존재할 때 안정적인 성능을 제공합니다.

[//]: # (heading_4 is not supported)

- **손실 함수 (Cost Function):
**$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n} \vert{}\theta_j\vert{} $

- **특징 및 효과:**

	- 중요도가 낮은 특성의 가중치를 **완전히 0**으로 만듭니다.

	- 0이 된 특성을 제거함으로써 자동으로 **Feature Selection(변수 선택)** 및 희소성(Sparsity) 확보 효과를 얻습니다.

[//]: # (heading_4 is not supported)

- **손실 함수 (Cost Function):
**$ J(\theta) = \text{MSE} + r \cdot \lambda \sum_{j=1}^{n} \vert{}\theta_j\vert{} + \frac{1 - r}{2} \cdot \lambda \sum_{j=1}^{n} \theta_j^2 $

- **특징 및 효과:**

	- Ridge와 Lasso의 규제항을 동시에 결합한 방식입니다 (r은 L1 규제 비율).

	- 특성의 수(n)가 샘플 수(m)보다 많거나, 변수 간 상관관계가 매우 높은 데이터셋에서 Lasso의 단점을 보완하며 우수한 성능을 보입니다.

---

[//]: # (heading_4 is not supported)


```python
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
```

---

[//]: # (heading_4 is not supported)

- [Medium - Regularization Techniques: Lasso and Ridge](https://medium.com/@m.shihab.akhtar/regularization-techniques-lasso-and-ridge-185e3fbe3f6d)

- [Scikit-learn Official Guide - Generalized Linear Models (Ridge, Lasso, ElasticNet)](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)

- [Scikit-learn API Reference - ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)`[sklearn.linear_model.ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)`

