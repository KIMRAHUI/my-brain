---
notion-url: https://app.notion.com/p/Linear_Regression_and_Gradient_Descent-3c277fa64f7f808193d7f6f793afc02a
title: Linear_Regression_and_Gradient_Descent
date: '2026-08-20 09:03:00.000'
from_notion: https://app.notion.com/p/Linear_Regression_and_Gradient_Descent-3c277fa64f7f808193d7f6f793afc02a
author: rahui kim
last_edited_time: '2026-08-20 09:28:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- **정의:** 하나 이상의 입력 변수(X)와 연속적인 목표 변수(Y) 사이의 관계를 최적의 직선(또는 초평면) 방정식 형태로 표현하는 모델입니다.

- **가설 함수 수식 (Hypothesis Function):**h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n = \theta^T x

	- \theta_0: **편향 (Bias)** - 모든 특성이 0일 때의 기본 예측값

	- \theta_1, \theta_2, \dots, \theta_n: **가중치 (Weight / 회귀계수)** - 각 입력 변수가 예측값에 미치는 영향력 크기

	- \theta^T x: 가중치 벡터 \theta와 입력 벡터 x의 행렬 내적 표현

---

[//]: # (heading_4 is not supported)

- **목적:** 훈련 데이터의 전체 실제값(y)과 모델의 예측값(h_\theta(x)) 사이의 오차(Residual)를 종합적으로 평가하기 위함입니다.

[//]: # (heading_4 is not supported)

- **수식 구성 요소 및 원리:**

	1. **오차의 제곱 \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2:** 오차(예측값 - 실제값)를 그대로 더하면 양의 오차와 음의 오차가 상쇄되어 전체 오차가 0에 가까워지는 왜곡이 발생합니다. 이를 방지하고 큰 오차에 더 큰 패널티를 부여하기 위해 제곱을 취합니다.

	1. **\frac{1}{m} 계수:** 전체 m개 데이터 샘플의 오차 제곱합을 데이터 개수로 나누어 **평균 오차**로 만들어 줍니다.

	1. **\frac{1}{2} 계수:** 경사하강법 적용 시 손실 함수 J(\theta)를 미분할 때 발생되는 미분 계수 2 (\frac{d}{d\theta} \theta^2 = 2\theta)와 서로 상쇄시켜 연산 수식을 간결하게 만들기 위한 수학적 조치입니다.

---

[//]: # (heading_4 is not supported)

- **목적:** 손실 함수 J(\theta)의 값을 최소화하는 최적의 가중치 패러미터 \theta를 찾기 위해 반복적으로 기울기(Gradient)를 계산하여 매개변수를 갱신하는 최적화 알고리즘입니다.

[//]: # (heading_4 is not supported)


```plain text
[현재 위치의 기울기(Gradient) 계산] ──> [기울기의 반대 방향으로 이동] ──> [가중치 θ 갱신] ──> [최솟값 수렴시 종료]
```

- **핵심 요소:**

	- **\alpha (학습률, Learning Rate):** 한 번의 업데이트 단계에서 가중치를 얼마나 이동시킬지 결정하는 하이퍼파라미터입니다.

		- **학습률이 너무 큰 경우:** 최적점(Minimum)을 지나쳐 오차가 오히려 증가하며 발산(Divergence)합니다.

		- **학습률이 너무 작은 경우:** 기울기 보폭이 작아 수렴에 지나치게 오랜 시간이 소요되거나 국소 최솟값(Local Minimum)에 갇힐 위험이 있습니다.

	- **기울기 방향성:** 편미분값(기울기)이 양수이면 가중치를 감소시키고, 음수이면 가중치를 증가시켜 항상 Cost 함수 곡선의 골짜기(최소점) 방향으로 파라미터를 경사 하강시킵니다.

---

[//]: # (heading_4 is not supported)


```python
import numpy as np

# 1. 가상 데이터 생성 (y = 2x + 4 + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# X0 = 1 (Bias 추가)
X_b = np.c_[np.ones((100, 1)), X]

# 2. 경사하강법을 활용한 파라미터(theta) 학습
alpha = 0.1       # 학습률 (Learning Rate)
n_iterations = 1000
m = 100           # 샘플 개수

theta = np.random.randn(2, 1) # 가중치 무작위 초기화

for iteration in range(n_iterations):
    # 1) 예측값(Hypothesis) 계산
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    # 2) 가중치 갱신 (Gradient Descent)
    theta = theta - alpha * gradients

print("=== 경사하강법 추정 가중치 ===")
print("편향 (theta0):", theta[0][0])  # 실제값 4 근사
print("가중치 (theta1):", theta[1][0])  # 실제값 3 근사
```

---

[//]: # (heading_4 is not supported)

- [Jeremy Jordan - Learning Rate에 따른 모델 수렴 분석 블로그](https://www.jeremyjordan.me/nn-learning-rate/)

- [Scikit-learn Official Guide - Linear Models (Linear Regression)](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)

- [Scikit-learn API Reference - ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)`[sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)`

