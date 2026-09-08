---
notion-url: https://app.notion.com/p/Logistic_Regression_and_Metrics-2b177fa64f7f83979d2c8134b7f006bf
title: Logistic_Regression_and_Metrics
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Logistic_Regression_and_Metrics-2b177fa64f7f83979d2c8134b7f006bf
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- **정의:** 이름에는 '회귀'가 들어가지만 실제로는 분류 문제를 해결하는 대표적인 **확률적 선형 분류 모델**입니다.

- **시그모이드 함수 (Sigmoid / Logistic Function):**

	- 선형 회귀의 결과값 $ z = \theta^T x $를 받아 $ 0 \sim 1 $ 사이의 확률값으로 압축(변환)합니다.

			- $ z \to +\infty $ 이면 출력은 1에 수렴합니다.

	- $ z \to -\infty  $이면 출력은 0에 수렴합니다.

	- z = 0 이면 출력은 0.5 (결정 경계)가 됩니다.


```plain text
[선형 연산: z = θᵀx] ──> [Sigmoid 변환: σ(z)] ──> [0 ~ 1 사이 확률값 반환]
```

- **손실 함수 (Binary Cross-Entropy / Log Loss):**

	- 분류 문제에서 평균 제곱 오차(MSE)를 사용할 경우, 손실 함수 형태가 **비볼록(Non-convex)** 구조가 되어 경사하강법으로 글로벌 최솟값(Global Minimum)을 찾기 어렵습니다.

	- 따라서 볼록(Convex) 형태로 최적화가 용이한 로그 손실 함수(Binary Cross-Entropy)를 사용합니다.
$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] $

- **다중 분류(Multi-class)로의 확장:**

	- **One-vs-All (One-vs-Rest):** 각 클래스별로 이진 분류기를 개별적으로 학습시켜 가장 높은 확률을 가진 클래스를 선택합니다.

	- **Softmax 함수:** 모든 클래스의 예측 확률 합이 1이 되도록 일반화하여 한 번에 다중 클래스를 분류합니다.

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

 |  | Predicted Negative (0) | Predicted Positive (1) | 
 | ---- | ---- | ---- | 
 | **Actual Negative (0)** | **TN** (True Negative) | **FP** (False Positive - Type I Error) | 
 | **Actual Positive (1)** | **FN** (False Negative - Type II Error) | **TP** (True Positive) | 

---

[//]: # (heading_4 is not supported)

- **정확도 (Accuracy):** 전체 샘플 중 모델이 올바르게 분류한 비율입니다 (데이터 불균형 시 신뢰도가 낮아짐).

	- **정밀도 (Precision):** 모델이 Positive라고 예측한 것 중에서 실제 Positive인 데이터의 비율입니다 (스팸 메일 분류 등에 중요).

	- **재현율 (Recall / Sensitivity):** 실제 Positive인 데이터 중에서 모델이 Positive라고 맞힌 비율입니다 (암 환자 진단, 금융 사기 탐지 등에 중요).

	- **F1 Score:** 정밀도와 재현율의 균형을 평가하기 위한 **조화 평균** 값입니다.

	---

[//]: # (heading_4 is not supported)


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 암 진단 데이터셋 로드
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 1. 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# 2. 예측 및 확률값 산출
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 3. 오차 행렬 및 종합 분류 리포트 출력
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
```

---

[//]: # (heading_4 is not supported)

- [DataScience Beehive - 로지스틱 시그모이드 함수 설명](https://datasciencebeehive.tistory.com/80)

- [Scikit-learn Official User Guide - Metrics (Classification Report & Confusion Matrix)](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

- [Scikit-learn Official API - ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)`[sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)`

