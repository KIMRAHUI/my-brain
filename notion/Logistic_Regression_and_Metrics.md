# Logistic_Regression_and_Metrics

개요

선형 방정식의 결과를 0 \sim 1 사이의 확률값으로 변환하여 이진 및 다중 분류를 수행하는 로지스틱 회귀(Logistic Regression)의 수학적 구조와 로그 손실 함수(Binary Cross-Entropy), 그리고 분류 모델의 성능을 객관적으로 정밀 평가하기 위한 오차 행렬(Confusion Matrix) 및 평가 지표(Accuracy, Precision, Recall, F1 Score)를 다룹니다.

로지스틱 회귀 (Logistic Regression)

정의: 이름에는 '회귀'가 들어가지만 실제로는 분류 문제를 해결하는 대표적인 확률적 선형 분류 모델입니다.

시그모이드 함수 (Sigmoid / Logistic Function):

[선형 연산: z = θᵀx] ──> [Sigmoid 변환: σ(z)] ──> [0 ~ 1 사이 확률값 반환]

손실 함수 (Binary Cross-Entropy / Log Loss):

다중 분류(Multi-class)로의 확장:

분류 모델 평가 지표 (Evaluation Metrics)

① 오차 행렬 (Confusion Matrix)

학습된 분류 모델이 예측을 수행하면서 얼마나 헷갈리고 있는지, 어떤 유형의 오차를 내고 있는지 한눈에 보여주는 평가 표입니다.

② 핵심 평가 지표 공식

정확도 (Accuracy): 전체 샘플 중 모델이 올바르게 분류한 비율입니다 (데이터 불균형 시 신뢰도가 낮아짐).

정밀도 (Precision): 모델이 Positive라고 예측한 것 중에서 실제 Positive인 데이터의 비율입니다 (스팸 메일 분류 등에 중요).

재현율 (Recall / Sensitivity): 실제 Positive인 데이터 중에서 모델이 Positive라고 맞힌 비율입니다 (암 환자 진단, 금융 사기 탐지 등에 중요).

F1 Score: 정밀도와 재현율의 균형을 평가하기 위한 조화 평균 값입니다.

Python 코드로 보는 로지스틱 회귀 및 평가 지표 연산

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

공식 문서 및 참고 링크

DataScience Beehive - 로지스틱 시그모이드 함수 설명

Scikit-learn Official User Guide - Metrics (Classification Report & Confusion Matrix)

Scikit-learn Official API - sklearn.linear_model.LogisticRegression

