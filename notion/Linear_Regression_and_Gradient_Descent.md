# Linear_Regression_and_Gradient_Descent

개요

수치형 연속 데이터를 예측하는 대표적인 지도학습 모델인 선형 회귀(Linear Regression)의 가설 함수(Hypothesis)부터, 모델의 오차를 측정하는 손실 함수(Cost Function / MSE)의 수학적 원리, 그리고 손실을 최소화하는 최적의 가중치를 탐색하는 경사하강법(Gradient Descent) 알고리즘을 상세히 정리합니다.

선형 회귀의 가설 (Hypothesis)

정의: 하나 이상의 입력 변수(X)와 연속적인 목표 변수(Y) 사이의 관계를 최적의 직선(또는 초평면) 방정식 형태로 표현하는 모델입니다.

가설 함수 수식 (Hypothesis Function):h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n = \theta^T x

선형 회귀의 Cost 함수 (MSE와 부호 상쇄 방지)

목적: 훈련 데이터의 전체 실제값(y)과 모델의 예측값(h_\theta(x)) 사이의 오차(Residual)를 종합적으로 평가하기 위함입니다.

평균 제곱 오차 (MSE, Mean Squared Error) 수식 및 풀이

J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2

수식 구성 요소 및 원리:

경사하강법 (Gradient Descent Algorithm)

목적: 손실 함수 J(\theta)의 값을 최소화하는 최적의 가중치 패러미터 \theta를 찾기 위해 반복적으로 기울기(Gradient)를 계산하여 매개변수를 갱신하는 최적화 알고리즘입니다.

가중치 업데이트 수식 및 풀이

\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) \cdot x_j^{(i)}

[현재 위치의 기울기(Gradient) 계산] ──> [기울기의 반대 방향으로 이동] ──> [가중치 θ 갱신] ──> [최솟값 수렴시 종료]

핵심 요소:

코드로 보는 선형 회귀 및 경사하강법 구현 예시

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

공식 문서 및 참고 링크

Jeremy Jordan - Learning Rate에 따른 모델 수렴 분석 블로그

Scikit-learn Official Guide - Linear Models (Linear Regression)

Scikit-learn API Reference - sklearn.linear_model.LinearRegression

