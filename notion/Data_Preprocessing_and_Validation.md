# Data_Preprocessing_and_Validation

개요

머신러닝 모델의 학습 안정성과 예측 성능을 확보하기 위한 핵심 전처리 단계인 피처 스케일링(Feature Scaling)의 필요성과 종류, 모델의 일반화 상태를 판가름하는 과적합/과소적합(Overfitting/Underfitting), 그리고 학습 데이터만을 활용해 객관적인 성능 평가를 수행하는 K-Fold 교차 검증(Cross Validation)을 상세히 정리합니다.

Feature Scaling (피처 스케일링)의 필요성

정의: 머신러닝 모델에 입력할 각 변수(Feature)들의 수치 범위(Scale)를 일치시키거나 분포를 균일하게 정규화하는 전처리 작업입니다.

스케일링이 필수적인 이유:

주요 스케일링 기법 수식 및 특징

Min-Max Scaling (정규화, Normalization)

Standardization (표준화, Z-Score Scaling)

Robust Scaling

과적합(Overfitting)과 과소적합(Underfitting)

머신러닝 모델 학습 과정에서 편향(Bias)과 분산(Variance)의 트레이드오프 관계를 나타내는 핵심 개념입니다.

[과소적합 (Underfitting)]        [적정 적합 (Good Fit)]        [과적합 (Overfitting)]
   모델이 너무 단순함              일반화 성능 확보             모델이 너무 복잡함
(Train 낮음, Test 낮음)         (Train 높음, Test 높음)      (Train 높음, Test 낮음)

과소적합 (Underfitting):

과적합 (Overfitting):

해결책:

교차 검증 (Cross Validation)과 K-Fold

목적: 테스트 데이터(Test Set)는 최종 성능 평가 및 배포 단계용으로 완전히 아껴두고, 훈련 데이터(Train Set)만을 여러 방식으로 분할하여 모델의 일반화 성능을 객관적으로 검증하기 위함입니다.

K-Fold 교차 검증 작동 원리:

Fold 1    Fold 2    Fold 3    ...    Fold K
[ Validation ] [   Train    ] [   Train    ] ... [   Train    ] -> Iteration 1
[   Train    ] [ Validation ] [   Train    ] ... [   Train    ] -> Iteration 2
...
[   Train    ] [   Train    ] [   Train    ] ... [ Validation ] -> Iteration K

Python 코드로 보는 스케일링 및 교차 검증 구현 예시

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 1. Feature Scaling 적용 (StandardScaler 예시)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. K-Fold 교차 검증 객체 생성 (5-Fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. 모델 정의 및 교차 검증 연산 수행
model = LogisticRegression(max_iter=200)
scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')

print("=== K-Fold 교차 검증 결과 ===")
print("각 Fold별 정확도:", scores)
print(f"평균 정확도: {scores.mean():.4f}")

공식 문서 및 참고 링크

Hands-On Machine Learning (Aurélien Géron) GitHub Notebooks

Scikit-learn Official User Guide - Preprocessing Data

Scikit-learn Official User Guide - Cross-validation



