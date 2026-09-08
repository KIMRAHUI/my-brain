# Clustering_and_PCA

개요

비지도학습(Unsupervised Learning)의 대표적인 분야인 데이터를 그룹화하는 K-Means Clustering과, 데이터의 정보 손실을 최소화하면서 차원의 저주를 해결하는 주성분 분석(PCA, Principal Component Analysis)의 수학적 원리 및 핵심 알고리즘을 정리합니다.

K-Means Clustering (K-평균 군집화)

정의: 사전에 지정한 군집 개수 K개로 데이터를 그룹화하는 대표적인 비지도학습 Hard Clustering 알고리즘입니다.

알고리즘 순환 과정:

[K개 중심점 무작위 설정] ──> [가장 가까운 중심점에 할당] ──> [데이터 평균 지점으로 중심점 이동] ──> [수렴 시 종료]

최적의 K 선정 (Elbow Method):

성능 평가 지표 (Silhouette Score):

주성분 분석 (PCA, Principal Component Analysis)

목적: 차원이 커질수록 데이터가 희소해지는 차원의 저주(Curse of Dimensionality) 문제를 해결하기 위해, 원 데이터의 변동성(분산)을 가장 크게 보존하는 새로운 축(주성분)을 찾아 저차원 공간으로 압축/투영(Projection)합니다.

작동 방식 및 단계별 알고리즘:

Python 코드로 보는 K-Means 및 PCA 연산 예시

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 데이터 로드 및 표준화
iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)

# 1. K-Means 클러스터링 수행 (K=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

print("=== K-Means 실루엣 스코어 ===")
print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")

# 2. PCA 차원 축소 수행 (4차원 -> 2차원)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n=== PCA 차원 축소 결과 ===")
print("원래 데이터 Shape:", X_scaled.shape)
print("축소 데이터 Shape:", X_pca.shape)
print("주성분별 설명된 분산 비율:", pca.explained_variance_ratio_)

공식 문서 및 참고 링크

공돌이의 수학정리노트 - 주성분 분석(PCA) 기하학적 이해

Scikit-learn Official User Guide - Decomposition (PCA 구현)

Scikit-learn Official User Guide - Clustering (K-Means 구현)

