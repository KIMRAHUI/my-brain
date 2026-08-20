---
notion-url: https://app.notion.com/p/Clustering_and_PCA-3c277fa64f7f805b89a4d58f7afc8ff4
title: Clustering_and_PCA
date: '2026-08-20 09:11:00.000'
from_notion: https://app.notion.com/p/Clustering_and_PCA-3c277fa64f7f805b89a4d58f7afc8ff4
author: rahui kim
last_edited_time: '2026-08-20 09:28:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- **정의:** 사전에 지정한 군집 개수 K개로 데이터를 그룹화하는 대표적인 비지도학습 Hard Clustering 알고리즘입니다.

- **알고리즘 순환 과정:**

	1. **초기화:** K개의 중심점(Centroid)을 무작위로 설정합니다.

	1. **할당 (Assignment):** 모든 데이터 포인트를 유클리디안 거리 기준으로 가장 가까운 중심점에 할당합니다.

	1. **업데이트 (Update):** 각 클러스터에 할당된 데이터들의 평균 위치 지점으로 중심점을 새롭게 이동시킵니다.

	1. **수렴 (Convergence):** 중심점의 위치 변화가 없을 때까지 2단계와 3단계를 반복 수행합니다.


```plain text
[K개 중심점 무작위 설정] ──> [가장 가까운 중심점에 할당] ──> [데이터 평균 지점으로 중심점 이동] ──> [수렴 시 종료]
```

- **최적의 K 선정 (Elbow Method):**

	- 군집 수 K를 늘려가며 클러스터 내 오차 제곱 합(SSE, Distortion)을 계산하여, 비용 함수 그래프의 감소율이 급격히 꺾이는 지점(Elbow Point)의 K를 선택합니다.

- **성능 평가 지표 (Silhouette Score):**

	- 데이터가 자신이 속한 군집 내부 데이터들과 얼마나 밀집해 있고(a(i)), 인접한 다른 군집과는 얼마나 멀리 떨어져 있는지(b(i))를 평가하는 지표입니다 ($ -1 \sim 1  $사이 값).

	---

[//]: # (heading_4 is not supported)

- **목적:** 차원이 커질수록 데이터가 희소해지는 **차원의 저주(Curse of Dimensionality)** 문제를 해결하기 위해, 원 데이터의 변동성(분산)을 가장 크게 보존하는 새로운 축(주성분)을 찾아 저차원 공간으로 압축/투영(Projection)합니다.

- **작동 방식 및 단계별 알고리즘:**

	1. **데이터 표준화 (Standardization):** 변수 간 스케일 차이에 의한 왜곡을 방지하기 위해 데이터를 평균 0, 분산 1로 표준화합니다.

	1. **공분산 행렬 (Covariance Matrix) 계산:** 변수 간 분산과 상관관계를 담고 있는 공분산 행렬을 계산합니다.

	1. **고유값 분해 (Eigenvalue Decomposition):** 공분산 행렬을 고유값 분해하여 데이터 분산의 방향을 나타내는 고유벡터(Eigenvector)와 분산의 크기를 나타내는 고유값(Eigenvalue)을 산출합니다.

			1. **주성분 선택 및 투영:** 고유값이 큰 순서대로 최상위 k개의 주성분을 선택하고, 해당 고유벡터 축으로 원 데이터를 정사영(Projection)하여 저차원 데이터셋을 완성합니다.

---

[//]: # (heading_4 is not supported)


```python
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
```

---

[//]: # (heading_4 is not supported)

- [공돌이의 수학정리노트 - 주성분 분석(PCA) 기하학적 이해](https://angeloyeo.github.io/2019/07/27/PCA.html)

- [Scikit-learn Official User Guide - Decomposition (PCA 구현)](https://scikit-learn.org/stable/modules/decomposition.html#pca)

- [Scikit-learn Official User Guide - Clustering (K-Means 구현)](https://scikit-learn.org/stable/modules/clustering.html#k-means)

