---
notion-url: https://app.notion.com/p/SVM_and_Decision_Tree-3c277fa64f7f8078aa62c90694416d5f
title: SVM_and_Decision_Tree
date: '2026-08-20 09:08:00.000'
from_notion: https://app.notion.com/p/SVM_and_Decision_Tree-3c277fa64f7f8078aa62c90694416d5f
author: rahui kim
last_edited_time: '2026-08-20 09:29:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- **정의:** 확률값이 아닌 카테고리를 직관적으로 구분짓는 선형 결정 경계(Decision Boundary)를 찾는 판별(Non-probabilistic) 모델입니다.

- **핵심 개념:**

	- **마진 (Margin):** 결정 경계선과 각 클래스별 가장 가까운 데이터 샘플 사이의 폭(거리)을 의미합니다.

	- **서포트 벡터 (Support Vector):** 마진의 경계선(Hyperplane)에 접해 있으면서 마진의 크기를 결정하는 기준이 되는 최외곽 훈련 데이터 포인트들입니다.

- **마진 최대화 전략:**

	- 데이터 구분 정확도를 확보함과 동시에 마진을 최대한 넓혀(마진 최대화) 새로운 데이터에 대한 일반화 성능을 올립니다.

- **손실 함수 (Hinge Loss):**

	- 결정 경계 바깥쪽에 정상적으로 배치된 샘플의 오차는 0으로 처리하고, 마진 내부나 반대편으로 침범한 오차 샘플에 대해서만 선형적으로 패널티를 부여하는 Hinge Loss를 사용합니다.

	- **수식 형태:** $ Cost(h_theta(x), y) = max(0, 1 - y * (theta^T x))  $(또는 조건별 분할식 구조)

---

[//]: # (heading_4 is not supported)

- **정의:** 나무(Tree) 구조를 기반으로 데이터에 질문(Question)을 던져가며 계층적인 의사결정 규칙을 만들어내는 분류 및 회귀 모델입니다.

- **학습 방식 (Recursive Partitioning):**

	- 각 단계마다 불순도를 가장 크게 낮추는 하나의 특성(Feature)과 임계값을 탐색하여 수직/수평의 초평면으로 반복 분할(Recursive Partitioning)을 진행합니다.

[//]: # (heading_4 is not supported)

- p_i: 해당 데이터 세트 내에서 특정 클래스 i가 차지하는 비율

[//]: # (heading_4 is not supported)

- $ \text{Gini}_{\text{parent}} $: 분할 전 부모 노드의 지니 불순도

- $ N_{\text{left}}, N_{\text{right}} $: 왼쪽, 오른쪽 자식 노드의 데이터 수

- N: 전체 데이터 수

[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)


```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 1. Linear SVM 모델 학습
svm_clf = SVC(kernel='linear', C=1.0)
svm_clf.fit(X_train, y_train)

print("=== SVM Support Vector 개수 ===")
print(svm_clf.n_support_)

# 2. Decision Tree 모델 학습 (지니 불순도 기반, max_depth로 가지치기)
tree_clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)

print("\n=== Decision Tree 속성 중요도 ===")
for name, importance in zip(iris.feature_names, tree_clf.feature_importances_):
    print(f"{name}: {importance:.4f}")
```

---

[//]: # (heading_4 is not supported)

- [ML Wiki - Support Vector Machines 이론 및 수학적 증명](http://mlwiki.org/index.php/Support_Vector_Machines)

- [Scikit-learn Official User Guide - Decision Trees](https://scikit-learn.org/stable/modules/tree.html)

- [Scikit-learn Official User Guide - Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)

