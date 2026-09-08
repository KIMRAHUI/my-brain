# SVM_and_Decision_Tree

개요

판별적(Non-probabilistic) 선형 분류 알고리즘으로 마진(Margin)을 최대화하는 Linear Support Vector Machine (Linear SVM)과, 데이터를 계층적으로 분할하여 규칙을 생성하는 결정 트리(Decision Tree)의 수학적 불순도 지표(Gini, Information Gain) 및 학습 원리를 정리합니다.

Linear Support Vector Machine (Linear SVM)

정의: 확률값이 아닌 카테고리를 직관적으로 구분짓는 선형 결정 경계(Decision Boundary)를 찾는 판별(Non-probabilistic) 모델입니다.

핵심 개념:

마진 최대화 전략:

손실 함수 (Hinge Loss):

결정 트리 (Decision Tree) 및 불순도 지표

정의: 나무(Tree) 구조를 기반으로 데이터에 질문(Question)을 던져가며 계층적인 의사결정 규칙을 만들어내는 분류 및 회귀 모델입니다.

학습 방식 (Recursive Partitioning):

① 지니 불순도 (Gini Impurity)

데이터 집단 내에 서로 다른 클래스가 얼마나 혼합되어 있는지를 나타내는 지표로, 데이터가 한 클래스로만 구성되어 순수할수록 0에 가까워집니다.

G(S) = 1 - \sum_{i=1}^{c} p_i^2

p_i: 해당 데이터 세트 내에서 특정 클래스 i가 차지하는 비율

② 정보 이득 (Information Gain)

분할 전 부모 노드의 불순도와 분할 후 자식 노드들의 가중 평균 불순도 간의 차이(불순도 감소량)를 의미하며, 결정 트리는 이 정보 이득이 최대가 되는 질문을 선택하여 분기합니다.

\text{Information Gain} = \text{Gini}_{\text{parent}} - \left( \frac{N_{\text{left}}}{N} \cdot \text{Gini}_{\text{left}} + \frac{N_{\text{right}}}{N} \cdot \text{Gini}_{\text{right}} \right)

\text{Gini}_{\text{parent}}: 분할 전 부모 노드의 지니 불순도

N_{\text{left}}, N_{\text{right}}: 왼쪽, 오른쪽 자식 노드의 데이터 수

N: 전체 데이터 수

③ 과적합 방지: 가지치기 (Pruning)

트리의 깊이(Depth)가 깊어질수록 훈련 데이터의 아주 세밀한 노이즈까지 과도하게 학습하여 오버피팅(Overfitting)이 발생합니다. 이를 방지하기 위해 트리의 최대 깊이(max_depth)를 제한하거나 가지치기(Pruning)를 수행해야 합니다.

Python 코드로 보는 SVM 및 Decision Tree 연산 예시

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

공식 문서 및 참고 링크

ML Wiki - Support Vector Machines 이론 및 수학적 증명

Scikit-learn Official User Guide - Decision Trees

Scikit-learn Official User Guide - Support Vector Machines

