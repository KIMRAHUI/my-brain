# Classification Model

고객의 정기예금 가입 여부 예측을 위한 은행 마케팅 분류 모델 구축

1. 분석 목적 및 목표

분석 목적: 포르투갈 A은행의 마케팅 담당자로서 고객이 정기 예금을 가입할 가능성을 예측하고, 정기 예금과 관련이 있는 주요 요소들을 파악하여 마케팅 캠페인의 효율성을 제고합니다.

분석 목표: 결정 트리와 앙상블 기법을 사용하여 고객의 정기 예금 가입 여부를 예측하는 분류 모델을 구축하고, 도출된 인사이트를 바탕으로 비즈니스 전략을 제시합니다.

2. 데이터 소개

출처: UC Irvine Machine Learning Repository Bank Marketing Dataset

설명: 2008년부터 2010년까지의 포르투갈 은행의 전화 통화 기반 마케팅 캠페인 결과를 담고 있으며, 고객이 정기 예금 가입에 동의한 경우 목표 변수(target) y에 yes로 표시됩니다.

3. 데이터 불러오기 및 전처리

라이브러리 설치 및 임포트

# !pip install optuna
# !pip install catboost
# !pip install xgboost
# !pip install lightgbm
# !pip install optuna-integration[sklearn]

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import optuna
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb

from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

데이터 로드 및 중복 제거

file_path = '/content/bank-additional-full.csv'
data = pd.read_csv(file_path, sep=';')

# 중복 데이터 제거 및 인덱스 재정렬
data = data.drop_duplicates()
data = data.reset_index(drop=True)

데이터셋은 총 41,188행에서 중복 12건을 제거하여 정제되었습니다. 결측치는 관찰되지 않았습니다.

4. 탐색적 데이터 분석 (EDA)

목표 변수(y) 불균형: no가 약 89%, yes가 약 11%로 불균형한 분포를 보입니다. 이를 해결하기 위해 SMOTE 오버샘플링 실험을 병행합니다.

주요 인사이트:

5. 베이스라인 모델 구축

5.1 데이터 전처리 및 인코딩

cat_columns = list(data.select_dtypes(include=['object']).columns)[:-1]
cat_columns_w_target = list(data.select_dtypes(include=['object']).columns)

data.loc[:, cat_columns] = data.loc[:, cat_columns].astype(str)

enc = OrdinalEncoder()
encoded_values = enc.fit_transform(data.loc[:, cat_columns_w_target])
data[cat_columns_w_target] = pd.DataFrame(encoded_values, columns=cat_columns_w_target)

5.2 데이터 분할 및 SMOTE 적용

X = data.columns[:-1]
y = data.columns[-1]

X_train, X_test, y_train, y_test = train_test_split(
    data[X], data[y], stratify=data[y], shuffle=True, test_size=0.2, random_state=2023
)

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

smt = SMOTE(random_state=42)
X_train_smote, y_train_smote = smt.fit_resample(X_train, y_train)
X_train_scale_smote, y_train_scale_smote = smt.fit_resample(X_train_scale, y_train)

5.3 교차 검증 성능 평가

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2023)

a = [X_train, X_train_scale, X_train_smote, X_train_scale_smote]
b = [y_train, y_train, y_train_smote, y_train_smote]
c = [X_test, X_test_scale, X_test, X_test_scale]
d = [y_test, y_test, y_test, y_test]

def cv_f1(model, X, y, X_t, y_t):
    f1_train = cross_val_score(model, X, y, scoring="f1", cv=skf)
    model.fit(X, y)
    pred = model.predict(X_t)
    f1_test = f1_score(y_t, pred)
    roc_test = roc_auc_score(y_t, pred)
    acc_test = accuracy_score(y_t, pred)
    pre_test = precision_score(y_t, pred)
    rec_test = recall_score(y_t, pred, average='macro')
    return f1_train.mean(), f1_test, roc_test, acc_test, pre_test, rec_test

결과 요약: 데이터 불균형으로 인해 정확도(Accuracy)보다는 소수 클래스(yes)를 정확히 판별하는 F1-Score와 재현율(Recall)을 기준으로 평가했을 때, SMOTE를 적용한 데이터셋을 사용한 모델들의 성능이 더 우수하게 나타났습니다.

6. 하이퍼파라미터 튜닝 (Optuna)

SMOTE 전처리가 적용된 데이터를 기반으로 OptunaSearchCV를 활용해 각 모델별 최적의 하이퍼파라미터를 탐색했습니다 (n_trials=50).

# 1. Random Forest 튜닝
param_rf = {
    "max_depth": optuna.distributions.IntDistribution(1, 10),
    'min_samples_leaf': optuna.distributions.IntDistribution(1, 6),
    'n_estimators': optuna.distributions.IntDistribution(80, 200)
}
rf = RandomForestClassifier(random_state=2023)
optuna_search = optuna.integration.OptunaSearchCV(
    rf, param_rf, cv=skf, scoring='f1', n_jobs=-1, n_trials=50, random_state=2023
)
optuna_search.fit(X_train_smote, y_train_smote)
rf_fin = RandomForestClassifier(random_state=2023, **optuna_search.best_params_)
rf_fin.fit(X_train_smote, y_train_smote)
pred_rf = rf_fin.predict(X_test)

# 2. LightGBM 튜닝
param_lgb = {
    "learning_rate": optuna.distributions.FloatDistribution(0.005, 0.5),
    "n_estimators": optuna.distributions.IntDistribution(80, 200),
    "colsample_bytree": optuna.distributions.FloatDistribution(0.3, 1.0)
}
lgbm = lgb.LGBMClassifier(random_seed=2023, device="cpu")
optuna_search_lgb = optuna.integration.OptunaSearchCV(
    lgbm, param_lgb, cv=3, scoring='f1', n_jobs=-1, n_trials=50
)
optuna_search_lgb.fit(X_train_smote, y_train_smote)
lgbm_fin = lgb.LGBMClassifier(random_state=2023, **optuna_search_lgb.best_params_)
lgbm_fin.fit(X_train_smote, y_train_smote)
pred_lgbm = lgbm_fin.predict(X_test)

# 3. CatBoost 튜닝
param_cat = {
    "learning_rate": optuna.distributions.FloatDistribution(0.005, 0.5),
    "iterations": optuna.distributions.IntDistribution(300, 1500),
    "depth": optuna.distributions.IntDistribution(1, 10)
}
cat = CatBoostClassifier(random_seed=2023, verbose=False, task_type='CPU')
optuna_search_cat = optuna.integration.OptunaSearchCV(
    cat, param_cat, cv=3, scoring='f1', n_jobs=-1, n_trials=50
)
optuna_search_cat.fit(X_train_smote, y_train_smote)
cat_fin = CatBoostClassifier(random_seed=2023, verbose=False, **optuna_search_cat.best_params_)
cat_fin.fit(X_train_smote, y_train_smote)
pred_cat = cat_fin.predict(X_test)

7. 결론 및 모델 비교

정밀도(Precision): 음성 클래스(0)는 전반적으로 매우 높으며, 양성 클래스(1)는 모델 간 유사한 수준을 보였습니다.

재현율(Recall): LGBM이 양성 클래스(1)에서 0.83으로 가장 높은 재현율을 기록했습니다.

F1-Score 및 종합 평가:

