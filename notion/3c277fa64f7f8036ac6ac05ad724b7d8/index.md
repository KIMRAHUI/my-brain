---
notion-url: https://app.notion.com/p/Seaborn-Advanced-Visualization-3c277fa64f7f8036ac6ac05ad724b7d8
title: Seaborn Advanced Visualization
date: '2026-08-20 08:30:00.000'
from_notion: https://app.notion.com/p/Seaborn-Advanced-Visualization-3c277fa64f7f8036ac6ac05ad724b7d8
author: rahui kim
last_edited_time: '2026-08-20 08:31:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- `**sns.set_theme()**`**:** 전반적인 그래프 스타일(테마, 폰트, 색상 팔레트)을 일괄 적용합니다.

- `**sns.load_dataset()**`**:** Seaborn 공식 깃허브 저장소에서 온라인 샘플 데이터셋을 DataFrame 형태로 불러옵니다.

[//]: # (heading_4 is not supported)


```python
import seaborn as sns
import matplotlib.pyplot as plt

# 한글 폰트 및 마이너스 깨짐 방지 설정
plt.rc('font', family='Malgun Gothic')  # Windows: Malgun Gothic / macOS: AppleGothic
plt.rc('axes', unicode_minus=False)

# 테마 스타일 설정 (whitegrid, darkgrid, ticks 등)
sns.set_theme(style="whitegrid", font="Malgun Gothic")

# 샘플 데이터셋 로드 (식당 팁 데이터)
tips = sns.load_dataset("tips")

print("=== Tips 데이터셋 상위 5행 ===")
print(tips.head())
```

---

[//]: # (heading_4 is not supported)

- `**sns.scatterplot()**`**:** 두 수치형 변수 간의 관계를 점으로 표현합니다. `hue`(색상), `style`(점 모양), `size`(점 크기) 파라미터를 사용해 추가적인 범주형 변수를 한 차원에 동시에 시각화할 수 있습니다.

- `**sns.histplot()**`** / **`**sns.kdeplot()**`**:** 데이터의 빈도 구간 분포(히스토그램)와 커널 밀도 추정(KDE) 곡선을 생성하여 연속형 변수의 확률 밀도 함수 형태를 시각화합니다.

[//]: # (heading_4 is not supported)


```python
# 1) 산점도 (Scatter Plot) - 요일별(hue), 성별(style) 구분
plt.figure(figsize=(8, 5))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", style="sex", s=100)
plt.title("전체 지불 금액 대비 팁 비율")
plt.show()

# 2) 히스토그램 및 KDE 분포 곡선
plt.figure(figsize=(8, 5))
sns.histplot(data=tips, x="total_bill", kde=True, bins=20, color="skyblue")
plt.title("전체 지불 금액 분포")
plt.show()
```

---

[//]: # (heading_4 is not supported)

- `**sns.barplot()**`**:** 범주별 평균값(기본값)을 막대 길이로 표현하며, 검은색 막대(Error Bar)로 신뢰구간(CI)을 나타냅니다.

- `**sns.boxplot()**`**:** 사분위수 기반의 5수치 요약(최소, Q1, 중앙값, Q3, 최대)을 상자 형태로 표현하며, IQR 범위를 벗어난 이상치(Outlier)를 쉽게 탐지합니다.

[//]: # (heading_4 is not supported)


```python
# 1) 막대 그래프 (Bar Plot)
plt.figure(figsize=(8, 5))
sns.barplot(data=tips, x="day", y="total_bill", hue="sex", palette="Set2")
plt.title("요일별/성별 평균 지불 금액")
plt.show()

# 2) 박스 플롯 (Box Plot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=tips, x="day", y="total_bill", palette="Pastel1")
plt.title("요일별 지불 금액 분포 및 이상치")
plt.show()
```

---

[//]: # (heading_4 is not supported)

- `**corr()**`**:** 수치형 컬럼 간 피어슨 상관계수(-1 ~ 1)를 산출합니다.

- `**sns.heatmap()**`**:** 2차원 데이터를 색상으로 시각화합니다.

	- `annot=True`: 각 셀에 해당 수치를 직접 명시

	- `fmt=".2f"`: 소수점 둘째 자리까지 표시

	- `cmap`: 색상 지도(Color Map) 지정 (`Blues`, `coolwarm` 등)

[//]: # (heading_4 is not supported)


```python
# 수치형 컬럼 간 상관계수 계산
corr = tips.select_dtypes(include=['float64', 'int64']).corr()

# 히트맵 시각화
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=-1, vmax=1)
plt.title("수치형 변수 간 상관관계 히트맵")
plt.show()
```

---

[//]: # (heading_4 is not supported)

- [Seaborn Official Documentation & User Guide](https://seaborn.pydata.org/)

- [Seaborn Example Gallery](https://seaborn.pydata.org/examples/index.html)

- [Seaborn API Reference - Categorical Plots](https://seaborn.pydata.org/api.html#categorical-api)

