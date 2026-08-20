---
notion-url: https://app.notion.com/p/matplotlib_basics-3c277fa64f7f80a3a153c272d6835c70
title: matplotlib_basics
date: '2026-08-20 08:30:00.000'
from_notion: https://app.notion.com/p/matplotlib_basics-3c277fa64f7f80a3a153c272d6835c70
author: rahui kim
last_edited_time: '2026-08-20 08:31:00.000'
---
[//]: # (heading_4 is not supported)

---

[//]: # (heading_4 is not supported)

- `**plt.plot(x, y)**`**:** x축과 y축 데이터를 받아 선 그래프를 생성합니다.

- `**plt.title()**`** / **`**plt.xlabel()**`** / **`**plt.ylabel()**`**:** 그래프의 제목 및 각 축의 이름(라벨)을 지정합니다.

- `**plt.show()**`**:** 생성된 그래프를 화면에 출력합니다. (주피터 노트북 환경에서는 인라인 출력)

[//]: # (heading_4 is not supported)


```python
import matplotlib.pyplot as plt

# 데이터 준비
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

# 그래프 그리기
plt.plot(x, y)

# 타이틀 및 축 라벨 설정
plt.title("Line Plot Example")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")

# 그래프 출력
plt.show()
```

---

[//]: # (heading_4 is not supported)

- **OS별 한글 폰트 설정:**

	- **Windows:** `Malgun Gothic` (맑은 고딕)

	- **macOS:** `AppleGothic` (애플고딕)

	- **Linux (Colab 등):** `NanumGothic` (나눔고딕 설치 필요)

- `**plt.rc('axes', unicode_minus=False)**`**:** 마이너스 기호 깨짐 현상을 방지합니다.

[//]: # (heading_4 is not supported)


```python
import matplotlib.pyplot as plt

# 운영체제에 맞는 한글 폰트 및 마이너스 깨짐 방지 설정
plt.rc('font', family='Malgun Gothic')  # macOS: 'AppleGothic'
plt.rc('axes', unicode_minus=False)    # 마이너스 기호(-) 깨짐 방지

# 음수 데이터 포함 샘플 데이터
x = [-2, -1, 0, 1, 2]
y = [-4, -2, 0, 2, 4]

plt.plot(x, y)
plt.title("한글 제목 및 음수 데이터 테스트")
plt.xlabel("X 축 (음수 포함)")
plt.ylabel("Y 축 (음수 포함)")

plt.show()
```

---

[//]: # (heading_4 is not supported)

- `**color**`**:** 선 색상 지정 (`'red'`, `'blue'`, `'#FF5733'` 등 Hex 코드 지원)

- `**linestyle**`** / **`**ls**`**:** 선 모양 지정 (`'-'` 실선, `'--'` 점선, `':'` 짧은 점선 등)

- `**marker**`**:** 데이터 포인트 위치에 표시할 기호 (`'o'` 원, `'s'` 사각형, `'^'` 삼각형 등)

- `**label**`**:** 각 그래프 시리즈의 이름 지정 (`plt.legend()` 호출 시 범례 표기)

- `**plt.grid(True)**`**:** 데이터 수치 비교가 용이하도록 격자 눈금선 추가

[//]: # (heading_4 is not supported)


```python
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [1, 2, 3, 4, 5]

# 스타일 옵션 적용 (색상, 선 스타일, 마커, 범례 라벨)
plt.plot(x, y1, color='red', linestyle='--', marker='o', label='y = x^2')
plt.plot(x, y2, color='blue', linestyle='-', marker='s', label='y = x')

# 구성 요소 추가
plt.title("스타일링이 적용된 다중 선 그래프")
plt.xlabel("X 축")
plt.ylabel("Y 축")
plt.legend(loc='upper left')  # 범례 위치 지정
plt.grid(True)               # 격자 표시

plt.show()
```

---

[//]: # (heading_4 is not supported)

- [Matplotlib Official Documentation - Pyplot Tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)

- [Matplotlib API Reference - ](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)`[matplotlib.pyplot.plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html)`

- [Matplotlib Customizing - Fonts & Styles](https://matplotlib.org/stable/tutorials/introductory/customizing.html)

