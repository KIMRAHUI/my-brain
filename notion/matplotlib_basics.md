# matplotlib_basics

개요

파이썬 데이터 시각화의 가장 대표적이자 기본이 되는 라이브러리인 Matplotlib의 기본 구조를 다룹니다. 선 그래프(Line Plot) 생성 방법, OS별 한글 폰트 깨짐 및 마이너스 기호 설정, 그리고 선 색상·스타일·마커·범례 등을 적용하여 시각화 가독성을 높이는 커스터마이징 기법을 상세히 학습합니다.

기본 그래프 생성 (plt.plot)

pyplot 모듈의 plt.plot() 함수를 활용하여 가장 기본이 되는 선 그래프(Line Plot)를 생성하고 캔버스에 출력합니다.

plt.plot(x, y): x축과 y축 데이터를 받아 선 그래프를 생성합니다.

plt.title() / plt.xlabel() / plt.ylabel(): 그래프의 제목 및 각 축의 이름(라벨)을 지정합니다.

plt.show(): 생성된 그래프를 화면에 출력합니다. (주피터 노트북 환경에서는 인라인 출력)

기본 선 그래프 생성 코드 예시

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

한글 폰트 및 마이너스 기호 설정 (폰트 깨짐 방지)

Matplotlib은 기본 폰트로 영문만 지원하는 DejaVu Sans를 사용하므로, 그래프의 제목이나 축 이름에 한글을 입력하면 네모(□) 형태로 폰트가 깨져 출력됩니다. 또한 음수 데이터 표현 시 마이너스 기호(-)가 유니코드 문제로 깨질 수 있어 설정이 필요합니다.

OS별 한글 폰트 설정:

plt.rc('axes', unicode_minus=False): 마이너스 기호 깨짐 현상을 방지합니다.

한글 폰트 및 마이너스 기호 설정 코드 예시

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

그래프 스타일링 및 커스터마이징

선 색상, 선 스타일, 데이터 포인트 마커, 범례(Legend), 격자(Grid) 등을 설정하여 전달하고자 하는 데이터의 가독성과 완성도를 높입니다.

color: 선 색상 지정 ('red', 'blue', '#FF5733' 등 Hex 코드 지원)

linestyle / ls: 선 모양 지정 ('-' 실선, '--' 점선, ':' 짧은 점선 등)

marker: 데이터 포인트 위치에 표시할 기호 ('o' 원, 's' 사각형, '^' 삼각형 등)

label: 각 그래프 시리즈의 이름 지정 (plt.legend() 호출 시 범례 표기)

plt.grid(True): 데이터 수치 비교가 용이하도록 격자 눈금선 추가

스타일 커스터마이징 코드 예시

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

공식 문서 및 참고 링크

Matplotlib Official Documentation - Pyplot Tutorial

Matplotlib API Reference - matplotlib.pyplot.plot

Matplotlib Customizing - Fonts & Styles

