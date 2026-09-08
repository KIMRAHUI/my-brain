---
notion-url: https://app.notion.com/p/Image-Processing-and-Augmentation-64277fa64f7f83edb11f81614d90a767
title: Image-Processing-and-Augmentation
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Image-Processing-and-Augmentation-64277fa64f7f83edb11f81614d90a767
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

1. **합성곱(Convolution) 연산과 필터링 원리**

	- **작동 메커니즘:** 사전에 정의된 작은 크기의 행렬인 필터(Kernel)를 원본 이미지 위에서 일정한 보폭(Stride)으로 슬라이딩시키며, 겹치는 픽셀 값과 커널 가중치의 원소별 곱(Element-wise Multiplication)을 합산하여 특징 맵을 생성합니다.

	- **에지 검출 필터 예시:** 수평 에지 검출 필터(가로 선 강조), 수직 에지 검출 필터(세로 선 강조), 대각선 필터 등을 적용하여 이미지의 구조적 윤곽선을 추출합니다.

1. **Hough Transform을 이용한 선 검출 실습 파이프라인**

	- **Step 1 - 이미지 로드 및 Grayscale 변환:** `cv2.imread()` \rightarrow `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`를 통해 채널을 1개로 축소하여 연산 효율을 높입니다.

	- **Step 2 - Canny Edge Detection (**`**cv2.Canny**`**):** 하한 임계값(50)과 상한 임계값(150)의 이중 임계값(Hysteresis Thresholding)을 적용하여 신뢰할 수 있는 이진 에지 맵을 추출합니다.

	- **Step 3 - 확률적 허프 변환 (**`**cv2.HoughLinesP**`**):**

		- `**rho**`** (거리 해상도):** 누적 배열의 거리 분해능 (일반적으로 `1` 픽셀)

		- `**theta**`** (각도 해상도):** 각도 분해능 (일반적으로 `np.pi / 180`, 1도)

		- `**threshold**`**:** 직선으로 판정하기 위한 최소 교차 누적 수

		- `**minLineLength**`**:** 검출할 선분의 최소 길이

		- `**maxLineGap**`**:** 동일한 선분으로 간주할 점 사이의 최대 허용 간격

	- **Step 4 - 시각화:** 검출된 좌표 세그먼트를 `cv2.line()`으로 원본 이미지 위에 오버레이 렌더링합니다.

[//]: # (heading_4 is not supported)

1. **전처리의 필요성 및 핵심 기법**

	- **전처리의 목적:** 데이터셋 내 다양한 해상도/조명 조건의 일관성 확보, 노이즈 제거를 통한 신호 대 잡음비 향상, GPU 메모리 규격에 맞춘 텐서 표준화.

	- **주요 전처리 기법 상세:**

		- **색상 공간 변환 (Color Space Conversion):** RGB \rightarrow Grayscale(얼굴 인식, 구조 분석) 또는 RGB \rightarrow HSV(피부색/객체 색상 분리).

		- **히스토그램 평활화 (Histogram Equalization):** 명암 대비가 낮은 저조도 영상의 밝기 분포를 균일하게 재분배하여 객체 시인성을 대폭 강화합니다.

		- **크기 조정 (Resizing):** 모든 이미지를 모델의 고정 입력 크기(예: 224\times224)로 맞춥니다.

		- **노이즈 제거 (Denoising):** Gaussian Blur, Median Filter를 적용하여 카메라 센서 결함이나 고주파 노이즈를 제거합니다.

		- **정규화 (Normalization):** 0 \sim 255 픽셀 값을 255.0으로 나누어 0.0 \sim 1.0 또는 표준 정규분포(평균 0, 표준편차 1)로 변환하여 신경망의 경사 수렴을 안정화합니다.

		- **자르기 (Cropping / ROI 추출):** 객체가 포함된 관심 영역(Region of Interest)만을 슬라이싱하여 불필요한 배경 노이즈를 제거합니다.

[//]: # (heading_4 is not supported)

1. **데이터 증강의 필요성**

	- **과적합(Overfitting) 방지:** 소규모 데이터셋 환경에서 다양한 변형을 주어 모델이 불변하는 핵심 특징을 학습하도록 유도합니다.

	- **현실 조건 시뮬레이션:** 회전, 가림, 조명 변화 등 실제 배포 환경에서 발생할 수 있는 데이터 왜곡에 강건한 모델을 구축합니다.

1. **분류별 데이터 증강 기법 상세**

	- **1. 기하학적 변환 (Geometric Transformations):**

		- **회전 (Rotation):** 지정 범위 내 회전(예: \pm15^\circ, 30^\circ). 회전 시 발생하는 테두리 여백의 패딩(Padding) 방식을 고려해야 합니다.

		- **자르기 (Random Crop):** 무작위 영역 크롭을 통해 객체의 위치 이동에 대한 불변성을 부여합니다. 과도한 크롭 시 핵심 객체 유실에 주의합니다.

		- **반전 (Flipping):** 좌우 반전(Horizontal Flip) 및 상하 반전(Vertical Flip). 상하 관계가 명확한 데이터(예: 풍경, 도로)에 상하 반전을 적용하지 않도록 주의합니다.

		- **스케일링 (Scaling):** 확대/축소를 통해 객체 크기 변화에 유연하게 대응합니다.

	- **2. 픽셀 기반 변환 (Pixel-level Transformations):**

		- **밝기/대비 조정 (Brightness & Contrast):** 다양한 주간/야간 조명 조건을 시뮬레이션합니다.

		- **색상 변경 (Color Jitter):** 밝기, 대비, 채도, 색조를 임의로 조절하여 색감 변화에 대한 모델의 민감도를 낮춥니다.

		- **노이즈 추가 (Adding Noise):** 가우시안 노이즈 등을 주입하여 센서 불량 환경에 대비합니다.

	- **3. 고급 증강 기법 (Advanced Augmentation):**

		- **아핀 변환 (Affine Transformation):** 평행성을 유지하면서 이동, 확대/축소, 회전, 전단(Shearing/뒤틀림)을 포괄하는 선형 변환입니다.

		- **랜덤 삭제 (Random Erasing / Cutout):** 이미지의 사각형 영역을 무작위로 지워 객체 일부가 가려진 오클루전(Occlusion) 상황을 학습시킵니다.

		- **믹스업 (Mixup):** 두 개의 서로 다른 이미지와 레이블을 선형 보간(Linear Interpolation) 비율로 혼합하여 결정 경계(Decision Boundary)를 매끄럽게 정규화합니다.

[//]: # (heading_4 is not supported)

- [OpenCV Canny Edge Detection Tutorial](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)

- [OpenCV Hough Lines Transform Guide](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)

- [PyTorch Torchvision Transforms v2 Documentation](https://pytorch.org/vision/stable/transforms.html)

- [Affine Transformation Mathematical Explanation](https://angeloyeo.github.io/2024/06/28/Affine_Transformation.html)

- [Google Colab Image Processing & Convolution Practice Notebook](https://colab.research.google.com/drive/14H2ryOzX27LgAakQpAbQNuZuP9842h-L)

