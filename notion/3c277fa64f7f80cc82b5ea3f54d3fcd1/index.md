---
notion-url: https://app.notion.com/p/Image-Data-Fundamentals-3c277fa64f7f80cc82b5ea3f54d3fcd1
title: Image-Data-Fundamentals
date: '2026-08-20 09:44:00.000'
from_notion: https://app.notion.com/p/Image-Data-Fundamentals-3c277fa64f7f80cc82b5ea3f54d3fcd1
author: rahui kim
last_edited_time: '2026-08-20 09:44:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

1. **픽셀(Pixel)과 채널(Channel) 구조**

	- **픽셀의 정의와 좌표계:** 디지털 이미지를 구성하는 최소 단위(Picture Element)로, 좌측 상단 (0, 0)을 원점으로 하여 우측 방향으로 x(가로), 하단 방향으로 y(세로)가 증가합니다. 각 픽셀은 밝기 및 색상 수치를 저장합니다.

	- **주요 색상 공간(Color Space) 및 채널 구성:**

		- **Grayscale (1채널):** 0(\text{검정}) \sim 255(\text{흰색}) 범위의 밝기 정보만 단일 채널로 표현합니다.

		- **RGB (3채널):** Red, Green, Blue 채널의 조합(0 \sim 255)으로 인간이 인지하는 대부분의 가시광선 색상을 표현합니다.

		- **CMYK (4채널):** Cyan, Magenta, Yellow, Key(Black) 채널로 구성되며 인쇄 및 출력 환경에 특화되어 있습니다.

		- **YCbCr (3채널):** 밝기(Y)와 색차(Cb, Cr)를 분리하여 인간의 시각이 밝기에 더 민감한 특성을 이용, 디지털 영상 및 JPEG 압축에 널리 활용됩니다.

		- **HSV (3채널):** 색상(Hue), 채도(Saturation), 명도(Value)로 분리되어 색상 기반 객체 검출(예: 피부색 분리) 및 조명 변화 대응에 매우 유리합니다.

1. **해상도, DPI 및 이미지 파일 포맷 비교**

	- **해상도와 픽셀 밀도:** 해상도는 \text{가로}(W) \times \text{세로}(H) 픽셀 수이며(Full HD: 1920\times1080, 4K: 3840\times2160), DPI(Dots Per Inch)는 인쇄/스캔 시 물리적 인치당 점(Dot)의 수를 의미합니다.

	- **주요 파일 포맷과 압축 방식:**

		- **JPEG:** 손실 압축(Lossy) 방식으로 인간의 눈이 둔감한 고주파 색차 정보를 제거하여 파일 용량을 크게 절감합니다. 웹 환경에서 표준으로 쓰입니다.

		- **PNG:** 무손실 압축(Lossless) 방식으로 원본 손실이 없으며 투명도를 지원하는 알파(Alpha) 채널을 포함합니다.

		- **BMP:** 비압축 래스터 포맷으로 무손실 고품질을 유지하나 용량이 매우 큽니다.

		- **TIFF:** 비압축/무손실 고해상도 포맷으로 의료, 지리 정보, 출판 등 전문 분야에서 표준으로 사용됩니다.

		- **GIF:** 256색 팔레트 기반의 간단한 애니메이션을 지원합니다.

[//]: # (heading_4 is not supported)

1. **고차원 텐서(Tensor) 표현과 메모리 요구량**

	- **3차원 텐서 표현:** 이미지는 컴퓨터 내부에서 높이(H) \times 너비(W) \times 채널(C)의 3차원 텐서로 표현됩니다.

	- **메모리 병목과 배치 처리:** 1920\times1080 RGB 이미지 1장은 메모리 상에서 약 6\text{MB}를 차지하며, 수만 장의 고해상도 이미지를 직접 처리할 경우 수백 GB 이상의 메모리 부하가 발생합니다. 따라서 미니배치(Mini-batch) 분할 처리와 적절한 다운샘플링 전처리가 필수적입니다.

1. **공간적 연속성(Spatial Locality)과 다중 스케일 특성**

	- **Spatial Locality (공간적 국소성):** 이미지 내 인접한 픽셀들은 서로 강한 통계적 상관관계를 갖습니다. 전체 이미지를 한 번에 전결합하지 않고 작은 커널을 슬라이딩시키는 CNN의 합성곱 연산이 효율적인 근본 이유입니다.

	- **다중 스케일(Multi-Scale) 특징:** 저해상도에서는 전체적인 형상과 윤곽선(Global Outline)이 지배적이며, 고해상도에서는 세밀한 질감과 미세 엣지(Local Detail)가 드러납니다. SIFT나 Feature Pyramid Network(FPN) 등은 이러한 스케일 불변성을 다루도록 설계됩니다.

[//]: # (heading_4 is not supported)

1. **히스토그램 분석 및 밝기/대비**

	- **히스토그램(Histogram):** 픽셀 밝기/색상 값의 빈도 분포를 나타냅니다. 편향된 분포를 평탄화하는 **히스토그램 평활화(Histogram Equalization)**를 통해 어둡거나 흐릿한 저조도 이미지의 명암 대비(Contrast)를 개선할 수 있습니다.

	- **밝기와 대비:** 밝기는 픽셀들의 산술 평균값에 비례하며, 대비는 최소 밝기 영역과 최대 밝기 영역의 편차에 비례합니다.

1. **텍스처 분석 기법 (GLCM & LBP)**

	- **GLCM (Gray-Level Co-occurrence Matrix):** 특정 공간적 거리와 방향에 위치한 두 픽셀 간의 밝기 조합 빈도를 행렬로 표현하여 질감의 규칙성과 균일성을 수치화합니다.

	- **LBP (Local Binary Patterns):** 중심 픽셀과 주변 인접 픽셀들의 밝기를 비교하여 이진 코드로 변환함으로써, 연산량이 매우 적으면서도 회전 및 조명 변화에 강인한 텍스처 패턴을 추출합니다.

1. **이미지의 주파수 특성과 공간 필터링**

	- **저주파수 성분 (Low Frequency):** 픽셀 값이 완만하게 변화하는 영역으로, 이미지의 전반적인 배경, 전체 윤곽, 큰 형상 정보를 담당합니다. (저주파 통과 필터 = 블러링)

	- **고주파수 성분 (High Frequency):** 픽셀 값이 급격하게 변화하는 영역으로, 세밀한 텍스처, 엣지, 미세 노이즈를 형성합니다. (고주파 통과 필터 = 엣지 검출 및 샤프닝)

[//]: # (heading_4 is not supported)

- [OpenCV Color Spaces Documentation](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)

- [Scikit-Image GLCM Texture Feature Tutorial](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html)

- [Scikit-Image Local Binary Pattern Guide](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html)

- [OpenCV Histograms Equalization Tutorial](https://docs.opencv.org/4.x/d4/d1b/tutorial_py_histogram_equalization.html)

