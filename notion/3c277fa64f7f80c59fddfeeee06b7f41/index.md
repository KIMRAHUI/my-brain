---
notion-url: https://app.notion.com/p/Computer-Vision-Applications-3c277fa64f7f80c59fddfeeee06b7f41
title: Computer-Vision-Applications
date: '2026-08-20 09:44:00.000'
from_notion: https://app.notion.com/p/Computer-Vision-Applications-3c277fa64f7f80c59fddfeeee06b7f41
author: rahui kim
last_edited_time: '2026-08-20 09:44:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

1. **자율주행의 비전 파이프라인과 엔드투엔드(E2E) 아키텍처**

	- **주요 목적 및 요소 기술:** 주행 환경의 3차원 공간 이해와 실시간 안전 궤적 설계를 위해 객체 검출(차량, 보행자, 신호 체계), 차선 및 주행 가능 구역 세그멘테이션 기술을 융합합니다.

	- **엔드투엔드(End-to-End, E2E) 아키텍처 비교 분석:**

		- **주요 장점 (Pros):**

			- **병목현상 제거 및 정보 손실 최소화:** 원본 센서 스트림(카메라, 라이다 등)에서 최종 제어 명령(조향, 제동)까지 단일 신경망으로 전달되어 모듈 간 변환 오차가 제거됩니다.

			- **인간과 유사한 부드러운 주행감:** 대규모 주행 실증 데이터를 모방 학습하여 복잡한 비정형 도로에서도 유연하고 자연스러운 대처가 가능합니다.

			- **파이프라인 단순화 및 스케일링:** 복잡한 규칙 기반(Rule-based) 코딩을 축소하고, 데이터와 컴퓨팅 자원 확장에 비례해 성능이 향상되는 스케일링 법칙(Scaling Law) 적용이 용이합니다.

		- **주요 단점 및 한계 (Cons):**

			- **블랙박스 문제와 설명 가능성(Explainability) 결여:** 제어 명령의 역추적이 어려워 안전 검증 및 사고 발생 시 법적 책임 소재 규명이 어렵습니다.

			- **희귀 엣지 케이스(Edge Case) 취약성:** 학습에 포함되지 않은 비정형 낙하물이나 특이 수신호 상황에서 비정상적 제어 명령(Hallucination)을 내릴 위험이 존재합니다.

			- **방대한 학습 비용과 데이터 정제 부담:** 수많은 주행 영상과 대규모 GPU 클러스터가 필수적이며 편향 데이터 제거를 위한 큐레이션 비용이 높습니다.

1. **의료 영상 분석의 진단 편차 극복과 데이터 모델링**

	- **진단 편차와 실제값(Ground Truth)의 모호성:** 동일한 CT/MRI 영상에 대해서도 전문의 간 숙련도와 기준에 따른 판독 편차(Variability)가 발생하며, 조직 검사 확진 데이터 역시 채취 위치 오차를 내포할 수 있어 단일 권위자의 라벨만을 절대적 기준으로 삼기 어렵습니다.

	- **현대 의료 AI의 타겟 정합 접근법:**

		- **다중 주석자(Multi-Annotator) 확률적 모델링:** 단일 확정 라벨 대신 복수 의사의 판독 결과를 확률 분포(Probabilistic Label, 예: 악성 확률 70%) 형태로 보존하여 학습합니다.

		- **합의(Consensus) 기반 라벨링 및 델파이 기법:** 독립 판독 후 구조화된 합의 프로토콜(Consensus Voting)을 거쳐 표준화된 타겟 데이터셋을 확정합니다.

		- **불확실성 반영 손실 함수(Robust Loss Function):** 라벨 노이즈(Label Noise)를 감안한 손실 함수를 설계하여 과적합을 방지하고 일반화 성능을 확보합니다.

[//]: # (heading_4 is not supported)

1. **얼굴 인식(Face Recognition) 시스템 4단계 파이프라인**

	- **Step 1 - 얼굴 검출 (Face Detection):** 입력 이미지에서 배경과 얼굴을 분리하고 Bounding Box 위치를 특정합니다. (예: YOLO, SSD, MTCNN, RetinaFace)

	- **Step 2 - 얼굴 정렬 및 전처리 (Alignment & Preprocessing):** 눈, 코, 입 등 주요 랜드마크(Landmark) 위치를 정렬하고 크기 및 조명을 표준화합니다.

	- **Step 3 - 얼굴 임베딩 및 특징 추출 (Face Embedding & Feature Extraction):** 심층 CNN 백본을 통해 고유한 생체 특징을 고차원 수치 벡터(Embedding Vector)로 압축 변환합니다. (예: ArcFace Loss)

	- **Step 4 - 유사도 비교 및 매칭 (Similarity Matching):** 저장된 DB 벡터와 입력 벡터 간의 거리(코사인 유사도, 유클리디안 거리)를 산출하여 임계값 기반 인증을 수행합니다.

1. **주요 얼굴 인식 오픈소스 생태계**

	- `**face_recognition**`** (dlib 기반):** 간단한 파이썬 인터페이스로 얼굴 검출 및 인식을 지원하는 교육 및 프로토타이핑용 표준 라이브러리입니다.

	- `**InsightFace**`**:** ResNet/MobileNet 등 다양한 백본과 ArcFace 손실 함수를 탑재하여 산업 현장에서 가장 널리 쓰이는 SOTA 프레임워크입니다.

	- `**RetinaFace**`**:** 단일 단계 랜드마크 정밀 검출에 특화되어 실시간 영상 스트리밍 분석에 최적화된 고속 검출 모델입니다.

1. **AR/VR을 위한 SLAM(동시적 위치 추정 및 지도 작성) 메커니즘**

	- **SLAM 5단계 작동 프로세스:**

		1. **초기화(Initialization):** 센서(카메라, 라이다) 데이터를 바탕으로 초기 기준 좌표계 및 지도를 생성합니다.

		1. **데이터 획득(Data Acquisition):** 이동에 따라 연속적인 환경 특징점(Landmark)을 수집합니다.

		1. **위치 추정(Localization):** 이전 시점과 현재 관측 데이터를 대조하여 센서의 이동 궤적을 계산합니다.

		1. **지도 업데이트(Mapping):** 추정된 위치 좌표를 기반으로 새로운 3차원 지형지물을 맵에 추가합니다.

		1. **루프 클로저(Loop Closure):** 이전에 방문한 위치에 재진입했을 때 누적 오차를 감지하고 전역 지도를 일관성 있게 보정합니다.

	- **SLAM 핵심 분류 및 특성 비교표:**

 | 분류 | 사용 센서 / 기법 | 핵심 장점 | 주요 단점 / 한계 | 대표 오픈소스 | 
 | ---- | ---- | ---- | ---- | ---- | 
 | **Visual SLAM** | 모노/스테레오/RGB-D 카메라 | 하드웨어 비용 저렴, 풍부한 시각 텍스처 정보 | 조명 급변 및 텍스처 없는 벽면 환경에 취약 | `ORB-SLAM3` | 
 | **LiDAR SLAM** | 3D 라이다 + IMU 센서 | 초정밀 거리 측정, 조명 무관 작동 | 고가의 센서 비용 및 무게/부피 부담 | `LIO-SAM` | 
 | **2D/3D 매핑 최적화** | 다중 센서 융합 + 그래프 최적화 | 실시간 매핑 최적화 및 루프 클로저 우수 | 환경 규모 증가에 따른 연산 비용 증가 | `Cartographer` (Google) | 

[//]: # (heading_4 is not supported)

1. **대조 학습과 생성형 비전 모델**

	- **CLIP (Contrastive Language-Image Pre-training):** 수억 쌍의 이미지-텍스트 데이터를 대조 학습(Contrastive Learning)시켜, 동일한 의미의 이미지와 텍스트 벡터 거리는 좁히고 다른 의미는 멀어지도록 학습합니다. 별도의 라벨링 없이도 강력한 제로샷(Zero-shot) 분류 성능을 발휘합니다.

	- **DALL-E (Text-to-Image Generation):** 텍스트 프롬프트를 조건(Condition)으로 입력받아 노이즈 상태에서 시작해 점진적으로 깨끗한 이미지를 복원해내는 확산 모델(Diffusion Model) 아키텍처를 기반으로 시각 콘텐츠를 생성합니다.

1. **거대 멀티모달 모델(LMM)과 AI 이미지 캡셔닝**

	- **LLaVA (Large Language-and-Vision Assistant):** CLIP 비전 인코더와 거대 언어 모델(LLM)을 투영 계층(Projection Layer)으로 결합하여 시각적 질의응답 및 상세 추론을 수행합니다.

	- **AI 이미지 캡셔닝 (Image Captioning) Encoder-Decoder 구조:**

		- **비전 인코더 (Vision Encoder):** CNN 또는 ViT를 통해 입력 이미지로부터 고차원 시각적 특징 벡터(Visual Feature)를 추출합니다.

		- **어텐션 메커니즘 (Attention Mechanism):** 단어를 순차적으로 생성할 때 이미지의 특정 공간 영역에 선택적으로 가중치를 부여합니다.

		- **텍스트 디코더 (Text Decoder):** Transformer 디코더가 문맥과 시각 가중치를 결합하여 자연어 문장을 순차적으로 예측/완성합니다.

[//]: # (heading_4 is not supported)

- [Waymo Open Dataset Official Repository](https://github.com/waymo-research/waymo-open-dataset)

- [face_recognition Python Package GitHub](https://github.com/ageitgey/face_recognition)

- [InsightFace Deep Face Analysis GitHub](https://github.com/deepinsight/insightface)

- [ArcFace: Additive Angular Margin Loss Paper](https://arxiv.org/abs/1801.07698)

- [RetinaFace PyTorch Implementation](https://github.com/biubug6/Pytorch_Retinaface)

- [LIO-SAM LiDAR SLAM Repository](https://github.com/TixiaoShan/LIO-SAM)

- [ORB-SLAM3 Visual SLAM Framework](https://github.com/UZ-SLAMLab/ORB_SLAM3)

- [Google Cartographer Real-time SLAM](https://github.com/cartographer-project/cartographer)

- [OpenAI CLIP Official Research & Code](https://github.com/openai/CLIP)

- [OpenAI DALL-E Official Overview](https://openai.com/research/dall-e)

- [LLaVA GitHub Repository](https://github.com/haotian-liu/LLaVA)

- [Salesforce BLIP / BLIP-2 GitHub](https://github.com/salesforce/BLIP)

- [NVIDIA Image Captioner Repository](https://github.com/NVIDIA/image-captioner)

