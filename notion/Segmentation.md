# Segmentation

개요

본 문서는 Image Segmentation(이미지 분할) 의 개념부터 현대 딥러닝 기반 세그멘테이션의 핵심 아키텍처까지의 발전 과정을 다룹니다. Segmentation의 정의와 Semantic/Instance/Panoptic 세 가지 유형을 시작으로, 초기 Patch 기반 접근의 한계, Up-sampling/Down-sampling 기법, Fully Convolutional Networks(FCN, 2014), 의료 영상 분할에 특화된 U-Net(2015), 그리고 Instance Segmentation의 대표 모델인 Mask R-CNN(2017) 까지 체계적으로 정리하였습니다.

Part 1. Image Segmentation의 개념과 분류

Image Segmentation이란?

Segmentation의 세 가지 유형

초기 Computer Vision의 Segmentation 접근 (Selective Search)

Part 2. Semantic Segmentation — Patch 기반 접근에서 Encoder-Decoder 구조로

Semantic Segmentation의 정의와 목표

초기 아이디어와 한계

최종 아이디어 — Auto Encoder 형식의 Down/Up Sampling

Up-Sampling 방법론

Part 3. Fully Convolutional Networks (FCN, 2014)

FCN 개념 및 접근법

핵심 설계 — FC Layer 제거

- **FC Layer의 단점:**

- **전환(Converting Classifiers to FCN):** 완전 연결층을 모두 컨볼루션 연산으로 대체하여, 공간 정보를 보존하면서 임의 크기 입력에 대해 예측을 수행합니다.

10. FCN 구조

Convolution Layer를 통해 Feature를 추출합니다.

1×1 Convolution Layer로 피처맵의 채널 수를 데이터셋 객체 개수와 동일하게 변경합니다(Class Presence Heat Map 추출).

Up-sampling으로 낮은 해상도의 Heat Map을 입력 이미지와 같은 크기의 Map으로 복원합니다.

최종 피처 맵과 라벨 피처맵의 차이를 이용하여 네트워크를 학습합니다.

핵심 구성: Conv Layer만 사용(입력 크기 무관) + Transpose Convolution(Deconvolution 효과로 Featuremap upsample) + Skip Connection(세밀한 정보를 Decoder에 전달).

Skip Connection과 FCN-32s / 16s / 8s

Up-sampling 방법: Bilinear Interpolation과 Deconvolution을 사용해 원본 픽셀과 가까운 dense-map으로 변환합니다.

Skip Connections: 다운샘플링 과정에서 얻는 깊은 층의 특징은 의미(semantic) 정보는 풍부하지만 해상도가 낮아 세부 정보가 희석됩니다. 반대로 얕은 층은 해상도는 높으나 의미 정보가 부족합니다. FCN은 Skip Connection을 도입하여 깊은 층의 coarse한 예측과 얕은 층의 세밀한 공간 정보를 결합합니다.

FCN-32s / FCN-16s / FCN-8s: pool3, pool4, pool5 단계의 정보를 각각 다른 비율로 결합하여 업샘플링하는 변형으로, 숫자가 작을수록(8s) 더 많은 저수준 정보를 결합해 세밀한 결과를 냅니다.

Loss Function

픽셀 단위로 Cross-Entropy를 계산하여 모델 성능을 평가합니다. 각 픽셀의 클래스 확률을 Softmax 함수로 계산한 후 크로스 엔트로피를 적용합니다.

FCN의 장단점

장점: 픽셀 단위 End-to-End 학습 가능 / 다양한 네트워크(VGG, ResNet 등)와 결합 용이 / Skip Connection으로 경계 복원 능력 향상.

단점: 단순한 업샘플링 방식으로 세밀한 경계 복원에 한계가 있으며, 복잡한 객체가 많은 장면에서 성능이 저하됩니다.

Part 4. U-Net (2015) — 의료 영상 분할 특화 아키텍처

U-Net 개념 및 접근법

FCN의 한계와 U-Net의 목표

아키텍처 — 인코더(Contracting Path)와 디코더(Expansive Path)

Input Image Processing

Data Augmentation 기법

Loss Function

U-Net의 장단점

장점: 스킵 연결과 대칭 구조를 통한 정밀한(픽셀 단위) 분할 / 데이터 증강과 경계선 가중치로 소량 데이터로도 학습 가능 / 최신 GPU에서 빠른 추론 속도로 실시간 처리 가능.

단점 및 한계: 의료 영상에 특화되어 일반 이미지 분할에는 다른 구조가 더 적합할 수 있음 / 인코더·디코더의 다채널 특징 맵과 스킵 연결로 메모리 사용량이 많아질 수 있음 / 패딩 없이 처리하므로 경계부 정보 처리에 주의가 필요하며 overlap-tile 등 보완 기법을 반드시 적용해야 함.

Part 5. Instance Segmentation과 Mask R-CNN (2017)

Semantic Segmentation vs Instance Segmentation

Mask R-CNN 개념 및 Main Idea

Faster R-CNN 복습

Mask R-CNN 아키텍처

Mask R-CNN 파이프라인 (Backbone: ResNet + FPN)

입력 이미지를 800~1024 픽셀 크기로 bilinear interpolation을 사용해 리사이즈.

ResNet-101로 각 레이어(stage)에서 특징 맵(C1~C5) 생성.

FPN(Feature Pyramid Network)으로 다중 스케일 특징 맵(P2~P6) 생성.

각 FPN 특징 맵에 RPN을 적용해 분류(classification)와 bbox regression 값 예측.

RPN의 bbox regression 출력을 앵커(anchor)에 적용해 후보 영역(proposal) 생성.

Non-Maximum Suppression(NMS)으로 중복·겹치는 영역 제거, 최종 후보 선택.

RoI Align으로 서로 다른 크기의 후보 영역을 고정 크기 특징 맵으로 정밀 정렬.

고정 크기 특징 맵을 Fast R-CNN의 classification/bbox regression 브랜치와 mask branch에 통과시켜, 클래스 분류·경계 상자 회귀·픽셀 단위 분할 마스크를 최종 산출.

Loss

Mask R-CNN은 분류, 박스 회귀, 마스크 손실을 포함한 다중 작업(multi-task) Loss를 동시에 최적화합니다.

RoI Align vs RoI Pooling

RoI Pooling의 문제: 각 RoI를 특징 맵의 격자(grid) 단위로 양자화(quantization)하여 고정 크기 특징 맵(예: 7×7)을 추출하는데, 이 양자화 과정에서 RoI의 경계와 내부 세부 정보가 손실되어 픽셀 단위 정렬이 어려워지고 특히 마스크 분할 작업에서 성능이 저하됩니다.

RoI Align: RoIPooling의 양자화 문제를 해결하기 위해 고안되었으며, 양자화를 제거하고 정확한 위치의 특징을 추출하는 것이 핵심 아이디어입니다. RoI의 경계와 내부 빈(bin)을 양자화하지 않고 원래의 실수 좌표를 그대로 사용하며, 각 빈마다 여러 샘플 포인트를 선정해 주변 격자 점으로부터 bilinear interpolation으로 값을 계산합니다.

동작 순서: ① RoI를 소수점 그대로 매핑하고 개별 Grid에 4개의 포인트를 균등 배열 → ② 개별 포인트에서 가장 가까운 feature map grid를 고려해 보간법으로 계산 → ③ 계산된 포인트를 기반으로 MaxPooling 진행. 이를 통해 양자화를 제거하고 정확한 픽셀 정렬을 보장하여 마스크·키포인트 등 세밀한 작업에서 높은 정확도를 얻습니다.

최종 결과 및 후처리

RoI별 예측: 각 RoI에서 Fast R-CNN 헤드와 mask branch를 통해 클래스 확률, 박스 회귀, 고정 크기 마스크(예: 28×28)를 예측합니다.

후처리 단계:

최종 출력: 각 객체 별 클래스 레이블, 신뢰도 점수(Confidence Score), 경계 상자, 분할 마스크.

참고 논문 및 링크

Fully Convolutional Networks for Semantic Segmentation (FCN, 2014) - arXiv

U-Net: Convolutional Networks for Biomedical Image Segmentation - arXiv

Mask R-CNN - arXiv

Image Segmentation: The Deep Learning Approach

A Guide to Semantic Segmentation - Encord

What is Transposed Convolutional Layer - Towards Data Science

Bilinear Interpolation - Wikipedia

FCN 정리 - wikidocs

U-Net 정리 - wikidocs

Fast R-CNN 논문 리뷰: RoI Pooling Layer와 Truncated SVD - Medium

How to Use RoI Pool and RoI Align in Your Neural Networks (PyTorch) - Medium

RoI Pooling vs RoI Align - Medium

