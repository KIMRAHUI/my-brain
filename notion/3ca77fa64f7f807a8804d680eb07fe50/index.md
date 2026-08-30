---
notion-url: https://app.notion.com/p/Segmentation-3ca77fa64f7f807a8804d680eb07fe50
title: Segmentation
date: '2026-08-28 08:13:00.000'
from_notion: https://app.notion.com/p/Segmentation-3ca77fa64f7f807a8804d680eb07fe50
author: rahui kim
last_edited_time: '2026-08-28 08:13:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

1. **Image Segmentation이란?**

	- **정의:** Image Segmentation은 이미지 내에서 픽셀 단위로 객체와 배경, 또는 서로 다른 객체를 구분하는 작업입니다. 각 픽셀에 대해 어떤 클래스에 속하는지 라벨을 부여하여 이미지의 구조와 형태를 정밀하게 파악합니다.

	- **세밀한 정보 추출:** 단순한 전체 이미지 분류나 객체 위치 추정(Object Detection)보다 훨씬 정밀하게 객체의 경계와 형태를 파악합니다.

	- **응용 분야:** 자율주행, 의료 영상 분석, 위성 사진 분석 등에서 객체의 정확한 영역 구분이 필수적인 분야에 핵심적으로 활용됩니다.

1. **Segmentation의 세 가지 유형**

	- **Semantic Segmentation:**

		- 동일 클래스에 동일 태그를 부여합니다. 같은 클래스에 속한 모든 픽셀은 동일하게 분류되며, 개별 객체 간의 차이는 구분하지 않습니다.

		- 예: 도로 사진에서 차량, 도로, 보행자를 각각 다른 색으로 표시(같은 클래스 내 개체 구분 없음).

	- **Instance Segmentation:**

		- 동일 클래스 내에서도 개별 객체를 구분합니다. 클래스와 개별 객체를 함께 태깅하여 객체의 위치와 경계를 명확히 파악합니다.

		- 예: 도로 위 "차량 A"와 "차량 B"를 각각 다른 색으로 표시.

	- **Panoptic Segmentation:**

		- Semantic + Instance를 통합한 방식으로, 개별 객체(사람, 자동차 등 Instance)와 배경 맥락("도로", "하늘" 같은 Stuff 클래스)을 함께 구분합니다.

1. **초기 Computer Vision의 Segmentation 접근 (Selective Search)**

	- 초기 Computer Vision에서는 Selective Search와 같이 주변 픽셀과의 관계를 이용해 Segmentation을 수행했습니다.

	- **한계:** 맥락(context) 파악 없이 인접 픽셀 관계에만 의존하여 경계가 명확하게 분리되지 않고, 연산에 오랜 시간이 소요되는 문제가 있었습니다.

[//]: # (heading_4 is not supported)

1. **Semantic Segmentation의 정의와 목표**

	- 이미지 내에서 어떤 물체(또는 영역)가 어디에 있는지를 픽셀 단위로 파악하는 작업입니다.

	- 객체의 윤곽(경계)을 정밀하게 추론해야 하므로 고해상도 공간 정보가 중요합니다.

	- 인스턴스 간 구분("소가 몇 마리?")은 고려하지 않는다는 점에서 Instance Segmentation과 구분됩니다.

1. **초기 아이디어와 한계**

	- **패치(Patch) 기반 분류:** 타겟 픽셀 주변의 작은 패치를 잘라내어 소규모 CNN에 입력, 중앙 픽셀의 클래스를 예측하는 방식에서 출발했습니다.

	- **맥락 정보의 필요성:** 하나의 픽셀만으로는 클래스를 판단하기 어려우므로, 픽셀 주변의 맥락 정보를 함께 파악하기 위해 CNN을 활용해야 합니다.

	- **전통적 분류 CNN의 한계:** AlexNet, VGG, ResNet 등 전통적 CNN은 입력 이미지를 받아 하나의 분류 결과만 출력하며, 중간 단계에서 공간 해상도가 점차 축소되어 최종적으로 단일 벡터로 압축됩니다. 의미 분할을 위해서는 "이미지 전체의 단일 라벨"이 아니라 "각 픽셀에 대한 라벨"이 필요하며, 즉 출력 해상도가 입력과 동일해야 합니다.

	- **다운샘플링을 없앤다면?** 모든 연산을 컨볼루션 레이어로만 구성하고 다운샘플링(풀링, 스트라이드)을 제거하면 미세 구조는 보존되지만, 메모리 사용량과 연산량이 기하급수적으로 증가하고 Receptive Field가 좁아 객체 간 관계 파악이 어려워지는 문제가 발생합니다.

1. **최종 아이디어 — Auto Encoder 형식의 Down/Up Sampling**

	- CNN 구조를 유지하면서 Down Sampling과 Up Sampling을 연속으로 적용하는 Auto Encoder 형식이 해법으로 제시되었습니다.

	- **다운샘플링(Pooling, Stride 등):** 계산량 감소와 넓은 Receptive Field 확보를 통해 추상적이고 강력한 특징을 얻고, 작은 이동·스케일 변화에 둔감한 표현 불변성을 갖게 되지만 공간 정보가 손실되는 단점이 있습니다.

	- **업샘플링(Deconvolution, Transposed Convolution 등):** 줄어든 해상도를 복원하여 최종적으로 (H×W) 크기의 예측을 만듭니다. 단순 보간(bilinear)보다 학습 가능한 파라미터를 두면 객체 경계를 더 정확히 복원할 수 있습니다.

1. **Up-Sampling 방법론**

	- **Nearest Neighbor:** 가장 가까운 기존 데이터 포인트의 값을 참조합니다.

	- **Bed of Nails:** 작은 해상도의 값을 큰 해상도로 "복사"하고 나머지는 0으로 채웁니다.

	- **Max Unpooling:** 풀링 시 기록해 둔 위치 정보를 기반으로, 업샘플링 시 해당 위치에 값을 다시 채워 넣습니다.

	- **Bilinear Interpolation(이중 선형 보간):** 인접한 4픽셀 값의 가중합으로 새 픽셀 값을 계산합니다. 학습 파라미터는 없지만 빠르고 부드러운 보간이 가능합니다(연산량은 증가).

		- **선형 보간법(1차 보간법):** 두 점 $(x_0, y_0)$, $(x_1, y_1)$이 주어졌을 때, 이 두 점을 잇는 직선 위의 임의의 점 $(x, y)$ 값을 직선의 방정식으로 계산하는 방법입니다.

		- **이중 선형 보간법:** 2차원 격자(이미지 픽셀 그리드)에서 네 점을 기반으로 중간 값을 예측합니다. 먼저 한 방향(가로)으로 선형 보간을 수행한 후, 그 결과를 이용해 다른 방향(세로)으로 보간합니다.

	- **Deconvolution(Transposed Convolution):** 3×3 필터, stride=2 등으로 학습 가능한 업샘플링을 수행합니다. 필터를 데이터에 맞춰 최적화하여 객체 경계 등 세부 표현을 향상시킵니다. PyTorch의 `ConvTranspose2d`가 이를 구현하며, 주로 업샘플링·디코딩 단계에서 활용됩니다.

		- **출력 크기 공식:** $o = (i-1) \times s + k - 2p$ (padding 추가, input 행렬 사이에 zero 삽입 후 Convolution 수행)

[//]: # (heading_4 is not supported)

1. **FCN 개념 및 접근법**

	- **정의:** 기존 이미지 분류용 ConvNet을 변형하여, 입력 이미지 크기에 상관없이 각 픽셀마다 예측을 수행할 수 있는 모델입니다. (논문: *"Fully Convolutional Networks for Semantic Segmentation", 2014*, [arXiv:1411.4038](https://arxiv.org/pdf/1411.4038))

	- **End-to-End 학습:** 입력 이미지 전체를 한 번에 처리하여 패치 단위 학습보다 효율적입니다.

	- **유연한 입력 크기:** 고정 크기가 아닌 임의 크기의 이미지를 입력받아 그에 맞는 분할 결과를 생성합니다.

	- **밀집 예측(Dense Prediction):** 각 픽셀마다 클래스를 예측하여 객체의 정확한 경계와 구조를 파악합니다.

1. **핵심 설계 — FC Layer 제거**

	- FCN의 가장 큰 특징은 VGG16에서 **Fully Connected(FC) Layer를 제거**하는 것입니다. FC Layer 통과를 위해서는 동일한 크기의 데이터가 입력되어야 하는 제약이 있었습니다.

	- **FC Layer의 장점:**

 | 장점 | 상세 설명 | 
 | ---- | ---- | 
 | 전역 의존성 학습 | 이미지 전체 픽셀을 한꺼번에 연결하므로 "고양이 귀+꼬리 → 고양이"처럼 멀리 떨어진 특징 조합을 쉽게 학습 | 
 | 강한 비선형 결정 경계 | 대량의 파라미터가 복잡한 클래스 구분에 유리 | 
 | 전이학습 시 유연성 | 마지막 FC 층만 교체·미세조정하면 새로운 클래스를 빠르게 학습 | 


```plain text
- **FC Layer의 단점:**
```

 | 단점 | 영향 | 
 | ---- | ---- | 
 | 파라미터·메모리 증가 | Nin×Nout 규모 가중치로 무겁고 학습이 느림 | 
 | 공간 정보 소실 | Flatten 과정에서 위치·형태 정보가 사라져 픽셀 단위 작업 정확도 급락 | 
 | 고정 입력 크기 요구 | 가중치 행렬이 입력 해상도에 종속되어 가변 크기 이미지 처리 어려움 | 
 | 계층적 지역 패턴 학습 부재 | Conv의 weight sharing·국소 불변성이 없어 일반화력이 낮음 | 


```plain text
- **전환(Converting Classifiers to FCN):** 완전 연결층을 모두 컨볼루션 연산으로 대체하여, 공간 정보를 보존하면서 임의 크기 입력에 대해 예측을 수행합니다.
```

- Convolution Layer를 통해 Feature를 추출합니다.

- 1×1 Convolution Layer로 피처맵의 채널 수를 데이터셋 객체 개수와 동일하게 변경합니다(Class Presence Heat Map 추출).

- Up-sampling으로 낮은 해상도의 Heat Map을 입력 이미지와 같은 크기의 Map으로 복원합니다.

- 최종 피처 맵과 라벨 피처맵의 차이를 이용하여 네트워크를 학습합니다.

- 핵심 구성: **Conv Layer만 사용**(입력 크기 무관) + **Transpose Convolution**(Deconvolution 효과로 Featuremap upsample) + **Skip Connection**(세밀한 정보를 Decoder에 전달).

1. **Skip Connection과 FCN-32s / 16s / 8s**

- **Up-sampling 방법:** Bilinear Interpolation과 Deconvolution을 사용해 원본 픽셀과 가까운 dense-map으로 변환합니다.

- **Skip Connections:** 다운샘플링 과정에서 얻는 깊은 층의 특징은 의미(semantic) 정보는 풍부하지만 해상도가 낮아 세부 정보가 희석됩니다. 반대로 얕은 층은 해상도는 높으나 의미 정보가 부족합니다. FCN은 Skip Connection을 도입하여 깊은 층의 coarse한 예측과 얕은 층의 세밀한 공간 정보를 결합합니다.

- **FCN-32s / FCN-16s / FCN-8s:** pool3, pool4, pool5 단계의 정보를 각각 다른 비율로 결합하여 업샘플링하는 변형으로, 숫자가 작을수록(8s) 더 많은 저수준 정보를 결합해 세밀한 결과를 냅니다.

1. **Loss Function**

- 픽셀 단위로 Cross-Entropy를 계산하여 모델 성능을 평가합니다. 각 픽셀의 클래스 확률을 Softmax 함수로 계산한 후 크로스 엔트로피를 적용합니다.

1. **FCN의 장단점**

- **장점:** 픽셀 단위 End-to-End 학습 가능 / 다양한 네트워크(VGG, ResNet 등)와 결합 용이 / Skip Connection으로 경계 복원 능력 향상.

- **단점:** 단순한 업샘플링 방식으로 세밀한 경계 복원에 한계가 있으며, 복잡한 객체가 많은 장면에서 성능이 저하됩니다.

[//]: # (heading_4 is not supported)

1. **U-Net 개념 및 접근법**

	- **정의:** 생물의학(의료) 영상 분할에 특화된 CNN 구조로, 좌우 대칭적인 U자형 아키텍처를 가집니다. (논문: *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*, [arXiv:1505.04597](https://arxiv.org/pdf/1505.04597))

	- **Skip Connections:** 인코더의 특징을 디코더의 대응 레이어에 직접 연결하여 로컬 정보와 문맥 정보를 동시에 활용합니다.

	- **엔드투엔드 학습:** 완전 연결층 없이 전 과정을 컨볼루션으로 처리하여 소량의 주석 이미지로도 효과적으로 학습합니다.

	- **데이터 증강:** Elastic Deformation 등의 기법으로 제한된 데이터셋에서도 강력한 성능을 발휘합니다.

1. **FCN의 한계와 U-Net의 목표**

	- **FCN의 한계:** 업샘플링 과정에서 정보 손실이 발생해 세부 경계 복원이 어렵고, Encoder에서 추출한 고해상도 정보를 Decoder가 효과적으로 활용하지 못합니다. 다층 특징을 결합하지만 구조가 U-Net만큼 대칭적이지 않고 스킵 연결도 단순한 경우가 많습니다.

	- **U-Net의 목표:** FCN의 단점을 보완하여 세부 경계까지 복원 가능한 모델을 제안합니다. 인코더와 디코더가 명확히 대칭을 이루며, 인코더 각 단계의 고해상도 정보를 디코더에 직접 연결해 세밀한 위치 정보를 복원하고, Skip Connection으로 고해상도·저해상도 특징을 결합합니다.

1. **아키텍처 — 인코더(Contracting Path)와 디코더(Expansive Path)**

	- **인코더(Contracting Path):** 입력 이미지에서 점진적으로 해상도를 줄이며 중요한 특징을 추출합니다. 반복적으로 3×3 컨볼루션 → ReLU 활성화를 적용하고, 2×2 Max Pooling으로 다운샘플링(해상도 감소, 채널 증가)합니다.

	- **디코더(Expansive Path):** 인코더에서 추출된 정보를 바탕으로 점진적으로 해상도를 복원하며 세밀한 분할 결과를 생성합니다. 업샘플링(업컨볼루션)으로 해상도를 복원한 뒤, 대응하는 인코더 단계에서 잘라낸(cropped) 특징 맵과 결합(스킵 연결)하고, 이어서 두 번의 3×3 컨볼루션과 ReLU를 적용합니다.

	- **최종 출력:** 1×1 컨볼루션으로 각 픽셀의 64차원 특징 벡터를 최종 클래스 수에 맞게 매핑하며, 입력 이미지의 전체 문맥 정보를 반영해 각 픽셀을 분류합니다.

1. **Input Image Processing**

	- 의료 데이터는 해상도와 크기가 매우 다양하고 종종 매우 큰 이미지를 다루어야 하므로, 저자들은 여러 전략을 사용했습니다.

	- **Overlap-Tile 전략:** 기존 슬라이딩 윈도우 방식은 중복 연산이 많아 계산량이 큽니다. 저자들은 이미지를 일정 크기의 Patch(격자)로 잘라 중복을 줄이되, 타일 간에 일정 부분 겹치도록(overlap) 분할하여 경계부 정보가 끊기지 않고 매끄럽게 연결되도록 했습니다.

	- **Mirroring Extrapolation:** 타일 분할 시 입력 영역이 실제 이미지 밖으로 넘어가 컨볼루션에 필요한 문맥이 부족해지는 경우, 이미지를 좌우대칭(mirroring)으로 확장하여 부족한 부분을 채웁니다. Zero-padding보다 의료 영상(세포, 조직 등)에서 더 자연스럽고 데이터 증강 효과도 얻을 수 있습니다.

1. **Data Augmentation 기법**

	- 의료 영상은 주석 데이터가 매우 제한적인 경우가 많아, 다음 기법으로 데이터를 늘리고 일반화 성능을 높였습니다.

	- **Shift(이동 변환):** 이미지를 x, y 방향으로 조금씩 이동.

	- **Rotation(회전 변환):** 90도, 180도, 270도 등으로 회전.

	- **Gray value 변환:** 조도(밝기)나 명암비를 조절해 다양한 조명 환경을 모사.

	- **Elastic Deformation(탄성 변형):** 3×3 격자에서 무작위 변위 벡터를 샘플링하고 이미지 전체에 부드럽게 적용하여 뒤틀리는 효과를 주며, 세포 조직에서 자주 발생하는 실제 변형을 흉내냅니다.

1. **Loss Function**

	- **픽셀 단위 Softmax:** 최종 특징 맵의 각 픽셀에 대해 K개 클래스에 대한 확률을 계산합니다.

	- **Cross Entropy 손실:** 각 픽셀에서 예측 확률과 실제 레이블의 차이를 벌점으로 환산합니다.

	- **경계선 가중치:** 경계선 픽셀에 더 큰 가중치(weight)를 부여해 경계가 정확히 분리되도록 학습합니다.

1. **U-Net의 장단점**

- **장점:** 스킵 연결과 대칭 구조를 통한 정밀한(픽셀 단위) 분할 / 데이터 증강과 경계선 가중치로 소량 데이터로도 학습 가능 / 최신 GPU에서 빠른 추론 속도로 실시간 처리 가능.

- **단점 및 한계:** 의료 영상에 특화되어 일반 이미지 분할에는 다른 구조가 더 적합할 수 있음 / 인코더·디코더의 다채널 특징 맵과 스킵 연결로 메모리 사용량이 많아질 수 있음 / 패딩 없이 처리하므로 경계부 정보 처리에 주의가 필요하며 overlap-tile 등 보완 기법을 반드시 적용해야 함.

[//]: # (heading_4 is not supported)

1. **Semantic Segmentation vs Instance Segmentation**

	- **Semantic Segmentation:** 이미지의 각 픽셀을 미리 정의된 클래스(사람, 자동차, 배경 등)로 분류합니다. 같은 클래스에 속한 픽셀은 동일 레이블을 가지며 개별 인스턴스는 구분하지 않습니다. (예: 여러 명의 사람을 모두 "사람" 클래스 하나로 분류)

	- **Instance Segmentation:** 이미지 내 각 객체 인스턴스를 개별적으로 탐지하고, 각 인스턴스에 대해 픽셀 단위 마스크를 생성합니다. 동일 클래스 내에서도 개별 객체를 구분합니다. (예: 각 사람을 별도의 인스턴스로 구분해 마스크 생성)

	- **비교 정리:**

		- **출력 형태/목표:** Semantic은 모든 픽셀에 클래스 레이블 예측(개별 구분 X) / Instance는 객체 탐지 후 인스턴스별 개별 마스크 예측.

		- **네트워크 구조:** Semantic은 FCN 기반 End-to-End 픽셀 분류(Softmax) / Instance는 Faster R-CNN 등 Detection 네트워크 + Segmentation Branch(RoI 기반, 이진 마스크 예측).

		- **정밀한 위치 정보:** Semantic은 경계 정렬이 상대적으로 덜 중요 / Instance는 정확한 경계를 위해 RoIAlign(양자화 제거, bilinear interpolation) 사용.

		- **학습 전략:** Semantic은 픽셀 단위 다중 클래스 교차 엔트로피 손실 / Instance는 분류·박스 회귀·마스크 손실을 동시에 최적화하는 다중 작업 손실.

1. **Mask R-CNN 개념 및 Main Idea**

	- **정의:** Faster R-CNN을 확장하여 경계 상자(bounding box) 예측뿐 아니라 각 객체 인스턴스의 정확한 픽셀 단위 분할(mask)까지 동시에 예측하는 모델입니다. (논문: *"Mask R-CNN"*, [arXiv:1703.06870](https://arxiv.org/pdf/1703.06870))

	- **주요 특징:** 객체 탐지와 분할의 통합(위치·클래스·마스크를 하나의 모델에서 예측) / Faster R-CNN에 브랜치를 추가하는 유연한 구조로 키포인트 검출·자세 추정 등으로 확장 용이 / 약 5fps의 비교적 빠른 추론 속도로 실무·연구에 널리 사용.

	- **Main Idea:** Faster R-CNN이 RoI에 대해 클래스 분류와 경계 상자 회귀를 수행하는 데 더해, Mask R-CNN은 마스크 분할 브랜치를 추가하여 각 RoI에 대해 픽셀 단위 이진 마스크를 예측합니다. 분류(Classification), 경계 상자 회귀(Box Regression), 마스크 예측(Mask Prediction) 세 가지를 동시에 수행하는 **다중 작업 학습(Multi-Task Learning)** 구조이며, 클래스별로 독립적인 이진 마스크를 예측하여 서로 간섭 없이 결과를 산출합니다.

1. **Faster R-CNN 복습**

	- Region Proposal에 대해 이진 분류(객체 존재 Y/N)와 BB Regression(X, Y, W, H)을 수행하는 2-stage detector입니다.

1. **Mask R-CNN 아키텍처**

	- **백본 네트워크(Backbone):** ResNet, ResNeXt, FPN 등 CNN으로 이미지에서 특징을 추출합니다.

	- **영역 제안 네트워크(RPN):** 객체 후보 영역(Proposal)을 생성합니다(Faster R-CNN과 동일하게 작동).

	- **RoI 연산:** 각 후보 영역의 특징 맵 정보를 추출합니다. 기존 RoIPool 대신 정밀한 마스크 예측을 위해 **RoIAlign**을 사용합니다.

	- **헤드(Head):** 각 RoI의 정보로 분류, 박스 회귀, 마스크 예측 세 작업을 수행합니다. 마스크 예측은 작은 FCN으로 구현되어 객체의 공간적 레이아웃을 유지합니다.

1. **Mask R-CNN 파이프라인 (Backbone: ResNet + FPN)**

1. 입력 이미지를 800~1024 픽셀 크기로 bilinear interpolation을 사용해 리사이즈.

1. ResNet-101로 각 레이어(stage)에서 특징 맵(C1~C5) 생성.

1. FPN(Feature Pyramid Network)으로 다중 스케일 특징 맵(P2~P6) 생성.

1. 각 FPN 특징 맵에 RPN을 적용해 분류(classification)와 bbox regression 값 예측.

1. RPN의 bbox regression 출력을 앵커(anchor)에 적용해 후보 영역(proposal) 생성.

1. Non-Maximum Suppression(NMS)으로 중복·겹치는 영역 제거, 최종 후보 선택.

1. RoI Align으로 서로 다른 크기의 후보 영역을 고정 크기 특징 맵으로 정밀 정렬.

1. 고정 크기 특징 맵을 Fast R-CNN의 classification/bbox regression 브랜치와 mask branch에 통과시켜, 클래스 분류·경계 상자 회귀·픽셀 단위 분할 마스크를 최종 산출.

1. **Loss**

- Mask R-CNN은 분류, 박스 회귀, 마스크 손실을 포함한 다중 작업(multi-task) Loss를 동시에 최적화합니다.

1. **RoI Align vs RoI Pooling**

- **RoI Pooling의 문제:** 각 RoI를 특징 맵의 격자(grid) 단위로 양자화(quantization)하여 고정 크기 특징 맵(예: 7×7)을 추출하는데, 이 양자화 과정에서 RoI의 경계와 내부 세부 정보가 손실되어 픽셀 단위 정렬이 어려워지고 특히 마스크 분할 작업에서 성능이 저하됩니다.

- **RoI Align:** RoIPooling의 양자화 문제를 해결하기 위해 고안되었으며, 양자화를 제거하고 정확한 위치의 특징을 추출하는 것이 핵심 아이디어입니다. RoI의 경계와 내부 빈(bin)을 양자화하지 않고 원래의 실수 좌표를 그대로 사용하며, 각 빈마다 여러 샘플 포인트를 선정해 주변 격자 점으로부터 bilinear interpolation으로 값을 계산합니다.

- **동작 순서:** ① RoI를 소수점 그대로 매핑하고 개별 Grid에 4개의 포인트를 균등 배열 → ② 개별 포인트에서 가장 가까운 feature map grid를 고려해 보간법으로 계산 → ③ 계산된 포인트를 기반으로 MaxPooling 진행. 이를 통해 양자화를 제거하고 정확한 픽셀 정렬을 보장하여 마스크·키포인트 등 세밀한 작업에서 높은 정확도를 얻습니다.

1. **최종 결과 및 후처리**

- **RoI별 예측:** 각 RoI에서 Fast R-CNN 헤드와 mask branch를 통해 클래스 확률, 박스 회귀, 고정 크기 마스크(예: 28×28)를 예측합니다.

- **후처리 단계:**

	- 후보 영역 정리: RPN 및 RoIAlign으로 얻은 다수의 RoI 중 NMS를 적용해 겹치는 검출을 제거하고 신뢰도 높은 후보만 선택.

	- 박스 및 분류 확정: bbox 회귀 결과를 반영해 최종 경계 상자를 결정하고 분류 브랜치 확률로 최종 클래스를 할당.

	- 마스크 선택 및 보정: 한 RoI당 K개의 마스크 중 분류 결과에 해당하는 클래스의 마스크를 선택하고, 예측된 경계 상자 크기에 맞게 2D 리사이즈 및 이진화하여 최종 마스크를 생성.

- **최종 출력:** 각 객체 별 클래스 레이블, 신뢰도 점수(Confidence Score), 경계 상자, 분할 마스크.

[//]: # (heading_4 is not supported)

- [Fully Convolutional Networks for Semantic Segmentation (FCN, 2014) - arXiv](https://arxiv.org/pdf/1411.4038)

- [U-Net: Convolutional Networks for Biomedical Image Segmentation - arXiv](https://arxiv.org/pdf/1505.04597)

- [Mask R-CNN - arXiv](https://arxiv.org/pdf/1703.06870)

- [Image Segmentation: The Deep Learning Approach](https://indiaai.gov.in/article/image-segmentation-the-deep-learning-approach)

- [A Guide to Semantic Segmentation - Encord](https://encord.com/blog/guide-to-semantic-segmentation/)

- [What is Transposed Convolutional Layer - Towards Data Science](https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11/)

- [Bilinear Interpolation - Wikipedia](https://en.wikipedia.org/wiki/Bilinear_interpolation)

- [FCN 정리 - wikidocs](https://wikidocs.net/147359)

- [U-Net 정리 - wikidocs](https://wikidocs.net/148870)

- [Fast R-CNN 논문 리뷰: RoI Pooling Layer와 Truncated SVD - Medium](https://medium.com/@parkie0517/fast-r-cnn-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-roi-pooling-layer%EC%99%80-truncated-svd-a1147f7267be)

- [How to Use RoI Pool and RoI Align in Your Neural Networks (PyTorch) - Medium](https://medium.com/@andrewjong/how-to-use-roi-pool-and-roi-align-in-your-neural-networks-pytorch-1-0-b43e3d22d073)

- [RoI Pooling vs RoI Align - Medium](https://firiuza.medium.com/roi-pooling-vs-roi-align-65293ab741db)

