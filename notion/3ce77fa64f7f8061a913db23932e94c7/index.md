---
notion-url: https://app.notion.com/p/Computer-Vision-Full-Arch-3ce77fa64f7f8061a913db23932e94c7
title: Computer Vision Full Arch
date: '2026-09-01 01:35:00.000'
from_notion: https://app.notion.com/p/Computer-Vision-Full-Arch-3ce77fa64f7f8061a913db23932e94c7
author: rahui kim
last_edited_time: '2026-09-01 01:35:00.000'
---
- **주요 개념:** 5개의 합성곱 층과 3개의 완전 연결 층 구조로 딥러닝 부흥을 이끈 모델입니다. ReLU 활성화 함수로 기울기 소실을 완화하고 Dropout과 데이터 증강을 적용했습니다.

- **주요 코드:**


```python
import torch.nn as nn
alexnet = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
)
```

- **단점 및 면접 꼬리물기 방어:** 마지막 완전 연결(FC) 층이 전체 파라미터의 대부분을 차지해 메모리 병목이 극심합니다.

- **대응 및 대체 방안:** 글로벌 평균 풀링(GAP)을 도입하거나 현대적인 경량 아키텍처(MobileNet 등)로 대체합니다.

- **주요 개념:** 3x3 크기의 작은 합성곱 필터만 중첩(16~19층)하여 수용장을 넓히고 비선형 표현력을 극대화했습니다.

- **주요 코드:**


```python
vgg16_block = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2)
)
```

- **단점 및 면접 꼬리물기 방어:** 파라미터 수가 과도하게 많아 가중치 파일 용량이 크고 추론 속도가 느립니다.

- **대응 및 대체 방안:** 전이학습 백본으로는 여전히 유용하나, 실무 배포 시에는 모델 프루닝(Pruning)이나 가중치 양자화(Quantization), 혹은 ResNet/EfficientNet 계열로 전환합니다.

- **주요 개념:** 인셉션 모듈을 통해 다양한 커널 사이즈(1x1, 3x3, 5x5)를 병렬 수행하고, 1x1 합성곱으로 채널을 축소(Bottleneck)해 연산량을 절감했습니다.

- **주요 코드:**


```python
class InceptionModule(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_red, n3x3, n5x5_red, n5x5):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, n1x1, 1), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3_red, 1), nn.ReLU(),
            nn.Conv2d(n3x3_red, n3x3, 3, padding=1), nn.ReLU()
        )
```

- **단점 및 면접 꼬리물기 방어:** 멀티 브랜치 구조가 복잡하여 구현 및 디버깅 난이도가 높고 메모리 효율이 비선형적입니다.

- **대응 및 대체 방안:** 단순하면서도 성능이 검증된 ResNet이나 구조가 체계화된 후속 Inception 버전(v3/v4)을 채택합니다.

- **주요 개념:** 스킵 연결(Skip Connection / $F(x) + x$)을 도입해 층이 깊어져도 기울기가 손실되지 않고 역전파되도록 설계했습니다.

- **주요 코드:**


```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        identity = x
        out = nn.ReLU()(self.conv1(x))
        out = self.conv2(out)
        return nn.ReLU()(out + identity)
```

- **단점 및 면접 꼬리물기 방어:** 블록 간 채널 수나 이미지 크기(Stride)가 바뀔 때 덧셈 연산 시 텐서 크기 불일치 에러가 발생합니다.

- **대응 및 대체 방안:** 차원이 달라지는 지점에는 1x1 합성곱을 적용하는 프로젝션 숏컷(Projection Shortcut)을 반드시 구현해 채널과 해상도를 맞춥니다.

- **주요 개념:** 분류 CNN의 FC 층을 1x1 합성곱으로 대체하고 전치 합성곱으로 픽셀 단위 예측을 복원하는 시맨틱 세그멘테이션의 시초입니다.

- **주요 코드:**


```python
fcn_head = nn.Sequential(
    nn.Conv2d(512, num_classes, kernel_size=1),
    nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=32)
)
```

- **단점 및 면접 꼬리물기 방어:** 반복적 풀링으로 인해 공간 정보가 손실되어 객체의 경계선(Boundary)이 뭉개집니다.

- **대응 및 대체 방안:** 인코더의 고해상도 피처를 디코더와 결합하는 U-Net 구조나 확장 합성곱(Atrous Convolution) 기반의 DeepLab 모델을 사용합니다.

- **주요 개념:** 대칭적인 인코더-디코더 구조에 스킵 연결(Concatenation)을 결합해 미세한 공간 정보를 디코더로 직접 전달합니다.

- **주요 코드:**


```python
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(out_ch * 2, out_ch, 3, padding=1), nn.ReLU())
    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))
```

- **단점 및 면접 꼬리물기 방어:** 입력 이미지 크기가 2의 거듭제곱 배수가 아니면 홀/짝수 차이로 인해 `torch.cat` 시 크기 불일치 에러가 빈발합니다.

- **대응 및 대체 방안:** 전처리 단계에서 이미지 크기를 고정(예: 256x256, 512x512)하거나 패딩(Padding) 로직을 엄격하게 설계합니다.

- **주요 개념:** 외부 영역 제안 알고리즘을 제거하고 신경망 내부의 RPN(Region Proposal Network)이 후보 영역을 직접 추출하는 Two-Stage 객체 탐지 모델입니다.

- **주요 코드:**


```python
import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=True)
```

- **단점 및 면접 꼬리물기 방어:** Two-Stage 구조 특성상 객체 제안 후 분류를 순차적으로 수행해 추론 속도가 느립니다(실시간 서비스 부적합).

- **대응 및 대체 방안:** 실시간성이 요구되는 경우 YOLO 계열(YOLOv8 등)의 One-Stage 모델로 대체합니다.

- **주요 개념:** Faster R-CNN에 마스크 예측 분기를 추가하고, 정수 반올림 오차를 해결하기 위해 **RoIAlign**을 도입한 인스턴스 세그멘테이션 모델입니다.

- **주요 코드:**


```python
import torchvision
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=True)
```

- **단점 및 면접 꼬리물기 방어:** 바운딩 박스와 픽셀 마스크 손실을 동시에 최적화해야 하므로 메모리 사용량이 극단적으로 높고 학습 속도가 매우 느립니다.

- **대응 및 대체 방안:** 경량 인스턴스 세그멘테이션 모델(YOLACT 또는 YOLOv8-seg)을 활용해 트레이드오프를 조정합니다.

- **주요 개념:** 80개 클래스, 대규모 이미지, 정교한 세그멘테이션 마스크와 키포인트를 제공하는 컴퓨터 비전 표준 벤치마크입니다.

- **단점 및 면접 꼬리물기 방어:** 객체 크기 편차가 심하고 다중 클래스 불균형 문제가 내재되어 있습니다.

- **대응 및 대체 방안:** 평가 시 소형 객체 성능 지표인 $AP_S$를 별도로 모니터링하고, Focal Loss나 Mosaic 데이터 증강 기법을 적용합니다.

- **주요 개념:** CNN의 판단 근거를 히트맵으로 시각화합니다. CAM은 GAP 제약이 있으나, **Grad-CAM**은 역방향 기울기를 가중치로 써서 모든 아키텍처에 범용 적용 가능합니다.

- **주요 코드 (Grad-CAM 핵심):**


```python
score = model(input_tensor)[:, class_idx]
score.backward()
gradients = model.get_activations_gradient()
pooled_gradients = torch.mean(gradients, dim=[1, 2])
activation = model.get_activations(input_tensor)
for i in range(activation.shape[1]):
    activation[:, i, :, :] *= pooled_gradients[i]
```

- **단점 및 면접 꼬리물기 방어:** 마지막 합성곱 층의 저해상도 특성 때문에 생성된 히트맵이 거칠(Coarse)어 픽셀 단위 국소화가 정밀하지 못합니다.

- **대응 및 대체 방안:** Score-CAM이나 HiResCAM 등 기울기 없이 점수 변화 기반으로 가중치를 산출하는 방식을 도입합니다.

 | **영역** | **대표 모델** | **공통점** | **차이점 및 핵심 구분 포인트** | 
 | ---- | ---- | ---- | ---- | 
 | **Classification** | AlexNet ~ ResNet | • CNN 백본 기반

• 계층적 피처 추출 | • 층의 깊이 확장 방식 및 기울기 소실 극복 방식(ReLU, Skip Connection)의 진화 | 
 | **Object Detection** | Faster R-CNN | • 분류 백본 공유

• 피처 맵 재사용 | • RPN과 RoI Pooling/Align을 통해 객체의 위치(BBox)와 클래스를 동시 예측 | 
 | **Segmentation** | FCN, U-Net, Mask R-CNN | • 픽셀 단위 매핑

• 업샘플링 구조 활용 | • FCN/U-Net은 시맨틱(배경/객체 통합), Mask R-CNN은 인스턴스(개별 객체 분리) 단위 처리 | 
 | **Explainability** | CAM, Grad-CAM | • 블랙박스 모델 해석

• Conv 층 가중치 활용 | • 모델 구조 수정 여부 및 기울기(Gradient) 사용 유무에 따른 범용성 차이 | 

<br/>

