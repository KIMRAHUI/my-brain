---
notion-url: https://app.notion.com/p/Football-Semantic-Segmentation_UNET-3d077fa64f7f805ebba4e3ea668acc55
title: Football Semantic Segmentation_UNET
date: '2026-09-03 05:13:00.000'
from_notion: https://app.notion.com/p/Football-Semantic-Segmentation_UNET-3d077fa64f7f805ebba4e3ea668acc55
author: rahui kim
last_edited_time: '2026-09-03 05:35:00.000'
---
- **목표**: Football Semantic Segmentation Dataset을 활용하여 축구장 환경에서 객체 및 영역을 탐지하는 시맨틱 세그멘테이션 모델(U-Net)을 구현하고, 데이터 전처리부터 모델 학습 및 평가까지의 실무 파이프라인을 구축합니다.

- **주요 기술**: PyTorch, Torchvision, OpenCV, Matplotlib, KaggleHub, U-Net 구조 설계, 픽셀 단위 세그멘테이션 평가

- **보완 및 개선 반영 사항**:

	- 데이터셋 로드 전 파일 누락 및 해상도 정합성 검증 로직 도입

	- 원본 이미지와 마스크(`fuse`, `save`)의 정확한 페어링 및 예외 처리 강화

	- 텐서 폼 표준화(`.permute(2, 0, 1)`) 및 디바이스(GPU) 최적화 반영

	- 체크포인트 분리 저장 및 시각화 검증 단계 체계화


```python
import os
import random
import collections
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as F_vision
from torchvision.tv_tensors import Image, Mask
from tqdm import tqdm
import kagglehub

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

SEED = 42
set_seed(SEED)

from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

- **라희님 코드 계승 부분**: 최신 딥러닝 파이프라인 구성에 필요한 `torchvision.transforms.v2` 및 `tv_tensors` 등 선진 라이브러리 구성을 빠짐없이 포함했습니다. 구글 코랩 환경에서 구글 드라이브 연동을 설정했습니다.

- **강사님 코드 계승 부분**: 실험의 재현성을 보장하기 위해 난수 생성 시드 고정 함수(`set_seed`)를 최상단에 배치하여 실행할 때마다 결과가 흔들리지 않도록 했습니다.

- **실무적 보완 포인트**: 딥러닝 실험에서 무작위성에 의한 성능 편차를 막기 위한 시드 고정은 재현 가능한 실험의 핵심 기본기입니다.


```python
# Kaggle Hub를 통한 데이터셋 다운로드
path = kagglehub.dataset_download("sadhliroomyprime/football-semantic-segmentation")
print("Path to dataset files:", path)

image_folder = os.path.join(path, "images")
file_list = os.listdir(image_folder)
print(f"Total file count: {len(file_list)}")

# 원본이미지와 fuse이미지, save이미지 분류
original_images = sorted([f for f in file_list if f.endswith(".jpg")])
fuse_images = sorted([f for f in file_list if "fuse" in f])
save_images = sorted([f for f in file_list if "save" in f])

print(f"Original: {len(original_images)}, Fuse: {len(fuse_images)}, Save: {len(save_images)}")
```

- **라희님 코드 계승 부분**: `kagglehub` API를 활용하여 수동 압축 해제 없이 코랩 환경에 즉시 데이터를 연동하고, 리스트 컴프리헨션으로 파일 카테고리를 정확히 분리했습니다.


```python
def unique_sizes(image_files, image_folder):
    unique_sizes_set = set()

    for image in image_files:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        if img is None:
            continue

        height, width, channels = img.shape
        img_size = (height, width, channels)

        if img_size in unique_sizes_set:
            continue

        print(f"Image Size (H, W, C): {height}, {width}, {channels}")
        unique_sizes_set.add(img_size)

print("Unique sizes of Original Image")
unique_sizes(original_images, image_folder)
print("Unique sizes of Save Image")
unique_sizes(save_images, image_folder)
```

- **라희님 코드 계승 부분**: 훈련 전 `unique_sizes` 함수를 작성하여 이미지 해상도나 채널 수의 이상 유무, 결측 이미지(`None`)를 사전 검증하는 훌륭한 방어적 프로그래밍 기법을 적용했습니다.


```python
# 이미지 페어링
image_pairs = []
for original_image in original_images:
    base_name = os.path.basename(original_image)

    fuse_image = next((f for f in fuse_images if base_name in f), None)
    save_image = next((f for f in save_images if base_name in f), None)

    if fuse_image and save_image:
        original_path = os.path.join(image_folder, original_image)
        fuse_path = os.path.join(image_folder, fuse_image)
        save_path = os.path.join(image_folder, save_image)

        orig = cv2.imread(original_path)
        fuse = cv2.imread(fuse_path)
        save = cv2.imread(save_path)

        if orig is not None and fuse is not None and save is not None:
            image_pairs.append((orig, fuse, save))

print(f"{len(image_pairs)} pairs successfully matched.")

# 샘플 시각화
num_samples = 3
sample_image_pairs = random.sample(image_pairs, num_samples)

fig, axes = plt.subplots(num_samples, 3, figsize=(11, 3 * num_samples))
for i, (orig, fuse, save) in enumerate(sample_image_pairs):
    axes[i, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(cv2.cvtColor(fuse, cv2.COLOR_BGR2RGB))
    axes[i, 1].set_title("Fuse")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(cv2.cvtColor(save, cv2.COLOR_BGR2RGB))
    axes[i, 2].set_title("Save")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
```

- **라희님 코드 계승 부분**: `next()` 함수와 제너레이터 표현식을 활용해 파일 이름 기반으로 원본과 마스크 파일들을 매칭 에러 없이 안전하게 묶어냈고, OpenCV의 BGR 색상 체계를 Matplotlib 표준인 RGB로 변환(`cv2.cvtColor`)하여 시각화했습니다.

- **실무적 보완 포인트**: 시각화 샘플 추출 시 난수 시드를 고정해 두면 리포트나 제출물 작성 시 결과 이미지가 달라지지 않아 일관성을 유지할 수 있습니다.


```python
class FootballSegmentationDataset(Dataset):
    def __init__(self, image_pairs):
        self.image_pairs = image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        orig, fuse, save = self.image_pairs[idx]

        # 이미지 전처리 (RGB 변환, 채널 순서 변경, 스케일링)
        image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # 마스크 전처리 (그레이스케일 변환 및 이진화)
        mask = cv2.cvtColor(save, cv2.COLOR_BGR2GRAY)
        mask = (mask > 127).astype(np.int64)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

# 데이터셋 분할 및 로더 생성
dataset = FootballSegmentationDataset(image_pairs)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
```

- **실무적 보완 반영**: 텐서 표준 형식인 `(C, H, W)` 준수를 위한 `.permute(2, 0, 1)` 처리와 0~1 스케일링을 `Dataset` 내부에 명확히 캡슐화하여 딥러닝 학습 파이프라인과 완벽히 연동되도록 보완했습니다


```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(SimpleUNet, self).__init__()
        self.dconv_down1 = DoubleConv(in_channels, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.maxpool = nn.MaxPool2d(2)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)

        x = self.upsample(conv2)
        if x.shape[2:] != conv1.shape[2:]:
            x = F.interpolate(x, size=conv1.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out

model = SimpleUNet(in_channels=3, out_channels=2).to(device)
```

- **실무적 보완 반영**: 세그멘테이션 표준 네트워크인 U-Net의 인코더, 디코더, 스킵 커넥션 구조를 적용하고, 업샘플링 시 텐서 크기 불일치 오류를 방지하는 크기 보정 로직을 포함시켰습니다


```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3
print("--- 모델 학습 루프 시작 ---")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
```

- **실무적 보완 반영**: 픽셀 분류에 적합한 `CrossEntropyLoss`와 `Adam` 옵티마이저를 연결하고, `tqdm`을 활용해 학습 진행 상황을 가시화했습니다


```python
model.eval()
print("--- 검증 결과 시각화 ---")
with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(images[0].cpu().permute(1, 2, 0).numpy())
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(masks[0].cpu().numpy(), cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        axes[2].imshow(preds[0], cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()
        break
```

- **실무적 보완 반영**: 학습이 완료된 모델의 일반화 성능을 정성적으로 평가하기 위해 원본 이미지, 정답 마스크, 예측 마스크를 1행 3열로 나란히 출력하는 검증 시각화 코드를 완성했습니다

- **인사이트**:

	- 철저한 데이터 무결성 검증과 안전한 페어링 로직이 파이프라인의 에러를 최소화합니다

- **재사용 체크포인트**:

	- [ ] 데이터셋 경로 및 클래스 개수만 수정하여 다른 세그멘테이션 과제에도 동일하게 활용할 수 있습니다

<br/>

