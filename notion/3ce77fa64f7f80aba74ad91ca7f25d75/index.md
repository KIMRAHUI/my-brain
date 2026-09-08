---
notion-url: https://app.notion.com/p/FCN-8s-3ce77fa64f7f80aba74ad91ca7f25d75
title: FCN-8s
date: '2026-09-01 00:04:00.000'
from_notion: https://app.notion.com/p/FCN-8s-3ce77fa64f7f80aba74ad91ca7f25d75
author: rahui kim
last_edited_time: '2026-09-01 00:05:00.000'
---
# FCN-8s 시맨틱 세그멘테이션 코드 정리

---

## 1. 데이터셋 로드 및 전처리

### 1-1. Import


```python
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
```

### 1-2. `SegmentationDataset` 클래스

- `train=True/False`로 train/val `image_set` 선택

- `classes.json`이 있으면 클래스 정보(이름, 색상)를 로드, 없으면 클래스 1~21에 대한 기본값 생성

- 마스크의 `255`(ignore label)는 `0`(배경)으로 치환

- `target_transform`을 통해 마스크 크기 조정 (nearest interpolation으로 클래스 값 보존)


```python
class SegmentationDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """
        VOCSegmentation 데이터셋을 로드하고, 이미지와 pixel-level segmentation mask를 반환하는 Dataset 클래스입니다.

        Args:
            root (str): 데이터셋이 저장될 루트 경로
            train (bool): True이면 training 데이터를, False이면 validation 데이터를 사용
            transform: 입력 이미지에 적용할 변환(예: 크기 조정, tensor 변환 등)
            target_transform: segmentation mask에 적용할 변환(보통 크기 조정 등, 클래스 값 보존을 위해 nearest interpolation 사용)
            download (bool): 데이터가 없으면 자동으로 다운로드할지 여부
        """
        image_set = 'train' if train else 'val'

        self.voc = datasets.VOCSegmentation(
            root=root,
            year='2012',
            image_set=image_set,
            download=download
        )

        classes_json_path = os.path.join(root, "VOCdevkit", "VOC2012", "classes.json")
        if os.path.exists(classes_json_path):
            with open(classes_json_path, "r") as file:
                self.categories = json.load(file)
        else:
            # 1번부터 21번까지의 클래스(보통 background가 0이고 1~20은 객체 클래스)를 만듭니다.
            self.categories = {str(i): {"class": f"class_{i}",
                                        "color": [i * 10 % 256, i * 20 % 256, i * 30 % 256]}
                               for i in range(1, 22)}
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, mask = self.voc[index]

        mask = np.array(mask)
        # VOC 마스크에서는 255가 'ignore label'로 사용됩니다.
        mask = np.where(mask == 255, 0, mask)

        target = torch.LongTensor(mask).unsqueeze(0)  # (1, H, W)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.voc)
```

### 1-3. Transform 정의


```python
# 이미지: PIL → Tensor → float → (224, 224) 리사이즈
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((224, 224))
])

# 마스크: nearest interpolation으로 클래스 값 보존하며 (224, 224) 리사이즈
target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
])
```

### 1-4. Dataset / DataLoader 생성


```python
root = "./datasets"

train_dataset = SegmentationDataset(root, train=True, transform=transform,
                                     target_transform=target_transform, download=True)
test_dataset = SegmentationDataset(root, train=False, transform=transform,
                                    target_transform=target_transform, download=False)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, drop_last=True)
```

---

## 2. 시각화 함수 `draw_mask`


```python
def draw_mask(images, masks, outputs=None, plot_size=4):
    """
    학습 중 또는 결과 확인을 위해, 이미지와 segmentation mask를 시각화합니다.

    Args:
        images (Tensor): 배치 이미지 (shape: [B, C, H, W])
        masks (Tensor): 배치 마스크 (shape: [B, 1, H, W])
        outputs (Tensor, optional): 예측 마스크 (있을 경우)
        plot_size (int): 시각화할 배치의 수
    """
    def color_mask(image, target):
        m = target.squeeze().numpy().astype(np.uint8)
        cm = np.zeros_like(image, dtype=np.uint8)
        for i in range(1, 21):
            cm[m == i] = train_dataset.categories.get(str(i), {"color": [0, 0, 0]})["color"]
        classes = [train_dataset.categories.get(str(idx), {"class": f"class_{idx}"})["class"]
                   for idx in np.unique(m)]
        return cm, classes

    col = 3 if outputs is not None else 2
    figsize = 8 if outputs is not None else 12
    fig, ax = plt.subplots(plot_size, col, figsize=(14, figsize), constrained_layout=True)

    for batch in range(plot_size):
        im = images[batch].numpy().transpose(1, 2, 0)
        ax[batch][0].imshow(im)
        ax[batch][0].set_title("Original Image")
        ax[batch][0].axis("off")

        cm, classes = color_mask(im, masks[batch])
        ax[batch][1].set_title("GT Mask\n" + " / ".join(classes))
        ax[batch][1].imshow(cm)
        ax[batch][1].axis("off")

        if outputs is not None:
            cm, classes = color_mask(im, outputs[batch])
            ax[batch][2].set_title("Pred Mask\n" + " / ".join(classes))
            ax[batch][2].imshow(cm)
            ax[batch][2].axis("off")
    plt.show()
```


```python
images, masks = next(iter(train_dataloader))
draw_mask(images, masks, plot_size=4)
```

---

## 3. FCN-8s 모델 정의 (VGG16 백본)

### 3-1. 아키텍처 요약

 | 구간 | VGG16 layer 범위 | 출력 해상도 | 채널 | 
 | ---- | ---- | ---- | ---- | 
 | `pool3` | `features[:17]` | 1/8 | 256 | 
 | `pool4` | `features[17:24]` | 1/16 | 512 | 
 | `pool5` | `features[24:]` | 1/32 | 512 | 


```plain text
pool5 → score_fr (1x1 conv) → upscore2 (x2)  ──┐
pool4 → score_pool4 (1x1 conv) ────────────────┼─ fuse_pool4 → upscore_pool4 (x2) ──┐
                                                                                      │
pool3 → score_pool3 (1x1 conv) ──────────────────────────────────────────────────── fuse_pool3 → upscore8 (x8) → 원본 해상도
```

- `upscore2`: `(7-1)*2 + 4 - 2 = 14`

- `upscore_pool4`: `(14-1)*2 + 4 - 2 = 28`

- `upscore8`: `(28-1)*8 + 16 - 8 = 224`

### 3-2. 코드


```python
import torch.nn as nn
import torchvision.models as models

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())

        # VGG16에서 중간 feature map 추출
        self.pool3 = nn.Sequential(*features[:17])   # 1/8 해상도
        self.pool4 = nn.Sequential(*features[17:24]) # 1/16 해상도
        self.pool5 = nn.Sequential(*features[24:])   # 1/32 해상도

        # 1x1 convolution으로 채널 수를 num_classes로 축소
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_fr    = nn.Conv2d(512, num_classes, kernel_size=1)

        # Transposed convolution (upsampling) layers
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4,
                                           stride=2, padding=1, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes,
                                                kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes,
                                           kernel_size=16, stride=8, padding=4, bias=False)

    def forward(self, x):
        input_size = x.size()  # [B, C, H, W]
        pool3 = self.pool3(x)        # 1/8,  채널 256
        pool4 = self.pool4(pool3)    # 1/16, 채널 512
        pool5 = self.pool5(pool4)    # 1/32, 채널 512

        score_fr = self.score_fr(pool5)         # [B, num_classes, H/32, W/32]
        upscore2 = self.upscore2(score_fr)      # → [B, num_classes, H/16, W/16]

        score_pool4 = self.score_pool4(pool4)
        fuse_pool4 = upscore2 + score_pool4     # skip connection

        upscore_pool4 = self.upscore_pool4(fuse_pool4)  # → [B, num_classes, H/8, W/8]

        score_pool3 = self.score_pool3(pool3)
        fuse_pool3 = upscore_pool4 + score_pool3        # skip connection

        upscore8 = self.upscore8(fuse_pool3)    # → [B, num_classes, H, W]

        # 입력 크기와 정확히 맞지 않을 경우 crop
        out = upscore8[:, :, :input_size[2], :input_size[3]]
        return out
```

---

## 4. 학습 (Training)

### 4-1. 설정


```python
import torch.optim as optim
from tqdm import tqdm

num_classes = 21
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FCN8s(num_classes=num_classes).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()
```

### 4-2. 학습 루프


```python
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        targets = targets.to(device)  # [B, H, W]

        # 타겟의 차원 확인 후 squeeze
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        optimizer.zero_grad()
        outputs = model(images)  # [B, num_classes, H, W]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
```

### 4-3. 모델 저장 / 로드


```python
torch.save(model.state_dict(), 'fcn-8.pth')

model_loaded = FCN8s(num_classes=num_classes).to(device)
model_loaded.load_state_dict(torch.load('fcn-8.pth'))
```

---

## 5. 평가 (Evaluation)

### 5-1. 결과 시각화


```python
with torch.no_grad():
    model.eval()
    images, masks = next(iter(test_dataloader))
    outputs = model(images.to(device))
    outputs = outputs.argmax(dim=1).to("cpu")
    draw_mask(images, masks, outputs, 2)
```

### 5-2. mIoU 계산


```python
from collections import defaultdict

def calculate_iou(targets, outputs, ious, class_count, num_classes=21):
    for i in range(num_classes):
        intersection = np.float32(np.sum((outputs == targets) * (targets == i)))
        union = np.sum(targets == i) + np.sum(outputs == i) - intersection
        if union > 0:
            ious[i] += intersection / union
            class_count[i] += 1
    return ious, class_count

ious = np.zeros(num_classes)
class_count = defaultdict(int)

with torch.no_grad():
    model.eval()
    for images, targets in tqdm(test_dataloader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images).detach().to("cpu").numpy()  # [B, num_classes, H, W]
        targets = targets.detach().to("cpu").numpy()        # [B, H, W]
        outputs = outputs.argmax(axis=1)
        ious, class_count = calculate_iou(targets, outputs, ious, class_count, num_classes)

miou = 0.0
for idx in range(1, num_classes):
    miou += ious[idx] / class_count[idx]
miou /= (num_classes - 1)
print(f"mIoU 계산 결과 : {miou}")
```

> ⚠️ 배경 클래스(index 0)를 제외하고 1~20 클래스만 평균내는 방식입니다.

### 5-3. 저장된 모델로 재평가


```python
with torch.no_grad():
    model_loaded.eval()
    images, masks = next(iter(test_dataloader))
    outputs = model_loaded(images.to(device))
    outputs = outputs.argmax(dim=1).to("cpu")
    draw_mask(images, masks, outputs, 2)
```

---

## 6. 부록: PennFudan Pedestrian 데이터셋 다운로드


```bash
!wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -P data
!cd data && unzip PennFudanPed.zip
```

---

## 전체 파이프라인 요약

1. **데이터셋**: `VOCSegmentation` 래핑 → 224×224 리사이즈, ignore label(255) → 0 처리

1. **모델**: VGG16 백본 기반 FCN-8s, `pool3/4/5` skip connection으로 세밀한 디테일 보존

1. **학습**: SGD + CrossEntropyLoss, 20 epoch

1. **평가**: 시각화(`draw_mask`) + 클래스별 mIoU

