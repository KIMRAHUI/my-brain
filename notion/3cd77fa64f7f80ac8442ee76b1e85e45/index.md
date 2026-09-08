---
notion-url: https://app.notion.com/p/Segmentation_COCO-VOC-Dataset-3cd77fa64f7f80ac8442ee76b1e85e45
title: Segmentation_COCO/VOC Dataset
date: '2026-08-31 07:41:00.000'
from_notion: https://app.notion.com/p/Segmentation_COCO-VOC-Dataset-3cd77fa64f7f80ac8442ee76b1e85e45
author: rahui kim
last_edited_time: '2026-08-31 07:43:00.000'
---
#  컴퓨터 비전 세그멘테이션(Segmentation) 및 COCO/VOC 데이터셋 구조 정리

## 1. PASCAL VOC2012 데이터셋 구조 및 전처리

- **데이터 구조**:

	- `JPEGImages`: 원본 이미지 폴더

	- `Annotations`: XML 형식의 객체 메타데이터(바운딩 박스, 클래스명 등) 폴더

	- `SegmentationClass`: 픽셀 단위 마스크 이미지(PNG) 폴더

- **Ignore Label (255)**:

	- VOC 마스크에서 객체의 경계선이나 모호한 영역은 `255` 값(ignore label)으로 지정되어 있습니다. 모델 학습 시 오류를 방지하기 위해 이를 통상 `0`(배경)으로 변환하여 처리합니다.

- **클래스 구성**:

	- 배경(`background`)인 `0번`을 포함하여 총 21개의 클래스로 구성됩니다.


```python
# VOC 마스크 전처리 예시 (255 -> 0 변환)
import numpy as np

mask_np = np.array(raw_mask)
mask_np_fixed = np.where(mask_np == 255, 0, mask_np)
```

## 2. PyTorch Dataset 및 DataLoader 구현 (SegmentationDataset)


```python
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class SegmentationDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        image_set = 'train' if train else 'val'
        self.voc = datasets.VOCSegmentation(root=root, year='2012', image_set=image_set, download=download)
        self.categories = {str(i): {"class": f"class_{i}", "color": [i * 10 % 256, i * 20 % 256, i * 30 % 256]} for i in range(1, 22)}
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, mask = self.voc[index]
        mask = np.array(mask)
        mask = np.where(mask == 255, 0, mask)  # Ignore label 처리
        target = torch.LongTensor(mask).unsqueeze(0)  # (1, H, W) 형태 변환

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.voc)

# Transform 정의
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((224, 224))
])

target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
])
```

## 3. 주요 데이터 타입(Data Types) 요약

 | **데이터 타입** | **크기** | **표현 범위 / 설명** | **주요 용도** | 
 | ---- | ---- | ---- | ---- | 
 | `**uint8**` | 8비트 (1바이트) | `0 ~ 255` 부호 없는 정수 | 이미지 픽셀 데이터(RGB/Gray), 마스크 | 
 | `**int64**` | 64비트 (8바이트) | 정수형 (`LongTensor`) | PyTorch 기본 정수 텐서, 클래스 인덱스 | 
 | `**float32**` | 32비트 (4바이트) | 단정밀도 부동소수점 | 모델 가중치(Weight), 정규화된 이미지 입력 (`0.0~1.0`) | 
 | `**bool**` | 8비트 (1바이트) | `True` 또는 `False` | 마스크 영역 지정, 조건부 필터링 | 

##  COCO Dataset 구조 및 API 활용

- **주요 JSON 키 구성**:

	- `images`: 이미지 파일명, 고유 ID, 해상도 등 메타데이터

	- `annotations`: 바운딩 박스, 다각형(Polygon) 형태의 Segmentation 정보

	- `categories`: 객체 카테고리 정보

- **PyCOCOTools 활용**:

	- `COCO(annFile)`를 통해 JSON 파일을 로드한 뒤, `getCatIds()`, `getImgIds()`, `loadAnns()` 등의 API로 손쉽게 데이터를 탐색할 수 있습니다.


```python
from pycocotools.coco import COCO

annFile = '/content/data/annotations/instances_val2017.json'
coco = COCO(annFile)

# 특정 카테고리 및 이미지 ID 조회 예시
catIds = coco.getCatIds(catNms=['person', 'dog'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[0])[0]

# Polygon 정보를 이진 마스크(Binary Mask)로 변환
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
mask = coco.annToMask(anns[0])
```

##  OpenCV를 활용한 Polygon 및 Mask 시각화


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 다각형 외곽선 시각화
draw_img = image_array.copy()
polygon_xy = np.array(polygon_xy, np.int32)
draw_img = cv2.polylines(draw_img, [polygon_xy], True, (0, 255, 0))

# 다각형 내부 채우기 시각화
draw_filled = image_array.copy()
draw_filled = cv2.fillPoly(draw_filled, [polygon_xy], (0, 255, 0))

# 마스크 블렌딩(Alpha Blending) 함수 예시
def apply_mask_01(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                   image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                   image[:, :, c])
    return image
```

## 6. 핵심 복습 포인트 (Key Takeaways)

1. **마스크 리사이즈 시 주의점**: 이미지 리사이즈에는 양선형(Bilinear) 보간법을 쓰지만, 정수형 클래스 라벨이 담긴 마스크에는 반드시 **Nearest Interpolation**을 적용하여 값의 변형을 막아야 합니다.

1. **COCO와 VOC의 차이**: VOC는 개별 XML 파일로 어노테이션을 관리하는 반면, COCO는 거대한 단일 JSON 파일 내에 이미지, 카테고리, 그리고 다각형(Polygon) 좌표 기반의 `segmentation` 정보를 통합 관리합니다.

1. **마스크 처리 방식**: 다각형(Polygon) 형태의 어노테이션은 `cv2.fillPoly`나 `coco.annToMask()`를 통해 2차원 이진 마스크(Binary Mask)로 변환한 뒤 세그멘테이션 모델 학습이나 시각화에 활용됩니다.

