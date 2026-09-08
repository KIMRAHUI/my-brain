# Segmentation_COCO/VOC Dataset

 컴퓨터 비전 세그멘테이션(Segmentation) 및 COCO/VOC 데이터셋 구조 정리

1. PASCAL VOC2012 데이터셋 구조 및 전처리

PASCAL VOC 데이터셋은 객체 검출(Object Detection)과 의미론적 분할(Semantic Segmentation)을 위한 대표적인 벤치마크 데이터셋입니다.

데이터 구조:

Ignore Label (255):

클래스 구성:

# VOC 마스크 전처리 예시 (255 -> 0 변환)
import numpy as np

mask_np = np.array(raw_mask)
mask_np_fixed = np.where(mask_np == 255, 0, mask_np)

2. PyTorch Dataset 및 DataLoader 구현 (SegmentationDataset)

이미지와 픽셀 마스크를 동시에 다루기 위해 커스텀 Dataset을 구현하며, 특히 마스크 크기를 조절할 때는 클래스 인덱스 값이 변형되지 않도록 Nearest Interpolation을 적용해야 합니다.

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

3. 주요 데이터 타입(Data Types) 요약

딥러닝 파이프라인에서 자주 사용되는 데이터 타입의 특징과 용도입니다.

 COCO Dataset 구조 및 API 활용

COCO 데이터셋은 대규모 객체 검출 및 인스턴스 세그멘테이션을 위한 데이터셋으로, 모든 메타데이터와 어노테이션이 단일 JSON 파일(instances_*.json)에 구조화되어 있습니다.

주요 JSON 키 구성:

PyCOCOTools 활용:

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

 OpenCV를 활용한 Polygon 및 Mask 시각화

Polygon 좌표를 활용해 이미지 위에 객체 영역을 외곽선(polylines)으로 그리거나 내부를 채우는(fillPoly) 방법입니다.

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

6. 핵심 복습 포인트 (Key Takeaways)

마스크 리사이즈 시 주의점: 이미지 리사이즈에는 양선형(Bilinear) 보간법을 쓰지만, 정수형 클래스 라벨이 담긴 마스크에는 반드시 Nearest Interpolation을 적용하여 값의 변형을 막아야 합니다.

COCO와 VOC의 차이: VOC는 개별 XML 파일로 어노테이션을 관리하는 반면, COCO는 거대한 단일 JSON 파일 내에 이미지, 카테고리, 그리고 다각형(Polygon) 좌표 기반의 segmentation 정보를 통합 관리합니다.

마스크 처리 방식: 다각형(Polygon) 형태의 어노테이션은 cv2.fillPoly나 coco.annToMask()를 통해 2차원 이진 마스크(Binary Mask)로 변환한 뒤 세그멘테이션 모델 학습이나 시각화에 활용됩니다.

