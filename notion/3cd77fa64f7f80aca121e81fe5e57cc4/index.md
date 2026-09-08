---
notion-url: https://app.notion.com/p/VOC2012DATA-XMLANNOTATION-3cd77fa64f7f80aca121e81fe5e57cc4
title: VOC2012DATA/ XMLANNOTATION
date: '2026-08-31 02:41:00.000'
from_notion: https://app.notion.com/p/VOC2012DATA-XMLANNOTATION-3cd77fa64f7f80aca121e81fe5e57cc4
author: rahui kim
last_edited_time: '2026-08-31 02:44:00.000'
---
# VOC2012 데이터 및 XML 어노테이션 다루기


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


```python
from torchvision.datasets import voc
voc.DATASET_YEAR_DICT["2012"]["url"] = "http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar"
```


```python
# VOC 데이터셋의 첫 샘플을 불러와 이미지와 마스크의 형태를 확인합니다.
voc_dataset = datasets.VOCSegmentation(
    root='../datasets',      # VOC 데이터가 저장될 경로
    year='2012',
    image_set='train',
    download=True             # 처음 실행 시 자동 다운로드
)
```


```python
# 첫 번째 이미지와 마스크 불러오기
voc_dataset[0]
```


```python
# 원본 데이터를 시각화해봅니다.
raw_image, raw_mask = voc_dataset[0]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(raw_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(raw_mask)
plt.title("Original Mask")
plt.axis("off")
plt.show()
```


```python
########################################
# Step 3: 마스크 데이터의 전처리 이해하기
########################################
# VOC 마스크는 PIL.Image 형태이므로 numpy 배열로 변환하여 픽셀값을 확인합니다.
mask_np = np.array(raw_mask)
print("마스크의 고유 값:", np.unique(mask_np))

# VOC에서는 255가 ignore label로 사용되므로, 이를 0으로 바꿔줍니다.
mask_np_fixed = np.where(mask_np == 255, 0, mask_np)
print("전처리 후 마스크의 고유 값:", np.unique(mask_np_fixed))
```

### Annotation (XML 데이터 다루기 주요 개념 및 활용)

1. **구조 파악 (**`**Root**`**와 태그):** XML 파일은 계층형 트리 구조를 가집니다. 최상위 루트 엘리먼트(`root = tree.getroot()`)를 기준으로 하위 태그들에 접근해야 합니다.

1. **필수 메타데이터 추출 (**`**filename**`**, **`**size**`**):** 객체 탐지(Object Detection)나 세그멘테이션 작업 시 이미지 파일명과 원본 이미지의 크기(`width`, `height`, `depth`) 정보를 알아야 바운딩 박스 좌표를 정확하게 매핑할 수 있습니다.

1. **객체 정보 순회 (**`**object**`** 및 **`**bndbox**`**):** 이미지 내에 여러 객체가 존재할 수 있으므로 `root.findall("object")`를 통해 반복문을 돌며 각 객체의 클래스 이름(`name`)과 바운딩 박스 좌표(`xmin`, `ymin`, `xmax`, `ymax`)를 안전하게 파싱해야 합니다.


```python
import xml.etree.ElementTree as ET

# XML 파일을 파싱합니다.
tree = ET.parse('sample_annotation.xml')
root = tree.getroot()

# 파일명 추출 (예: <filename> 태그의 내용)
filename = root.find("filename").text
print("파일명:", filename)

# 이미지 크기 정보 추출 (예: <size> 태그 내부의 <width>, <height>, <depth> 등)
size = root.find("size")
width = size.find("width").text
height = size.find("height").text
depth = size.find("depth").text
print("이미지 크기 - Width:", width, "Height:", height, "Depth:", depth)

# 모든 객체 정보 추출 (예: <object> 태그들)
for obj in root.findall("object"):
    name = obj.find("name").text
    print("객체 이름:", name)
```

> **참고 자료:** [ 공식 문서 - xml.etree.ElementTree](https://docs.python.org/3/library/xml.etree.elementtree.html)


```python
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image

# VOC 데이터셋의 기본 경로 설정 (VOCdevkit/VOC2012)
voc_root = "../datasets/VOCdevkit/VOC2012"

# Annotations 폴더 경로 지정
annotation_dir = os.path.join(voc_root, "Annotations")

# Annotations 폴더 내의 모든 XML 파일 목록 확인
annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]
print("전체 annotation 파일 수:", len(annotation_files))

# 예시로 첫 번째 annotation 파일 선택
sample_annotation_file = annotation_files[0]
sample_annotation_path = os.path.join(annotation_dir, sample_annotation_file)
print("예시 annotation 파일 경로:", sample_annotation_path)
```


```python
########################################
# Step 1: XML 파싱을 통해 annotation 데이터 로드하기
########################################
# xml.etree.ElementTree를 이용해 XML 파일을 파싱합니다.
tree = ET.parse(sample_annotation_path)
root_xml = tree.getroot()

# 파일명 출력 (이미지 파일명)
filename = root_xml.find("filename").text
print("이미지 파일명:", filename)

# 이미지 크기 정보 확인
size_elem = root_xml.find("size")
width = size_elem.find("width").text
height = size_elem.find("height").text
depth = size_elem.find("depth").text
print("이미지 크기 - Width:", width, ", Height:", height, ", Depth:", depth)
```


```python
########################################
# Step 2: 객체 정보(Annotation) 확인하기
########################################
print("\n이미지 내 객체 정보:")
for obj in root_xml.findall("object"):
    # 객체 이름(클래스) 가져오기
    obj_name = obj.find("name").text

    # 객체의 bounding box 정보 가져오기
    bndbox = obj.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    print(f"- 객체: {obj_name}, Bounding Box: ({xmin}, {ymin}, {xmax}, {ymax})")
```


```python
########################################
# Step 3: 이미지와 bounding box 시각화하기
########################################
# 이미지 파일 경로: VOCdevkit/VOC2012/JPEGImages 폴더 내에 있습니다.
image_path = os.path.join(voc_root, "JPEGImages", filename)
image = Image.open(image_path)

# 시각화를 위해 matplotlib의 기능을 사용합니다.
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title(f"Annotation example")
```


```python
# Pascal VOC 데이터셋의 segmentation 결과를 시각화하기 위해,
# 각 클래스에 대해 고유한 색상 코드를 지정합니다.
# 일반적으로 background는 0번, 객체 클래스는 1~20번으로 정의되며,
# 각 객체 클래스에 대해 클래스 번호에 따라 서로다른 RGB 색상 코드를 생성합니다.
# [i * 10 % 256, i * 20 % 256, i * 30 % 256]
# % 256 연산을 통해 각 색상 채널의 값이 0에서 255 사이로 유지됩니다
categories = {
    str(i): {
        "class": f"class_{i}",
        "color": [i * 10 % 256, i * 20 % 256, i * 30 % 256]
    } for i in range(1, 22)
}
```


```python
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

###############################################
# Step 1: VOCSegmentation 데이터셋 로드 및 클래스 정보 설정하기
###############################################
class SegmentationDataset(Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False):
        """
        VOCSegmentation 데이터셋을 로드하고, 이미지와 pixel-level segmentation mask를 반환하는 Dataset 클래스입니다.

        Args:
            root (str): 데이터셋이 저장될 루트 경로
            train (bool): True이면 training 데이터를, False이면 validation 데이터를 사용
            transform: 입력 이미지에 적용할 변환(예: 크기 조정, tensor 변환 등)
            target_transform: segmentation mask에 적용할 변환(보통 크기 조정 등, 클래스 값 보존을 위해 nearest interpolation 사용)
            download (bool): 데이터가 없으면 자동으로 다운로드할지 여부
        """
        # train이면 'train', 아니면 'val' 이미지를 선택합니다.
        image_set = 'train' if train else 'val'

        # torchvision의 VOCSegmentation 클래스를 사용하여 VOC 데이터셋을 불러옵니다.
        self.voc = datasets.VOCSegmentation(
            root=root,
            year='2012',
            image_set=image_set,
            download=download
        )

        # 일관된 시각화를 위해 클래스 별로 색상코드 지정
        self.categories = {str(i): {"class": f"class_{i}",
                                    "color": [i * 10 % 256, i * 20 % 256, i * 30 % 256]}
                           for i in range(1, 22)}
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        주어진 index에 해당하는 이미지와 segmentation mask를 반환합니다.
        """
        # VOCSegmentation 객체로부터 image와 mask를 불러옵니다.
        image, mask = self.voc[index]

        # mask는 PIL.Image 형태로 로드되므로 numpy 배열로 변환하여 각 픽셀 값을 확인할 수 있습니다.
        mask = np.array(mask)

        # VOC 마스크에서는 255가 'ignore label'로 사용됩니다.
        # 모델 학습 시 ignore label은 보통 처리하지 않으므로, 여기서는 0 (배경)으로 대체합니다.
        mask = np.where(mask == 255, 0, mask)

        # numpy 배열을 파이토치의 LongTensor로 변환합니다.
        # unsqueeze(0)은 마스크에 채널 차원을 추가하여 (1, H, W) 형태로 만듭니다.
        target = torch.LongTensor(mask).unsqueeze(0)

        # 입력 이미지에 설정된 transform(예: 크기 조정, tensor 변환 등)을 적용합니다.
        if self.transform is not None:
            image = self.transform(image)
        # 마스크에도 target_transform이 있다면 적용합니다.
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        """
        데이터셋에 포함된 샘플의 총 개수를 반환합니다.
        """
        return len(self.voc)


###############################################
# Step 2: 이미지와 마스크에 적용할 전처리(Transform) 정의하기
###############################################
# 이미지 변환:
# 1. PIL 이미지를 tensor로 변환
# 2. tensor의 데이터 타입을 float로 변경
# 3. 이미지 크기를 (224, 224)로 조정
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((224, 224))
])

# 마스크 변환:
# 마스크는 정수값(클래스 id)을 가지므로, 크기 조정 시 nearest interpolation을 사용하여 값이 보존되도록 합니다.
target_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
])
```


```python
###############################################
# Step 3: 데이터셋과 DataLoader 생성하기
###############################################
# 데이터셋 경로 (VOC 데이터셋의 루트 폴더)
root = "../datasets"

# train 데이터셋 생성 (필요시 download=True 옵션 사용)
train_dataset = SegmentationDataset(root, train=True,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=True)
# test 또는 validation 데이터셋 생성
test_dataset = SegmentationDataset(root, train=False, transform=transform, target_transform=target_transform, download=False)

# DataLoader 생성: 데이터를 배치 단위로 불러와 학습에 사용할 수 있도록 합니다.
train_dataloader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, drop_last=True)
```


```python
# 클래스 별 RGB값 딕셔너리에서 꺼내오기 (default 없음)
train_dataset.categories.get(str(1))
```


```python
train_dataset.categories.get(str(100)) # return None
```


```python
# 클래스 별 RGB값 딕셔너리에서 꺼내오기 (키가 없을 때 default 설정)
train_dataset.categories.get(str(100), {"color": [0, 0, 0]})
```


```python
###############################################
# Step 4: 시각화 함수 정의하기
###############################################
def draw_mask(images, masks, outputs=None, plot_size=4):
    """
    학습 중 또는 결과 확인을 위해, 이미지와 segmentation mask를 시각화합니다.

    Args:
        images (Tensor): 배치 이미지 (shape: [B, C, H, W])
        masks (Tensor): 배치 마스크 (shape: [B, 1, H, W])
        outputs (Tensor, optional): 예측 마스크 (있을 경우)
        plot_size (int): 시각화할 배치의 수
    """
    # 내부 함수: 마스크에 클래스별 색상을 입히는 함수
    def color_mask(image, target):
        # target: (1, H, W) 텐서를 squeeze하여 (H, W) 형태로 변경
        m = target.squeeze().numpy().astype(np.uint8)
        # image와 같은 크기를 가지는 빈 컬러 이미지를 생성 (H, W, 3)
        cm = np.zeros_like(image, dtype=np.uint8)
        # 클래스 1~20에 대해 색상 적용 (background는 0으로 남음)
        for i in range(1, 21):
            cm[m == i] = train_dataset.categories.get(str(i), {"color": [0, 0, 0]})["color"]

        # 마스크에 포함된 클래스 번호를 이용해 클래스 이름 리스트 생성
        classes = [train_dataset.categories.get(str(idx), {"class": f"class_{idx}"})["class"]
                   for idx in np.unique(m)]
        return cm, classes

    # 출력할 열 수: 예측 결과(outputs)가 있으면 3, 없으면 2
    col = 3 if outputs is not None else 2
    # figsize: outputs가 있으면 8, 없으면 10 (원하는 크기에 맞춰 조절 가능)
    figsize = 8 if outputs is not None else 12
    fig, ax = plt.subplots(plot_size, col, figsize=(10, figsize), constrained_layout=True)

    for batch in range(plot_size):
        # 이미지 tensor를 numpy 배열로 변환 후, (C, H, W) -> (H, W, C) 형태로 전환
        im = images[batch].numpy().transpose(1, 2, 0)
        ax[batch][0].imshow(im)
        ax[batch][0].set_title("input image")
        ax[batch][0].axis("off")

        # 실제 마스크를 색칠한 결과
        cm, classes = color_mask(im, masks[batch])
        ax[batch][1].set_title("real mask\n" + " / ".join(classes))
        ax[batch][1].imshow(cm)
        ax[batch][1].axis("off")

        # 만약 예측 결과가 있다면 추가로 시각화
        if outputs is not None:
            cm, classes = color_mask(im, outputs[batch])
            ax[batch][2].set_title("pred mask\n" + " / ".join(classes))
            ax[batch][2].imshow(cm)
            ax[batch][2].axis("off")
```


```python

###############################################
# Step 5: DataLoader에서 한 배치 가져와 시각화하기
###############################################
# train DataLoader에서 한 배치를 가져옵니다.
images, masks = next(iter(train_dataloader))
# 첫 배치의 결과를 시각화하여 데이터 로드와 전처리 과정이 올바르게 작동하는지 확인합니다.
draw_mask(images, masks, plot_size=4)
```

<br/>

