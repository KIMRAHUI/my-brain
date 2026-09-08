---
notion-url: https://app.notion.com/p/SSD-5c577fa64f7f82e7909f810124c79481
title: SSD
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/SSD-5c577fa64f7f82e7909f810124c79481
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
# SSD (torchvision 기반 구현)

> [!info] 이 노트는
`torchvision.models.detection.ssd`를 활용해 **커스텀 백본(ResNet34) + SSD 헤드**를 조립하고,
COCO 형식 데이터셋으로 학습·평가·시각화까지 전체 파이프라인을 구현한 코드입니다.
프로젝트에서 "커스텀 데이터로 SSD 학습"이 필요할 때 이 노트를 그대로 템플릿으로 재사용 가능.

## 전체 파이프라인 흐름


```plain text
백본 정의(SSDBackbone) → 앵커 생성자(DefaultBoxGenerator) → SSD 모델 조립
→ COCODataset 정의 → DataLoader → 학습 루프 → 시각화 → COCOeval 평가
```

---

## 1. SSD 백본(Backbone) 정의

> [!note] 핵심 아이디어
SSD는 **여러 해상도의 feature map**에서 동시에 물체를 검출합니다.
그러기 위해 backbone 뒤에 점점 채널을 줄이며 다운샘플링하는 `extra` 레이어들을 덧붙여
총 6단계(scale)의 feature map을 뽑아냅니다.


```python
from torch import nn
from collections import OrderedDict

class SSDBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # backbone의 초기 층: conv1, bn1, relu를 하나의 sequential로 묶습니다.
        layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        # ResNet의 레이어들 (layer1 ~ layer4)를 저장
        layer1 = backbone.layer1
        layer2 = backbone.layer2
        layer3 = backbone.layer3
        layer4 = backbone.layer4

        # features는 layer0부터 layer3까지 연결한 것입니다. (layer4는 extra에서 사용)
        self.features = nn.Sequential(layer0, layer1, layer2, layer3)

        # upsampling 모듈: features의 출력 채널 수(256)를 512로 맞춰 첫 예측 스케일로 사용
        self.upsampling = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # extra 모듈: SSD 다중 스케일 feature map을 위한 추가 계층들
        # 점진적으로 다운샘플링하면서 채널 수를 조정 (총 6개 scale block)
        self.extra = nn.ModuleList(
            [
                # ① layer4 통과 후 1x1 conv로 채널 증가 (512 -> 1024)
                nn.Sequential(
                    layer4,
                    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
                    nn.ReLU(inplace=True),
                ),
                # ② 1x1로 채널 축소 후 3x3 stride=2 로 다운샘플 (1024 -> 256 -> 512)
                nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True),
                ),
                # ③ 위와 동일 패턴, 채널만 축소 (512 -> 128 -> 256)
                nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True),
                ),
                # ④ stride=1, padding 없음 → 해상도 추가 축소 (256 -> 128 -> 256)
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
                # ⑤ ④와 동일 패턴 반복
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
                # ⑥ kernel_size=4 로 마지막 최소 해상도 feature map 생성
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=4),
                    nn.ReLU(inplace=True),
                )
            ]
        )

    def forward(self, x):
        # 기본 feature 추출 (features: layer0 ~ layer3)
        x = self.features(x)
        # upsampling을 적용하여 첫 번째 출력 feature map을 만듭니다. (scale 1)
        output = [self.upsampling(x)]
        # extra 모듈을 순차적으로 통과시키며 나머지 5개 scale feature map 생성
        for block in self.extra:
            x = block(x)
            output.append(x)
        # OrderedDict 형태로 반환해야 torchvision SSD 모델이 인식 가능
        return OrderedDict([(str(i), feat) for i, feat in enumerate(output)])
```

> [!tip] 왜 OrderedDict인가?
torchvision의 `SSD` 모델은 backbone의 출력이 `Dict[str, Tensor]` 형태여야
여러 스케일의 feature map을 순서대로 인식해 앵커/헤드에 매칭합니다.

---

## 2. 모델 & 앵커 생성자(Anchor Generator) 초기화


```python
import torch
from torchvision.models import resnet34
from torchvision.models.detection import ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

# 사전 학습된 ResNet34을 backbone base로 로드 (ImageNet pretrained)
backbone_base = resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
# 위에서 정의한 SSDBackbone으로 래핑 → 6개 스케일 feature map을 뽑는 구조로 변환
backbone = SSDBackbone(backbone_base)

# DefaultBoxGenerator: 각 feature map 셀마다 놓을 기본 박스(anchor) 설정
anchor_generator = DefaultBoxGenerator(
    aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],  # 스케일별 종횡비
    scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05, 1.20],        # 스케일별 박스 크기(비율)
    steps=[8, 16, 32, 64, 100, 300, 512],                            # 각 feature map의 stride
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# SSD 모델 생성: backbone + anchor_generator + 입력 크기 + 클래스 수
model = ssd.SSD(
    backbone=backbone,
    anchor_generator=anchor_generator,
    size=(512, 512),
    num_classes=3   # 배경 포함하지 않은 실제 클래스 수 (프로젝트 데이터에 맞게 변경)
).to(device)
```

> [!warning] 프로젝트 적용 시 체크포인트

	- `num_classes`는 **배경(background) 클래스를 제외한 값**인지 torchvision 버전별로 확인 필요

	- `size`는 backbone의 extra 구조와 맞물려 있으므로 임의로 바꾸면 anchor 개수/크기와 어긋날 수 있음

---

## 3. COCO 데이터셋 클래스 정의


```python
import os
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, root, train, transform=None):
        super().__init__()
        # 학습/검증 구분에 따라 annotation 파일 경로 결정
        directory = "train" if train else "val"
        annotations = os.path.join(root, "annotations", f"{directory}_annotations.json")

        # pycocotools로 COCO 형식 annotation 로드
        self.coco = COCO(annotations)
        self.image_path = os.path.join(root, directory)
        self.transform = transform

        self.categories = self._get_categories()
        self.data = self._load_data()

    def _get_categories(self):
        # {id: name} 딕셔너리, 0번은 background로 예약
        categories = {0: "background"}
        for category in self.coco.cats.values():
            categories[category["id"]] = category["name"]
        return categories

    def _load_data(self):
        data = []
        # 모든 이미지에 대해 (image, target) 쌍을 미리 메모리에 로드
        # 주의: 데이터가 크면 메모리 부담 → 대용량 데이터셋은 __getitem__에서 lazy loading 권장
        for _id in self.coco.imgs:
            file_name = self.coco.loadImgs(_id)[0]["file_name"]
            image_path = os.path.join(self.image_path, file_name)
            image = Image.open(image_path).convert("RGB")

            boxes = []
            labels = []
            anns = self.coco.loadAnns(self.coco.getAnnIds(_id))
            for ann in anns:
                # COCO bbox 포맷: [x, y, w, h] → torchvision이 요구하는 [x1, y1, x2, y2]로 변환
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

            target = {
                "image_id": torch.LongTensor([_id]),
                "boxes": torch.FloatTensor(boxes),
                "labels": torch.LongTensor(labels)
            }
            data.append([image, target])
        return data

    def __getitem__(self, index):
        image, target = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.data)
```

---

## 4. DataLoader & 전처리


```python
from torchvision import transforms
from torch.utils.data import DataLoader

# 객체 검출 데이터는 이미지마다 박스 개수가 달라 기본 collate_fn을 못 씀
# → 배치를 (images_tuple, targets_tuple)로 그대로 묶어주는 custom collator 필요
def collator(batch):
    return tuple(zip(*batch))

transform = transforms.Compose(
    [
        transforms.PILToTensor(),                       # PIL 이미지 → Tensor (uint8)
        transforms.ConvertImageDtype(dtype=torch.float)  # float32로 변환 (0~1 정규화 포함)
    ]
)

train_dataset = COCODataset("../datasets/coco", train=True, transform=transform)
test_dataset = COCODataset("../datasets/coco", train=False, transform=transform)

train_dataloader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, drop_last=True, collate_fn=collator
)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=collator
)
```

---

## 5. 학습 루프


```python
from torch import optim
from tqdm import tqdm

# 학습 가능한 파라미터만 필터링해서 optimizer에 전달
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(10):
    train_cost = 0.0
    model.train()  # torchvision detection 모델은 train() 모드에서 forward 시 loss dict를 반환
    for images, targets in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # model(images, targets) → {"bbox_regression": ..., "classification": ...} 형태의 loss dict
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_cost += losses.item()

    lr_scheduler.step()
    avg_train_loss = train_cost / len(train_dataloader)
    print(f"Epoch: {epoch+1:4d}, Train Loss: {avg_train_loss:.3f}")
```

> [!tip] torchvision detection 모델의 관례

	- `model.train()` + `model(images, targets)` → **loss dict 반환**

	- `model.eval()` + `model(images)` → **예측 결과(boxes/labels/scores) 반환**
이 관례는 SSD, Faster R-CNN, RetinaNet 등 torchvision detection 계열 공통.

---

## 6. 결과 시각화


```python
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image

def draw_bbox(ax, box, text, color):
    # matplotlib Rectangle patch로 박스를 그리고 텍스트 라벨을 추가
    ax.add_patch(
        plt.Rectangle(
            xy=(box[0], box[1]),
            width=box[2] - box[0],
            height=box[3] - box[1],
            fill=False, edgecolor=color, linewidth=2,
        )
    )
    ax.annotate(text=text, xy=(box[0] - 5, box[1] - 5), color=color, weight="bold", fontsize=13)

threshold = 0.5
categories = test_dataset.categories

with torch.no_grad():
    model.eval()
    for images, targets in test_dataloader:
        images = [image.to(device) for image in images]
        outputs = model(images)

        boxes = outputs[0]["boxes"].to("cpu").numpy()
        labels = outputs[0]["labels"].to("cpu").numpy()
        scores = outputs[0]["scores"].to("cpu").numpy()

        # confidence threshold 이하 박스 제거
        boxes = boxes[scores >= threshold].astype(np.int32)
        labels = labels[scores >= threshold]
        scores = scores[scores >= threshold]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(to_pil_image(images[0]))

        # 예측 결과 = 빨간색
        for box, label, score in zip(boxes, labels, scores):
            draw_bbox(ax, box, f"{categories[label]} - {score:.4f}", "red")

        # 정답(GT) = 파란색
        tboxes = targets[0]["boxes"].numpy()
        tlabels = targets[0]["labels"].numpy()
        for box, label in zip(tboxes, tlabels):
            draw_bbox(ax, box, f"{categories[label]}", "blue")

        plt.show()
```

---

## 7. COCO 평가지표 (mAP)


```python
import numpy as np
from pycocotools.cocoeval import COCOeval

with torch.no_grad():
    model.eval()
    coco_detections = []
    for images, targets in test_dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i in range(len(targets)):
            image_id = targets[i]["image_id"].data.cpu().numpy().tolist()[0]
            boxes = outputs[i]["boxes"].data.cpu().numpy()
            # COCOeval은 [x, y, w, h] 포맷을 요구 → [x1,y1,x2,y2]에서 변환
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            scores = outputs[i]["scores"].data.cpu().numpy()
            labels = outputs[i]["labels"].data.cpu().numpy()

            for instance_id in range(len(boxes)):
                box = boxes[instance_id, :].tolist()
                prediction = np.array([
                    image_id, box[0], box[1], box[2], box[3],
                    float(scores[instance_id]), int(labels[instance_id]),
                ])
                coco_detections.append(prediction)
    coco_detections = np.asarray(coco_detections)

    coco_gt = test_dataloader.dataset.coco
    coco_dt = coco_gt.loadRes(coco_detections)  # 검출 결과를 COCO 형식으로 등록
    coco_evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()   # mAP@[.5:.95], mAP@.5, mAP@.75 등 출력
```

---

## 8. 백본 출력 채널 확인 (디버깅 유틸)


```python
def retrieve_out_channels(model, size):
    # 실제 이미지 없이 더미(zero) 텐서를 흘려보내 각 feature map의 채널 수를 파악
    # → 헤드/앵커 설계 시 채널 수가 맞는지 검증할 때 유용
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        image = torch.zeros((1, 3, size[1], size[0]), device=device)
        features = model(image)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        out_channels = [x.size(1) for x in features.values()]

    model.train()
    return out_channels

print(retrieve_out_channels(backbone, (512, 512)))
# 예: [512, 1024, 512, 256, 256, 256] → SSD 6개 스케일의 채널 수
```

---

## 🎯 프로젝트 대비 체크리스트

- [ ] 커스텀 데이터셋을 COCO 형식(json)으로 변환할 수 있는가?

- [ ] `num_classes`가 데이터셋의 실제 클래스 수 + background 규칙과 일치하는가?

- [ ] backbone 교체 시 `retrieve_out_channels`로 feature map 채널 수 검증했는가?

- [ ] `DefaultBoxGenerator`의 `scales`/`aspect_ratios`를 객체 크기 분포에 맞게 튜닝했는가?

- [ ] 평가 시 `COCOeval`로 mAP를 산출해 다른 모델과 비교 가능한가?

## 관련 노트

- [[03_YOLOv1]] — 1-stage 검출의 원조, SSD와 비교되는 grid-cell 방식

- [[05_YOLOv3]] — FPN 스타일 다중 스케일 검출과 SSD의 다중 스케일 방식 비교

