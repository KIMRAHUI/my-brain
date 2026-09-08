---
notion-url: https://app.notion.com/p/YOLOv1-eaf77fa64f7f826dbe0f813aa3ceb1e3
title: YOLOv1
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/YOLOv1-eaf77fa64f7f826dbe0f813aa3ceb1e3
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
# YOLOv1 (PyTorch From-Scratch 구현)

> [!info] 이 노트는
YOLOv1을 원 논문 구조 그대로 **직접 구현**한 코드입니다.
핵심은 3가지: ① Architecture(conv 24층+FC 2층) ② IoU 계산 ③ Loss(좌표+confidence+class).
"YOLO의 grid-cell 회귀 방식이 코드로 어떻게 구현되는가"를 이해하는 데 최적.

## 구성 순서


```plain text
architecture_config (레이어 설계도) → create_conv_layers (설계도 → nn.Sequential)
→ YOLOv1 클래스 (conv + FC) → IoU 함수 → NMS 함수 → yolo_loss 함수
```

---

## 1. Architecture 설계

> [!note] 왜 config를 리스트로 따로 뺐나?
반복되는 conv 블록이 많아 `(kernel, filters, stride, padding)` 튜플과
`[conv1, conv2, 반복횟수]` 리스트로 표현 → 코드 재사용성과 가독성을 높임.


```python
# 논문에 제시된 아키텍처 구성 (YOLOv1)
architecture_config = [
    (7, 64, 2, 3),       # (kernel_size, filters, stride, padding)
    "M",                 # maxpool (2x2, stride 2)
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # 해당 conv 2개 조합을 4번 반복
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],  # 2번 반복
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),   # 여기서 다운샘플 (stride=2)
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]
```

## 2. Config → 실제 네트워크로 변환


```python
def create_conv_layers(config, in_channels):
    layers = []
    for module in config:
        if type(module) == tuple:
            # 단일 conv 블록: Conv2d + LeakyReLU(0.1)
            kernel_size, filters, stride, padding = module
            layers.append(nn.Conv2d(in_channels, filters, kernel_size, stride, padding))
            layers.append(nn.LeakyReLU(0.1))   # YOLO 논문은 전 구간 LeakyReLU(0.1) 사용
            in_channels = filters
        elif module == "M":
            # 맥스풀링으로 해상도 절반 축소
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif type(module) == list:
            # 반복 블록: [conv1 튜플, conv2 튜플, 반복 횟수]
            conv1, conv2, num_repeats = module
            for _ in range(num_repeats):
                k, f, s, p = conv1
                layers.append(nn.Conv2d(in_channels, f, k, s, p))
                layers.append(nn.LeakyReLU(0.1))
                in_channels = f
                k, f, s, p = conv2
                layers.append(nn.Conv2d(in_channels, f, k, s, p))
                layers.append(nn.LeakyReLU(0.1))
                in_channels = f
    return nn.Sequential(*layers)

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, num_classes=20, split_size=7, num_boxes=2):
        super(YOLOv1, self).__init__()
        self.conv_layers = create_conv_layers(architecture_config, in_channels)
        # 입력 448x448 → conv 통과 후 최종 feature map 7x7 (논문 기준)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),  # 논문에서 사용한 dropout, 과적합 방지
            # 최종 출력: S*S*(C + B*5) = 7*7*(20+2*5) = 7*7*30 = 1470
            nn.Linear(4096, split_size * split_size * (num_classes + num_boxes * 5))
        )
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x   # shape: [batch, 1470] → 이후 [batch, 7, 7, 30]으로 reshape해서 사용
```

> [!tip] 출력 텐서 30채널의 의미 (S=7, B=2, C=20 기준)
`[0:20]` = 클래스 확률(20개) / `[20]` = box1 confidence / `[21:25]` = box1 (x,y,w,h)
`[25]` = box2 confidence / `[26:30]` = box2 (x,y,w,h)

---

## 3. IoU (Intersection over Union)


```python
# IoU 계산 함수 (bbox 형식: [x_center, y_center, width, height])
def iou(boxes1, boxes2, eps=1e-6):
    """
    boxes1, boxes2: 텐서, 마지막 차원이 [x_center, y_center, width, height]
    """
    # 중심좌표+너비/높이 → 좌측상단(x1,y1), 우측하단(x2,y2) 좌표로 변환
    box1_x1 = boxes1[..., 0:1] - boxes1[..., 2:3] / 2
    box1_y1 = boxes1[..., 1:2] - boxes1[..., 3:4] / 2
    box1_x2 = boxes1[..., 0:1] + boxes1[..., 2:3] / 2
    box1_y2 = boxes1[..., 1:2] + boxes1[..., 3:4] / 2

    box2_x1 = boxes2[..., 0:1] - boxes2[..., 2:3] / 2
    box2_y1 = boxes2[..., 1:2] - boxes2[..., 3:4] / 2
    box2_x2 = boxes2[..., 0:1] + boxes2[..., 2:3] / 2
    box2_y2 = boxes2[..., 1:2] + boxes2[..., 3:4] / 2

    # 교집합 영역의 좌표 (겹치는 부분이 없으면 clamp(min=0)로 0 처리)
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    # IoU = 교집합 / 합집합 (eps로 0나눗셈 방지)
    iou_val = inter / (box1_area + box2_area - inter + eps)
    return iou_val
```

---

## 4. NMS (Non-Maximum Suppression)


```python
# NMS: confidence 낮은 박스 제거 후, 같은 클래스 내에서 겹치는 중복 박스 제거
def nms(bboxes, iou_threshold, conf_threshold):
    """
    bboxes: [pred_class, confidence, x1, y1, x2, y2] 형태 리스트
    """
    # 1) confidence threshold 이하 박스 제거
    bboxes = [box for box in bboxes if box[1] > conf_threshold]
    # 2) confidence 내림차순 정렬 (가장 확신하는 박스부터 채택)
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)  # 가장 confidence 높은 박스 선택
        # 같은 클래스 && IoU가 threshold 이상인 박스는 제거 (같은 물체로 간주)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] or iou(
                torch.tensor(chosen_box[2:]).unsqueeze(0),
                torch.tensor(box[2:]).unsqueeze(0)
            ).item() < iou_threshold
        ]
        bboxes_nms.append(chosen_box)
    return bboxes_nms
```

---

## 5. Loss 함수

> [!note] YOLOv1 Loss = 4가지 요소의 가중합
① Box 좌표 손실(λ_coord=5) ② Object confidence 손실 ③ No-object confidence 손실(λ_noobj=0.5) ④ Class 손실
그리드 셀당 2개 박스 중 **GT와 IoU가 더 높은 박스만** "책임(responsible)" 지도록 선택하는 것이 핵심.


```python
def yolo_loss(predictions, target, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
    """
    predictions: [batch, 7*7*30] (모델 raw 출력)
    target:      [batch, 7, 7, 30] (GT, 각 그리드 셀에 인코딩됨)
    """
    predictions = predictions.view(-1, S, S, C + B * 5)

    # 두 예측 박스(box1: 21:25, box2: 26:30) 각각을 GT 박스(21:25)와 IoU 비교
    iou_b1 = iou(predictions[..., 21:25], target[..., 21:25])
    iou_b2 = iou(predictions[..., 26:30], target[..., 21:25])
    ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

    # IoU가 더 높은 박스를 "책임 박스(bestbox)"로 선택 (0=box1, 1=box2)
    iou_maxes, bestbox = torch.max(ious, dim=0)
    bestbox = bestbox.float().unsqueeze(-1)

    # exists_box: 해당 그리드 셀에 실제 객체가 있는지 여부 (1 or 0)
    exists_box = target[..., 20].unsqueeze(-1)

    # --- ① Box 좌표 손실 ---
    # bestbox 값에 따라 두 예측 박스 중 하나만 선택 (책임 박스만 학습 대상)
    box_pred = exists_box * (bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25])
    box_target = exists_box * target[..., 21:25]

    # w, h에 sqrt 적용 → 큰 박스와 작은 박스의 오차 민감도를 맞춤 (논문의 핵심 트릭)
    box_pred_wh = torch.sqrt(torch.clamp(box_pred[..., 2:4], min=1e-6))
    box_target_wh = torch.sqrt(torch.clamp(box_target[..., 2:4], min=1e-6))
    box_pred = torch.cat([box_pred[..., :2], box_pred_wh], dim=-1)
    box_target = torch.cat([box_target[..., :2], box_target_wh], dim=-1)

    box_loss = torch.sum((box_pred - box_target) ** 2)

    # --- ② Object Confidence 손실 (물체가 있는 셀) ---
    pred_conf = bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
    confidence_target = exists_box * iou_maxes.unsqueeze(-1)  # target=IoU 값 (soft label)
    object_loss = torch.sum((exists_box * (pred_conf - confidence_target)) ** 2)

    # --- ③ No-object Confidence 손실 (물체가 없는 셀, 두 박스 모두 패널티) ---
    noobj_loss = torch.sum(((1 - exists_box) * predictions[..., 20:21]) ** 2)
    noobj_loss += torch.sum(((1 - exists_box) * predictions[..., 25:26]) ** 2)

    # --- ④ Class 손실 (물체가 있는 셀에서만 클래스 확률 학습) ---
    class_loss = torch.sum((exists_box * (predictions[..., :20] - target[..., :20])) ** 2)

    # 최종 loss: 좌표 손실 가중치↑, no-object 손실 가중치↓ (클래스 불균형 보정)
    total_loss = lambda_coord * box_loss + object_loss + lambda_noobj * noobj_loss + class_loss
    return total_loss
```

---

## 6. 모델 요약 확인


```python
from torchinfo import summary

model = YOLOv1(in_channels=3, num_classes=20, split_size=7, num_boxes=2)
summary(model, input_size=(1, 3, 448, 448))
```

---

## 🎯 프로젝트 대비 체크리스트

- [ ] `architecture_config` 패턴으로 나만의 conv 네트워크 설계도를 짤 수 있는가?

- [ ] IoU 계산에서 `x_center,y_center,w,h` ↔ `x1,y1,x2,y2` 변환을 헷갈리지 않는가?

- [ ] Loss에서 "책임 박스(responsible box)" 선택 로직(bestbox)을 설명할 수 있는가?

- [ ] w,h에 sqrt를 취하는 이유(작은 박스 민감도 보정)를 설명할 수 있는가?

- [ ] `lambda_coord=5`, `lambda_noobj=0.5`가 왜 필요한지(클래스 불균형: 배경 셀이 훨씬 많음) 이해했는가?

## 한계점 (면접/발표용 요약)

- 그리드 셀당 최대 B개 박스만 예측 → 겹친 작은 객체 다수 검출 취약

- 단일 스케일(7x7)만 사용 → 작은 물체 검출 성능 낮음 (SSD/YOLOv3가 이를 다중 스케일로 개선)

## 관련 노트

- [[02_SSD]] — 다중 스케일 anchor 기반 검출과 비교

- [[04_YOLOv2]] — anchor box 도입으로 YOLOv1의 좌표 회귀 정밀도 문제 개선

