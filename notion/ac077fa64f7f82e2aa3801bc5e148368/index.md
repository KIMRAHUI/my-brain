---
notion-url: https://app.notion.com/p/YOLOv3-ac077fa64f7f82e2aa3801bc5e148368
title: YOLOv3
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/YOLOv3-ac077fa64f7f82e2aa3801bc5e148368
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
# YOLOv3 (Darknet-53 + FPN 스타일 Neck 구현)

> [!info] 이 노트는
YOLOv3의 핵심인 **① Darknet-53(Residual Block 기반 백본)**, **② 3-스케일 FPN 스타일 Neck**,
**③ 출력 디코딩 함수**, **④ Multi-label 대응 Loss(BCE)** 를 구현합니다.
Backbone → Neck → Head 3단 구조로 나눠서 이해하는 것이 핵심.

## 구성 순서


```plain text
ResidualBlock → Darknet53 (backbone, route1/route2/x_small 3개 feature 반환)
→ YOLOv3Head (스케일별 예측 head) → YOLOv3 (전체 조립: backbone+FPN neck+3 head)
→ yolo_output_to_boxes (추론 시 디코딩) → YOLOv3Loss
```

---

## 1. Residual Block (Darknet-53 기본 단위)

> [!note] ResNet과의 관계
YOLOv2까지는 잔차 연결이 없었지만, YOLOv3는 **Residual Connection**을 도입해
53개 conv layer의 깊은 네트워크에서도 기울기 소실 없이 학습 가능해졌습니다.


```python
class ResidualBlock(nn.Module):
    # in_channels: 입력 채널, hidden_channels: 내부 1x1 conv의 출력 채널 (보통 in_channels // 2)
    def __init__(self, in_channels, hidden_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)  # 채널 축소
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1, bias=False)  # 채널 복원
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x   # skip connection을 위해 입력 보존
        out = self.leaky(self.bn1(self.conv1(x)))
        out = self.leaky(self.bn2(self.conv2(out)))
        return out + residual   # 입력 + 변환 결과를 더함 (gradient가 직접 흐르는 경로 확보)
```

---

## 2. Darknet-53 백본

> [!tip] 왜 3개의 feature map(route1/route2/x)을 반환하는가?
YOLOv3의 다중 스케일 검출을 위해 **얕은 층(고해상도, 작은 객체용)**부터
**깊은 층(저해상도, 큰 객체용)**까지 서로 다른 3단계 feature가 모두 필요하기 때문.


```python
class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.1)
        )
        # _make_layer: [다운샘플 conv(stride=2)] + [ResidualBlock * num_blocks]
        self.layer1 = self._make_layer(32, 64, num_blocks=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=8)   # route1 (~52x52) ★큰 객체용 아님, 작은 객체용
        self.layer4 = self._make_layer(256, 512, num_blocks=8)   # route2 (~26x26) ★중간 객체용
        self.layer5 = self._make_layer(512, 1024, num_blocks=4)  # 최종 (~13x13)  ★큰 객체용

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        # 첫 conv는 항상 stride=2로 다운샘플링 (MaxPool 대신 strided conv 사용 — Darknet 특징)
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.1)
        ))
        # 지정된 수만큼 ResidualBlock 반복 (block 내부 채널은 out_channels//2)
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels // 2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        route1 = self.layer3(x)        # 큰 해상도 feature (52x52, 256채널)
        route2 = self.layer4(route1)   # 중간 해상도 feature (26x26, 512채널)
        x = self.layer5(route2)        # 작은 해상도 feature (13x13, 1024채널)
        return route1, route2, x       # 3개 스케일 feature 모두 반환
```

---

## 3. Detection Head (스케일별 예측 모듈)


```python
class YOLOv3Head(nn.Module):
    # out_channels = num_anchors * (5 + num_classes)  (5 = tx,ty,tw,th,objectness)
    def __init__(self, in_channels, out_channels):
        super(YOLOv3Head, self).__init__()
        # 1x1(채널축소) -> 3x3(특징추출) 패턴을 3번 반복하는 YOLOv3 특유의 head 구조
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2), nn.LeakyReLU(0.1)
        )
        # 최종 1x1 conv: 그리드 셀 x anchor 별 예측 벡터 생성
        self.pred = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pred(x)
        return x
```

---

## 4. YOLOv3 전체 모델 (Backbone + FPN 스타일 Neck + 3 Head)

> [!note] FPN과의 관계
`[[00_Object-Detection-MOC|FPN]]`의 **top-down + lateral connection** 아이디어를
YOLOv3가 detection에 그대로 적용한 구조. 깊은 feature를 업샘플링해서
얕은 feature와 concat하는 패턴이 반복됩니다 (13→26→52).


```python
class YOLOv3(nn.Module):
    def __init__(self, num_classes=80, num_anchors=3):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors   # 스케일당 3개 (총 9개, k-means로 산출)
        self.backbone = Darknet53()
        out_channels = num_anchors * (5 + num_classes)

        # --- 작은 스케일(13x13): 가장 깊은 feature를 바로 head에 입력 ---
        self.head_small = YOLOv3Head(1024, out_channels)

        # --- 중간 스케일(26x26): 작은 스케일 feature를 업샘플링 + route2와 결합 ---
        self.conv_small_to_medium = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.1)
        )
        self.head_medium = YOLOv3Head(768, out_channels)  # 256(업샘플) + 512(route2) = 768

        # --- 큰 스케일(52x52): 중간 스케일 feature를 업샘플링 + route1과 결합 ---
        self.conv_medium_to_large = nn.Sequential(
            nn.Conv2d(768, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1)
        )
        self.head_large = YOLOv3Head(384, out_channels)   # 128(업샘플) + 256(route1) = 384

        # 업샘플링: nearest neighbor 2배 확대 (연산량 적고 구현 간단 — FPN 표준 방식)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # route feature를 head에 넣기 전 채널 정리용 1x1 conv (lateral connection)
        self.conv_route2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.1)
        )
        self.conv_route1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # 1. 백본에서 3개 스케일 feature 추출
        route1, route2, x_small = self.backbone(x)
        # route1: [batch,256,52,52] / route2: [batch,512,26,26] / x_small: [batch,1024,13,13]

        # 2. 작은 스케일(13x13) 예측 — 가장 깊은 feature 직접 사용
        small_out = self.head_small(x_small)

        # 3. 중간 스케일(26x26) 예측 — top-down: 13x13을 업샘플링해서 26x26 route와 concat
        x_small_to_medium = self.conv_small_to_medium(x_small)   # 채널 축소 1024->256
        x_small_to_medium = self.upsample(x_small_to_medium)      # 13x13 -> 26x26
        route2_processed = self.conv_route2(route2)               # lateral connection
        medium_input = torch.cat([route2_processed, x_small_to_medium], dim=1)  # concat: 768채널
        medium_out = self.head_medium(medium_input)

        # 4. 큰 스케일(52x52) 예측 — 동일 패턴 반복
        x_medium_to_large = self.conv_medium_to_large(medium_input)  # 768->128
        x_medium_to_large = self.upsample(x_medium_to_large)          # 26x26 -> 52x52
        route1_processed = self.conv_route1(route1)                   # lateral connection
        large_input = torch.cat([route1_processed, x_medium_to_large], dim=1)  # concat: 384채널
        large_out = self.head_large(large_input)

        # 5. 3개 스케일 예측 결과 모두 반환 (작은 객체~큰 객체까지 커버)
        return small_out, medium_out, large_out
```


```python
model = YOLOv3(num_classes=80, num_anchors=3)
x = torch.randn(1, 3, 416, 416)
outputs = model(x)
print("Small scale output shape (13x13):", outputs[0].shape)
print("Medium scale output shape (26x26):", outputs[1].shape)
print("Large scale output shape (52x52):", outputs[2].shape)
```

---

## 5. 출력 디코딩 (추론 시: raw 출력 → 실제 박스 좌표)


```python
def yolo_output_to_boxes(outputs, anchors, num_classes, img_size):
    """
    outputs: (small_out, medium_out, large_out) — 모델의 raw 출력 3개
    anchors: 스케일별 anchor 크기 리스트 (픽셀 단위)
             예: [[(10,13),(16,30),(33,23)], [(30,61),(62,45),(59,119)], [(116,90),(156,198),(373,326)]]
    반환: [batch_idx, cx, cy, w, h, objectness, class_id, class_score] 리스트 (0~1 정규화 좌표)
    """
    if isinstance(img_size, int):
        img_height, img_width = img_size, img_size
    else:
        img_height, img_width = img_size

    all_detections = []
    num_anchors_per_scale = len(anchors[0])

    for scale_idx, output in enumerate(outputs):
        batch_size, _, grid_h, grid_w = output.shape

        # (batch, A*(5+C), H, W) -> (batch, A, H, W, 5+C)
        output = output.view(batch_size, num_anchors_per_scale, 5 + num_classes, grid_h, grid_w).permute(0, 1, 3, 4, 2)

        # 각 요소 분리 + 활성화 함수 적용
        box_centers = torch.sigmoid(output[..., 0:2])   # tx, ty -> sigmoid (그리드 셀 내부 상대좌표)
        box_dims_raw = output[..., 2:4]                  # tw, th (아직 exp 적용 전)
        objectness = torch.sigmoid(output[..., 4:5])     # objectness -> sigmoid
        class_scores = torch.sigmoid(output[..., 5:])    # ★YOLOv3는 softmax 대신 sigmoid(독립 로지스틱)
                                                            # → 다중 라벨(예: "사람"+"운동선수") 동시 예측 가능

        # 그리드 셀 좌상단 좌표 (offset) 계산
        grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).to(output.device)

        anchors_tensor = torch.tensor(anchors[scale_idx]).float().to(output.device)

        # 중심 좌표: (sigmoid(t) + grid_offset) / grid_size → 이미지 기준 0~1 스케일
        final_centers = (box_centers + grid.view(1, 1, grid_h, grid_w, 2))
        final_centers[..., 0] /= grid_w
        final_centers[..., 1] /= grid_h

        # 너비/높이: exp(t) * anchor_size → 이미지 기준 0~1 스케일
        final_dims = torch.exp(box_dims_raw) * anchors_tensor.view(1, num_anchors_per_scale, 1, 1, 2)
        final_dims[..., 0] /= img_width
        final_dims[..., 1] /= img_height

        predictions = torch.cat([final_centers, final_dims, objectness, class_scores], dim=-1)
        predictions = predictions.view(batch_size, -1, 5 + num_classes)

        batch_indices = torch.arange(batch_size).view(batch_size, 1, 1).repeat(1, predictions.shape[1], 1).to(output.device)
        detections_this_scale = torch.cat([batch_indices.float(), predictions], dim=-1)
        all_detections.append(detections_this_scale.view(-1, 6 + num_classes))

    # 3개 스케일 결과를 하나로 합침
    all_detections = torch.cat(all_detections, dim=0)

    # 최종 점수 = objectness * class_score (NMS 적용 전 단계)
    objectness_scores = all_detections[:, 5:6]
    class_probs = all_detections[:, 6:]
    final_scores = objectness_scores * class_probs
    max_class_scores, max_class_ids = torch.max(final_scores, dim=1)
    # 이후 threshold 필터링 + NMS(예: 03_YOLOv1의 nms 함수 재사용)로 최종 박스 확정
```

> [!tip] YOLOv3 라벨링의 특징
클래스 예측에 **softmax 대신 sigmoid**를 쓰기 때문에 한 박스가 여러 클래스에 동시 속할 수 있음
(예: "Woman" + "Person" 동시 예측). Multi-label 데이터셋(Open Images 등)에 유리.

---

## 6. YOLOv3Loss


```python
class YOLOv3Loss(nn.Module):
    def __init__(self, num_classes, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        # 좌표: MSE / confidence·class: BCEWithLogitsLoss (softmax 아님 → multi-label 대응)
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        """
        predictions/targets: 리스트, 각 원소 shape = (batch, grid_h, grid_w, num_anchors, 5+num_classes)
        """
        total_loss = 0.0
        batch_size = predictions[0].shape[0]

        for pred, target in zip(predictions, targets):
            pred_xy = torch.sigmoid(pred[..., :2])   # 중심 좌표
            pred_wh = pred[..., 2:4]                   # 너비/높이 (raw)
            pred_obj = pred[..., 4]                    # objectness logit
            pred_cls = pred[..., 5:]                   # class logits

            target_xy = target[..., :2]
            target_wh = target[..., 2:4]
            target_obj = target[..., 4]
            target_cls = target[..., 5:]

            obj_mask = target_obj        # 객체가 할당된 위치만 1
            noobj_mask = 1 - target_obj

            # 좌표 손실 (object 있는 곳만, mask 곱해서 나머지는 0으로 무시)
            loss_xy = self.mse_loss(pred_xy * obj_mask.unsqueeze(-1), target_xy * obj_mask.unsqueeze(-1))
            # w,h는 sqrt로 스케일 보정 (YOLOv1/v2와 동일 원리)
            loss_wh = self.mse_loss(
                torch.sqrt(torch.clamp(pred_wh, min=1e-6)) * obj_mask.unsqueeze(-1),
                torch.sqrt(target_wh + 1e-6) * obj_mask.unsqueeze(-1)
            )

            # objectness 손실: object 있는 곳/없는 곳 따로 계산, no-object는 가중치 축소
            loss_obj = self.bce_loss(pred_obj * obj_mask, target_obj * obj_mask)
            loss_noobj = self.bce_loss(pred_obj * noobj_mask, target_obj * noobj_mask)

            # 클래스 손실: BCE (softmax 아님 → multi-label 대응)
            loss_cls = self.bce_loss(pred_cls * obj_mask.unsqueeze(-1), target_cls * obj_mask.unsqueeze(-1))

            scale_loss = (self.lambda_coord * (loss_xy + loss_wh)
                          + loss_obj + self.lambda_noobj * loss_noobj + loss_cls)
            total_loss += scale_loss

        return total_loss / batch_size
```

> [!warning] 실전 학습 시
이 노트북은 개념 구현용입니다. 실제 학습(target encoding, anchor matching, augmentation)은
검증된 구현체(`eriklindernoren/PyTorch-YOLOv3`)를 참고하는 것을 권장.

---

## 🎯 프로젝트 대비 체크리스트

- [ ] Darknet-53의 Residual Block이 왜 필요한지(깊은 네트워크 gradient 흐름) 설명 가능한가?

- [ ] route1/route2/x_small 3개 feature가 각각 어떤 크기 객체를 담당하는지 설명 가능한가?

- [ ] FPN의 top-down + lateral connection 패턴을 YOLOv3의 forward 코드에서 짚을 수 있는가?

- [ ] YOLOv3가 softmax 대신 sigmoid를 쓰는 이유(multi-label)를 설명할 수 있는가?

- [ ] Backbone/Neck/Head 3단 구조로 다른 1-stage detector(RetinaNet 등)도 설명할 수 있는가?

## 관련 노트

- [[04_YOLOv2]] — 단일 스케일(13x13) → 3-스케일로 확장된 배경

- [[02_SSD]] — SSD의 다중 스케일 방식(concat 없이 독립 예측) vs YOLOv3(top-down concat)

- [[_참고_ultralytics_YOLO]] — 실전에서는 이 구조를 직접 구현하지 않고 ultralytics 라이브러리 사용

