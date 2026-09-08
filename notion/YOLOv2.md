# YOLOv2

 (Darknet-19 + Passthrough 구현)

[!info] 이 노트는
YOLOv2의 핵심 3요소를 코드로 구현합니다: ① Darknet-19 백본 ② **Passthrough(Space-to-Depth)**로
작은 객체 detail 보존 ③ Anchor 기반 YOLOv2Loss. YOLOv1과 달리 FC layer 없이 완전 conv로만 구성.

구성 순서

SpaceToDepth (passthrough용 변환) → Darknet19 (백본, 13x13 + 26x26 feature 반환)
→ YOLOv2 (백본 + passthrough 결합 + detection head) → YOLOv2Loss

1. SpaceToDepth (Passthrough 연결의 핵심 트릭)

[!note] 왜 필요한가?
YOLOv2는 최종 13x13 feature map만 쓰면 작은 객체 정보가 손실됩니다.
26x26의 더 고해상도 feature를 채널 방향으로 압축해서 13x13과 concat하는 것이 passthrough.
"공간 해상도를 채널 수로 바꿔치기"하는 연산이 SpaceToDepth.

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size   # 보통 2 (26x26 -> 13x13, 채널 4배)

    def forward(self, x):
        # x: [batch, C, H, W]
        batch, channels, height, width = x.size()
        new_h = height // self.block_size
        new_w = width // self.block_size
        # H, W를 block_size 단위로 쪼개서 별도 차원으로 분리
        x = x.view(batch, channels, new_h, self.block_size, new_w, self.block_size)
        # 공간 축(block_size들)을 채널 앞쪽으로 옮김
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        # 채널 수를 block_size^2 배로 압축, 공간 해상도는 1/block_size로 축소
        x = x.view(batch, channels * (self.block_size ** 2), new_h, new_w)
        return x
        # 예: [batch, 64, 26, 26] -> [batch, 256, 13, 13]

2. Darknet-19 백본

[!tip] YOLOv1과의 차이

class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()
        # Layer1: 416x416 -> 208x208 (conv 3x3 + BN + LeakyReLU + MaxPool)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2)
        )
        # Layer2: 208x208 -> 104x104
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2)
        )
        # Layer3~4: 1x1 conv로 채널을 줄였다가(bottleneck) 다시 3x3으로 확장하는 패턴 시작
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),  # bottleneck (1x1)
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1)
        )
        # Layer5: 104x104 -> 52x52
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1)
        )
        # Layer8: 52x52 -> 26x26
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2)
        )
        # Layer9: 26x26, 512채널 → ★이 시점 출력을 passthrough로 저장★
        self.layer9 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.1)
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.1)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.1)
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.1)
        )
        # Layer13: 26x26 -> 13x13
        self.layer13 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2)
        )
        # Layer14~18: 13x13 해상도 유지, 채널만 1024 <-> 512 오가며 깊게 쌓음
        self.layer14 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024), nn.LeakyReLU(0.1)
        )
        self.layer15 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.1)
        )
        self.layer16 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024), nn.LeakyReLU(0.1)
        )
        self.layer17 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.1)
        )
        self.layer18 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024), nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x = self.layer1(x)   # [batch, 32, 208, 208]
        x = self.layer2(x)   # [batch, 64, 104, 104]
        x = self.layer3(x)   # [batch, 128, 104, 104]
        x = self.layer4(x)   # [batch, 64, 104, 104]
        x = self.layer5(x)   # [batch, 128, 52, 52]
        x = self.layer6(x)   # [batch, 256, 52, 52]
        x = self.layer7(x)   # [batch, 128, 52, 52]
        x = self.layer8(x)   # [batch, 256, 26, 26]
        x = self.layer9(x)   # [batch, 512, 26, 26] -> passthrough 저장 지점
        passthrough = x      # ★ 나중에 detection head에서 concat할 feature 저장
        x = self.layer10(x)  # [batch, 256, 26, 26]
        x = self.layer11(x)  # [batch, 512, 26, 26]
        x = self.layer12(x)  # [batch, 256, 26, 26]
        x = self.layer13(x)  # [batch, 512, 13, 13]
        x = self.layer14(x)  # [batch, 1024, 13, 13]
        x = self.layer15(x)  # [batch, 512, 13, 13]
        x = self.layer16(x)  # [batch, 1024, 13, 13]
        x = self.layer17(x)  # [batch, 512, 13, 13]
        x = self.layer18(x)  # [batch, 1024, 13, 13]
        return x, passthrough   # 최종 feature + passthrough feature 둘 다 반환

3. YOLOv2 전체 모델 (백본 + Passthrough + Detection Head)

class YOLOv2(nn.Module):
    def __init__(self, num_classes=20, num_anchors=5):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors   # k-means로 산출한 anchor 5개 (논문 기준)
        self.backbone = Darknet19()

        # passthrough feature(512채널)를 1x1 conv로 64채널로 축소 후 space-to-depth
        self.passthrough_conv = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1)
        )
        self.space_to_depth = SpaceToDepth(block_size=2)

        # concat 후 채널 수 = 1024(최종) + 64*(2^2)=256(passthrough) = 1280
        self.det_head = nn.Sequential(
            nn.Conv2d(1280, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024), nn.LeakyReLU(0.1),
            # 최종 1x1 conv: anchor마다 (5 + num_classes)개 값 예측
            nn.Conv2d(1024, self.num_anchors * (5 + num_classes), kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # x: [batch, 3, 416, 416]
        final_feat, passthrough_feat = self.backbone(x)
        # final_feat: [batch, 1024, 13, 13], passthrough_feat: [batch, 512, 26, 26]

        # passthrough 처리: 1x1 conv → space-to-depth로 [batch, 256, 13, 13]까지 압축
        passthrough_feat = self.passthrough_conv(passthrough_feat)
        passthrough_feat = self.space_to_depth(passthrough_feat)

        # 채널 방향으로 concat (저해상도 detail + 고해상도 semantic 결합)
        concat = torch.cat([passthrough_feat, final_feat], dim=1)  # [batch, 1280, 13, 13]
        detections = self.det_head(concat)
        return detections  # [batch, num_anchors*(5+num_classes), 13, 13]

model = YOLOv2(num_classes=20, num_anchors=5)
x = torch.randn(1, 3, 416, 416)
output = model(x)
print(output.shape)  # 예상: [1, 125, 13, 13]  (5*(5+20)=125)

4. YOLOv2Loss

[!warning] 실전 주의
이 구현은 target이 이미 anchor별로 할당된 형식으로 준비되어 있다고 가정합니다.
실제로는 GT box를 어느 anchor/그리드셀에 할당할지 결정하는 전처리(target encoding)가 별도로 필요합니다.

class YOLOv2Loss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, lambda_coord=5, lambda_noobj=0.5):
        """
        anchors: [(w, h), ...] 원본 이미지 기준 앵커 박스 크기
        img_size: 입력 이미지 크기 (정방형)
        """
        super(YOLOv2Loss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        """
        predictions: (batch, A*(5+num_classes), grid_h, grid_w)
        target: (batch, grid_h, grid_w, A, 5+num_classes)
        """
        batch_size = predictions.size(0)
        grid_h = predictions.size(2)
        grid_w = predictions.size(3)

        # (batch, A, 5+C, H, W) -> (batch, H, W, A, 5+C) 로 재배치
        prediction = predictions.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_h, grid_w)
        prediction = prediction.permute(0, 3, 4, 1, 2).contiguous()

        # tx, ty, tw, th, confidence, class logits로 분리
        pred_tx = prediction[..., 0]
        pred_ty = prediction[..., 1]
        pred_tw = prediction[..., 2]
        pred_th = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])   # objectness는 sigmoid
        pred_cls = prediction[..., 5:]

        # YOLOv2의 "직접 위치 예측" 방식: sigmoid로 그리드 셀 내부 상대좌표(0~1) 강제
        pred_x = torch.sigmoid(pred_tx)
        pred_y = torch.sigmoid(pred_ty)
        pred_w = torch.exp(pred_tw)   # anchor 대비 배율 (exp로 양수 보장)
        pred_h = torch.exp(pred_th)

        # 그리드 셀의 좌상단 offset 좌표 생성
        device = predictions.device
        grid_x = torch.arange(grid_w, device=device).repeat(grid_h, 1).float()
        grid_y = torch.arange(grid_h, device=device).unsqueeze(1).repeat(1, grid_w).float()
        grid_x = grid_x.unsqueeze(0).unsqueeze(3)
        grid_y = grid_y.unsqueeze(0).unsqueeze(3)

        # anchor 크기를 텐서로 변환
        anchors = torch.tensor(self.anchors, device=device).float()
        anchor_w = anchors[:, 0].view(1, 1, 1, self.num_anchors)
        anchor_h = anchors[:, 1].view(1, 1, 1, self.num_anchors)

        # 최종 박스 좌표 계산 (이미지 전체 기준 0~1 정규화 값)
        box_x = (pred_x + grid_x) / grid_w
        box_y = (pred_y + grid_y) / grid_h
        box_w = (pred_w * anchor_w) / self.img_size
        box_h = (pred_h * anchor_h) / self.img_size

        obj_mask = target[..., 4]      # 객체가 할당된 (anchor, 셀) 위치
        noobj_mask = 1 - obj_mask

        target_x = target[..., 0]
        target_y = target[..., 1]
        target_w = target[..., 2]
        target_h = target[..., 3]

        # 좌표 손실 (object 있는 위치만)
        loss_x = torch.sum(obj_mask * (box_x - target_x) ** 2)
        loss_y = torch.sum(obj_mask * (box_y - target_y) ** 2)
        # w, h는 sqrt로 스케일 보정 (YOLOv1과 동일한 이유)
        loss_w = torch.sum(obj_mask * (torch.sqrt(box_w + 1e-6) - torch.sqrt(target_w + 1e-6)) ** 2)
        loss_h = torch.sum(obj_mask * (torch.sqrt(box_h + 1e-6) - torch.sqrt(target_h + 1e-6)) ** 2)
        coord_loss = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)

        # confidence 손실 (object 있는 곳 / 없는 곳 분리, no-object는 가중치 축소)
        loss_conf_obj = torch.sum(obj_mask * (pred_conf - target[..., 4]) ** 2)
        loss_conf_noobj = torch.sum(noobj_mask * (pred_conf - target[..., 4]) ** 2)
        conf_loss = loss_conf_obj + self.lambda_noobj * loss_conf_noobj

        # 클래스 손실 (MSE 방식, 실전에서는 CrossEntropy로도 구현 가능)
        loss_cls = torch.sum(obj_mask.unsqueeze(-1) * (pred_cls - target[..., 5:]) ** 2)

        total_loss = (coord_loss + conf_loss + loss_cls) / batch_size
        return total_loss

🎯 프로젝트 대비 체크리스트

Passthrough(=Space-to-Depth)가 "해상도를 채널로 바꾸는 연산"이라는 걸 그림으로 설명 가능한가?

Darknet-19가 YOLOv1 대비 FC layer가 없는 이유(임의 크기 입력 대응, Multi-Scale Training)를 설명할 수 있는가?

anchor box 도입 시 좌표를 직접 회귀하지 않고 sigmoid(tx)+grid_offset, exp(tw)*anchor_w로 예측하는 이유(학습 안정성)를 아는가?

YOLO9000의 WordTree 개념(계층적 분류, 조건부 확률 곱)과 YOLOv2Loss의 class loss 차이를 구분할 수 있는가?

관련 노트

[[03_YOLOv1]] — anchor 도입 이전의 직접 회귀 방식과 비교

[[05_YOLOv3]] — 단일 스케일(13x13) → 다중 스케일(3종)로 확장되는 지점

