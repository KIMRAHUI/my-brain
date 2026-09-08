# U-net

UNet 시맨틱 세그멘테이션 (PennFudan Pedestrian) 코드 정리

PennFudan 보행자 데이터셋으로 이진(배경/보행자) 세그멘테이션을 학습하는 UNet 파이프라인 정리 노트입니다.

0. 데이터 다운로드

!wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -P data
!cd data && unzip PennFudanPed.zip

데이터 구조: data/PennFudanPed/PNGImages (원본 이미지), data/PennFudanPed/PedMasks (인스턴스 마스크)

1. 데이터셋 로드 및 전처리

1-1. Import

import os
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

1-2. PennFudanDataset 클래스

PNGImages / PedMasks 폴더에서 이미지·마스크 쌍을 읽어옴

원본 마스크는 인스턴스별로 다른 픽셀값을 가지지만, 이진 분할을 위해 0(배경) / 1(보행자)로 단순화

transform은 이미지와 마스크에 동일하게 적용해야 함 (좌표 정합성 유지)

class PennFudanDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        root: PennFudanPed 폴더의 상위 경로 (예: "./data/PennFudanPed")
        transform: 이미지 및 마스크에 적용할 transform (동일하게 적용)
        """
        self.root = root
        self.imgs_dir = os.path.join(root, "PNGImages")
        self.masks_dir = os.path.join(root, "PedMasks")
        self.imgs = list(sorted(os.listdir(self.imgs_dir)))
        self.masks = list(sorted(os.listdir(self.masks_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 이미지 로드
        img_path = os.path.join(self.imgs_dir, self.imgs[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # mask는 각 인스턴스마다 다른 값이지만, 이진 분할을 위해
        # 0은 배경, 그 외는 보행자(1)로 처리
        mask = np.array(mask)
        # 여러 인스턴스 픽셀값 > 0을 1로 변경[중요]
        mask = np.where(mask > 0, 1, 0).astype('uint8')
        mask = Image.fromarray(mask)

        # transform 적용 (동일하게 이미지와 마스크에 적용)
        if self.transform is not None:
            # mask의 경우 PIL.Image.NEAREST를 사용해야 함
            img, mask = self.transform(img, mask)

        return img, mask


1-3. joint_transform 함수

이미지·마스크를 함께 리사이즈 + 텐서 변환합니다.

def joint_transform(img, mask, size=(256, 256)):
    # Resize: 이미지와 mask에 동일하게 적용
    img = TF.resize(img, size)
    mask = TF.resize(mask, size, interpolation=Image.NEAREST)  # 클래스 값 보존

    # ToTensor: 이미지는 [0,1] 실수 tensor, mask는 int tensor
    img = TF.to_tensor(img)
    mask = torch.as_tensor(np.array(mask), dtype=torch.long)
    return img, mask

⚠️ 마스크 리사이즈는 반드시 NEAREST 보간을 사용해야 클래스 라벨(0/1)이 보존됩니다.

2. UNet 모델 정의

2-1. 아키텍처 요약

핵심 구조: Encoder에서 저장한 feature map(x1~x4)을 Decoder 단계에서 concat하여 공간 정보를 복원하는 skip connection 구조입니다. FCN이 덧셈(add) 방식이라면, UNet은 채널 축 결합(concat) 방식이라는 차이가 있습니다.

2-2. DoubleConv 블록

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

2-3. UNet 전체 코드

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))

        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 256)   # 512(up) + 512(skip x4) = 1024
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 128)    # 256(up) + 256(skip x3) = 512
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 64)     # 128(up) + 128(skip x2) = 256
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)     # 64(up) + 64(skip x1) = 128

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)     # [B,64,H,W]
        x2 = self.down1(x1)  # [B,128,H/2,W/2]
        x3 = self.down2(x2)  # [B,256,H/4,W/4]
        x4 = self.down3(x3)  # [B,512,H/8,W/8]
        x5 = self.down4(x4)  # [B,512,H/16,W/16]  (bottleneck)

        x = self.up1(x5)               # H/8
        x = torch.cat([x, x4], dim=1)  # skip connection
        x = self.conv1(x)

        x = self.up2(x)                # H/4
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)                # H/2
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)                # H
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)          # [B, n_classes, H, W]
        return logits

3. 학습 / 평가 함수

3-1. train, evaluate

# ======================
# 학습 및 평가 루프
# ======================
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        # masks: [B, H, W] -> loss 함수(CrossEntropyLoss)는 class index를 기대하므로 shape 그대로 사용
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # outputs: [B, n_classes, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

4. 메인 실행 (학습 파이프라인)

4-1. 하이퍼파라미터 & 디바이스

# 하이퍼파라미터 설정
num_epochs = 25
batch_size = 4
learning_rate = 1e-4

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

4-2. Dataset / DataLoader (Train/Val 분할)

# Dataset 및 DataLoader
dataset = PennFudanDataset(root="./data/PennFudanPed", transform=joint_transform)
# 학습/검증 분할 (예: 80% train, 20% val)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

⚠️ 버그 주의: train_loader가 train_dataset이 아닌 val_dataset으로 생성되어 있습니다. 실제 학습 시에는 아래처럼 고쳐야 의도대로 동작합니다.

4-3. 모델 / 손실함수 / 옵티마이저

# 모델, 손실함수, optimizer
model = UNet(n_channels=3, n_classes=2).to(device)
criterion = nn.CrossEntropyLoss()  # output: [B, 2, H, W], mask: [B, H, W] (각 픽셀 0 또는 1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

4-4. 학습 루프 (Best 모델 저장)

# 학습 루프
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    # 가장 낮은 검증 손실 모델 저장 (옵션)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_unet.pth")

5. 예측 결과 시각화

원본 이미지 / GT 마스크 / 예측 마스크를 나란히 표시합니다.

import matplotlib.pyplot as plt

def visualize_predictions(model, dataloader, device, num_images=4):
    model.eval()
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
             outputs = model(imgs)  # 출력: [B, n_classes, H, W]
            # 예측: 각 픽셀에서 채널의 argmax (배경:0, 보행자:1)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]로 변경
            masks = masks.cpu().numpy()

            for i in range(min(num_images, imgs.shape[0])):
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(imgs[i])
                axs[0].set_title("Input Image")
                axs[0].axis("off")

                axs[1].imshow(masks[i], cmap="gray")
                axs[1].set_title("Ground Truth")
                axs[1].axis("off")

                axs[2].imshow(preds[i], cmap="gray")
                axs[2].set_title("Predicted Mask")
                axs[2].axis("off")
                plt.show()
            break  # 첫 배치만 시각화

실행:

val_dataloader_shuffle = DataLoader(val_dataset, batch_size=4, shuffle=True)

visualize_predictions(model, val_loader, device)

전체 파이프라인 요약

데이터셋: PennFudan 이미지/마스크 → 인스턴스 마스크를 이진(배경/보행자)으로 단순화 → 256×256 리사이즈

모델: 4-depth UNet (encoder-decoder + concat skip connection), 클래스 수 2 (배경/보행자)

학습: Adam(lr=1e-4) + CrossEntropyLoss, 25 epoch, val loss 기준 best 모델 저장

평가: 이미지/GT/예측 마스크 3열 시각화로 정성 평가

FCN-8s와의 구조적 차이 비교

