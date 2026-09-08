# AutoEncoder_MNIST

사전처리

import 추가

파일 위치 주소 관련 : os, glob

torch : torch, torch.nn, torch.optim. torch.nn.functional

torch.utils.data : Dataset, DataLoader, random_split

torchvision : torchvision.transform, v2

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import cv2
import matplotlib.pyplot as plt
import math
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import v2
from glob import glob

!pip install torchinfo
from torchinfo import summary

device 설정

CUDA gpu가 있는 경우 cuda로 디바이스 설정

mac - Apple 실리콘의 gpu 사용을 위해 mps 설정추가

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.mps.is_available() else device)
print(f"Using device: {device}")

데이터 처리

데이터 설정

데이터 메인 폴더와 데이터 파일 리스트 생성

데이터 길이와 내용 일부 확인

Train dataset 크기와 Val dataset 크기 설정

from google.colab import drive
drive.mount('/content/drive')

main_folder = "/content/drive/MyDrive/강의/Code_It/Data/denoising-dirty-documents"
datafiles = glob(os.path.join(main_folder, "train", "*.png"))
testfiles = glob(os.path.join(main_folder, "test", "*.png"))

# 데이터 셋 길이와 샘플 출력
print(f"Number of images: {len(datafiles)}")
print(f"First N images: {datafiles[:1]}")

# Train set과 Test set 크기 설정
train_size = int(0.8 * len(datafiles))
val_size = len(datafiles) - train_size

print(f"Train set size: {train_size}")
print(f"Test set size: {val_size}")

Image Size check 함수

이미지 file 리스트를 입력해 이미지의 shape 리스트 확인

unique를 통해 중복 제거

def check_image_shapes(image_paths):
    shapes = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            shapes.append(img.shape)
    return np.unique(shapes, axis=0)

이미지 크기 설정

Train set 에 포함된 이미지의 크기를 종류별로 확인 하여 train에 사용할 이미지 크기 설정

[258 540] 크기와 [420 540] 크기 두 종류가 포함되어 있음.

이들중에 큰 값으로 학습 이미지 크기 설정

unique_shapes = check_image_shapes(datafiles)
print(f"Unique image shapes in the dataset: \n{unique_shapes}")

set_size = (unique_shapes.max(axis=0))
print(f"Set size: {set_size}")

PadToSize class

이미지 크기 조절처리

이미지에 padding을 추가해 비율을 유지하며 이미지 크기 변경

class PadToSize:
    def __init__(self, target_size=(420, 540)):
        self.target_size = target_size

    def __call__(self, *images):
        outputs = []
        for image in images:
            _, h, w = image.shape
            target_h, target_w = self.target_size

            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left

            outputs.append(TF.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0))

        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

Grayscale 변환

class Grayscale:
    def __call__(self, *imgs):
        outputs = []
        for img in imgs:
            # 그레이스케일 변환
            grayscale_img = TF.rgb_to_grayscale(img)  # RGB -> Grayscale
            outputs.append(grayscale_img)
        # Return a tuple if multiple images were passed, otherwise return the single image
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

Transform 설정

PadToSize class를 적용할 수 있음

transform_train = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0], std=[1.0]),
        v2.RandomRotation(10),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        Grayscale(),
        PadToSize(target_size=set_size),
    ]
)
transform_test = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0], std=[1.0]),
        Grayscale(),
        PadToSize(target_size=set_size),
    ]
)

DataSet Class

Train 이미지와 Train_cleaned 데이터를 pair로 묶어 데이터 셋 설정

입력으로 이미지 파일 리스트와 transform 추가

pad to size 함수를 입력으로 받아와 처리

class Dataset(Dataset):
  def __init__(self, files, transform=None):
    self.files = files
    self.transform = transform

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    data_path = os.path.join(self.files[idx])
    img = Image.open(data_path)                                   # '.../train/72.png'
    clean_path = data_path.replace("train", "train_cleaned")  # '.../train_cleaned/72.png'
    clean_img = Image.open(clean_path)

    if self.transform:
        img, clean_img = self.transform(img, clean_img)

    return img.to(device), clean_img.to(device)

class TestDataset(Dataset):
  def __init__(self, files, transform=None):
    self.files = files
    self.transform = transform

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    data_path = os.path.join(self.files[idx])
    img = Image.open(data_path)
    if self.transform:
        img = self.transform(img)
    return img.to(device)

Dataset 및 Dataloader 설정

random_split을 이용해 데이터 분할

학습 batch_size : 32

train_file, val_file = random_split(datafiles, [train_size, val_size])
train_dataset = Dataset(train_file, transform=transform_train)
val_dataset = Dataset(val_file, transform=transform_test)
test_dataset = TestDataset(testfiles, transform=transform_test)

print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=30, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

print(f"Train DataLoader Batches: {len(train_loader)}")
print(f"Val DataLoader Batches: {len(val_loader)}")
print(f"Test DataLoader Batches: {len(test_loader)}")

# 첫 번째 배치 확인
for batch_inputs, batch_targets in train_loader:
    print(f"Batch Input Shape: {batch_inputs.shape}")
    print(f"Batch Target Shape: {batch_targets.shape}")
    break

이미지 확인 함수

데이터 셋을 입력 받아 이미지를 확인

def image_show(dataset, index=0, num_images_to_show=1):
    if index >= len(dataset):
        print(f"Warning: Index {index} is out of bounds.")
        return

    # Get the number of images in a single dataset item
    num_images_in_item = len(dataset[index])

    fig, axes = plt.subplots(num_images_to_show, num_images_in_item, figsize=(num_images_in_item * 5, num_images_to_show * 3))

    # Ensure axes is always a 2D array for consistent indexing
    if num_images_to_show == 1 and num_images_in_item == 1:
        axes = np.array([[axes]])
    elif num_images_to_show == 1:
        axes = np.array([axes])
    elif num_images_in_item == 1:
        axes = np.array(axes).reshape(-1, 1)

    for i in range(num_images_to_show):
        if index + i >= len(dataset):
            print(f"Warning: Skipping index {index + i} as it is out of bounds.")
            continue

        item = dataset[index + i]
        for j in range(num_images_in_item):
            if len(item[j].shape)<3:
                data = item[j].unsqueeze(0)
            else:
                data = item[j]

            img = torch.permute(data, (1, 2, 0)).detach().cpu().numpy()
            min_val, max_val = data.min(), data.max()

            axes[i, j].imshow(img, cmap='gray', vmin=min_val, vmax=max_val)
            axes[i, j].set_title(f"Image {j} (Index {index + i})")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()

# Train dataset 샘플 이미지 확인
image_show(train_dataset, 2)
image_show(val_dataset, 5)
image_show(test_dataset, 0)

모델 선언

CNN 기반 Autoencoder Class

Conv2D 4개층, ConvTranspose2d 4개층으로 이루어진 오토인코더 모델

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=4),  # 420x540 -> 212x272
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=3),  # 212x272 -> 108x138
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2),  # 108x138 -> 55x70
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 55x70 -> 28x35
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=1, output_padding=0),  # 28x35 -> 56x70
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=1, output_padding=0),  # 56x70 -> 110x138
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=1, output_padding=0),   # 110x138 -> 216x272
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=1, output_padding=0),     # 216x272 -> 428x540
            nn.Sigmoid()  # 픽셀 값 [0, 1]로 정규화
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        resized_output = F.interpolate(decoded, size=(420, 540), mode='bilinear', align_corners=False)
        return resized_output

# 모델 생성
model = Autoencoder().to(device)

summary(model, input_size=(1, 1, 420, 540))

모델 학습

학습 사전 설정

loss function 설정

optimizer 설정

loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

def train(num_epochs, model, loss_fn, opt, train_loader, device):
  model.to(device)
  model.train()
  train_losses = []

  old_loss = 0
  total_loss = 0

  for epoch in range(num_epochs):
    train_loss = 0.
    for x, y in train_loader:
      opt.zero_grad()
      x, y = x.to(device), y.to(device)
      pred = model(x)
      loss = loss_fn(pred, y)
      loss.backward()
      opt.step()

      train_loss += loss.item()

    total_loss = train_loss / len(train_loader)
    train_losses.append(total_loss)
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss}")

    if old_loss > total_loss:
      image_show(torch.stack((x, y, pred), dim=1), 0)
      torch.save(model.state_dict(), os.path.join(main_folder, "model.pth"))
    old_loss = total_loss

  return train_losses

모델 학습

# 모델 학습
train_losses = train(10, model, loss_fn, opt, train_loader, device)

# 기존의 학습을 이어서 하기 위해 필요시 모델 load
model.load_state_dict(torch.load(os.path.join(main_folder, "model.pth")))
train_losses = train(40, model, loss_fn, opt, train_loader, device)

# 학습 손실 시각화
plt.figure(figsize=(6,3))
plt.plot(train_losses, label="Train Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()

모델 평가

evaluation 함수를 이용해 test dataset 추론

테스트 결과를 이미지로 확인

RMSE & PSNR 계산

def model_test(model, loss_fn, loader, device):
  model.to(device)
  preds_list = []
  yy_list = []
  model.eval()
  for batch, (x, y) in enumerate(loader):
    x, y = x.to(device), y.to(device)
    pred = model(x)
    loss = loss_fn(pred, y)
    preds_list.append(pred.cpu().detach())
    yy_list.append(y.cpu().detach())
  preds = torch.cat(preds_list, dim=0)
  yy = torch.cat(yy_list, dim=0)
  print(f"Test Loss: {loss.item()}")
  return preds, yy

def model_infer(model, loader, device):
  model.to(device)
  preds_list = []
  model.eval()
  for batch, x in enumerate(loader):
    x = x.to(device)
    pred = model(x)
    preds_list.append(pred.cpu().detach())
  preds = torch.cat(preds_list, dim=0)
  return preds

model.load_state_dict(torch.load(os.path.join(main_folder, "model.pth")))
preds, yy = model_test(model, loss_fn, val_loader, device)

print(preds.size())
print(yy.size())

for i in range(5):
  image_show(torch.stack((yy, preds), dim=1), i)

test_preds = model_infer(model, test_loader, device)

for i in range(5):
  image_show(test_preds, i)

# 샘플 3개 시각화
num_samples = 3
fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))

test_images_tensor, _ = next(iter(val_loader))
test_images_tensor = test_images_tensor[:num_samples]
test_outputs = preds[:num_samples]

for i in range(num_samples):
    axes[i, 0].imshow(test_images_tensor[i].cpu().squeeze(), cmap="gray")
    axes[i, 0].set_title("original Document")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(test_outputs[i].cpu().squeeze(), cmap="gray")
    axes[i, 1].set_title("Processed Document")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()

# RMSE 와 PSNR 계산
def calculate_rmse(original, restored):
    return torch.sqrt(F.mse_loss(original, restored)).item()

def calculate_psnr(original, restored, max_pixel=1.0):
    mse = F.mse_loss(original, restored).item()
    if mse == 0:
        return 100
    return 20 * np.log10(max_pixel / np.sqrt(mse))

rmse_scores = []
psnr_scores = []

with torch.no_grad():
    for i in range(num_samples):
        original = test_images_tensor[i].to(device)
        restored = preds[i].to(device)

        rmse = calculate_rmse(original, restored)
        psnr = calculate_psnr(original, restored)

        rmse_scores.append(rmse)
        psnr_scores.append(psnr)

print(f"평균 RMSE: {np.mean(rmse_scores):.6f}")
print(f"평균 PSNR: {np.mean(psnr_scores):.2f} dB")

UNet Test

메모리 문제로 인해서 학습 어려움

batch size 조정을 통해서 해결

class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)  # 420x540 → 210x270
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)  # 210x270 → 105x135
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)  # 105x135 → 52x67
        )

        # 연결층
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, output_padding=1),  # 52x67 → 105x135
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=2, stride=2, output_padding=0),  # 105x135 → 210x270
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=2, stride=2, output_padding=0),  # 210x270 → 420x540
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.final = nn.Conv2d(64 + 64, 1, kernel_size=1)

    def forward(self, x):
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # 연결층
        b = self.bottleneck(e3)

        # Decoding + Skip Connections
        d3 = self.dec3(b)
        d3 = torch.cat((d3, F.interpolate(e3, size=d3.shape[2:], mode="bilinear", align_corners=False)), dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat((d2, F.interpolate(e2, size=d2.shape[2:], mode="bilinear", align_corners=False)), dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat((d1, F.interpolate(e1, size=d1.shape[2:], mode="bilinear", align_corners=False)), dim=1)

        out = self.final(d1)
        out = torch.sigmoid(out)  # 픽셀 값 [0,1] 범위로 조정
        return out

# 모델 생성
model_unet = UNetAutoencoder().to(device)

summary(model_unet, input_size=(1, 1, 420, 540))

DataLoader 재실행

GPU 메모리 문제로 인해 batch-size 수정

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

모델 학습

Loss Function 과 Optimizer

모델 학습 함수 실행

loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model_unet.parameters(), lr=1e-3)

# 모델 학습
# model_unet.load_state_dict(torch.load(os.path.join(main_folder, "model_unet.pth")))
train_losses = train(20, model_unet, loss_fn, opt, train_loader, device)

모델 평가

preds, yy = model_test(model_unet, loss_fn, val_loader, device)

# 학습 손실 시각화
plt.figure(figsize=(6,3))
plt.plot(train_losses, label="Train Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()

# 샘플 시각화
num_samples = 4
fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))

test_images_tensor, _ = next(iter(val_loader))
test_images_tensor = test_images_tensor[:num_samples]
test_outputs = preds[:num_samples]

for i in range(num_samples):
    axes[i, 0].imshow(test_images_tensor[i].cpu().squeeze(), cmap="gray")
    axes[i, 0].set_title("original Document")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(test_outputs[i].cpu().squeeze(), cmap="gray")
    axes[i, 1].set_title("Processed Document")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()

rmse_scores = []
psnr_scores = []

with torch.no_grad():
    for i in range(num_samples):
        original = test_images_tensor[i].to(device)
        restored = preds[i].to(device)

        rmse = calculate_rmse(original, restored)
        psnr = calculate_psnr(original, restored)

        rmse_scores.append(rmse)
        psnr_scores.append(psnr)

print(f"평균 RMSE: {np.mean(rmse_scores):.6f}")
print(f"평균 PSNR: {np.mean(psnr_scores):.2f} dB")



