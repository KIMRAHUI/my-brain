---
notion-url: https://app.notion.com/p/Trasnfer_FineTuning-0b677fa64f7f82409f3e817cb46d1412
title: Trasnfer_FineTuning
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Trasnfer_FineTuning-0b677fa64f7f82409f3e817cb46d1412
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
# 🩺 PyTorch 흉부 X-ray 폐렴 분류 및 전이 학습

## 1. 데이터 불러오기 및 전처리


```python
import kagglehub
import torch
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np

# 1. Kaggle Hub를 통해 흉부 X-ray 폐렴 데이터셋 다운로드
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Dataset path:", path)

# 2. GPU 사용 가능 여부 확인 및 디바이스 설정 (GPU가 있으면 'cuda', 없으면 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 3. 데이터셋 내의 train, val, test 디렉토리 경로 지정
train_dir = os.path.join(path, "chest_xray", "train")
val_dir = os.path.join(path, "chest_xray", "val")
test_dir = os.path.join(path, "chest_xray", "test")

# 4. 데이터 전처리 및 증강(Data Augmentation) 파이프라인 정의
data_transforms = {
    'train': v2.Compose([
        v2.Resize((224, 224)),                # 이미지 크기를 224x224로 통일
        v2.RandomCrop((200, 200)),            # 200x200 크기로 랜덤 크롭 (과적합 방지)
        v2.RandomRotation(20),                # -20도에서 20도 사이로 랜덤 회전
        v2.ToImage(),                         # 이미지를 텐서 이미지로 변환
        v2.ToDtype(dtype=torch.float32, scale=True), # 픽셀 값을 0~1 사이의 float32로 스케일링
        v2.Normalize([0.5], [0.5])            # 평균 0.5, 표준편차 0.5로 정규화 (-1 ~ 1 범위)
    ]),
    'val': v2.Compose([
        v2.Resize((224, 224)),                # 검증 데이터는 크롭/회전 없이 리사이즈만 수행
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize([0.5], [0.5])
    ]),
    'test': v2.Compose([
        v2.Resize((224, 224)),                # 테스트 데이터도 리사이즈 및 정규화만 수행
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize([0.5], [0.5])
    ])
}

# 5. ImageFolder를 이용해 디렉토리 구조에서 자동으로 데이터셋과 레이블 로드
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

# 6. DataLoader 생성 (배치 단위로 데이터를 묶어주고 셔플 여부 결정)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)   # 학습용은 셔플 O
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)     # 검증용은 셔플 X
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)   # 테스트용은 셔플 X

# 7. 클래스 이름 확인 및 데이터셋 크기 출력
class_names = train_dataset.classes
print("Class names:", class_names)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
```

## 2. Custom CNN 모델 구현 및 학습 루프


```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1. 사용자 정의 CNN 모델 클래스 선언
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 첫 번째 합성곱 층 (입력 채널: 3(RGB), 출력 채널: 32, 커널 크기: 3x3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # 최대 풀링 층 (특징맵 크기를 절반으로 줄임)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 두 번째 합성곱 층 (입력: 32, 출력: 64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 어댑티브 평균 풀링 (이미지 크기가 달라도 항상 7x7 크기로 출력맵 고정)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # Fully Connected (Linear) 층 1 (64채널 * 7 * 7 -> 128차원)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 최종 출력 층 (128차원 -> 클래스 개수)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv1 통과 후 ReLU 활성화 함수 적용, 그리고 MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        # Conv2 통과 후 ReLU, MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        # Adaptive Average Pooling 적용
        x = self.avgpool(x)
        # 2D 텐서를 1D 벡터로 펼치기 (Flatten)
        x = torch.flatten(x, 1)
        # 첫 번째 전결합층 + ReLU
        x = F.relu(self.fc1(x))
        # 최종 출력층 (Softmax는 CrossEntropyLoss 내부에 포함되어 있으므로 원본 로짓 출력)
        x = self.fc2(x)
        return x

# 2. 모델 인스턴스 생성 및 지정한 장치(GPU/CPU)로 이동
num_classes = len(class_names)
model = CustomCNN(num_classes).to(device)

# 3. 손실 함수(CrossEntropy)와 최적화 알고리즘(Adam) 설정
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 모델 학습을 수행하는 함수
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # 모델을 학습 모드로 설정 (Dropout, BatchNorm 등이 활성화됨)
        running_loss = 0.0
        correct = 0
        total = 0

        # 미니배치 단위로 데이터 반복 로드
        for inputs, labels in train_loader:
            # 데이터를 GPU 또는 CPU 메모리로 이동
            inputs, labels = inputs.to(device), labels.to(device)

            # 이전 스텝에서 계산된 기울기(Gradient) 초기화
            optimizer.zero_grad()

            # Forward Pass: 모델에 입력 데이터를 넣고 예측값(로짓) 출력
            outputs = model(inputs)

            # 손실(Loss) 계산
            loss = criterion(outputs, labels)

            # Backward Pass: 역전파를 통해 각 파라미터의 기울기 계산
            loss.backward()

            # Optimizer Step: 기울기를 바탕으로 가중치 업데이트
            optimizer.step()

            # 통계 누적
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1) # 가장 확률이 높은 클래스 선택
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # 에폭 단위 평균 손실 및 정확도 계산
        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # 검증(Validation) 데이터셋으로 성능 평가 수행
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# 5. 모델 평가를 수행하는 함수
def evaluate(model, dataloader, loss_fn):
    model.eval()  # 모델을 평가 모드로 설정
    losses = []
    correct = 0
    total = 0

    # 평가 단계에서는 기울기를 계산할 필요가 없으므로 메모리 절약을 위해 torch.no_grad() 사용
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())

            pred_labels = torch.argmax(outputs, dim=1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

    avg_loss = sum(losses) / len(losses)
    accuracy = correct / total
    return avg_loss, accuracy

# 6. Custom CNN 학습 및 테스트 실행
train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=10)
test_loss, test_acc = evaluate(model, test_loader, loss_fn)
print(f"Test Accuracy: {test_acc:.4f}")
```

## 3. 전이 학습 (Transfer Learning)

### ① Feature Extraction (특징 추출기 동결)

<br/>


```python
import torchvision.models as models

# 1. 사전학습된 ResNet18 모델 로드 (ImageNet으로 학습된 가중치 사용)
pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. 특징 추출기(Feature Extractor)의 모든 가중치를 동결(Freeze)하여 학습 시 변경되지 않도록 함
for param in pretrained_model.parameters():
    param.requires_grad = False

# 3. 마지막 Fully Connected Layer(fc)를 우리 문제의 클래스 개수에 맞게 새로 교체 (이 층만 학습됨)
num_features = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(num_features, num_classes)
pretrained_model = pretrained_model.to(device)

loss_fn = nn.CrossEntropyLoss()
# 4. 새로 교체된 출력층 파라미터만 옵티마이저에 전달하여 학습 설정
optimizer = optim.Adam(pretrained_model.fc.parameters(), lr=0.001)

# 5. 전이 학습(Feature Extraction) 실행
train(pretrained_model, train_loader, val_loader, loss_fn, optimizer, num_epochs=5)
```

### ② Partial Fine-Tuning (부분 미세조정)


```python
pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 1. ResNet의 마지막 블록(layer4)과 분류기(fc)만 학습 가능하도록 설정하고 나머지는 동결
for name, param in pretrained_model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True  # 학습 허용
    else:
        param.requires_grad = False # 동결

# 2. 분류기(Classifier) 교체
num_features = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(num_features, num_classes)
pretrained_model = pretrained_model.to(device)

loss_fn = nn.CrossEntropyLoss()
# 3. requires_grad가 True인 파라미터들만 골라서 옵티마이저에 전달
optimizer = optim.Adam(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr=0.001)

# 4. 부분 미세조정 학습 실행
train(pretrained_model, train_loader, val_loader, loss_fn, optimizer, num_epochs=5)
```

### ③ Full Fine-Tuning (전체 미세조정)


```python
pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 1. 모델의 모든 층을 학습 가능하도록 설정 (Unfreeze)
for param in pretrained_model.parameters():
    param.requires_grad = True

# 2. 분류기 교체
num_features = pretrained_model.fc.in_features
pretrained_model.fc = nn.Linear(num_features, num_classes)
pretrained_model = pretrained_model.to(device)

criterion = nn.CrossEntropyLoss()
# 3. 전체 가중치를 미세조정하므로 기존 학습된 지식이 망가지지 않도록 매우 낮은 학습률(Learning Rate) 사용
optimizer = optim.Adam(pretrained_model.parameters(), lr=0.00001)

# 4. 전체 미세조정 학습 실행
train(pretrained_model, train_loader, val_loader, criterion, optimizer, num_epochs=5)
```

