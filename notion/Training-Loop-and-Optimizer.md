# Training-Loop-and-Optimizer

개요

전복(abalone) 데이터를 기반으로 회귀 문제를 푸는 과정을 통해 데이터 준비, 모델 생성, 학습 루프의 뼈대를 살펴봅니다.

데이터 준비

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

abalone_df = pd.read_csv(
    'https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv',
    names=['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
           'Viscera weight', 'Shell weight', 'Age']
)

input_data = abalone_df.drop(columns=['Age']).to_numpy().astype(np.float32)
target_data = abalone_df['Age'].to_numpy().astype(np.float32)

class AbaloneDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.input_data[index], dtype=torch.float32)
        target_tensor = torch.tensor(self.target_data[index], dtype=torch.float32)
        return input_tensor, target_tensor

# 학습/검증/테스트 분할 (80% / 10% / 10%)
train_size = int(len(input_data) * 0.8)
val_size = int(len(input_data) * 0.1)

train_inputs, train_targets = input_data[:train_size], target_data[:train_size]
val_inputs = input_data[train_size:train_size + val_size]
val_targets = target_data[train_size:train_size + val_size]
test_inputs = input_data[train_size + val_size:]
test_targets = target_data[train_size + val_size:]

# 표준화(학습 데이터 기준으로 fit)
scaler = StandardScaler()
scaler.fit(train_inputs)
train_inputs_scaled = scaler.transform(train_inputs)
val_inputs_scaled = scaler.transform(val_inputs)
test_inputs_scaled = scaler.transform(test_inputs)

train_dataset = AbaloneDataset(train_inputs_scaled, train_targets)
val_dataset = AbaloneDataset(val_inputs_scaled, val_targets)
test_dataset = AbaloneDataset(test_inputs_scaled, test_targets)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, drop_last=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=32)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32)

모델 정의

class AbaloneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)   # 회귀 문제이므로 마지막에는 활성화 함수를 적용하지 않음
        return x

model = AbaloneModel()

구조: Fully Connected Layers로 구성. 첫 레이어는 입력 피처 7개, 출력 32개. 마지막 레이어는 출력 1개로 나이를 예측.

활성화 함수: 은닉층에는 ReLU 사용.

출력: 회귀 문제에 적합하게 마지막 레이어에서는 활성화 함수를 적용하지 않음.

트레이닝 루프의 기본 뼈대

epochs = 10
for epoch in range(epochs):
    for train_batch in train_dataloader:
        x_train, y_train = train_batch
        pred = model(x_train)
        # pred와 y_train 사이에서 loss 계산
        # loss.backward()로 gradient 계산
        # optimizer.step()으로 모델 파라미터 업데이트

설명

데이터 루프: train_dataloader에서 배치를 하나씩 가져옵니다.

에폭 루프: 데이터 전체를 반복 학습하기 위해 바깥쪽에 에폭(epoch) 루프를 둡니다.

예측 및 업데이트: 예측값(pred)과 실제값(y_train)으로 손실을 계산하고, 그 손실을 기반으로 모델 파라미터를 업데이트합니다.

Loss 계산하기

MSE 손실 함수 (회귀 문제)

전복 데이터는 회귀 문제이므로 평균제곱오차(MSE, Mean Squared Error)를 손실 함수로 사용합니다.

loss_fn = nn.MSELoss()

epochs = 10
for epoch in range(epochs):
    for train_batch in train_dataloader:
        x_train, y_train = train_batch
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        # loss를 통해 gradient 계산
        # gradient를 활용해 모델 파라미터 업데이트

예측값 계산: pred = model(x_train) — 입력 데이터를 모델에 통과시켜 예측값을 계산합니다.

손실 값 계산: loss = loss_fn(pred, y_train) — MSE 손실 함수 객체에 모델 예측값과 실제 타깃값을 넣어 손실 값을 계산합니다. 예측값(pred)과 타깃값(y_train)의 순서를 바꾸지 않도록 주의해야 합니다.

shape 불일치 경고와 squeeze()

loss = loss_fn(pred, y_train)
print(loss)

예측값 텐서(pred)는 크기가 (32, 1)이고, 타깃값 텐서(y_train)는 크기가 (32,)입니다.

두 텐서의 크기가 다르면 PyTorch가 Tensor-Operations-Broadcasting|브로드캐스팅을 수행하며, 이로 인해 경고 메시지가 표시됩니다.

예측값 텐서에서 불필요한 차원을 제거하기 위해 squeeze() 메서드를 사용하면 경고가 사라집니다.

pred = model(x_train).squeeze()  # (32, 1) → (32,)
loss = loss_fn(pred, y_train)
print(loss)  # 정상적으로 손실 값이 출력됨, 경고 메시지 없음

다른 손실 함수들

평균 제곱 오차(MSE) — 직접 확인

inputs = torch.tensor([2.0, 2.0, 2.0](./2.0, 2.0, 2.0.md))
targets = torch.tensor([0.0, 0.0, 1.0](./0.0, 0.0, 1.0.md))

loss_fn = nn.MSELoss()
loss = loss_fn(inputs, targets)  # tensor(3.)

크로스 엔트로피(Cross Entropy) — 다중 분류 문제

inputs = torch.randn(4, 3)          # 예측 값은 확률이 아닌 로짓(logit)
targets = torch.tensor([1, 1, 2, 0])  # 레이블은 정수 인덱스로 표현

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(inputs, targets)

크로스 엔트로피는 분류 문제에서 사용되는 손실 함수입니다. 모델이 예측한 클래스 분류 확률과 실제 타깃이 얼마나 다른지를 측정합니다.

분류 모델은 마지막에 softmax 활성화 함수를 적용하면 확률을 출력하고, 그러지 않으면 확률로 바꾸지 않은 원본 값인 로짓(logit) 을 출력합니다. nn.CrossEntropyLoss는 예측 값으로 로짓을 입력받도록 되어 있습니다. 즉, 모델의 forward()를 정의할 때 마지막에 따로 활성화 함수를 적용하지 않고 텐서를 그대로 출력하면 됩니다.

정답 레이블은 정수 인덱스로 나타내야 합니다. One-hot 인코딩된 형태로 표현하면 안 됩니다.

예시는 클래스 개수가 3개인 분류 문제에서, 배치 크기가 4인 데이터에 대해 크로스 엔트로피 값을 구하는 코드입니다.

이진 크로스 엔트로피(Binary Cross Entropy) — 이진 분류 문제

inputs = torch.randn(4)  # 예측 값은 확률이 아닌 로짓
targets = torch.tensor(
    [1, 1, 0, 0],
    dtype=torch.float32
)  # 레이블은 0 또는 1만 갖는 float 텐서

loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(inputs, targets)

이진 크로스 엔트로피는 클래스가 두 개인 이진 분류에서 사용하는 손실 함수입니다. 영어 표현의 첫 글자를 따 BCE라고도 부릅니다.

이진 분류 모델은 마지막에 sigmoid 활성화 함수를 적용하면 확률을 출력하고, 그러지 않으면 원본 값인 로짓을 출력합니다. nn.BCEWithLogitsLoss는 이름에서 짐작할 수 있듯이 예측 값으로 로짓을 입력받도록 되어 있습니다. 따라서 forward()를 정의할 때 마지막에 sigmoid를 적용하지 않고 텐서를 그대로 출력해야 합니다.

정답 레이블은 0 또는 1로 표현하되, 반드시 실수형(float) 텐서여야 합니다. (정수형이면 에러가 날 수 있습니다.)

손실 함수 요약 비교

Gradient 계산과 Optimizer

Gradient(그래디언트) 계산의 원리(requires_grad, .backward(), torch.no_grad())는 Autograd-Gradient-Computation 노트에서 자세히 다룹니다. 여기서는 학습 루프 안에서 실제로 어떻게 쓰이는지 정리합니다.

Optimizer 개념

PyTorch는 다양한 최적화 알고리즘을 제공하며, 이를 Optimizer 객체로 관리합니다. Optimizer는 자동 미분(autograd)으로 계산된 그래디언트를 활용하여 모델 파라미터를 업데이트합니다.

import torch.optim as optim

SGD 옵티마이저 만들기

# 기본 생성
optimizer = optim.SGD(model.parameters())

# 하이퍼파라미터를 함께 지정 (학습률 lr, 모멘텀 momentum)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model.parameters(): 모델이 가진 학습 가능한 모든 파라미터를 가져옵니다.

Optimizer로 파라미터 업데이트하기

파라미터 업데이트: Optimizer의 step() 메서드를 호출하면, 계산된 그래디언트를 바탕으로 모델 파라미터가 업데이트됩니다.

그래디언트 초기화: Optimizer의 zero_grad() 메서드를 호출해 기존 그래디언트를 초기화합니다. PyTorch에서는 그래디언트가 누적되므로, 각 배치 처리 후 초기화가 필요합니다. (자세한 이유는 Autograd-Gradient-Computation 참고)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epochs = 10
for epoch in range(epochs):
    for train_batch in train_dataloader:
        x_train, y_train = train_batch
        pred = model(x_train).squeeze()
        loss = loss_fn(pred, y_train)
        loss.backward()       # 그래디언트 계산
        optimizer.step()      # 파라미터 업데이트
        optimizer.zero_grad() # 그래디언트 초기화

트레이닝 루프 실행 후, 전복 데이터를 기반으로 한 회귀 모델이 10 에폭 동안 학습을 완료합니다.

Optimizer 주요 메서드

그래디언트 초기화가 중요한 이유

PyTorch에서는 그래디언트가 누적됩니다.

초기화를 하지 않으면 이전 배치의 그래디언트가 현재 배치의 그래디언트에 합산됩니다.

이를 방지하기 위해 각 배치 처리 후 반드시 zero_grad()를 호출해야 합니다.

Training Loop 정리 — 검증/평가까지 포함한 완전한 버전

실무에서 쓰는 완전한 학습 루프는 학습(train) 뿐 아니라 매 에폭마다 검증(validation) 도 함께 수행합니다.

import torch.optim as optim

class AbaloneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()   # 과적합 방지를 위한 드롭아웃

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.dropout(x)           # 드롭아웃 적용
        x = self.fc4(x)
        return x

model = AbaloneModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 10
step = 0
for epoch in range(epochs):
    model.train()  # 학습 모드 (Dropout, BatchNorm 등이 학습 방식으로 동작)
    for train_batch in train_dataloader:
        x_train, y_train = train_batch[0].to(device), train_batch[1].to(device)
        pred = model(x_train).squeeze()
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % 100 == 0:
            print(f'step {step}, train loss: {loss.item():.4f}')

    model.eval()  # 평가 모드 (Dropout, BatchNorm 등이 평가 방식으로 동작)
    with torch.no_grad():  # 검증 시에는 그래디언트 계산 불필요
        losses = []
        for val_batch in val_dataloader:
            x_val, y_val = val_batch[0].to(device), val_batch[1].to(device)
            pred = model(x_val).squeeze()
            loss = loss_fn(pred, y_val)
            losses.append(loss.item())

        val_loss_avg = sum(losses) / len(losses)
        print(f'epoch {epoch+1}/{epochs}, validation loss: {val_loss_avg:.4f}\n')

핵심 포인트

model.train() / model.eval(): Dropout이나 BatchNorm처럼 학습 때와 평가 때 동작이 달라지는 레이어가 있는 경우, 반드시 상황에 맞게 모드를 전환해줘야 합니다. train()은 학습 모드, eval()은 평가(추론) 모드로 전환합니다.

torch.no_grad(): 검증/평가 단계에서는 역전파가 필요 없으므로 with torch.no_grad(): 블록으로 감싸서 불필요한 메모리·연산 낭비를 막습니다.

loss.item(): 손실 텐서에서 순수한 파이썬 숫자 값만 뽑아낼 때 사용합니다. 리스트에 모아서 평균을 낼 때 유용합니다.

테스트 데이터로 최종 평가하기

model.eval()
with torch.no_grad():
    losses = []
    for test_batch in test_dataloader:
        x_test, y_test = test_batch[0].to(device), test_batch[1].to(device)
        pred = model(x_test).squeeze()
        loss = loss_fn(pred, y_test)
        losses.append(loss.item())

    test_loss_avg = sum(losses) / len(losses)
    print(f'test loss: {test_loss_avg:.4f}\n')

model.eval(): 모델을 평가 모드로 전환.

torch.no_grad(): 연산 기록 비활성화로 메모리와 연산 효율 향상.

테스트 데이터 평가: test_dataloader에서 배치를 순회하며 예측값(pred)과 타깃값(y_test)의 손실(loss)을 계산하고, 모든 배치의 손실 값을 평균 내어 테스트 데이터의 평균 손실(test_loss_avg)을 출력합니다.

테스트와 검증 데이터의 손실은 모두 MSE로 계산되었으므로, 제곱근을 사용해 RMSE(Root Mean Squared Error) 값으로 변환하면 원래 단위(나이)로 해석하기 쉬워집니다.

print(f'val RMSE: {np.sqrt(val_loss_avg):.4f}')
print(f'test RMSE: {np.sqrt(test_loss_avg):.4f}')

예측값 vs 실제값 시각화

model.eval()
with torch.no_grad():
    preds, targets = [], []
    for test_batch in test_dataloader:
        x_test, y_test = test_batch[0].to(device), test_batch[1].to(device)
        pred = model(x_test).squeeze()
        preds.extend(pred.cpu().numpy())   # GPU 텐서를 CPU로 이동 후 NumPy 변환
        targets.extend(y_test.cpu().numpy())

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(targets, preds, alpha=0.7, label='Predicted vs True', edgecolor='k')
min_val, max_val = min(min(targets), min(preds)), max(max(targets), max(preds))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
plt.title('Predicted vs True Values')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

파란 점: 모델이 예측한 값과 실제 타깃값의 산점도.

빨간 점선(y = x): 이상적인 예측 결과를 나타냄.

점들이 점선에 가까울수록 예측이 잘 된 것이며, 특정 구간에서 점들이 점선에서 많이 벗어나 있다면 그 구간에서 과소/과대 예측이 일어나고 있는 것으로 해석할 수 있습니다. ![Pasted image 20260805141841.png](./Pasted image 20260805141841.png.md)

모델 저장과 불러오기

학습한 모델을 저장하고 다시 불러오면, 학습된 결과를 재사용하고 효율적으로 실험을 반복할 수 있습니다.

모델 저장 — torch.save()

모델 저장은 보통 모델의 파라미터(state_dict) 만 저장합니다.

torch.save(model.state_dict(), 'model.pth')

model.state_dict(): 모델의 파라미터 정보를 담고 있는 딕셔너리.

'model.pth': 저장될 파일의 경로.

모델 불러오기 — torch.load()

state_dict_loaded = torch.load('model.pth')

결과는 파라미터 이름과 값으로 구성된 딕셔너리(state_dict_loaded)입니다.

1) 동일한 모델 구조 생성

저장된 파라미터는 모델 구조와 별개이므로, 파라미터를 불러오기 전에 동일한 모델 구조를 다시 만들어야 합니다.

model_loaded = AbaloneModel()

2) 파라미터 로드

load_state_dict() 메서드를 사용해 저장된 파라미터를 모델에 로드합니다.

model_loaded.load_state_dict(state_dict_loaded)

키 매칭: 저장된 파라미터 이름과 모델의 파라미터 이름이 정확히 매칭되어야 로드가 성공합니다. (모델 구조가 저장 당시와 다르면 에러가 납니다.)

3) 디바이스 설정

불러온 모델을 GPU 또는 CPU에 맞게 설정합니다.

