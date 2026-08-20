---
notion-url: https://app.notion.com/p/Recurrent-Neural-Networks-for-Time-Series-3c277fa64f7f80769271fee5bf57e9eb
title: Recurrent Neural Networks for Time Series
date: '2026-08-20 09:33:00.000'
from_notion: https://app.notion.com/p/Recurrent-Neural-Networks-for-Time-Series-3c277fa64f7f80769271fee5bf57e9eb
author: rahui kim
last_edited_time: '2026-08-20 09:40:00.000'
---
[//]: # (heading_4 is not supported)

1. **예나(Jena) 날씨 데이터**를 이용한 LSTM 기반 기온 시계열 예측 (단변량, Sequence-to-Sequence)

1. **삼성전자 주가 데이터**를 이용한 RNN 기반 종가 예측 (다변량, Many-to-One)

---

[//]: # (heading_4 is not supported)

- 2009년부터 2016년까지 독일 예나(Jena) 도시의 날씨 데이터를 10분 간격으로 기록

- 14개 칼럼: 기온(°C), 기압(hPa), 습도(%), 바람 속도 등 다양한 기상 정보 포함

- 총 데이터 개수: 약 42만 개, 총 칼럼 수: 15개(`Date Time` 포함)


```python
!wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip
!unzip jena_climate_2009_2016.csv.zip

import pandas as pd
df = pd.read_csv('./jena_climate_2009_2016.csv')
df.head()
```


```python
import matplotlib.pyplot as plt

temperatures = df['T (degC)']
temperatures.index = df['Date Time']

plt.figure(figsize=(10, 5))
temperatures.plot()
plt.xticks(rotation=20)
plt.xlabel("Date Time")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Over Time")
plt.show()
```

> ⚠️ **놓치기 쉬운 포인트**

	- 이 데이터는 10분 간격으로 기록되어 있어서 1년치만 해도 데이터 포인트가 수만 개에 달합니다. 모델에 계절적 흐름까지 전부 학습시키려면 매우 긴 시퀀스가 필요하지만, "다음 시간 기온"처럼 단기 예측이 목표라면 최근 며칠 치(예: 24~72시간 분량)만 입력 윈도우로 써도 충분한 경우가 많습니다.

	- 시계열 그래프를 볼 때는 "장기 패턴(계절성)"과 "단기 변동(일교차, 노이즈)"을 구분해서 보는 습관이 중요합니다. 이 둘을 구분하지 못하면 시퀀스 길이(`sequence_length`)를 얼마로 잡아야 할지 감을 잡기 어렵습니다.

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
import numpy as np

temperatures = df[['T (degC)']].to_numpy().astype(np.float32)
temperatures = temperatures[:50000]  # 실습용으로 앞부분 5만 개만 사용

num_data = len(temperatures)
train_size = int(num_data * 0.8)
val_size = int(num_data * 0.1)

train_data = temperatures[:train_size]
val_data = temperatures[train_size:train_size + val_size]
test_data = temperatures[train_size + val_size:]
```

[//]: # (heading_4 is not supported)


```python
print(f'max: {train_data.max()}, min: {train_data.min()}')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_data)

train_data_scaled = scaler.transform(train_data)
val_data_scaled = scaler.transform(val_data)
test_data_scaled = scaler.transform(test_data)

print(f'max: {train_data_scaled.max():.4f}, min: {train_data_scaled.min():.4f}')
print(f'mean: {train_data_scaled.mean():.4f}, std: {train_data_scaled.std():.4f}')
```

- 스케일링 전: `max: 32.98, min: -23.01` (예나 지역 실제 기온 범위)

- 스케일링 후: `max: 2.62, min: -3.82`, `mean: 0.0000, std: 1.0000`

- 기온 데이터는 영하 23°C에서 영상 33°C까지 넓은 범위를 가지고 있어서, 딥러닝 모델이 경사 하강법으로 안정적이고 빠르게 수렴할 수 있도록 평균 0, 표준편차 1로 정규화하는 과정입니다. `scaler.fit()`은 반드시 학습 데이터에만 호출하고, 검증/테스트 데이터에는 `transform()`만 적용해서 데이터 누수를 방지합니다.

[//]: # (heading_4 is not supported)


```python
import torch
from torch.utils.data import Dataset

class JenaTemperatureDataset(Dataset):
    def __init__(self, temperatures, sequence_length):
        self.temperatures = temperatures
        self.sequence_length = sequence_length

    def __len__(self):
        # 원본 데이터 개수에서 시퀀스 길이를 뺀 만큼
        return len(self.temperatures) - self.sequence_length

    def __getitem__(self, index):
        inputs = self.temperatures[index:index+self.sequence_length]
        targets = self.temperatures[index+1:index+self.sequence_length+1]
        return torch.tensor(inputs), torch.tensor(targets)
```

- 0번 데이터: `Input = [t_0, ..., t_9]`, `Target = [t_1, ..., t_10]`

- 1번 데이터: `Input = [t_1, ..., t_10]`, `Target = [t_2, ..., t_11]`


```python
sequence_length = 10
train_dataset = JenaTemperatureDataset(train_data_scaled, sequence_length)
val_dataset = JenaTemperatureDataset(val_data_scaled, sequence_length)
test_dataset = JenaTemperatureDataset(test_data_scaled, sequence_length)
```

[//]: # (heading_4 is not supported)


```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=128)
test_dataloader = DataLoader(test_dataset, batch_size=128)
```

- `**train_dataloader**`**에만 **`**shuffle=True**`: 시계열 데이터는 순서대로 연속되어 있어서, 매번 같은 순서로 학습하면 모델이 특정 순서에 과도하게 적응(오버피팅)할 수 있습니다. 에폭마다 섞어주면 시계열의 본질적인 패턴을 더 골고루 학습해서 일반화 성능이 좋아집니다.

- **검증/테스트에는 **`**shuffle**`** 안 함**: 검증과 테스트는 "실제로 시간 순서대로 흘러가는 미래를 얼마나 정확히 예측하는가"를 공정하게 평가하는 단계입니다. 섞어버리면 시간의 연속성이 깨져서 평가 결과가 왜곡됩니다.

> ⚠️ **놓치기 쉬운 포인트**

	- `Dataset`을 상속받은 `JenaTemperatureDataset`은 `__init__()`에서 `super().__init__()`을 호출하지 않아도 정상 작동합니다. `torch.utils.data.Dataset`은 별도로 초기화해야 할 내부 상태가 없는, 일종의 "규격 인터페이스" 역할만 하기 때문입니다. 다만 관례적으로 적어주는 경우도 많으니, 다른 사람 코드에서 보더라도 이상하게 여기지 마세요.

	- `__getitem__` 같은 코드는 모델(`nn.Module`) 클래스 안에는 들어가지 않습니다. `Dataset`은 "데이터를 어떻게 잘라서 꺼낼지" 결정하는 역할, 모델은 "이미 잘려서 들어온 배치를 받아 연산하는" 역할로 완전히 분리되어 있기 때문입니다.

---

[//]: # (heading_4 is not supported)


```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,     # 한 시점에 들어오는 피처 차원 (기온 단일 변수이므로 1)
            hidden_size=32,   # LSTM 내부 은닉 상태의 차원 (기억 용량)
            num_layers=2,     # LSTM 레이어를 2층으로 쌓음
            batch_first=True, # 입력 형태를 [Batch, Seq_Len, Feature]로 고정
        )
        self.linear = nn.Linear(32, 1)  # 32차원 은닉 상태를 1차원(기온 예측값)으로 변환

    def forward(self, x):
        # x: [Batch, Seq_Len, 1]
        lstm_output, _ = self.lstm(x)     # lstm_output: [Batch, Seq_Len, 32] (모든 타임스텝의 출력)
        output = self.linear(lstm_output) # output: [Batch, Seq_Len, 1] (타임스텝별 예측 기온)
        return output

model = LSTMModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```


```python
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
```


```python
epochs = 5
step = 0

for epoch in range(epochs):
    model.train()
    for train_batch in train_dataloader:
        inputs = train_batch[0].to(device)
        targets = train_batch[1].to(device)

        preds = model(inputs)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % 100 == 0:
            print(f'step {step}, train loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        losses = []
        for val_batch in val_dataloader:
            inputs = val_batch[0].to(device)
            targets = val_batch[1].to(device)
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            losses.append(loss.item())

    val_loss_avg = sum(losses) / len(losses)
    print(f'\nepoch {epoch+1}/{epochs}, val loss: {val_loss_avg:.4f}\n')
```

> ⚠️ **놓치기 쉬운 포인트**

	- `nn.LSTM`의 반환값은 `(lstm_output, (hidden, cell))` 형태의 튜플입니다. 위 코드에서 `lstm_output, _ = self.lstm(x)`처럼 두 번째 자리를 언더스코어(`_`)로 받아 버리고 있는데, 이는 "이번 모델에서는 마지막 은닉/셀 상태가 필요 없고, 모든 타임스텝의 출력(`lstm_output`)만 쓰겠다"는 의도입니다.

	- 이 모델은 Sequence-to-Sequence 구조라서 `linear` 레이어가 `lstm_output`의 **모든 타임스텝**에 적용됩니다. `nn.Linear`는 입력의 마지막 차원(여기서는 32)에만 작용하고 앞쪽 차원(배치, 시퀀스 길이)은 그대로 유지하기 때문에, `[Batch, Seq_Len, 32]` → `[Batch, Seq_Len, 1]`로 변환되는 것이 자연스럽게 가능합니다.

---

[//]: # (heading_4 is not supported)


```python
test_preds = []
model.eval()

with torch.no_grad():
    for test_batch in test_dataloader:
        inputs = test_batch[0].to(device)
        preds = model(inputs)  # [batch, seq_len, 1]

        # 시퀀스의 마지막 타임스텝(-1) 예측값만 추출
        test_preds.append(preds[:, -1, :])

    test_preds = torch.cat(test_preds, dim=0)
    test_preds = test_preds.cpu().numpy()

    # 표준화(StandardScaler)로 변환됐던 값을 원래 섭씨 단위로 되돌림
    test_preds = scaler.inverse_transform(test_preds)
```


```python
import matplotlib.pyplot as plt

test_targets = test_data[sequence_length:]

plt.plot(test_targets[:200], color='blue', label='Actual Temperature')
plt.plot(test_preds[:200], color='red', label='Predicted Temperature')
plt.legend()
plt.show()
```

> ⚠️ **놓치기 쉬운 포인트**

	- Sequence-to-Sequence 모델의 출력 `preds`는 `[batch, seq_len, 1]`처럼 시퀀스 전체에 대한 예측을 담고 있습니다. 이 실습처럼 "다음 한 시점"만 평가하고 싶다면 `preds[:, -1, :]`로 **마지막 타임스텝만** 꺼내야 합니다. 시퀀스 전체를 그대로 쓰면 평가 지표가 무엇을 의미하는지 헷갈리게 됩니다.

	- `scaler.inverse_transform()`을 잊으면 예측값이 표준화된 값(평균 0, 표준편차 1 근처의 작은 숫자)으로 남아있어서, 실제 섭씨 온도와 비교할 때 그래프가 완전히 어긋나 보입니다. 정규화를 했다면 시각화 직전에 반드시 역변환을 해줘야 합니다.

	- 이 그래프가 "너무 잘 맞아서 이상하다"고 느낄 수 있는데, 10분 뒤라는 아주 가까운 미래를 예측하는 것이므로 직전 값과 크게 다르지 않은 경우가 많다는 점을 감안해야 합니다. 먼 미래를 예측할수록 이런 정확도는 유지되기 어렵습니다.

---

[//]: # (heading_4 is not supported)

- **실제 데이터 연동**: `yfinance` 라이브러리를 통해 실시간 주가 데이터를 받아옵니다.

- **데이터 정규화**: 주가·거래량처럼 단위가 매우 큰 데이터는 `MinMaxScaler`로 0~1 사이로 압축하는 것이 사실상 필수적입니다.

- **다변량 입력, 단변량 출력**: 5개(시가, 고가, 저가, 종가, 거래량) 특징을 보고 1개(다음 날 종가)만 예측하는 **Many-to-One** 구조입니다. 앞서 다룬 기온 예측의 Sequence-to-Sequence 구조와는 다릅니다.

[//]: # (heading_4 is not supported)


```python
!pip install yfinance -q

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

stock_code = "005930.KS"  # 삼성전자
df = yf.download(stock_code, start="2020-01-01", end="2023-12-31")

data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
print(f"데이터 크기: {data.shape}")  # (일수, 5)
```


```python
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
```


```python
def create_dataset(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        # 입력: 과거 30일치 5개 지표 (형태: [30, 5])
        x_block = data[i : i + seq_length]
        # 정답: 30일 다음 날의 종가(Close, 3번 인덱스) 단 하나
        y_block = data[i + seq_length][3]

        xs.append(x_block)
        ys.append(y_block)

    return np.array(xs), np.array(ys)

seq_length = 30
X, Y = create_dataset(data_scaled, seq_length)

# 시간 순서를 유지한 채 80% 학습 / 20% 테스트로 분할
train_size = int(len(X) * 0.8)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).unsqueeze(1)

print(f"학습 데이터: {X_train_tensor.shape}")  # (N, 30, 5)
print(f"테스트 데이터: {X_test_tensor.shape}")  # (M, 30, 5)
```

> ⚠️ **놓치기 쉬운 포인트**

	- `Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)`에서 `.unsqueeze(1)`을 빼먹으면, `Y_train`은 `[N]` 형태의 1차원 텐서로 남습니다. 모델의 예측값은 `[N, 1]` 형태로 나오기 때문에, 손실 함수 계산 시 두 텐서의 차원이 맞지 않아 브로드캐스팅 경고가 뜨거나 의도치 않은 결과가 나올 수 있습니다.

	- 시계열 데이터를 학습/테스트로 나눌 때는 **반드시 시간 순서를 유지한 채 앞쪽을 학습, 뒤쪽을 테스트**로 나눠야 합니다. `train_test_split()`처럼 무작위로 섞는 함수를 쓰면 미래 데이터가 학습에 섞여 들어가는 데이터 누수가 발생합니다.

[//]: # (heading_4 is not supported)


```python
class StockRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StockRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, hidden = self.rnn(x)   # out: [Batch, Seq_Len, hidden_dim]
        last_out = out[:, -1, :]     # 마지막 타임스텝(30일째)의 출력만 추출
        prediction = self.fc(last_out)
        return prediction

input_dim = 5
hidden_dim = 64
output_dim = 1
learning_rate = 0.01
epochs = 100

model = StockRNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_history = []
for i in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if i % 10 == 0:
        print(f"Epoch {i}/{epochs}, Loss: {loss.item():.4f}")

plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.show()
```

> ⚠️ **놓치기 쉬운 포인트**

	- 이 예제는 `train_dataloader`로 미니배치를 나누지 않고, `X_train_tensor` 전체를 한 번에 모델에 통과시키는 **풀배치(full-batch) 학습** 방식입니다. 데이터가 상대적으로 작을 때는 이렇게 학습해도 되지만, 데이터가 커지면 메모리 문제로 `DataLoader`를 이용한 미니배치 학습이 필요합니다.

	- `out[:, -1, :]`로 마지막 타임스텝만 꺼내는 방식은 `nn.RNN`이 반환하는 두 번째 값(`hidden`, 마지막 시점의 최종 은닉 상태)과 사실상 같은 값입니다. `hidden[-1]`을 대신 사용해도 결과는 동일합니다.

[//]: # (heading_4 is not supported)


```python
model.eval()
with torch.no_grad():
    predicted = model(X_test_tensor).numpy()

plt.figure(figsize=(12, 6))
plt.plot(Y_test, label='Actual Price (Scaled)')
plt.plot(predicted, label='Predicted Price (Scaled)', linestyle='--')
plt.title('Samsung Electronics Stock Prediction (Test Set)')
plt.legend()
plt.show()
```

> ⚠️ **놓치기 쉬운 포인트**

	- 그래프의 값들이 실제 원화(KRW) 단위가 아니라 `MinMaxScaler`로 0~1 사이로 정규화된 값이라는 점에 주의하세요. 실제 가격 단위로 보고 싶다면 `scaler.inverse_transform()`으로 역변환해야 하는데, 이때는 5개 컬럼(시가·고가·저가·종가·거래량) 전체 구조에 맞춰 역변환해야 하므로 종가 하나만 따로 역변환하려면 나머지 컬럼을 더미 값으로 채우는 등의 추가 작업이 필요합니다.

	- "예측이 실제와 거의 똑같아 보여서 모델이 주가를 정말 잘 맞춘 것 같다"고 오해하기 쉬운데, 변곡점에서의 지연 현상(Lagging)은 이 모델이 사실상 "어제와 비슷한 값을 오늘도 예측"하는 것에 가깝다는 신호이기도 합니다. 시계열 예측 결과를 해석할 때는 항상 이런 지연 현상이 있는지 함께 확인하는 습관이 중요합니다.

---

[//]: # (heading_4 is not supported)

- [Jena Climate Dataset - Kaggle](https://www.kaggle.com/datasets/mnassrib/jena-climate)

- [torch.nn.LSTM - Official Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

- [torch.nn.RNN - Official Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)

- [torch.optim.Adam - Official Documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)

- [sklearn.preprocessing.StandardScaler - Official Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

- [sklearn.preprocessing.MinMaxScaler - Official Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

- [yfinance - PyPI Package Page](https://pypi.org/project/yfinance/)

