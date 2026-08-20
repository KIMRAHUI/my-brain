---
notion-url: https://app.notion.com/p/RNN-NLP-3c277fa64f7f8097ab55e8afab7de5d6
title: RNN & NLP
date: '2026-08-20 09:33:00.000'
from_notion: https://app.notion.com/p/RNN-NLP-3c277fa64f7f8097ab55e8afab7de5d6
author: rahui kim
last_edited_time: '2026-08-20 09:41:00.000'
---
[//]: # (heading_4 is not supported)

1. **문장 성분 태깅** — 매우 작은 장난감(toy) 데이터로 RNN의 입출력 구조와 내부 가중치 형태를 직접 확인하는 실습 (Many-to-Many)

1. **한국어 영화 리뷰 감성 분류** — 실제 데이터셋(NSMC)을 형태소 분석 → 정수 인코딩 → 임베딩 → LSTM 순서로 처리해 긍정/부정을 분류하는 실습 (Many-to-One)

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
# 단어 벡터
word_to_vec = {
    "John": [1, 0, 0, 0],
    "loves": [0, 1, 0, 0],
    "cats": [0, 0, 1, 0],
    "dogs": [0, 0, 0, 1]
}
```

[//]: # (heading_4 is not supported)


```python
# 입력 데이터
train_X = np.array([
    [word_to_vec["John"], word_to_vec["loves"], word_to_vec["cats"]],
    [word_to_vec["John"], word_to_vec["loves"], word_to_vec["dogs"]]
], dtype=np.float32)

# 타겟 데이터
S, V, O = 0, 1, 2

train_Y = np.array([
    [S, V, O],
    [S, V, O]
], dtype=np.int64)
```

[//]: # (heading_4 is not supported)


```python
# 하이퍼파라미터
num_classes = 3       # 출력 클래스 개수 - S, V, O
input_dim = 4          # 단어 벡터 차원 - [1, 0, 0, 0]
sequence_length = 3    # 문장의 길이 - 각 문장 단어 3개
hidden_dim = 3          # RNN hidden state 차원 - 단어 정보를 내부적으로 압축 표현 (자유롭게 조정 가능한 값)
learning_rate = 0.1
batch_size = 2
```

[//]: # (heading_4 is not supported)


```python
# PyTorch 모델 정의
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc(out)
        return out

model = SimpleRNN(input_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```


```python
# 데이터를 Tensor로 변환
train_X_tensor = torch.tensor(train_X)
train_Y_tensor = torch.tensor(train_Y)

# 학습
model.train()
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(train_X_tensor)
    # output은 (2,3,3)!
    # RNN 입력 데이터 (배치 크기 = 2, 시퀀스 길이 = 3, 단어 벡터 차원 = 4)
    # RNN 출력 데이터 (배치 크기 = 2, 시퀀스 길이 = 3, hidden_dim = 3)
    # FC 입력 (2,3,hidden_dim) FC 출력 (2,3,num_classes=3) → 최종 (2,3,3)
    loss = criterion(output.view(-1, num_classes), train_Y_tensor.view(-1))
    # output.view(-1, num_classes): (2,3,3) → (6,3)로 펼쳐 정렬
    # train_Y_tensor.view(-1): (2,3) → (6,)로 펼쳐 정답 레이블 정렬
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```


```plain text
Epoch [10/50], Loss: 0.5003
Epoch [20/50], Loss: 0.1579
Epoch [30/50], Loss: 0.0086
Epoch [40/50], Loss: 0.0022
Epoch [50/50], Loss: 0.0013
```


```python
model.eval()
with torch.no_grad():
    predictions = model(train_X_tensor)
    predicted_classes = torch.argmax(predictions, dim=2)

idx2tag = ['S', 'V', 'O']
for i, pred in enumerate(predicted_classes):
    pred_list = pred.numpy()
    result_str = [idx2tag[c] for c in pred_list]
    print(f"Sentence {i+1} Prediction: {' '.join(result_str)}")
```


```plain text
Sentence 1 Prediction: S V O
Sentence 2 Prediction: S V O
```

> ⚠️ **놓치기 쉬운 포인트**

	- `output.view(-1, num_classes)`와 `train_Y_tensor.view(-1)`은 **문장 단위로 나뉘어 있던 배치를 "단어 단위"로 한 줄로 쭉 펼치는** 작업입니다. `nn.CrossEntropyLoss`는 `(N, class_개수)` 형태의 예측값과 `(N,)` 형태의 정답을 기대하기 때문에, `(문장 수, 단어 수, class 수)`처럼 3차원인 출력을 그대로 넣으면 안 되고 `(문장 수 × 단어 수, class 수)`로 펼쳐줘야 합니다. 이렇게 펼치면 "문장이 어디서 끝나고 시작하는지" 구분 없이, 전체 단어 6개(문장 2개 × 단어 3개)를 한 줄로 놓고 한 번에 손실을 계산하는 것과 같아집니다.

	- `hidden_dim`은 정답이 정해진 값이 아니라 직접 조정할 수 있는 하이퍼파라미터입니다. 이 값을 늘리면 모델이 더 많은 정보를 기억할 수 있지만, 데이터가 매우 작은 이 실습에서는 크게 늘려도 성능 차이가 거의 없거나 오히려 과적합될 수 있습니다.

	- `torch.argmax(predictions, dim=2)`에서 `dim=2`를 지정한 이유는, `predictions`의 형태가 `(문장 수, 단어 수, class 수)`이고 이 중 마지막 축(class 수)에서 가장 확률이 높은 인덱스를 뽑아야 하기 때문입니다. `dim`을 잘못 지정하면 엉뚱한 축을 기준으로 최댓값을 찾게 됩니다.

---

[//]: # (heading_4 is not supported)


```python
# (참고용) 모델 가중치 확인
print("\nModel Summary:")
for name, param in model.named_parameters():
    print(f"{name} => {param.shape}")
```


```plain text
Model Summary:
rnn.weight_ih_l0 => torch.Size([3, 4])
rnn.weight_hh_l0 => torch.Size([3, 3])
rnn.bias_ih_l0 => torch.Size([3])
rnn.bias_hh_l0 => torch.Size([3])
fc.weight => torch.Size([3, 3])
fc.bias => torch.Size([3])
```

 | 이름 | 의미 | 형태 | 계산식 | 
 | ---- | ---- | ---- | ---- | 
 | `weight_ih_l0` (Wx) | 입력(x)을 은닉 상태로 변환하는 가중치 | `(hidden_dim, input_dim)` → `(3, 4)` | — | 
 | `weight_hh_l0` (Wh) | 이전 은닉 상태(h)를 현재 은닉 상태로 변환하는 가중치 | `(hidden_dim, hidden_dim)` → `(3, 3)` | — | 
 | `bias_ih_l0` | 입력 변환에 더해지는 편향 | `(hidden_dim,)` → `(3,)` | — | 
 | `bias_hh_l0` | 은닉 상태 변환에 더해지는 편향 | `(hidden_dim,)` → `(3,)` | — | 
 | `fc.weight` (Wy) | 은닉 상태를 최종 출력(class 개수)으로 변환하는 가중치 | `(output_dim, hidden_dim)` → `(3, 3)` | — | 
 | `fc.bias` | 최종 출력에 더해지는 편향 | `(output_dim,)` → `(3,)` | — | 


```python
!pip install torchinfo
from torchinfo import summary
summary(model, input_size=(2, 3, 4))
```


```plain text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
SimpleRNN                                [2, 3, 3]                 --
├─RNN: 1-1                               [2, 3, 3]                 27
├─Linear: 1-2                            [2, 3, 3]                 12
==========================================================================================
Total params: 39
```

> ⚠️ **놓치기 쉬운 포인트**

	- `weight_ih_l0`, `weight_hh_l0`처럼 이름에 붙은 `_l0`는 "0번째 레이어(layer 0)"라는 뜻입니다. `num_layers`를 2 이상으로 늘리면 `weight_ih_l1`, `weight_hh_l1`처럼 레이어 번호가 붙은 파라미터가 추가로 생깁니다.

	- `torchinfo`의 `summary()`는 실제 데이터를 넣지 않고 `input_size`만 지정해도 모델 구조와 파라미터 개수를 미리 확인할 수 있어서, 모델을 학습시키기 전에 설계가 의도한 대로 되어 있는지 검증하는 용도로 유용합니다.

	- RNN이 편향(bias)을 `bias_ih`와 `bias_hh` 두 개로 나눠서 가지고 있는 점이 특이한데, 이는 PyTorch의 구현 방식일 뿐이고 수학적으로는 두 편향을 더한 값 하나로도 표현할 수 있습니다.

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
!pip install Korpora
!pip install konlpy

import pandas as pd
from Korpora import Korpora

# NSMC(Naver Sentiment Movie Corpus) 데이터 로드
corpus = Korpora.load("nsmc")
```


```python
# 데이터를 DataFrame 형태로 변환
corpus_df = pd.DataFrame(corpus.test)
corpus_df
```


```plain text
                                                   text  label
0                                                   굳 ㅋ      1
1                                  GDNTOPCLASSINTHECLUB      0
2                뭐야 이 평점들은.... 나쁘진 않지만 10점 짜리는 더더욱 아니잖아      0
...
[50000 rows x 2 columns]
```

[//]: # (heading_4 is not supported)


```python
# 데이터를 학습(train)과 테스트(test)로 분리 (90% 학습, 10% 테스트)
train = corpus_df.sample(frac=0.9, random_state=42)
test = corpus_df.drop(train.index)

print("Training Data Size :", len(train))  # 45000
print("Testing Data Size :", len(test))    # 5000
```

[//]: # (heading_4 is not supported)


```python
from konlpy.tag import Okt  # 토크나이저
from collections import Counter

# 단어 사전을 구축하는 함수
def build_vocab(corpus, n_vocab, special_tokens):
    # corpus: 형태소 분석이 끝난 전체 문장 리스트 (train_tokens)
    # n_vocab: 보존할 단어 수 (예: 5000개)
    # special_tokens: <pad>, <unk> 등 특수 토큰을 단어 목록 맨 앞에 추가
    counter = Counter()
    for tokens in corpus:
        counter.update(tokens)  # 단어 등장 횟수 세기

    vocab = special_tokens  # 특수 토큰 추가
    for token, count in counter.most_common(n_vocab):  # 가장 많이 등장하는 단어 n_vocab개 선택
        vocab.append(token)
    return vocab

# 형태소 분석기 초기화
tokenizer = Okt()

# 텍스트를 형태소(단어) 단위로 분리
train_tokens = [tokenizer.morphs(review) for review in train.text]
test_tokens = [tokenizer.morphs(review) for review in test.text]

# 단어 사전 생성 (가장 많이 등장한 5000개 단어)
vocab = build_vocab(corpus=train_tokens, n_vocab=5000, special_tokens=["<pad>", "<unk>"])

# 단어와 ID 매핑
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}

print(vocab[:10])  # ['<pad>', '<unk>', '.', '이', '영화', '의', '..', '가', '에', '...']
print(len(vocab))  # 5002
```

- `<pad>`: 문장 길이를 통일하기 위해 채워 넣는 의미 없는 토큰. 예: `['감동', '있는', '영화', '<pad>', '<pad>']`

- `<unk>`: 단어 사전에 없는(등장 빈도가 낮아 제외된) 단어를 대체하는 토큰. 예: `['<unk>', '정말', '<unk>', '좋다']`

[//]: # (heading_4 is not supported)


```python
import numpy as np

# 문장을 일정한 길이로 패딩하는 함수
def pad_sequences(sequences, max_length, pad_value):
    result = []
    for sequence in sequences:
        sequence = sequence[:max_length]  # 최대 길이 초과하면 자르기
        pad_length = max_length - len(sequence)
        padded_sequence = sequence + [pad_value] * pad_length  # 부족한 부분 패딩 추가
        result.append(padded_sequence)
    return np.asarray(result)

# OOV(Out Of Vocabulary) 단어는 <unk>로 대체
unk_id = token_to_id["<unk>"]
train_ids = [[token_to_id.get(token, unk_id) for token in review] for review in train_tokens]
test_ids = [[token_to_id.get(token, unk_id) for token in review] for review in test_tokens]

# 문장의 최대 길이 설정 및 패딩
max_length = 32
pad_id = token_to_id["<pad>"]
train_ids = pad_sequences(train_ids, max_length, pad_id)
test_ids = pad_sequences(test_ids, max_length, pad_id)
```

[//]: # (heading_4 is not supported)


```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# 데이터를 PyTorch 텐서로 변환
train_ids = torch.tensor(train_ids)
test_ids = torch.tensor(test_ids)
train_labels = torch.tensor(train.label.values, dtype=torch.float32)
test_labels = torch.tensor(test.label.values, dtype=torch.float32)

# PyTorch 데이터셋 및 데이터 로더 생성
train_dataset = TensorDataset(train_ids, train_labels)
test_dataset = TensorDataset(test_ids, test_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
```

[//]: # (heading_4 is not supported)


```python
from torch import nn

# 문장 분류 모델 정의
class SentenceClassifier(nn.Module):
    def __init__(
        self,
        n_vocab,          # 전체 단어 사전의 크기 (단어 개수 + 특수 토큰 포함)
        hidden_dim,        # RNN/LSTM 은닉 상태(hidden state)의 차원
        embedding_dim,      # 임베딩 차원 (각 단어를 몇 차원 벡터로 바꿀지)
        n_layers,           # RNN/LSTM 층 수
        model_type="lstm"
    ):
        super().__init__()

        # 임베딩 레이어 (입력된 정수 인덱스를 실수 벡터로 변환)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0  # 패딩 토큰은 학습되지 않음
        )

        # RNN 또는 LSTM 모델 선택
        if model_type == "rnn":
            self.model = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
            )
        elif model_type == "lstm":
            self.model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
            )

        # 최종 분류를 위한 선형 레이어
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)   # 단어를 임베딩 벡터로 변환
        output, _ = self.model(embeddings)     # RNN/LSTM 모델 통과
        last_output = output[:, -1, :]          # 마지막 시퀀스의 출력을 사용 (이전 모든 단어의 정보를 압축해서 가짐)
        logits = self.classifier(last_output)   # 선형 레이어 통과하여 최종 출력
        return logits
```


```python
from torch import optim

n_vocab = len(token_to_id)
hidden_dim = 64
embedding_dim = 128
n_layers = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = SentenceClassifier(
    n_vocab=n_vocab, hidden_dim=hidden_dim, embedding_dim=embedding_dim, n_layers=n_layers
).to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
```

> ⚠️ **놓치기 쉬운 포인트**

	- `nn.Embedding(num_embeddings=n_vocab, embedding_dim=embedding_dim, padding_idx=0)`에서 `padding_idx=0`은, `<pad>` 토큰(ID가 0번으로 등록됨)에 대한 임베딩 벡터는 학습 과정에서 업데이트하지 않겠다는 뜻입니다. 의미 없는 채움 토큰까지 학습시키면 오히려 노이즈가 될 수 있기 때문입니다.

	- `embeddings = self.embedding(inputs)` 한 줄로 정수 인덱스 텐서 `(batch, seq_len)`가 실수 벡터 텐서 `(batch, seq_len, embedding_dim)`로 바뀝니다. RNN 계열 모델은 실수 벡터를 입력으로 기대하므로, 정수 인덱스를 그대로 넣으면 안 되고 항상 임베딩 레이어를 먼저 거쳐야 합니다.

	- `output[:, -1, :]`로 시퀀스의 마지막 타임스텝만 사용하는 것은, "문장 전체를 다 읽은 후의 최종 요약"을 가지고 긍정/부정을 판단하겠다는 뜻입니다. 이는 이전 문서에서 다룬 시계열의 Many-to-One 구조와 완전히 같은 패턴입니다.

[//]: # (heading_4 is not supported)


```python
# 학습 함수 정의
def train(model, datasets, criterion, optimizer, device, interval):
    model.train()
    losses = list()

    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)  # 차원 확장

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % interval == 0:
            print(f"Train Loss {step} : {np.mean(losses)}")


# 평가 함수 정의
def test(model, datasets, criterion, device):
    model.eval()
    losses = list()
    corrects = list()

    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        # 시그모이드 함수를 사용해 예측값 변환
        yhat = torch.sigmoid(logits) > 0.5
        corrects.extend(
            torch.eq(yhat, labels).cpu().tolist()
        )

    print(f"Val Loss : {np.mean(losses)}, Val Accuracy : {np.mean(corrects)}")


# 학습 실행
epochs = 5
interval = 500

for epoch in range(epochs):
    train(classifier, train_loader, criterion, optimizer, device, interval)
    test(classifier, test_loader, criterion, device)
```


```plain text
[Epoch 1] Val Loss : 0.6021, Val Accuracy : 0.6710
[Epoch 2] Val Loss : 0.4027, Val Accuracy : 0.8118
[Epoch 3] Val Loss : 0.3841, Val Accuracy : 0.8160
[Epoch 4] Val Loss : (계속 개선)
[Epoch 5] Val Loss : (계속 개선)
```

> ⚠️ **놓치기 쉬운 포인트**

	- `labels.to(device).unsqueeze(1)`에서 `.unsqueeze(1)`을 쓰는 이유는, `train_labels`가 `(batch,)` 형태의 1차원 텐서인 반면 모델 출력 `logits`는 `(batch, 1)` 형태의 2차원 텐서이기 때문입니다. 두 텐서의 차원을 맞춰주지 않으면 손실 계산 시 브로드캐스팅으로 인해 의도와 다른 결과가 나올 수 있습니다. (관련 개념은 `Training-Loop-and-Optimizer` 노트의 `squeeze()` 설명과 정반대 상황입니다.)

	- `test()` 함수 안에서 `torch.sigmoid(logits) > 0.5`로 예측 클래스를 결정하고 있습니다. 모델은 `BCEWithLogitsLoss`를 쓰기 위해 시그모이드를 거치지 않은 로짓을 출력하므로, "몇 퍼센트 확률로 긍정인가"를 사람이 보기 좋은 형태로 확인하려면 평가/추론 단계에서 직접 시그모이드를 적용해줘야 합니다.

	- `torch.eq(yhat, labels)`로 예측과 정답이 일치하는지 확인할 때, `yhat`은 `True`/`False`로 이루어진 불리언 텐서이고 `labels`는 `0.0`/`1.0`로 이루어진 실수 텐서입니다. PyTorch는 이런 타입 차이를 자동으로 비교 가능하게 처리해주지만, 두 텐서가 참/거짓을 나타내는 값이라는 전제가 깨지면 예상과 다른 비교 결과가 나올 수 있으니 값의 의미를 항상 염두에 둬야 합니다.

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
# 임베딩 매트릭스 저장
token_to_embedding = dict()
embedding_matrix = classifier.embedding.weight.detach().cpu().numpy()

for word, emb in zip(vocab, embedding_matrix):
    token_to_embedding[word] = emb

# 특정 단어의 임베딩 확인
token = vocab[1000]
print(token, token_to_embedding[token])
```


```plain text
보고싶다 [ 2.0375 -1.4278  0.1290  0.3207 ...]
```

[//]: # (heading_4 is not supported)


```python
# 새로운 문장의 감성 분석 함수
def predict_sentiment(model, tokenizer, text, device):
    model.eval()

    # 입력 문장을 형태소 단위로 분리
    tokens = tokenizer.morphs(text)

    # 토큰을 사전의 ID로 변환 (OOV는 <unk> 처리)
    token_ids = [token_to_id.get(token, token_to_id["<unk>"]) for token in tokens]

    # 패딩 적용
    padded_tokens = pad_sequences([token_ids], max_length, pad_id)

    # PyTorch 텐서로 변환 후 모델에 입력
    input_tensor = torch.tensor(padded_tokens).to(device)
    logits = model(input_tensor)

    # 예측값 변환 및 감성 판단
    prediction = torch.sigmoid(logits).item()
    sentiment = "Positive" if prediction > 0.5 else "Negative"

    print(f"문장: {text}\n예측 확률: {prediction:.4f}, 감정 분류: {sentiment}")
```


```plain text
문장: 이 영화 정말 재미있어요!
예측 확률: 0.9909, 감정 분류: Positive

문장: 이 영화 너무 재밌어요ㅠㅠㅠ
예측 확률: 0.9933, 감정 분류: Positive

문장: 돈이 아까운 영화
예측 확률: 0.0069, 감정 분류: Negative
```

> ⚠️ **놓치기 쉬운 포인트**

	- `predict_sentiment()` 함수 안에서 `pad_sequences([token_ids], max_length, pad_id)`처럼 `token_ids`를 리스트로 한 번 더 감싸고 있습니다. `pad_sequences` 함수가 "여러 문장이 담긴 리스트"를 입력으로 기대하도록 설계되어 있기 때문에, 문장 하나만 넣더라도 리스트 안에 리스트 형태(`[[...]]`)로 맞춰줘야 합니다.

	- 학습에 사용한 `token_to_id`, `max_length`, `pad_id` 같은 변수들을 함수 안에서 새로 정의하지 않고 그대로 재사용하고 있습니다. 이렇게 학습 때 썼던 것과 완전히 동일한 전처리 규칙을 추론 시점에도 그대로 적용해야, 모델이 학습한 것과 같은 방식으로 입력이 인코딩됩니다. 전처리 방식이 하나라도 어긋나면 모델이 엉뚱한 예측을 내놓을 수 있습니다.

---

[//]: # (heading_4 is not supported)

- [NSMC (Naver Sentiment Movie Corpus) - GitHub Repository](https://github.com/e9t/nsmc)

- [Korpora - PyPI Package Page](https://pypi.org/project/Korpora/)

- [KoNLPy - Official Documentation](https://konlpy.org/en/latest/)

- [torch.nn.Embedding - Official Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

- [torch.nn.BCEWithLogitsLoss - Official Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

- [torch.utils.data.TensorDataset - Official Documentation](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset)

- [torchinfo - GitHub Repository](https://github.com/TylerYep/torchinfo)

- [PyTorchTRF - 문장 분류 모델 실습 원본 코드 (Wikibook)](https://github.com/wikibook/pytorchtrf)

