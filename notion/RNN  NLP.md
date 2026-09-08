# RNN & NLP

개요

본 문서는 RNN 계열 모델을 자연어처리(NLP)에 적용하는 두 가지 실습을 다룹니다.

문장 성분 태깅 — 매우 작은 장난감(toy) 데이터로 RNN의 입출력 구조와 내부 가중치 형태를 직접 확인하는 실습 (Many-to-Many)

한국어 영화 리뷰 감성 분류 — 실제 데이터셋(NSMC)을 형태소 분석 → 정수 인코딩 → 임베딩 → LSTM 순서로 처리해 긍정/부정을 분류하는 실습 (Many-to-One)

첫 번째 실습은 RNN 내부에서 텐서가 어떤 모양으로 흘러가는지, 가중치 행렬의 크기가 왜 그렇게 정해지는지를 아주 작은 규모에서 확인하는 데 목적이 있고, 두 번째 실습은 실제 자연어 문장을 다루기 위한 전체 파이프라인(토큰화 → 단어 사전 → 패딩 → 임베딩)을 익히는 데 목적이 있습니다.

문장 구조 태깅 — RNN 입출력 구조 이해하기

실습 목표: "John loves cats", "John loves dogs"라는 두 문장을 입력받아, 각 단어가 문장 성분(S=주어, V=동사, O=목적어) 중 무엇인지 맞히는 아주 작은 RNN 모델을 만들어봅니다.

단어를 벡터로 바꾸기 (원-핫 인코딩)

# 단어 벡터
word_to_vec = {
    "John": [1, 0, 0, 0],
    "loves": [0, 1, 0, 0],
    "cats": [0, 0, 1, 0],
    "dogs": [0, 0, 0, 1]
}

각 단어를 4차원 원-핫(one-hot) 벡터로 표현합니다. 실제 NLP에서는 단어 수가 수만 개에 달해 이런 방식 대신 임베딩(embedding)을 사용하지만(3장 참고), 여기서는 구조를 이해하기 위해 아주 단순하게 표현했습니다.

입력 데이터와 타깃 데이터 구성

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

train_X의 형태는 (2, 3, 4) — 문장 2개, 각 문장은 단어 3개, 각 단어는 4차원 벡터입니다. train_Y는 (2, 3) — 문장 2개, 각 단어에 대한 정답 라벨(S/V/O) 3개입니다.

하이퍼파라미터 정의

# 하이퍼파라미터
num_classes = 3       # 출력 클래스 개수 - S, V, O
input_dim = 4          # 단어 벡터 차원 - [1, 0, 0, 0]
sequence_length = 3    # 문장의 길이 - 각 문장 단어 3개
hidden_dim = 3          # RNN hidden state 차원 - 단어 정보를 내부적으로 압축 표현 (자유롭게 조정 가능한 값)
learning_rate = 0.1
batch_size = 2

RNN 모델 정의 및 학습

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

학습 결과

Epoch [10/50], Loss: 0.5003
Epoch [20/50], Loss: 0.1579
Epoch [30/50], Loss: 0.0086
Epoch [40/50], Loss: 0.0022
Epoch [50/50], Loss: 0.0013

예측 및 결과 확인

model.eval()
with torch.no_grad():
    predictions = model(train_X_tensor)
    predicted_classes = torch.argmax(predictions, dim=2)

idx2tag = ['S', 'V', 'O']
for i, pred in enumerate(predicted_classes):
    pred_list = pred.numpy()
    result_str = [idx2tag[c] for c in pred_list]
    print(f"Sentence {i+1} Prediction: {' '.join(result_str)}")

Sentence 1 Prediction: S V O
Sentence 2 Prediction: S V O

두 문장 모두 각 단어의 문장 성분을 정확히 맞혔습니다.

⚠️ 놓치기 쉬운 포인트

RNN 내부 가중치 형태 분석

RNN이 내부적으로 어떤 가중치들을 가지고 있는지 직접 확인해봅니다.

# (참고용) 모델 가중치 확인
print("\nModel Summary:")
for name, param in model.named_parameters():
    print(f"{name} => {param.shape}")

Model Summary:
rnn.weight_ih_l0 => torch.Size([3, 4])
rnn.weight_hh_l0 => torch.Size([3, 3])
rnn.bias_ih_l0 => torch.Size([3])
rnn.bias_hh_l0 => torch.Size([3])
fc.weight => torch.Size([3, 3])
fc.bias => torch.Size([3])

각 가중치의 의미와 형태

Wx는 (hidden_dim, input_dim)로 설정됩니다. 즉, 가중치 행렬의 행이 은닉 차원, 열이 입력 차원입니다.

모델 구조 요약 (torchinfo)

!pip install torchinfo
from torchinfo import summary
summary(model, input_size=(2, 3, 4))

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
SimpleRNN                                [2, 3, 3]                 --
├─RNN: 1-1                               [2, 3, 3]                 27
├─Linear: 1-2                            [2, 3, 3]                 12
==========================================================================================
Total params: 39

RNN 파라미터 27개는 Wx(3×4=12) + Wh(3×3=9) + bias_ih(3) + bias_hh(3) = 27로 정확히 계산됩니다. Linear 파라미터 12개는 Wy(3×3=9) + bias(3) = 12입니다.

⚠️ 놓치기 쉬운 포인트

텍스트 분류 — 한국어 영화 리뷰 감성 분석

이번에는 장난감 데이터가 아닌, 네이버 영화 리뷰 실데이터(NSMC, Naver Sentiment Movie Corpus)로 긍정/부정을 분류하는 모델을 만듭니다.

데이터 불러오기

!pip install Korpora
!pip install konlpy

import pandas as pd
from Korpora import Korpora

# NSMC(Naver Sentiment Movie Corpus) 데이터 로드
corpus = Korpora.load("nsmc")

# 데이터를 DataFrame 형태로 변환
corpus_df = pd.DataFrame(corpus.test)
corpus_df

                                                   text  label
0                                                   굳 ㅋ      1
1                                  GDNTOPCLASSINTHECLUB      0
2                뭐야 이 평점들은.... 나쁘진 않지만 10점 짜리는 더더욱 아니잖아      0
...
[50000 rows x 2 columns]

label은 1(긍정) 또는 0(부정)입니다.

학습/테스트 분리

# 데이터를 학습(train)과 테스트(test)로 분리 (90% 학습, 10% 테스트)
train = corpus_df.sample(frac=0.9, random_state=42)
test = corpus_df.drop(train.index)

print("Training Data Size :", len(train))  # 45000
print("Testing Data Size :", len(test))    # 5000

random_state=42로 무작위성을 고정해서, 코드를 다시 실행해도 항상 같은 방식으로 분리되게 했습니다.

형태소 분석 및 단어 사전 구축

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

특수 토큰의 의미

<pad>: 문장 길이를 통일하기 위해 채워 넣는 의미 없는 토큰. 예: ['감동', '있는', '영화', '<pad>', '<pad>']

<unk>: 단어 사전에 없는(등장 빈도가 낮아 제외된) 단어를 대체하는 토큰. 예: ['<unk>', '정말', '<unk>', '좋다']

정수 인코딩 및 패딩

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

token_to_id.get(token, unk_id)는 "사전에 있는 단어면 해당 ID를, 사전에 없는 단어(OOV)면 <unk> ID를 사용하라"는 뜻입니다. dict.get(key, default) 문법으로 존재하지 않는 키에 대한 예외 처리를 간결하게 처리하고 있습니다.

PyTorch 텐서 및 DataLoader 준비

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

여기서는 직접 Dataset 클래스를 만들지 않고, PyTorch가 기본 제공하는 TensorDataset 을 사용했습니다. 입력 텐서와 라벨 텐서를 그대로 넘겨주기만 하면 Dataset이 요구하는 __len__, __getitem__을 자동으로 구현해줘서, 데이터가 이미 완성된 텐서 형태라면 커스텀 클래스를 따로 작성할 필요가 없습니다.

문장 분류 모델 정의 (임베딩 + RNN/LSTM)

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

모델 및 하이퍼파라미터 설정

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

이진 분류(긍정/부정 2개 클래스)이므로 손실 함수로 nn.BCEWithLogitsLoss()를 사용했습니다. 이름 그대로 로짓(logit, 활성화 함수를 거치지 않은 원시 출력)을 입력받는 손실 함수이기 때문에, 모델의 forward()에서 마지막에 시그모이드를 적용하지 않고 logits를 그대로 반환하고 있습니다.

⚠️ 놓치기 쉬운 포인트

학습 및 평가 함수 정의, 학습 실행

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

학습 결과 (에폭별 검증 성능)

[Epoch 1] Val Loss : 0.6021, Val Accuracy : 0.6710
[Epoch 2] Val Loss : 0.4027, Val Accuracy : 0.8118
[Epoch 3] Val Loss : 0.3841, Val Accuracy : 0.8160
[Epoch 4] Val Loss : (계속 개선)
[Epoch 5] Val Loss : (계속 개선)

1 에폭 만에 정확도가 67%에서 시작해, 2 에폭째 이미 81%를 넘어서는 것을 확인할 수 있습니다.

⚠️ 놓치기 쉬운 포인트

학습된 임베딩 확인 및 새 문장 예측

학습된 임베딩 벡터 확인

# 임베딩 매트릭스 저장
token_to_embedding = dict()
embedding_matrix = classifier.embedding.weight.detach().cpu().numpy()

for word, emb in zip(vocab, embedding_matrix):
    token_to_embedding[word] = emb

# 특정 단어의 임베딩 확인
token = vocab[1000]
print(token, token_to_embedding[token])

보고싶다 [ 2.0375 -1.4278  0.1290  0.3207 ...]

classifier.embedding.weight는 학습이 끝난 뒤 각 단어에 대응하는 128차원 벡터들의 모음입니다. .detach()로 그래디언트 추적을 끊고, .cpu().numpy()로 GPU 텐서를 NumPy 배열로 변환해서 확인했습니다.

새로운 문장으로 감성 분석하기

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

테스트 결과

문장: 이 영화 정말 재미있어요!
예측 확률: 0.9909, 감정 분류: Positive

문장: 이 영화 너무 재밌어요ㅠㅠㅠ
예측 확률: 0.9933, 감정 분류: Positive

문장: 돈이 아까운 영화
예측 확률: 0.0069, 감정 분류: Negative

"재밌어요ㅠㅠㅠ"처럼 초성체나 이모티콘성 표현이 섞인 구어체 문장도 정확히 긍정으로 분류하고, "돈이 아까운 영화"처럼 직접적인 감성 단어가 없어도 문맥을 통해 부정으로 정확히 분류하는 것을 확인할 수 있습니다.

⚠️ 놓치기 쉬운 포인트

참고 링크

NSMC (Naver Sentiment Movie Corpus) - GitHub Repository

Korpora - PyPI Package Page

KoNLPy - Official Documentation

torch.nn.Embedding - Official Documentation

torch.nn.BCEWithLogitsLoss - Official Documentation

torch.utils.data.TensorDataset - Official Documentation

torchinfo - GitHub Repository

PyTorchTRF - 문장 분류 모델 실습 원본 코드 (Wikibook)

