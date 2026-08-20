---
notion-url: https://app.notion.com/p/RNN-Architecture-PyTorch-NLP-Operations-3c277fa64f7f80ab8526f3d84cf2637d
title: RNN Architecture & PyTorch NLP Operations
date: '2026-08-20 09:42:00.000'
from_notion: https://app.notion.com/p/RNN-Architecture-PyTorch-NLP-Operations-3c277fa64f7f80ab8526f3d84cf2637d
author: rahui kim
last_edited_time: '2026-08-20 09:42:00.000'
---
[//]: # (heading_4 is not supported)

### 1. 기본 구조 및 작동 원리 (Basic RNN)

- **순차 데이터 (Sequential Data)**: 텍스트, 음성, 주가 데이터처럼 순서(Order)가 중요하며 이전 시점의 정보가 다음 시점에 영향을 미치는 데이터입니다.

- **RNN 셀 (RNN Cell)**: 은닉층에서 현재 시점의 입력값(x_t)과 이전 시점의 기억(h_{t-1})을 함께 받아 새로운 기억을 계산하는 기본 구성 단위입니다.

- **은닉 상태 (Hidden State, h_t)**: 특정 시점까지 입력된 데이터의 흐름과 문맥을 압축하여 가지고 있는 "과거의 기억" 역할을 하는 벡터입니다.

- **파라미터 공유 (Parameter Sharing)**: 시퀀스의 길이에 관계없이 모든 시점(Time Step)에서 동일한 가중치(W)와 편향(b)을 재사용하여 효율성을 극대화합니다.

- **BPTT (Backpropagation Through Time)**: 시간의 역방향(과거 시점)으로 오차를 거슬러 올라가며 기울기를 구하고 가중치를 업데이트하는 RNN 전용 역전파 알고리즘입니다.

---

[//]: # (heading_4 is not supported)

- **한계점 (기울기 소실 & 장기 의존성)**:

	- **기울기 소실 (Vanishing Gradient)**: BPTT 수행 시 시퀀스가 길어질수록 기울기가 연쇄적으로 곱해져 0에 수렴하며, 초기 시점의 정보가 학습되지 않는 현상입니다.

	- **장기 의존성 문제 (Long-term Dependency)**: 문장이 길어지면 먼 과거의 단어 정보가 뒤로 갈수록 희석되는 문제입니다.

- **개선 모델 (LSTM & GRU)**:

	- **LSTM (Long Short-Term Memory)**:

		- **셀 상태 (Cell State, C_t)**: 네트워크 전체를 관통하며 장기 기억을 안정적으로 전달하는 정보 고속도로 역할을 합니다.

		- **3가지 게이트 (Gate)**: 망각(Forget), 입력(Input), 출력(Output) 게이트를 통해 정보를 선택적으로 삭제, 저장, 추출하여 기울기 소실을 극복합니다.

	- **GRU (Gated Recurrent Unit)**:

		- Cell State와 Hidden State를 하나(h_t)로 통합하고, 게이트를 2개(Reset, Update)로 줄여 연산 속도를 향상시킨 간소화 모델입니다.

---

[//]: # (heading_4 is not supported)

- **원-핫 인코딩 (One-Hot Encoding)**: 단어를 표현할 때 해당 단어 위치만 1, 나머지는 0으로 표현하는 방식입니다. 차원이 과도하게 커지고 단어 간 의미적 유사성을 표현할 수 없다는 단점이 있습니다.

- **임베딩 (Embedding)**: 고차원의 희소 벡터(Sparse Vector)를 고정된 차원의 밀집 벡터(Dense Vector)로 변환하여, 단어 간 의미적 유사도와 연관성을 벡터 공간 상의 거리로 표현하는 기술입니다.

---

[//]: # (heading_4 is not supported)

 | 분류 용어 (Term) | 상세 설명 | 
 | ---- | ---- | 
 | **input_size** | 입력 데이터의 피처(Feature) 차원 크기입니다. (예: 임베딩 벡터 차원 수) | 
 | **hidden_size** | 은닉 상태(h_t) 및 셀 상태(C_t)의 벡터 차원 크기입니다. (모델의 기억 용량) | 
 | **num_layers** | RNN/LSTM/GRU 셀을 위로 몇 층(Multi-layer)으로 쌓을지 지정합니다. | 
 | **batch_first** | 입력 텐서의 차원 순서를 `(batch, seq, feature)`로 설정할지 여부입니다. (True 권장) | 
 | **bidirectional** | 양방향(Bidirectional) RNN 설정 여부입니다. (True 설정 시 정방향 + 역방향 학습) | 
 | **dropout** | 여러 층의 RNN 레이어 사이에 적용할 드롭아웃 확률값입니다. (과적합 방지) | 

---

[//]: # (heading_4 is not supported)

 | 분류 | 속성 / 메서드 (Method) | 설명 및 코드 활용 | 
 | ---- | ---- | ---- | 
 | **텐서 형태 및 속성** | `.shape` / `.size()` | 텐서의 차원 구조를 확인합니다. | 
 |  | `.dtype` | 텐서의 데이터 타입(`torch.float32`, `torch.int64` 등)을 확인합니다. | 
 |  | `.device` | 텐서가 위치한 장치(`cpu` 또는 `cuda:0`)를 확인합니다. | 
 | **특수 텐서 생성** | `torch.rand()` / `torch.randn()` | 균등 분포 및 정규 분포를 따르는 난수 텐서를 생성합니다. | 
 |  | `torch.zeros()` / `torch.ones()` | 모든 값이 0 또는 1로 채워진 텐서를 생성합니다. | 
 | **디바이스 관리** | `.to('cuda')` / `.cuda()` | 텐서를 CPU에서 GPU(또는 반대)로 이동시킵니다. (장치 불일치 에러 방지) | 
 | **타입 및 연동** | `.type(torch.float32)` | 텐서의 데이터 타입을 변환합니다. | 
 |  | `.numpy()` / `torch.from_numpy()` | PyTorch 텐서와 NumPy 배열 상호 변환 (메모리 공유 주의) | 
 | **연산 및 브로드캐스팅** | `torch.matmul()` / `*` | 행렬 곱셈 및 요소별(Element-wise) 연산을 수행합니다. | 
 |  | **Broadcasting** | 크기가 다른 텐서 간의 연산 시 작은 텐서를 자동으로 확장하여 연산합니다. | 

---

[//]: # (heading_4 is not supported)

 | 분류 | 속성 / 메서드 (Method) | 설명 및 코드 활용 | 
 | ---- | ---- | ---- | 
 | **레이어 선언** | `nn.RNN()` | 기본 Vanilla RNN 레이어를 생성합니다. | 
 |  | `nn.LSTM()` | Cell State 및 Gate 구조가 포함된 LSTM 레이어를 생성합니다. | 
 |  | `nn.GRU()` | 연산량이 적고 구조가 간소화된 GRU 레이어를 생성합니다. | 
 | **출력 및 상태값** | `output` | 모든 시점(Time Step)의 은닉 상태값들을 모아놓은 텐서입니다. | 
 |  | `hn (Hidden State)` | 가장 마지막 시점(Last Time Step)의 은닉 상태 벡터입니다. | 
 |  | `cn (Cell State)` | LSTM에서만 반환되며, 가장 마지막 시점의 셀 상태 벡터입니다. | 
 | **초기화 메서드** | `torch.zeros()` | 첫 번째 시점(t=0)에 주입할 초기 은닉 상태(h_0) 및 셀 상태(c_0)를 0으로 초기화할 때 사용합니다. | 

---

[//]: # (heading_4 is not supported)

 | 분류 | 속성 / 메서드 (Method) | 설명 및 코드 활용 | 
 | ---- | ---- | ---- | 
 | **단어 임베딩** | `nn.Embedding(num_embeddings, embedding_dim)` | 단어 사전 크기(`num_embeddings`)와 임베딩 차원(`embedding_dim`)을 받아 임베딩 레이어를 생성합니다. | 
 |  | `.weight` | 임베딩 레이어 내부의 학습 가능한 가중치 행렬에 접근합니다. | 
 | **패딩 처리** | `pad_sequence()` | 길이가 다른 가변 시퀀스 데이터를 동일한 길이로 맞추기 위해 패딩(0)을 추가합니다. | 
 |  | `pack_padded_sequence()` | 패딩 처리된 입력 중 실제 데이터 부분만 효율적으로 계산하도록 묶어주는 메서드입니다. (연산 속도 개선) | 
 |  | `pad_packed_sequence()` | `pack_padded_sequence` 처리된 결과를 다시 일반 패딩 텐서 형태로 복원합니다. | 

---

[//]: # (heading_4 is not supported)

- [PyTorch Official Documentation - torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)

- [PyTorch Official Documentation - torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

- [PyTorch Official Documentation - torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)

- [PyTorch Official Documentation - torch.Tensor](https://pytorch.org/docs/stable/tensors.html)

- [CS231n: Recurrent Neural Networks & Natural Language Processing](http://cs231n.stanford.edu/)

- [Understanding LSTM Networks (Chris Olah's Blog)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

