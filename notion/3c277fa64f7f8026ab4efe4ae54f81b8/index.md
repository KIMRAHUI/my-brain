---
notion-url: https://app.notion.com/p/RNN_Layer-3c277fa64f7f8026ab4efe4ae54f81b8
title: RNN_Layer
date: '2026-08-20 09:41:00.000'
from_notion: https://app.notion.com/p/RNN_Layer-3c277fa64f7f8026ab4efe4ae54f81b8
author: rahui kim
last_edited_time: '2026-08-20 09:41:00.000'
---
[//]: # (heading_4 is not supported)

> ⚠️  놓치기 쉬운 포인트

	- RNN 계열은 지금까지 다룬 `nn.Linear` 기반 모델과 다르게, **한 번에 하나의 데이터가 아니라 시간 순서(시퀀스)를 가진 데이터**를 처리합니다. 문장 속 단어들, 주가의 날짜별 흐름처럼 "순서가 의미를 가지는 데이터"에 사용한다는 점을 먼저 기억해두면 이후 개념들이 훨씬 잘 이해됩니다.

---

[//]: # (heading_4 is not supported)

- `**input_size**`: 입력 특성의 차원 (예: 단어 임베딩 차원 또는 피처 개수)

- `**hidden_size**`: 은닉 상태(Hidden State)의 차원 (뉴런 개수 및 내부 기억 용량)

- `**num_layers**`: 순환 레이어의 층 개수 (층을 깊게 쌓을수록 복잡한 시퀀스 패턴 학습 가능)

- `**batch_first**`: `True`로 설정하면 입력 텐서의 첫 번째 차원이 배치 크기(Batch Size)가 됨 (기본값은 `False`로 `[seq_len, batch, feature]` 순서)

- `**dropout**`: 레이어 간 드롭아웃 적용 여부 (과적합 방지, `num_layers > 1`일 때 유효)

- `**bidirectional**`: 양방향 RNN 사용 여부 (기본값 `False`)

> ⚠️ 놓치기 쉬운 포인트

	- `**batch_first**`**의 기본값은 **`**False**`**입니다.** 지금까지 다뤄온 `DataLoader`의 배치 텐서는 보통 `(batch, ...)` 순서였기 때문에, RNN 계열도 당연히 배치가 맨 앞일 거라 착각하기 쉽습니다. 기본 설정 그대로 두면 입력을 `(seq_len, batch, input_size)` 순서로 맞춰서 넣어야 하고, 헷갈린다면 `batch_first=True`로 명시적으로 설정해서 `(batch, seq_len, input_size)` 순서로 통일하는 것을 추천합니다.

	- `dropout` 파라미터는 `num_layers`가 **2 이상일 때만** 실제로 적용됩니다. `num_layers=1`인데 `dropout` 값을 넣으면 경고 메시지만 뜨고 아무 효과가 없습니다.

	- `bidirectional=True`로 설정하면 정방향과 역방향 두 개의 은닉 상태가 만들어져 이어붙여지므로, 출력의 마지막 차원 크기가 `hidden_size`가 아니라 `hidden_size * 2`가 됩니다.

---

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
import torch
import torch.nn as nn

# RNN 레이어 정의 (input_size=10, hidden_size=20)
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=False)

# 입력 텐서 생성: (seq_len=3, batch_size=32, input_size=10)
x = torch.randn(3, 32, 10)

# 순전파 실행
out, hidden = rnn(x)
print(out.shape, hidden.shape)  # 출력: torch.Size([3, 32, 20]) torch.Size([1, 32, 20])
```

[//]: # (heading_4 is not supported)

- **시퀀스 길이 (**`**seq_len=3**`**)**: 문장 내 단어의 개수 (타임스텝 t=0, 1, 2)

- **배치 크기 (**`**batch_size=32**`**)**: 한 번에 처리하는 문장의 개수

- **입력 특성 (**`**input_size=10**`**)**: 각 단어가 표현된 벡터의 차원 크기

- `**out**`** 텐서 **`**[3, 32, 20]**`: 모든 타임스텝(t)에서의 은닉 상태 기록이 차곡차곡 쌓인 형태

- `**hidden**`** 텐서 **`**[1, 32, 20]**`: 오직 마지막 타임스텝(t=2)에서의 최종 은닉 상태 요약본 (`num_layers` 개수만큼 존재)

> ⚠️ 놓치기 쉬운 포인트

	- `rnn(x)`의 반환값은 지금까지 봐온 다른 레이어들과 다르게 **텐서 하나가 아니라 튜플 **`**(out, hidden)**` 입니다. 이걸 놓치고 `out = rnn(x)`처럼 하나만 받으면, `out`에 튜플 전체가 담겨서 이후 연산에서 에러가 납니다.

	- `out`의 마지막 타임스텝 값(`out[-1]`, `batch_first=False` 기준)과 `hidden`의 값(`num_layers=1`일 때 `hidden[0]`)은 **사실상 동일한 값**입니다. `out`은 "모든 시점의 기록", `hidden`은 "가장 마지막 시점의 요약"이라는 관점 차이일 뿐입니다.

	- 분류(classification) 문제처럼 "문장 하나를 읽고 결론 하나만 필요"한 경우에는 보통 `hidden`(또는 `out`의 마지막 타임스텝)만 가져다 `nn.Linear`에 연결합니다. 반면 번역이나 품사 태깅처럼 "매 시점마다 출력이 필요"한 경우에는 `out` 전체를 사용합니다.

	- `num_layers=2` 이상으로 쌓으면 `hidden`의 0번째 차원 크기가 `num_layers`만큼 늘어납니다(예: `[2, 32, 20]`). 이때 `hidden[0]`은 첫 번째 레이어, `hidden[-1]`은 마지막 레이어의 최종 은닉 상태입니다.

---

[//]: # (heading_4 is not supported)

- **삭제 게이트 (Forget Gate, **`**f_t**`**)**: 과거의 기억 중 무엇을 버릴지 결정 (수식: `f_t = 시그모이드(W_f · [h_{t-1}, x_t] + b_f)`)

- **입력 게이트 (Input Gate, **`**i_t**`**)**: 현재 정보 중 어떤 것을 셀 스테이트에 기억시킬지 결정

- **출력 게이트 (Output Gate, **`**o_t**`**)**: 갱신된 장기 기억을 바탕으로 이번 시점의 출력(`h_t`)을 생성


```python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=False)

x = torch.randn(3, 32, 10)
out, (hidden, cell) = lstm(x)

print(out.shape, hidden.shape, cell.shape)
# torch.Size([3, 32, 20]) torch.Size([1, 32, 20]) torch.Size([1, 32, 20])
```

> ⚠️ 놓치기 쉬운 포인트

	- **왜 기울기 소실이 문제인가?** 바닐라 RNN은 같은 가중치 행렬을 시퀀스 길이만큼 반복해서 곱하는 구조라서, 역전파 과정에서 기울기가 지수적으로 작아지거나(소실) 커지는(폭주) 문제가 생깁니다. 그 결과 시퀀스가 길어질수록 "훨씬 앞쪽 타임스텝의 정보"가 뒤쪽까지 잘 전달되지 못합니다. LSTM의 셀 스테이트(`C_t`)는 덧셈 위주의 경로로 정보를 전달해서 이 문제를 완화합니다.

	- `nn.LSTM`의 반환값은 `nn.RNN`과 다르게 `**(out, (hidden, cell))**` 형태로, 두 번째 자리가 튜플 안에 튜플이 한 번 더 들어있습니다. `out, hidden = lstm(x)`처럼 그대로 받으면 `hidden` 변수에 `(hidden, cell)` 튜플 전체가 담기게 되어 이후 코드에서 혼동이 생기니, 반드시 `out, (hidden, cell) = lstm(x)`처럼 괄호를 정확히 맞춰 받아야 합니다.

	- **셀 스테이트(**`**C_t**`**)와 은닉 상태(**`**h_t**`**)는 다른 개념**입니다. 셀 스테이트는 오래 유지되는 "장기 기억 컨베이어 벨트"에 가깝고, 은닉 상태는 그 시점에 실제로 출력되는 "이번 시점의 요약"입니다.

---

[//]: # (heading_4 is not supported)

- **리셋 게이트 (Reset Gate, **`**r_t**`**)**: 새로운 후보 은닉 상태를 계산할 때 과거 기억을 얼마나 무시/참고할지 결정

- **업데이트 게이트 (Update Gate, **`**z_t**`**)**: 과거 기억을 유지하는 비율과 새로운 정보를 반영하는 비율을 동시에 제어


```python
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=False)

x = torch.randn(3, 32, 10)
out, hidden = gru(x)

print(out.shape, hidden.shape)
# torch.Size([3, 32, 20]) torch.Size([1, 32, 20])
```

> ⚠️ 놓치기 쉬운 포인트

	- GRU는 셀 스테이트가 따로 없기 때문에, `nn.RNN`과 마찬가지로 반환값이 `**(out, hidden)**` 두 개뿐입니다. LSTM처럼 `cell`을 따로 받으려 하면 에러가 납니다.

	- "LSTM과 GRU 중 뭘 써야 하나요?"에 정답은 없습니다. 일반적으로 데이터가 적거나 학습/추론 속도가 중요하면 GRU를, 시퀀스가 아주 길고 복잡한 장기 의존성을 다뤄야 하면 LSTM을 먼저 시도해보고 실험적으로 비교하는 것이 정석입니다.

---

[//]: # (heading_4 is not supported)

- [torch.nn.RNN - Official Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)

- [torch.nn.LSTM - Official Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

- [torch.nn.GRU - Official Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)

- [PyTorch Official - NLP From Scratch: Classifying Names with a Character-Level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)

