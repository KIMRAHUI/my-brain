# RNN_Layer

개요

본 문서는 PyTorch에서 제공하는 순환 신경망 클래스(torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU)의 공통 파라미터 구조와 각 모델별 코드 구현 및 텐서의 형태(Shape) 변화 과정을 다룹니다.

⚠️  놓치기 쉬운 포인트

PyTorch RNN 계열 공통 파라미터 구조

PyTorch에서는 순환 신경망을 구현할 때 공통적으로 다음과 같은 주요 파라미터를 사용합니다.

input_size: 입력 특성의 차원 (예: 단어 임베딩 차원 또는 피처 개수)

hidden_size: 은닉 상태(Hidden State)의 차원 (뉴런 개수 및 내부 기억 용량)

num_layers: 순환 레이어의 층 개수 (층을 깊게 쌓을수록 복잡한 시퀀스 패턴 학습 가능)

batch_first: True로 설정하면 입력 텐서의 첫 번째 차원이 배치 크기(Batch Size)가 됨 (기본값은 False로 [seq_len, batch, feature] 순서)

dropout: 레이어 간 드롭아웃 적용 여부 (과적합 방지, num_layers > 1일 때 유효)

bidirectional: 양방향 RNN 사용 여부 (기본값 False)

⚠️ 놓치기 쉬운 포인트

순환 신경망(RNN) 구현 및 텐서 Shape 분석

1) PyTorch 코드 구현 예시

import torch
import torch.nn as nn

# RNN 레이어 정의 (input_size=10, hidden_size=20)
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=False)

# 입력 텐서 생성: (seq_len=3, batch_size=32, input_size=10)
x = torch.randn(3, 32, 10)

# 순전파 실행
out, hidden = rnn(x)
print(out.shape, hidden.shape)  # 출력: torch.Size([3, 32, 20]) torch.Size([1, 32, 20])

2) 텐서 구조 해석

입력 텐서 (x) 구조: [seq_len, batch_size, input_size]

시퀀스 길이 (seq_len=3): 문장 내 단어의 개수 (타임스텝 t=0, 1, 2)

배치 크기 (batch_size=32): 한 번에 처리하는 문장의 개수

입력 특성 (input_size=10): 각 단어가 표현된 벡터의 차원 크기

출력 텐서 구조

out 텐서 [3, 32, 20]: 모든 타임스텝(t)에서의 은닉 상태 기록이 차곡차곡 쌓인 형태

hidden 텐서 [1, 32, 20]: 오직 마지막 타임스텝(t=2)에서의 최종 은닉 상태 요약본 (num_layers 개수만큼 존재)

⚠️ 놓치기 쉬운 포인트

고급 순환 신경망: LSTM (Long Short-Term Memory)

개념 바닐라 RNN의 고질적인 문제인 기울기 소실(Vanishing Gradient) 문제를 해결하기 위해 고안된 아키텍처로, 장기 기억을 보존하기 위해 셀 스테이트(Cell State, C_t)와 3개의 게이트를 도입했습니다.

주요 게이트 역할

삭제 게이트 (Forget Gate, f_t): 과거의 기억 중 무엇을 버릴지 결정 (수식: f_t = 시그모이드(W_f · [h_{t-1}, x_t] + b_f))

입력 게이트 (Input Gate, i_t): 현재 정보 중 어떤 것을 셀 스테이트에 기억시킬지 결정

출력 게이트 (Output Gate, o_t): 갱신된 장기 기억을 바탕으로 이번 시점의 출력(h_t)을 생성

lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=False)

x = torch.randn(3, 32, 10)
out, (hidden, cell) = lstm(x)

print(out.shape, hidden.shape, cell.shape)
# torch.Size([3, 32, 20]) torch.Size([1, 32, 20]) torch.Size([1, 32, 20])

⚠️ 놓치기 쉬운 포인트

고급 순환 신경망: GRU (Gated Recurrent Unit)

개념 LSTM의 복잡한 구조를 대폭 간소화하여, 내부 상태를 h_t 하나로 통합하고 게이트를 2개로 줄인 모델입니다.

주요 게이트 역할

리셋 게이트 (Reset Gate, r_t): 새로운 후보 은닉 상태를 계산할 때 과거 기억을 얼마나 무시/참고할지 결정

업데이트 게이트 (Update Gate, z_t): 과거 기억을 유지하는 비율과 새로운 정보를 반영하는 비율을 동시에 제어

특징 LSTM에 비해 파라미터 수가 적어 연산 속도가 빠르고 데이터가 상대적으로 적은 환경에서도 과적합 방지에 유리합니다.

gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=False)

x = torch.randn(3, 32, 10)
out, hidden = gru(x)

print(out.shape, hidden.shape)
# torch.Size([3, 32, 20]) torch.Size([1, 32, 20])

⚠️ 놓치기 쉬운 포인트

참고 링크 (공식 문서)

torch.nn.RNN - Official Documentation

torch.nn.LSTM - Official Documentation

torch.nn.GRU - Official Documentation

PyTorch Official - NLP From Scratch: Classifying Names with a Character-Level RNN

