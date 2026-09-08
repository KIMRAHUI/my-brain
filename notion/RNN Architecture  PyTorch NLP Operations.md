# RNN Architecture & PyTorch NLP Operations

딥러닝 핵심 개념: 순환 신경망(RNN)과 자연어 처리

1. 기본 구조 및 작동 원리 (Basic RNN)

순차 데이터 (Sequential Data): 텍스트, 음성, 주가 데이터처럼 순서(Order)가 중요하며 이전 시점의 정보가 다음 시점에 영향을 미치는 데이터입니다.

RNN 셀 (RNN Cell): 은닉층에서 현재 시점의 입력값(x_t)과 이전 시점의 기억(h_{t-1})을 함께 받아 새로운 기억을 계산하는 기본 구성 단위입니다.

은닉 상태 (Hidden State, h_t): 특정 시점까지 입력된 데이터의 흐름과 문맥을 압축하여 가지고 있는 "과거의 기억" 역할을 하는 벡터입니다.

파라미터 공유 (Parameter Sharing): 시퀀스의 길이에 관계없이 모든 시점(Time Step)에서 동일한 가중치(W)와 편향(b)을 재사용하여 효율성을 극대화합니다.

BPTT (Backpropagation Through Time): 시간의 역방향(과거 시점)으로 오차를 거슬러 올라가며 기울기를 구하고 가중치를 업데이트하는 RNN 전용 역전파 알고리즘입니다.

한계점 및 해결책 (LSTM & GRU)

한계점 (기울기 소실 & 장기 의존성):

개선 모델 (LSTM & GRU):

자연어 처리(NLP) 기초

원-핫 인코딩 (One-Hot Encoding): 단어를 표현할 때 해당 단어 위치만 1, 나머지는 0으로 표현하는 방식입니다. 차원이 과도하게 커지고 단어 간 의미적 유사성을 표현할 수 없다는 단점이 있습니다.

임베딩 (Embedding): 고차원의 희소 벡터(Sparse Vector)를 고정된 차원의 밀집 벡터(Dense Vector)로 변환하여, 단어 간 의미적 유사도와 연관성을 벡터 공간 상의 거리로 표현하는 기술입니다.

RNN 및 순차 모델 파라미터 요약

PyTorch 텐서 기초 및 속성 치트시트

PyTorch RNN & LSTM 모듈 주요 속성 및 메서드

NLP 전처리 및 임베딩 관련 속성 및 메서드

참고 공식 문서 및 학습 자료

PyTorch Official Documentation - torch.nn.RNN

PyTorch Official Documentation - torch.nn.LSTM

PyTorch Official Documentation - torch.nn.Embedding

PyTorch Official Documentation - torch.Tensor

CS231n: Recurrent Neural Networks & Natural Language Processing

Understanding LSTM Networks (Chris Olah's Blog)

