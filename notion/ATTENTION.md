# ATTENTION

import math
import random
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops import embedding
from torch.distributed.autograd import context

# Common_data.py에 만든 공통 데이터·전처리 도구 중,
# Attention 실습에서 필요한 이름만 가져온다.
from Common_data import (
    device,          # CPU 또는 GPU를 나타내는 장치 정보
    PAD,             # 패딩 토큰의 번호 (0): 문장 길이를 맞추는 빈칸
    V,               # 단어 사전의 크기: Embedding 입력 크기에 사용
    TRAIN_STORIES,   # 학습용 이야기 데이터 (문장 4개 묶음, 토픽 번호)
    VALID_STORIES,   # 검증용 이야기 데이터
    encode,          # 문장(str)을 토큰 번호 리스트(list[int])로 변환
    pad_batch,       # 길이가 다른 문장들을 PAD로 채워 하나의 텐서로 변환
    TOPIC_NAMES      # ['공부', '운동', '요리', '여행']
)


class BiLSTMWithAttention(nn.Module):
    # 파이토치 신경망 모듈(nn.Module)기본설정 불러오기
    def __init__(self,vocab_size:int,emd_dim:int=32,hidden:int=32,num_classes:int=4) -> None:
        """
        :param vocab_size:71
        :param emd_dim:32
        :param hidden:32
        :param num_classes:4
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,emd_dim,padding_idx=PAD)
        #단어ID를 32차원의 숫자 벡터로 변환, padding_idx를 주어 빈칸(PAD)토큰은 학습대상에서 제외
        self.lstm = nn.LSTM(emd_dim,hidden,batch_first=True,bidirectional=True)
        #bidirectional = True(양방향) -> hidden *2 가된다.
        self.attention = nn.Linear(hidden*2,hidden * 2)# W_att : h_t → u_t 로 변환
        #LSTM출력값(hidden*2)를 받아 선형변환을 거친다
        # 1) 차원 맞춤 : LSTM출력크기(64차원), 최종예측클래스(4) : 64 -> 32 -> 4
        # 2) Feature : 가중치(w)곱해주는 선형변환을 거쳐 어떤 정보에 더 큰 가중치를 주고, 무시해야하는지 스스로 학습
        # 3) Score계산 기초 작업 : 선형변환 거쳐 변형된 공간에 먼저 보낸뒤 최종 스칼라(1차원)로 압축하는과정거치기위해 선형변환
        self.context_vector= nn.Linear(hidden * 2,1,bias=False)# W_c   : u_t → 스칼라 점수 (출력 차원 1!)
        # 만약, 바로 선형변환 써서 1로 줄이면 표현력이 떨어짐
        # 1단계에서 특징을 입체적으로 가공되어 비선형 변환 거친 데이터를 받아, 최종 중요도 점수 매기기위해 1차원스코어로 압축
        self.layer_norm = nn.LayerNorm(hidden*2)
        #layer_norm : 굳이 hidden*2인이유는 BiLSTM을 거쳐 나온 컨텍스트벡터크기가 hidden *2.
        # nn.LayerNorm(hidden*2)통해 hidden*2 값들의 분포를 안정적으로 균형잡혀있게해줌
        # 1) 학습 안정화, 2)기울기소실/ 폭발 방지 3)과적합 방지 보조
        self.dropout = nn.Dropout(0.2) #20%뉴런을 무작위로 꺼서 Overfitting:과적합 방지
        self.fc1 = nn.Linear(hidden * 2, hidden)
        # 복잡한 패턴 학습위해 ReLu추가, fc1거친후 ReLu비선형함수통과하면 BinaryBoundary그릴수있어 똑똑하게 분류
        # 문장의 컨벡스트 벡터크기 hidden *2. 이 64차원의 정보를 4개의 클래스 억지로 쑤셔넣음 정보의 손실,왜곡발생
        # fc1이 중간역할로 32차원으로 한번 다듬고 압축
        self.fc2 = nn.Linear(hidden,num_classes)
        # 중간역할로 정제된 정보를 받아 최종적으로 4개의 토픽중 하나를 골라내도록 역할을 분담

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:                           # ← __init__ 과 같은 들여쓰기 수준!
        """
        :param x: 입력으로 들어오는 단어 ID들의 토치 텐서
        :param return_attn: 어텐션 가중치(a)까지 반환할지 여부를 결정 짓는 불리언 플래그
        """
        pad_mask = (x == PAD)
        # 문장 길이 맞추기위해 넣은 PAD인자리 찾아 True로 표시 : 마스킹 준비, # (B, L)  패딩 위치 True
        out,_ = self.lstm(self.embedding(x))
        #embedding(x) : 입력된 토큰을 32차원 벡터로 바꿈(B, L, 2H)
        #lstm (...) : 임베딩된 값을 LSTM에 통과
        u = torch.tanh(self.attention(out))
        #LSTM의 출력값에 선형변환을 거치고, tanh을 먹여서 비선형 특성을 가진 중간데이터 u 만든다.(B, L, 2H)
        score = self.context_vector(u).squeeze(-1)
        #컨텍스트 벡터를 통과시켜 1차원 스코어로 만든뒤, squeeze(-1)로 맨뒤 불필요한 1차원 없애 (B,L)형태
        score = score.masked_fill(pad_mask,-float('inf'))
        # 패딩은 softmax 후 0 이 되도록!
        #아까 찾아둔 pad_mask(빈자리)위치에 -inf(무한대)채워 넣어 나중에 softmax거칠때 가중치 완전 0되도록 차단
        a = torch.softmax(score,dim=-1)
        # (B, L)  토큰별 중요도 (합=1되는 확률값 a)
        #다중분류/정규화를 위해 softmax를취해, 문장 속 각 단어별 중요도
        context = (a.unsqueeze(-1) * out).sum(dim=1)
        # (B, 2H) 가중합 = 컨텍스트 벡터
        # 각 단어의 중요도(a)와 실제 LSTM출력(out)을 곱해 다 더한뒤에 문장 전체를 압축한
        # 단 하나의 컨벡스트벡터(context)완성
        h = self.dropout(self.layer_norm(context))
        # 완성된 컨벡스트 벡터에 레이어 정규화(LayerNorm)을 걸어 값을 안정화
        # dropout으로 과적합 방지
        logits = self.fc2(F.relu(self.fc1(h)))
        # (B, C)
        #fc1을 거친뒤에 ReLU를 통과시켜 최종 fc2통해 토픽 4개 개수 만큼의 최종 예측 점수
        #logits 뱉어냄
        return(logits,a) if return_attn else logits
        # 함수호출할때 return_attn = True로 물어보면 (로짓,어텐션가중치).
        #아니면 로짓만 리턴

#encode
"""
- **전:** `"오늘은 운동을 열심히 했다"` (사람이 읽는 글자)
- **후:** `[2, 15, 34, 12, 3]` (모델이 계산하는 숫자 ID 리스트) + `정답 토픽 번호`
"""
train_sents = [(encode(s),ti) for st,ti in TRAIN_STORIES for s in st]
valid_sents = [(encode(s),ti) for st,ti in VALID_STORIES for s in st]
"""
fot st,ti in TRAIN_STORIES : 전체 학습데이터에서 이야기(st)와 토픽(ti)을 하나씩 꺼낸다
for s in st : 낱개로 꺼낸 문장 s를 encode함수에 통과시켜 숫자 ID리스트로 바꾸고,
원래 그 이야기의 토픽인 ti와 짝을 지어 튜플 형태로 저장

encode(s)가 하는일
앞서 만든 단어장 stoi활용해서 글자를 모델이 계산 가능한 숫자 토큰ID바꿔주는 역할
"""

#데이터 뒤섞은뒤 배치 크기만큼 문장길이 맞추고(pad_batch),GPU/CPU로 실어나른다.
def batches (data: list[tuple[list[int],int]], bs:int,shuffle:bool=True) -> Iterator[tuple[torch.Tensor,torch.Tensor]]:
    """
    :param data: 데이터셋((문장ID리스트,정답레이블)튜플들의 리스트)
    :param bs: 배치사이즈(batch_size,한번에 묶을 데이터 개수)
    :param shuffle: 데이터를 무작위로 섞을지 여부(True면섞고, False면 순서대로 묶는다)
    """
    idx = list(range(len(data)))
    if shuffle: random.shuffle(idx)
    """
    데이터 전체 개수 만큼 번호표 idx만들고, shuffle=True면 그 번호표를 무작위로 섞는다.
    (학습할때 순서대로 외우지 않고 골고루 학습하기 위함)
    """
    for i in  range(0,len(idx),bs):
        #0번 크기부터 시작해서 배치크기(bs)만큼 뛰어가면서 데이터 자를 준비
        chunk = [data[j] for j in idx[i:i+bs]]
        #섞인 번호표 바탕으로 현재 배치 크기 만큼 데이터 조각(chunk)꺼내온다.
        yield (
            pad_batch([c[0] for c in chunk]).to(device),
            torch.tensor([c[1] for c in chunk]).to(device),
        )
        """
        문장 데이터처리 c[0]
        :안에 들어있는 문장들은 길이가 제각각. pad_batch통해 짧은 문장의 빈자리를 PAD(0)채워
         네모난 텐서형태 (B,L)맞춘뒤 -> 연산 장치 (GPU/CPU)로 보낸다.
        정답레이블 처리 c[1]
        :정답 토픽번호들을 모아 일반 토치텐서로 만든 뒤 장치로 보낸다.
        yield
        :일반 return과 달리, 데이터를 한번에 메모리에 올리지 않고, 필요할때마다 하나씩 (배치단위)
        꺼내 쓸 수 있게 (제너레이터) 만들어준다.
        """

model = BiLSTMWithAttention(V).to(device)
#앞서 정의한 모델을 생성. 사전크기(V)를 넘겨주고,빠른 계산위해 모델을 GPU/CPU로 보낸다.
opt = torch.optim.Adam(model.parameters(),lr=3e-3)
#옵티마이저설정
# 보편적으로 쓰이는 Adam알고리즘을 사용. 학습률을 lr은 0.003으로 지정. 모델의 가중치를 조금씩 수정해나간다
for epoch in range(8):
    # epoch를 총 8번 반복.
    model.train()
    tot = 0
    """
    학습모드로 전환 -> 이때 Dropout,LayerNorm이 학습용으로 작동
    tot는 이번 에포크 동안 총 손실(Loss)기록할 변수
    """
    for x, y in batches(train_sents, 64):
        """
        앞서 만든 batches함수를 호출,학습데이터를 배치크기 64개씩 묶어서 x(문장텐서)와 y(정답토픽)꺼내옴
        """
        loss = F.cross_entropy(model(x), y)
        #모델에 문장 x를 넣고, 예측값(Logits)을 뽑은뒤, 실제 정답 y와 비교해
        #손실(오차,Loss)를 계산
        #다중분류이므로 cross_entropy사용
        opt.zero_grad()
        #[딥러닝학습 3단계 공식] : 이전 step에 고여있던 기울기를 깨끗하게 초기화
        loss.backward()
        #[딥러닝학습 3단계 공식] : 오차를 바탕으로,미분(역전파)해서 각 가중치가 얼마나 잘못했는지 기울기 구함
        opt.step()
        #[딥러닝학습 3단계 공식] : 구한 기울기를 바탕으로 옵티마이저가 모델의 가중치를 수정(업데이트)한다.
        tot += loss.item()
        #현재 batch에 나온 오차값을 누적해서 더해준다. (나중에 평균내서 출력용쓰기위함)
    model.eval()
    #평가모드로 전환
    #학습이 끝나고 검증(Val)단계로 넘어갈때, 모델을 평가모드(eval)로변경
    #이때, Dropout등 꺼지고, 고정된 상태로 작동
    with torch.no_grad():
        #예측할때는 가중치를 업데이트할 필요없어서 기울기 계산을 꺼서 메모리와 연산량을 아낀다.
        correct = sum((model(x).argmax(dim=-1) == y).sum().item() for x,y in batches(valid_sents, 128,False))
        """
        검증 데이터를 배치(최대 128개)씩 모델에 넣어
        예측한 토픽 번호와 실제 토픽번호가 같은 문장수를 모두 더한다.
        -------------------------------------------------------
        model(x) : 각 문장별 토픽 점수, 형태(B,4)
        .argmax(1): 점수가 가장 큰 토픽 번호 선택. 형태(B,)
        == y :  예측과 정답이 같은지 True/False로 비교
        .sum.item(): True정답의 개수를 정수로 변환
        for x,y in batches(valid_sents, 128,False) : 검증데이터 섞지않고, 128개 단위로 꺼냄
        """
    print(f'epoch{epoch+1} | trains loss {tot / (len(train_sents)/64):.4f} | valid acc {correct/len(valid_sents):.4f}')
    """
    epoch1 | trains loss 0.3985 | valid acc 1.0000
    epoch2 | trains loss 0.0021 | valid acc 1.0000
    epoch3 | trains loss 0.0008 | valid acc 1.0000
    epoch4 | trains loss 0.0006 | valid acc 1.0000
    epoch5 | trains loss 0.0005 | valid acc 1.0000
    epoch6 | trains loss 0.0004 | valid acc 1.0000
    epoch7 | trains loss 0.0003 | valid acc 1.0000
    epoch8 | trains loss 0.0003 | valid acc 1.0000
    """

#어텐션 가중치 들여다 보기
model.eval()
#모델을 평가모드로 바꿈
for sent in ['지수는 숨이 차서 잠시 물을 마셨다', '하나는 시장에서 채소를 샀다']:
    # 테스트 문장으로 반복문
    x = torch.tensor([encode(sent)]).to(device)   # ← encode: 문장(str) → 숫자 id 리스트
    #문장문자열 sent를 encode함수 통해 숫자 ID리스트로바꾸고, 배치차원 [] 을 추가해,
    #2차원 텐서로 만든 뒤 GPU/CPU로 보냄
    with torch.no_grad():
        #예측할때는 가중치 업데이트 필요 없음
        logits, a = model(x, return_attn=True)
        """
        모델에 문장텐서(x)를 넣고, return_attn=True를 줘서 최종 예측 점수(logits)와 함께
        문장속 단어별 어텐션 가중치(a) 통째로 받아옴
        """
    print(f'문장: {sent} → 예측 토픽: {TOPIC_NAMES[logits.argmax(1).item()]}')
    #logits.argmax(1).item() : 4개의 토픽 점수 중 가장 높은 점수의 인덱스(0~3) 뽑아냄
    # TOPIC_NAMES[...] : 그 인덱스에 해당하는 실제 토픽 이름(운동,요리..)가져와서 문장과 함께 출력
    for w, wt in zip(sent.split(), a[0].tolist()):
        #문장을 단어단위(sent.split())로 쪼개고, 각 단어에 매겨진 어텐션 중요도 가중치 (a[0].tolist())와
        #하나씩 짝을지어 zip반복문을 돈다.
        print(f'   {w:<8} {wt:.3f} ' + '█' * int(wt * 40))
"""
문장: 지수는 숨이 차서 잠시 물을 마셨다 → 예측 토픽: 운동
   지수는      0.106 ████
   숨이       0.096 ███
   차서       0.072 ██
   잠시       0.192 ███████
   물을       0.327 █████████████
   마셨다      0.208 ████████
문장: 하나는 시장에서 채소를 샀다 → 예측 토픽: 요리
   하나는      0.045 █
   시장에서     0.287 ███████████
   채소를      0.484 ███████████████████
   샀다       0.184 ███████
"""


#scaled dot product attention : Transformer's attention
def scaled_dot_product_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor | None = None) -> tuple[torch.Tensor,torch.Tensor]:
    dk = q.size(-1)
    """query 벡터 차원의 크기 dk가져옴.
    # 나중에 값이 너무 커져서 생기는 기울기 소실 막기위해 나누어주는 스케일링 값으로 쓰임"""
    scores = torch.matmul(q,k.transpose(-1,-2) / math.sqrt(dk))
    """q와 k를 내적하여 모든 단어 쌍 간의 유사도 점수(Score)를 구한다.
    그후 math.sqrt(dk)로 나누어(scale)수치가 너무 발산하지 않도록 안정화한다."""
    if mask is not None:
        scores = scores.masked_fill(mask,float('-inf'))
        """
        마스크가 존재한다면,가려야할 위치에(True)에 inf 채워넣어 softmax를 거칠때
        가중치가 완전 0 되도록 만든다.
        """
    weights = torch.softmax(scores,dim=-1)
    # 마지막 축(dim=-1)을 따라 softmax를 취해 각 행의 합이 1이되는 어텐션 가중치(확률분포)구한다.
    return torch.matmul(weights,v),weights
    # 구해진 가중치(weights)와 v(value)를 곱해 최종 문맥 베터를 만들고,(최종출력,어텐션가중치)튜플형태로 반환

#마스크생성함수
def create_padding_mask(seq:torch.Tensor,pad:int = PAD) -> torch.Tensor:
    """
    입력 문장 속 PAD(빈칸) 위치를 찾아 attention이 보지 못하게 하는 마스크를 만든다.
    입력:  seq  → (B, T)  # B=배치 크기, T=문장 길이
    출력: mask → (B, 1, 1, T)
    True인 위치는 PAD이므로 attention score에서 -inf로 바꿔 가린다.
    ->unsqueeze(1).unsqueeze(2)는 (B, T)를 (B, 1, 1, T)로 바꿔서,
    멀티헤드 attention의 (B, heads, Tq, Tk) 점수에 자동 적용 되게 해주는 과정.
    """
    return (seq == pad).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(T:int) -> torch.Tensor:
    """
    길이 T인 문장에서 미래 단어를 보지 못하게 하는 causal mask를 만든다.

    True  : 가릴 위치 (현재 토큰보다 뒤쪽 = 미래)
    False : 볼 수 있는 위치 (자기 자신과 과거 토큰)

    예: T=4
    [[False, True,  True,  True ],
     [False, False, True,  True ],
     [False, False, False, True ],
     [False, False, False, False]]"""
    return torch.triu(torch.ones(T,T,dtype=torch.bool),diagonal=1)
    #diagonal =1 : 주 대각선 바로 위부터 True: 자기자신은 볼수있고, 미래만 가린다.

x = torch.randn(3,5,8)
#실행 및 테스트
"""배치크기:3, 문장길이:5, 특징 차원:8"""
out,w = scaled_dot_product_attention(x,x,x,mask=create_look_ahead_mask(5))
"""Query,Key,Value에 자기 자신 x를 넣는 Self-Attention을 테스트"""
"""mask=create_look_ahead_mask(5): 길이가 5인 문장에 대해 미래를 보지 못하게 하는 마스크 적용"""
print('out', out.shape, '| weights', w.shape)
print('첫 행(첫 토큰)은 자기 자신만 본다:', w[0, 0].round(decimals=2).tolist())
print('각 행의 합 = 1:', w.sum(-1)[0].tolist())
"""
out torch.Size([3, 5, 8]) | weights torch.Size([3, 5, 5])
첫 행(첫 토큰)은 자기 자신만 본다: [1.0, 0.0, 0.0, 0.0, 0.0]
각 행의 합 = 1: [1.0, 1.0, 1.0, 1.0, 1.0]
"""

#멀티헤드어텐션
class MultiHeadAttention(nn.Module):
    def __init__(self, em_dim: int, num_heads: int) -> None:
        """
        :param em_dim: 임베딩 차원(D). 각 토큰의 벡터길이, num_heads로 나누어 떨어져야한다.
        :param num_heads: attention head의 개수(H). 각 head는 em_dim // num_heads 차원씩 나누어 계산
        """
        super().__init__()  # nn.Module 초기화: Linear 등을 등록하기 전에 반드시 실행

        assert em_dim % num_heads == 0, "em_dim은 num_heads로 나누어떨어져야 합니다."
        #헤드별로 공평하게 차원을 쪼개야 하기때문
        self.h, self.dk, self.em_dim = num_heads, em_dim // num_heads, em_dim
        #h(헤드개수), dk(각 헤드가 담당할 차원 크기 : 전체차원 // 헤드개수)
        self.wq, self.wk, self.wv = (
            nn.Linear(em_dim, em_dim),
            nn.Linear(em_dim, em_dim),
            nn.Linear(em_dim, em_dim),
        )
        #query,key,value를 각각 선형 변환 하기 위한 가중치 행렬들
        self.dense = nn.Linear(em_dim, em_dim)
        #쪼개서 계산했던 여러 헤드들의 결과 하나로 합침(Concat)
        # -> 최종적으로 통과시켜 원래 차원으로 되돌려주는 출력 Projection층

    def split_heads(self,x:torch.Tensor) -> torch.Tensor:
        """(배치,길이,전체차원)형태의 텐서를 (배치,헤드개수,길이,헤드별차원)형태로 쪼개고, 자리를 바꿈
        이렇게 하면 여러개의 헤드가 동시에 각자의 관점(서로다른특징)에서 어텐션을 독립적으로 계산 가능)"""
        B,T,_ = x.size()
        return x.view(B,T,self.h,self.dk).transpose(1,2)


    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor | None = None) -> tuple[torch.Tensor,torch.Tensor]:
        B = q.size(0)
        q, k, v = self.split_heads(self.wq(q)), self.split_heads(self.wk(k)), self.split_heads(self.wv(v))
        """
        self.wq(q),wk(k),wv(v)를 통과시킨뒤,split_heads로 헤드별로 쪼갠다.
        앞서 만들었던 scaled_dot_product_attention함수에 집어 넣어 헤드별 어텐션 점수와 출려값을 구한다.
        """
        out, weights = scaled_dot_product_attention(q, k, v, mask)   # (B,h,Tq,dk)   ← scaled_dot_product_attention: softmax(QKᵀ/√dk)V → (출력, 어텐션 가중치)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.em_dim)   # head 병합 (concat)
        """
        쪼개져있던 헤드들을 다시 옆으로 이어붙여(Concat)원래의 3차원 형태(B,T,em_dim)로 복구한다.
        """
        return self.dense(out), weights
        #마지막으로 self.dense(out) -> 출력Projection을 거쳐 최종 결과를 반환한다.

#값 검증
D, H, T = 16, 4, 6
mine = MultiHeadAttention(D, H)
ref = nn.MultiheadAttention(D, H, batch_first=True)
with torch.no_grad():
    # 가중치를 복사 (ref 는 q,k,v 를 하나로 쌓아 저장)
    #파이토치 공식 nn.MultiheadAttention은 Q,K,V의 가중치를 하나로 묶어서in_proj_weight로관리
    ref.in_proj_weight.copy_(torch.cat([mine.wq.weight, mine.wk.weight, mine.wv.weight]))
    #따라서 내가 만든 모델(mine)의 가중치 wq,wk,wv들을 torch.cat으로 이어붙여 공식모델의 가중치 자리에 복사
    # 출발선을 똑같이 맞추는 작업
    ref.in_proj_bias.copy_(torch.cat([mine.wq.bias, mine.wk.bias, mine.wv.bias]))
    ref.out_proj.weight.copy_(mine.dense.weight); ref.out_proj.bias.copy_(mine.dense.bias)
x = torch.randn(2, T, D); cm = create_look_ahead_mask(T)   # ← create_look_ahead_mask: 미래 위치 True → 못 보게 가림
my_out, my_w = mine(x, x, x, cm)#똑같은 임의 텐서 x와 미래 차단 마스크 cm을 두 모델에 각각 집어 넣음
ref_out, ref_w = ref(x, x, x, attn_mask=cm, need_weights=True, average_attn_weights=False)
print('출력 일치   :', torch.allclose(my_out, ref_out, atol=1e-5))
print('어텐션맵 일치:', torch.allclose(my_w, ref_w, atol=1e-5), '| shape', tuple(my_w.shape), '= (B, heads, Tq, Tk)')
"""
torch.allclose(...,atol=1e-5) : 내가 만든 모델의 출력값 my_out과 공식모델의 출력값 ref_out이 
소수점 다섯 자리까지 완전 일치하는지 확인,
어텐션 맵 my_w와 ref_w의 형태가 (B,heads, Tq, Tk)로 정확히맞아 떨어지는지도 확인
"""
"""
출력 일치   : True
어텐션맵 일치: True | shape (2, 4, 6, 6) = (B, heads, Tq, Tk)
"""


# PyTorch가 자동으로 backward를 계산하지 않고,
# Scaled Dot-Product Attention의 미분 공식을 직접 구현한 함수
class ManualAttentionFn(torch.autograd.Function):
    """
    mask에서 True는 '가릴 위치'를 뜻한다.
    forward:  Attention(Q, K, V) = softmax(QKᵀ / √dk)V
    backward: 위 연산의 gradient를 수식대로 직접 계산한다.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # q, k, v의 마지막 차원 크기 = dk
        # QKᵀ 값이 너무 커져 softmax가 한쪽으로 치우치는 것을 막기 위한 스케일
        scale = 1.0 / math.sqrt(q.size(-1))

        # S: attention 점수, 형태 (..., Tq, Tk)
        # 각 Query 토큰과 Key 토큰 사이의 유사도를 계산한다.
        S = (q @ k.transpose(-1, -2)) * scale

        # mask가 True인 곳은 -inf로 바꾼다.
        # softmax(-inf) = 0 이므로 해당 위치에는 attention을 주지 않는다.
        if mask is not None:
            S = S.masked_fill(mask, float('-inf'))

        # P: attention 가중치, 마지막 차원(Tk)의 합은 1
        P = torch.softmax(S, dim=-1)

        # backward 계산에 필요한 값만 저장한다.
        # ctx는 forward에서 저장한 값을 backward에서 꺼내는 PyTorch용 공간이다.
        ctx.save_for_backward(q, k, v, P)
        ctx.scale = scale

        # 최종 attention 출력 O = PV, 형태 (..., Tq, Dv)
        return P @ v

    @staticmethod
    def backward(
        ctx,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        # forward 출력 O 하나에 대한 손실의 기울기
        # dO = ∂Loss / ∂O
        dO = grad_outputs[0]

        # forward에서 저장해 둔 값 복원
        q, k, v, P = ctx.saved_tensors

        # O = PV 이므로 V에 대한 gradient
        # dV = PᵀdO
        dV = P.transpose(-1, -2) @ dO

        # P에 대한 gradient
        # dP = dOVᵀ
        dP = dO @ v.transpose(-1, -2)

        # P = softmax(S)의 backward
        # dS = P * (dP - Σ(dP * P))
        dS = P * (dP - (dP * P).sum(-1, keepdim=True))

        # S = QKᵀ / √dk 이므로 Q와 K에 대한 gradient 계산
        dQ = dS @ k * ctx.scale
        dK = dS.transpose(-1, -2) @ q * ctx.scale

        # forward 입력은 q, k, v, mask 총 4개였으므로 gradient도 4개 반환
        # mask는 학습할 값이 아니므로 gradient가 없어 None 반환
        return dQ, dK, dV, None



# ── 직접 구현한 backward가 PyTorch autograd의 결과와 같은지 검증 ──

torch.manual_seed(0)  # 매번 같은 난수가 나오도록 고정

# Q, K, V 생성
# 형태: (B, heads, T, dk) = (2, 3, 5, 4)
# double: gradcheck(수치 미분 검사)는 정확도를 위해 float64를 권장한다.
# requires_grad=True: q, k, v에 대한 gradient를 계산하겠다는 뜻
q, k, v = [
    torch.randn(2, 3, 5, 4, dtype=torch.double, requires_grad=True)
    for _ in range(3)
]

# 길이 5 문장에서 미래 토큰을 가리는 causal mask
# 형태 (5, 5)지만 attention 계산 시 (B, heads, T, T)에 자동으로 적용된다.
cm = create_look_ahead_mask(5)

# 출력 O에 대해 위쪽(loss 쪽)에서 내려온다고 가정한 임의의 gradient
# dO = ∂Loss / ∂O, 형태는 attention 출력 O와 같다.
dO = torch.randn(2, 3, 5, 4, dtype=torch.double)

# 1) 일반 PyTorch 연산으로 attention을 계산한다.
# PyTorch autograd가 자동으로 구한 gradient를 '정답'으로 사용한다.
ref_out, _ = scaled_dot_product_attention(q, k, v, cm)
g_ref = torch.autograd.grad(ref_out, (q, k, v), dO)
# g_ref = (dQ, dK, dV)

# 2) ManualAttentionFn: 우리가 직접 작성한 backward로 gradient를 계산한다.
g_man = torch.autograd.grad(
    ManualAttentionFn.apply(q, k, v, cm),
    (q, k, v),
    dO,
)
# g_man = (dQ, dK, dV)

# 자동 미분 결과와 직접 구현한 결과가 같은지 확인
print(
    'dQ, dK, dV 일치:',
    [torch.allclose(a, b) for a, b in zip(g_ref, g_man)],
)

# gradcheck:
# backward 공식이 수치 미분 결과와도 일치하는지 검사한다.
# True면 직접 구현한 backward가 미분 관점에서도 정상이라는 뜻이다.
print(
    'gradcheck(수치미분) 통과:',
    torch.autograd.gradcheck(
        lambda a, b, c: ManualAttentionFn.apply(a, b, c, cm),
        (q, k, v),
    ),
)

"""
dQ, dK, dV 일치: [True, True, True]
gradcheck(수치미분) 통과: True
"""

# ── causal mask가 미래 토큰의 gradient도 막는지 확인 ──

# dO를 전부 0으로 만든 뒤,
# 두 번째 Query(인덱스 1)의 출력에서만 gradient가 내려오도록 1을 넣는다.
one = torch.zeros_like(dO)
one[:, :, 1] = 1

# 두 번째 Query 출력만의 영향을 받아 K에 생긴 gradient를 계산한다.
dk1 = torch.autograd.grad(
    ManualAttentionFn.apply(q, k, v, cm),
    k,
    one,
)[0]

# 두 번째 Query는 key 0, 1까지만 볼 수 있다.
# 따라서 미래 key인 2, 3, 4의 gradient 크기는 0이 된다.
print(
    '두 번째 query의 loss → key별 기울기 크기:',
    dk1[0, 0].norm(dim=-1).round(decimals=3).tolist(),
    '(key 2~4는 0)',
)
#참고로 “첫 query의 gradient가 0”인 이유는 볼 수 있는 key가 자기 자신 하나뿐이라 softmax 결과가 항상 `1`이고, Q·K 점수가 바뀌어도 attention 가중치가 바뀌지 않기 때문
#두 번째 query 의 loss → key 별 기울기 크기: [0.018, 0.018, 0.0, 0.0, 0.0] (key 2~4 는 0)




