# 🔥Transform

import math
import random
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops import embedding
from torch.distributed.autograd import context
import matplotlib.pyplot as plt

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
    TOPIC_NAMES,     # ['공부', '운동', '요리', '여행']
    Story,
    decode,
    EOS,BOS
)

#어텐션 + 마스크
def scaled_dot_product_attention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor | None = None) -> tuple[
    Tensor, Tensor]:
    scores = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(q.size(-1))
    #Q,K 행렬 내적한뒤, 값의 폭팔(기울기 소실)방지위해 Query의 차원제곱근(√dk)으로 나누어 유사도 점수(Score) 계산
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
        #마스크(패딩 또는 미래차단)가 존재한다면, True위치에 -inf채워
        # softmax후에 가중치가 정확이 0 이 되도록 한다.
    w = torch.softmax(scores, dim=-1)
    # 마지막 축 dim = -1을 기준으로 softmax를 취해 각 행의 합이 1이 되는 attention가중치(확률분포) w 구함
    return torch.matmul(w, v), w
    #가중치 w와 Value(v)를 곱해 최종 문맥 벡터를 완성하고, (최종 출력, 어텐션 가중치) 튜플로 반환



class MultiHeadAttention(nn.Module):
    def __init__(self, em_dim: int, num_heads: int) -> None:
        super().__init__()
        self.h, self.dk, self.em_dim = num_heads, em_dim // num_heads, em_dim
        #h: 헤드 개수, dk: 각 헤드가 담당할 차원 크기(전체 차원 // 헤드 개수), em_dim: 전체 임베딩 차원
        self.wq, self.wk, self.wv, self.dense = (nn.Linear(em_dim, em_dim) for _ in range(4))
        #Q, K, V 변환용 선형 층 3개와, 헤드들을 합친 뒤 원래 차원으로 되돌릴 출력 projection 층(dense) 1개를 한 번에 생성

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size(); return x.view(B, T, self.h, self.dk).transpose(1, 2)
        #(B, T, em_dim) 형태의 텐서를 (B, T, h, dk)로 쪼갠 뒤, transpose를 통해 (B, h, T, dk) 형태로 변환 (헤드별 독립 연산용)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B = q.size(0) # 배치 크기(Batch Size) 추출
        q, k, v = self.split_heads(self.wq(q)), self.split_heads(self.wk(k)), self.split_heads(self.wv(v))
        #입력 Q, K, V에 각각 선형 변환을 거친 뒤, split_heads를 통해 여러 헤드로 쪼갬 (형태: B, h, T, dk)
        out, w = scaled_dot_product_attention(q, k, v, mask)   # ← scaled_dot_product_attention: softmax(QKᵀ/√dk)V → (출력, 어텐션 가중치)
        #스케일드 닷 프로덕트 어텐션을 실행하여 (헤드별 출력 텐서, 어텐션 가중치 맵)을 획득:
        return self.dense(out.transpose(1, 2).contiguous().view(B, -1, self.em_dim)), w
        # head별 attention 결과를 다시 하나의 임베딩 벡터로 합친 뒤,
        #마지막 선형층(dense)을 통과시켜 최종 출력과 attention 가중치 w를 반환한다.


def create_padding_mask(seq: torch.Tensor, pad: int = PAD) -> torch.Tensor:
    """
    문장 속 PAD(빈칸) 위치를 찾아 attention이 보지 못하게 가리는 마스크 생성 함수
    (B, T) → (B, 1, 1, Tk)로 차원을 늘려
    모든 head와 모든 Query 위치에 같은 PAD 마스크가 적용되게 한다.
    """
    return (seq == pad).unsqueeze(1).unsqueeze(2)      # (B,1,1,Tk)


def create_look_ahead_mask(T: int) -> torch.Tensor:
    """
    현재 위치보다 뒤쪽의 미래 토큰을 보지 못하게 가리는 causal mask 생성 함수
    주대각선 바로 위(diagonal=1)부터 True:
    자기 자신과 과거는 볼 수 있고, 미래 토큰만 가린다.
    (T,T) 미래=True
    """
    return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)   # (T,T) 미래=True

#포지셔널 인코딩
class PositionalEncoding(nn.Module):
    """
    nn.Module을 상속받고, 임베딩 차원(em_dim)가 최대 문장 길이(max_pos=512)를 인자로 받음
    초기화 함수라 반환값 없음 -> None
    """
    def __init__(self,em_dim:int,max_pos:int=512)->None:
        super().__init__()
        pos = torch.arange(max_pos).unsqueeze(1)
        # 0 ~ 511 까지의 위치(Position)번호표를 만들고, 행렬 계산 위해 .unsqueeze(1)로 1차원을 늘려
        # (max_pos,1)형태의 세로 줄 모양으로 만든다.
        div = torch.exp(torch.arange(0,em_dim,2) * (-math.log(10000.0) / em_dim))
        """
        트랜스 포머 논문에 나오는 Frequency 계산 공식
        차원이 높아질수록 Frequency를 점점 낮게(완만하게)만들어주어, 모델이 단어들간의 상대적 거리나
        위치를 부드럽게 인식 가능하도록 주기함수(sin/cos)의 간격을 조절해주는 값
        """
        pe = torch.zeros(1, max_pos, em_dim)
        #미리 계산된 위치 정보드르을 담아둘 빈 도화지 (텐서)를 0으로 가득채워 만든다.(1*512*임베딩차원)
        pe[0, :, 0::2] = torch.sin(pos * div)  # 짝수 차원 sin
        pe[0, :, 1::2] = torch.cos(pos * div)  # 홀수 차원 cos
        # -> 리스트가 아닌 앞서만든 pe라는 공간에 텐서공간(행렬)에 직접 넣는것
        self.register_buffer('pe', pe)  # 학습되지 않는 상수 (state_dict 에는 저장)
        """
        pe는 사람이 직접훈련시키는것 아닌, 고정된 상수! 따라서 학습중에 변하지 않도록 딥러닝모델의
        뼈대에 버퍼로 등록해주는 코드
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        """
        실제 입력 데이터 x (B,T,D)형태가 들어왔을때 실행되는 순전파 함수
        문장의 실제 길이(x.size(1))만큼만 위치 정보(pe)를 잘라내서 원래 입력단어벡터(x)에
        위치 정보(pe)를 그대로 더해서 반환.
        transfomer는 순서 개념이 없기때문에 단어 의미 x에 위치 pe를 더해주는것
        """
        return x + self.pe[:, :x.size(1)]

pe = PositionalEncoding(16,50).pe[0]
#임베딩 차원16,최대길이50으로 positional Encoding객체를 만들고,
# 그안의 pe 값 중 1번째 배치[0]을 꺼낸다.
plt.figure(figsize=(7,2.6))
plt.imshow(pe.T, aspect='auto',cmap='RdBu')
plt.xlabel('Position')
plt.ylabel('dim')
plt.title('Sinusoidal PE')
plt.colorbar()
plt.show()

#인코더 디코더 레이어
"""
예를 들어 n_layers=2라면 구조는 이렇게 됩니다.
입력 토큰
  ↓
Encoder
  ├─ Embedding + Positional Encoding
  ├─ EncoderLayer 1
  └─ EncoderLayer 2
  ↓
enc_out
  ↓
Decoder
  ├─ Embedding + Positional Encoding
  ├─ DecoderLayer 1
  └─ DecoderLayer 2
  ↓
단어 예측
"""
def point_wise_ffn(em_dim:int,feed_dim:int) -> nn.Sequential:
    """
    선형변환 -> 활성화함수(ReLU)->선형변환으로 이어지는 기본 신경망 블록.
    어텐션을 거친 데이터의 특징을 한번 더 비 선형적으로 변환
    """
    return nn.Sequential(nn.Linear(em_dim,feed_dim),nn.ReLU(),nn.Linear(feed_dim,em_dim))

class EncoderLayer(nn.Module):
    """
    self_attention과 FFN을 거친다.
    이때 Add&Norm(잔차연결 + 레이어 정규화)구조가 겹겹이 쌓여 입력 문장의 의미를 깊게 이해
    """
    def __init__(self,em_dim:int,num_heads:int,feed_dim:int,rate:float = 0.1) -> None:
        super().__init__()
        self.mha,self.ffn = MultiHeadAttention(em_dim,num_heads),point_wise_ffn(em_dim,feed_dim)
        self.ln1,self.ln2 = nn.LayerNorm(em_dim,eps=1e-6), nn.LayerNorm(em_dim,eps=1e-6)
        self.do1,self.do2 = nn.Dropout(rate),nn.Dropout(rate)
    def forward(self,x:torch.Tensor,mask:torch.Tensor | None = None) -> torch.Tensor:
        a,_ = self.mha(x,x,x,mask)
        out1 = self.ln1(x+self.do1(a))
        return self.ln2(out1 + self.do2(self.ffn(out1)))

class DecoderLayer(nn.Module):
    """
    maksed-self=attention(mha1,미래토큰차단)과 cross-attention(mah2,인코더의결과물과 디코더 연결)
    그리고 FFN으로 구성.
    번역이나 문장 생성시 인코더가 읽은 원문 정보(enc_out)참고하면서 글을 써내려간다.
    """
    def __init__(self,em_dim:int,num_heads:int,feed_dim: int, rate:float = 0.1) -> None:
        super().__init__()
        self.mha1,self.mha2 = MultiHeadAttention(em_dim,num_heads),MultiHeadAttention(em_dim,num_heads)
        self.ffn = point_wise_ffn(em_dim,feed_dim)
        self.ln1,self.ln2,self.ln3 = (nn.LayerNorm(em_dim,eps=1e-6) for _ in range(3))
        self.do1, self.do2, self.do3 = (nn.Dropout(rate) for _ in range(3))

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, look_ahead_mask: torch.Tensor | None = None, padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a1,w1 = self.mha1(x,x,x,look_ahead_mask)
        out1 = self.ln1(x+self.do1(a1))
        a2,w2 = self.mha2(out1,enc_out,enc_out,padding_mask)
        out2 = self.ln2(out1 + self.do2(a2))
        out3 = self.ln3(out2 + self.do3(self.ffn(out2)))
        return out3, w1, w2


class Encoder(nn.Module):
    def __init__(self, n_layers: int, em_dim: int, heads: int, feed_dim: int, vocab: int, rate: float = 0.1) -> None:
        super().__init__()
        self.em_dim, self.emb, self.pe = em_dim, nn.Embedding(vocab, em_dim, padding_idx=PAD), PositionalEncoding(em_dim)
        self.layers = nn.ModuleList([EncoderLayer(em_dim, heads, feed_dim, rate) for _ in range(n_layers)])
        self.drop = nn.Dropout(rate)
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.drop(self.pe(self.emb(x) * math.sqrt(self.em_dim)))
        # 임베딩 × √D (논문 §3.4)
        # 포지셔널 인코딩을 더하기전, 임베딩값의 스케일을 키워줘서 학습을 안정화하는 트릭
        for layer in self.layers: x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers: int, em_dim: int, heads: int, feed_dim: int, vocab: int, rate: float = 0.1) -> None:
        super().__init__()
        self.em_dim, self.emb, self.pe = em_dim, nn.Embedding(vocab, em_dim, padding_idx=PAD), PositionalEncoding(em_dim)
        self.layers = nn.ModuleList([DecoderLayer(em_dim, heads, feed_dim, rate) for _ in range(n_layers)])
        self.drop = nn.Dropout(rate)
    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, look_ahead_mask: torch.Tensor | None = None, padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.drop(self.pe(self.emb(x) * math.sqrt(self.em_dim)))
        attn = {}
        for i, layer in enumerate(self.layers):
            x, w1, w2 = layer(x, enc_out, look_ahead_mask, padding_mask)
            attn[f'layer{i+1}_self'], attn[f'layer{i+1}_cross'] = w1, w2
        return x, attn

class Transformer(nn.Module):
    """
    내부적으로 Encoder와 Decoder를 품고있으며 크게 2가지 핵심 메서드를 제공.
    forward(학습용,Teacher Forcing) : 정답 문장을 통째로 한번에 집어넣어 병렬 연산으로 빠르게 학습
    generate(추론용,자기회귀Autogressive) : 단어를 하나씩 순서대로 생성, 문장을 완성(챗봇이나 번역기가 글을 뱉어내는방식)
    """
    def __init__(self,n_layers: int, em_dim: int, heads: int, feed_dim: int, vocab: int,tgt_vocab:int,rate:float = 0.1) -> None:
        super().__init__()
        self.encoder = Encoder(n_layers, em_dim, heads, feed_dim, tgt_vocab, rate)
        self.decoder = Decoder(n_layers, em_dim, heads, feed_dim, tgt_vocab, rate)
        self.final_layer = nn.Linear(em_dim, tgt_vocab)
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p) #nn.init.xavier_uniform_를 통해 모델이 처음 학습할때 가중치가 엉뚱하게 튀지않게 초기화

    def create_masks(self,src: torch.Tensor,tgt: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 인코더 self-attention: src의 PAD를 가림
        enc_pad = create_padding_mask(src)

        # 디코더 self-attention: 미래 토큰 + tgt의 PAD를 가림
        look = create_look_ahead_mask(tgt.size(1)).to(tgt.device)[None, None]
        dec_self = look | create_padding_mask(tgt)

        # 디코더 cross-attention: src의 PAD를 가림
        cross_mask = enc_pad

        return enc_pad, dec_self, cross_mask

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 학습: 모든 위치를 한 번에 (병렬, teacher forcing)
        enc_mask, dec_self_mask, cross_mask = self.create_masks(src, tgt_in)
        enc_out = self.encoder(src, enc_mask)
        dec_out, attn = self.decoder(tgt_in, enc_out, dec_self_mask, cross_mask)
        return self.final_layer(dec_out), attn

    @torch.no_grad()
    def generate(self, src: torch.Tensor, start: int, end: int, max_new_tokens: int = 20) -> list[int]:
        # 추론: 토큰을 하나씩 (자기회귀)
        self.eval(); src = src.to(next(self.parameters()).device)
        enc_mask = create_padding_mask(src); enc_out = self.encoder(src, enc_mask)
        # 인코더는 한 번만   ← create_padding_mask: PAD 자리 True → 어텐션이 무시
        out = [start]
        for _ in range(max_new_tokens):
            cur = torch.tensor([out], device=src.device)
            _, dec_self, cross = self.create_masks(src, cur)
            dec_out, _ = self.decoder(cur, enc_out, dec_self, cross)
            nxt = self.final_layer(dec_out[:, -1]).argmax(-1).item()
            # 마지막 위치 로짓만 사용 (greedy)
            out.append(nxt)
            if nxt == end: break
        return out

m = Transformer(2, 64, 4, 128, V, V).to(device)
print('파라미터 수:', sum(p.numel() for p in m.parameters())) #파라미터 수: 181127

# seq 2 seq
def make_pairs(stories: list[Story]) -> list[tuple[list[int], list[int]]]:
    pairs = []
    """
    하나의 이야기(st)안에 여러문장이 순서대로있을때,앞문장 st[i]을 보고, 다음 문장 st[i+1]을 맞히거나
    이어쓰는 학습을 하기 위해 문장들을 짝(Pair)으로 묶어줍니다.
    rage(3)을 통해 1번 -> 2번 / 2번 -> 3번 / 3번 -> 4번 문장 쌍을 만듦
    ----------------------------------------------------------------------------
    src(인코딩/입력문장) : 문장 끝에 EOS(End of Sentence)을 붙여 여기서 문장이 끝난다는 신호
    tgt(디코더/정답문장) : 문장 맨앞에 BOS(Beginning of Sentence)를 붙여 '지금부터 생성(시작)해라'
                         끝에 EOS를 붙여 끝맺음 알려줍니다. 
                       (전형적인 AutoGressive/생성모델의 정답 포멧)
    """
    for st, _ in stories:
        for i in range(3):# (문장1→2), (2→3), (3→4)
            pairs.append((encode(st[i]) + [EOS], [BOS] + encode(st[i + 1]) + [EOS]))
            # ← encode: 문장(str) → 숫자 id 리스트
    return pairs
train_pairs, valid_pairs = make_pairs(TRAIN_STORIES), make_pairs(VALID_STORIES)
print(len(train_pairs), len(valid_pairs)); print('src:', decode(train_pairs[0][0]), '\ntgt:', decode(train_pairs[0][1]))   # ← decode: 숫자 id 리스트 → 문장(str)
#decode함수를 사용해서 1번째 만들어진 데이터쌍(train_paris[0])의 숫자ID리스트들을 사람이 읽을 수 있는 원래 문장
#src와 tgt로 변환해서 데이터가 의도대로 잘 짝지어졋는지 눈으로 확인(디버깅)
"""
1440 360
src: 서아는 학교에서 영어를 공부했다 
tgt: 서아는 단어를 외우고 문제를 풀었다
"""

def pair_batches(pairs: list[tuple[list[int], list[int]]], bs: int, shuffle: bool = True) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Attention_prac의 batches함수가 거의 똑같지만, 입력 문장 c[0],c[1]과
    정답 문장c[1],tgt 한쌍으로 묶여서 움직인다는 점이 다름!
    배치 사이즈(bs)만큼 데이터를 무작위로 골라낸뒤, 입력 문장들과 정답문장들 각각에 대해
    길이 다르면 빈자리를 PAD(0)로 채워 네모난 텐서(B,L)로 맞춘 후 pad_batch,GPU/CPU로 실어 나른다.
    """
    idx = list(range(len(pairs)))
    if shuffle: random.shuffle(idx)
    for i in range(0, len(idx), bs):
        ch = [pairs[j] for j in idx[i:i + bs]]
        yield pad_batch([c[0] for c in ch]).to(device), pad_batch([c[1] for c in ch]).to(device)   # ← pad_batch: 길이가 다른 리스트들 → PAD(0)로 채워 (B, L) 텐서

def noam_lr(step: int, em_dim: int, warmup: int = 150, factor: float = 0.4) -> float:
    """
    트랜스포머는 학습 초기에 학습률(Learning Rate)이 너무 크면 모델이 엉뚱한 방향으로 튈수있고,너무작으면 학습안됨
    1.Warmup단계(step * warmup ** -1.5)
    : 학습초기에는 학습률을 0부터 서서히 높여가며 모델을 안정적으로 예열(warmup)합니다.
    2.감쇠단계(step ** -0.5)
    : 정해진 피크(warmup 시점)을 찍고나면 학습이 진행될수록 학습률을 서서히 낮춰가며 정교하게 수렴하도록 만듦
    """
    step = max(step, 1)
    return factor * em_dim ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)

model = Transformer(2, 64, 4, 128, V, V, rate=0.1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
"""
트랜스포머 논문에서 지정한 Adam 옵티마이저 표준 하이퍼파라미터 그대로
lt = 1.0으로 잡혀있지만, 어차피 매 step마다 noam_lr이 학습률을 알아서 덮어씌워 제어한다.
"""
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
"""
ignore_index = PAD : 문장 길이를 맞추기 위해 억지로 채워 넣은 빈자리(PAD)는 오차 계산(Loss)에 완전히 제외
label_smooting=0.1 : 정답을 너무 꽉 막힌 1.0으로 학습 시키면, 모델이 지나치게 자신감(OverConfident)을 갖게 되므로,
                     약간의 여유(노이즈)를 주어 일반화 성능을 높여주는 기법
"""

step, hist = 0, []
for epoch in range(12):
    model.train(); tot, n = 0, 0
    #Training Loop
    for src, tar in pair_batches(train_pairs, 64):
        #앞서 만든 pair_batches를 통해 입력문장(src)과 정답문장(tar)을배치크기 64개씩가져옴
        tar_in, tar_out = tar[:, :-1], tar[:, 1:]
        """
        Teacher-Forcing기법 : 디코더에게 공부 시킬때 사용하는 핵심트릭
        tar_in(입력용 정답) : 정답 문장의 맨 마지막 단어만 빼고,앞부분을 잘라서 디코더에 넣습니다.[BOS,단어1,단어2,단어3]
        tar_out(타겟용 정답) : 정답문장의 맨앞 BOS토큰만 빼고,뒤로 한칸씩 밀린상태를 만듭니다.[단어1,단어2,단어3,EOS]
        즉, 디코더에게 지금까지의 단어를 보고 바로 다음 단어가 무엇인지 맞춰봐! 하며 가르치는 방식
        """
        step += 1
        for g in opt.param_groups: g['lr'] = noam_lr(step, 64)
        #매 스텝마다 진행되는상황에 맞춰 noam스케줄러가 계산한 최적의 학습률(lr)을 옵티마이저에 갱신해줌
        logits, _ = model(src, tar_in)# (B, T, V)
        """
        트랜스포머모델에 원문(src)과 입력용 정답(tar_in)을 넣고 예측결과(logits)를 뽑아낸다.
        """
        loss = loss_fn(logits.reshape(-1, V), tar_out.reshape(-1))# (B*T, V) vs (B*T)
        """
        모델의 3차원 출력결과(B,T,V)와 2차원 정답(B,T)을 2차원/1차원 형태 길게 펼쳐서(reshape)
        cross-entropy 오차를 계산
        """
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        """
        딥러닝 학습 도중 값들이 너무 튀어서 기울기 폭발이 일어나는 것을 막기 위해,
        기울기의 최댓값을 1.0으로 단단하게 붙들어줍니다.
        """
        opt.step(); tot += loss.item(); n += 1
    hist.append(tot / n); print(f'epoch {epoch+1:2d} | loss {tot/n:.4f} | lr {opt.param_groups[0]["lr"]:.5f}')
    """
    epoch  1 | loss 4.0748 | lr 0.00063
    epoch  2 | loss 2.4942 | lr 0.00125
    epoch  3 | loss 1.0832 | lr 0.00188
    epoch  4 | loss 0.8195 | lr 0.00250
    epoch  5 | loss 0.7949 | lr 0.00313
    epoch  6 | loss 0.7810 | lr 0.00376
    epoch  7 | loss 0.7718 | lr 0.00394
    epoch  8 | loss 0.7651 | lr 0.00369
    epoch  9 | loss 0.7609 | lr 0.00348
    epoch 10 | loss 0.7577 | lr 0.00330
    epoch 11 | loss 0.7550 | lr 0.00314
    epoch 12 | loss 0.7530 | lr 0.00301
    """

#평가 : teacher forcing
model.eval()
# 정답 문장을 통째로 디코더에 밀어 넣었을때(Teacher Forcing)
# 모델이 다음 단어를 각각 얼마나 정확하게 맞히는지 글자(토큰)단위로 채점
with torch.no_grad():
    tf_ok = tf_all = 0
    for src, tar in pair_batches(valid_pairs, 128, shuffle=False):
        pred = model(src, tar[:, :-1])[0].argmax(-1);
        #모델이 예측한 가장 확률 높은 단어의 ID를 가져옵니다.
        gold = tar[:, 1:];
        # 실제 정답 단어 ID들
        keep = gold != PAD
        # 문장끝의 빈자리(PAD)는 채점대상에 제외하기위해 마스크를 만든다.
        tf_ok += ((pred == gold) & keep).sum().item();
        tf_all += keep.sum().item()
        """
        마스크된 유효한 단어들 중 모델이 정확히 맞힌 개수의 비율(정확도)을 계산하여 출력
        """
print(f'teacher-forcing 토큰 정확도: {tf_ok / tf_all:.4f}') #teacher-forcing 토큰 정확도: 1.0000

exact = 0
"""
힌트(정답)를 주고, 맞히는게 아니라 모델이 generate함수를 통해 처음부터 끝까지 스스로 단어를 하나씩
이어붙여 만든 문장(gen)이 실제로 정답 문장(tar_ids)과 쉼표 하나까지 완벽하게 일치하는지 채점
번역이나 챗봇 성능을 평가할때, 가장 신뢰할수있는 척도 중 하나.
100문장 중 몇 문장을 오탈자 없이 완벽하게 복원해 냈는지를 비율로 보여줌
"""
for src_ids, tar_ids in valid_pairs:
    gen = model.generate(torch.tensor([src_ids]), BOS, EOS)
    exact += int(gen == tar_ids)
print(f'generate 문장 완전일치(exact match): {exact / len(valid_pairs):.4f}') #generate 문장 완전일치(exact match): 1.0000
for src_ids, tar_ids in valid_pairs[:4]:
    """
    1. 입력 문장(src)이 무엇이었고
    2. 모델이 만들어야 할 진짜 정답(tar)무엇이고,
    3. 트랜스포머 모델이 스스로 생성해 낸 결과물 (gen) 무엇인지
    4. decode함수를 통해 사람이 읽을 수 있는 글자로 변환해서 나란히 출력
    -> 모델 번역이나 문장 생성을 사람처럼 자연스럽게 잘해내고있는 지 직관적으로 검사하는 최종 디버깅용 코드
    """
    gen = model.generate(torch.tensor([src_ids]), BOS, EOS)
    print(f'입력: {decode(src_ids)}\n  정답: {decode(tar_ids)}\n  생성: {decode(gen)}')  # ← decode: 숫자 id 리스트 → 문장(str)
    """
입력: 하나는 시장에서 채소를 샀다
  정답: 하나는 집에서 채소를 썰고 국을 끓였다
  생성: 하나는 집에서 채소를 썰고 국을 끓였다
입력: 하나는 집에서 채소를 썰고 국을 끓였다
  정답: 하나는 완성된 국을 식탁에 놓았다
  생성: 하나는 완성된 국을 식탁에 놓았다
입력: 하나는 완성된 국을 식탁에 놓았다
  정답: 하나는 가족과 함께 맛있게 먹었다
  생성: 하나는 가족과 함께 맛있게 먹었다
입력: 도하는 아침 일찍 기차를 탔다
  정답: 도하는 창밖으로 바다를 구경했다
  생성: 도하는 창밖으로 바다를 구경했다
    """

src, tar = next(pair_batches(train_pairs, 32))
model.train()

def grad_report(tag: str) -> None:
    """
    모델 내부에 있는 핵심 부품들(인코더 임베딩, 인코더 MHA 가중치, 디코더 크로스어텐션,최종 Linear 등)의 기울기 크기(|grad|)를
    측정해서 출력해주는 디버깅 도구.
    기울기가 크다는 것은 이 레이어가 오차를 학습하며 열심히 갱신되고 있다는 뜻.
    기울기가 0.0(또는 None)이라는것은 학습이 전혀 안되고 방치되어 있다는 뜻.
    """
    names = {'encoder.emb.weight': '인코더 임베딩', 'encoder.layers.0.mha.wq.weight': '인코더 L1 Wq',
             'decoder.layers.0.mha2.wk.weight': '디코더 L1 cross-attn Wk', 'decoder.emb.weight': '디코더 임베딩',
             'final_layer.weight': '최종 Linear'}
    print(f'[{tag}]')
    for n, p in model.named_parameters():
        if n in names:
            g = 0.0 if p.grad is None else p.grad.norm().item()
            print(f'   {names[n]:<24} |grad| = {g:.5f}')
            """
        [정상: 인코더에도 기울기 도달]
           인코더 임베딩                  |grad| = 0.00321
           인코더 L1 Wq                |grad| = 0.00127
           디코더 임베딩                  |grad| = 0.00515
           디코더 L1 cross-attn Wk     |grad| = 0.00107
           최종 Linear                |grad| = 0.01223
        [enc_out.detach(): 인코더 기울기 없음 (None→0)]
           인코더 임베딩                  |grad| = 0.00000
           인코더 L1 Wq                |grad| = 0.00000
           디코더 임베딩                  |grad| = 0.00520
           디코더 L1 cross-attn Wk     |grad| = 0.00120
           최종 Linear                |grad| = 0.01276
            """

# ① 정상 backward
model.zero_grad()
logits, _ = model(src, tar[:, :-1]); loss_fn(logits.reshape(-1, V), tar[:, 1:].reshape(-1)).backward()
grad_report('정상: 인코더에도 기울기 도달')
"""
평소처럼 인코더와 디코더를 통과시켜 오차(Loss)구한뒤 backword()를 합니다.
디코더에서 발생한 오차의 역전파(Backpropagation)가 cross-attention을 타고 인코더 영역까지 흘러감
인코더와 디코더가 하나의 팀으로 묶여 양쪽 다 정상적으로 학습
"""

# ② enc_out 을 끊고(detach) backward
model.zero_grad()
enc_mask, dec_self, cross = model.create_masks(src, tar[:, :-1])
enc_out = model.encoder(src, enc_mask).detach()                    # ← 여기서 그래프 절단
dec_out, _ = model.decoder(tar[:, :-1], enc_out, dec_self, cross)
loss_fn(model.final_layer(dec_out).reshape(-1, V), tar[:, 1:].reshape(-1)).backward()
grad_report('enc_out.detach(): 인코더 기울기 없음 (None→0)')
"""
인코더가 만든 결과물(enc_out)에 .detach()를 붙여버림

.detach()의 의미 
: 지금 이순간부터 이 텐서를 딥러닝 계산 그래프에서 잘라내어(Detach)고정된 상수 취급하겠다는 뜻
즉, 역전파(기울기)가 이선을 넘서 인코더 쪽으로 더 이상 과거로 거슬러 올라가지 못하고 차단.
디코더 쪽에 레이어들은 정상적으로 기울기가 찍히지만, 인코더 관련 레이어들의 기울기는 전부 0.0떨어짐
인코더는 전혀 학습 되지 않고 멈춰버림
"""


