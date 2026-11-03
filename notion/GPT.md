# GPT

import math
import random
from typing import Iterator
import torch
import numpy as np
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

BLOCK = 32

def pack_stories(stories: list[Story]) -> list[int]:
    """
    개별 문장이나 이야기들을 따로 노는 상태로 두지않고, 문장 시작엔 BOS, 끝엔 EOS를 붙여가며
    끝없이 이어지는 하나의 거대한 숫자 토큰 스트림(stream)으로 엮어버림.
    GPT같은 대형언어 모델은 문장 단위가 아니라 토큰틀의 긴 흐름 속에서 다음 단어를 예측하는 방식
    으로 학습하기때문에, 이렇게 일렬로 촘촘하게 엮어주는 작업이 필수적
    """
    stream = []
    for st, _ in stories:  # 문서 하나 = 이야기 4문장
        stream += [BOS] + [w for s in st for w in encode(s)] + [EOS]  # ← encode: 문장(str) → 숫자 id 리스트
    return stream

def make_chunks(stream: list[int], block: int = BLOCK) -> torch.Tensor:
    """
    길게 늘어선 토큰 스트림 BLOCK+1(여기서는 32+1=33개)크기만큼씩 칼로 자르듯 잘라냄
    맨 마지막 조각이 33개보다 모자라면 남는 공간을 PAD로 채워 텐서크기를 반듯하게 맞춤
    """
    chunks = [stream[i:i + block + 1] for i in range(0, len(stream), block + 1)]
    if len(chunks[-1]) < block + 1:                       # 마지막 자투리는 PAD 로 채움
        chunks[-1] += [PAD] * (block + 1 - len(chunks[-1]))
    return torch.tensor(chunks)

train_chunks, valid_chunks = make_chunks(pack_stories(TRAIN_STORIES)), make_chunks(pack_stories(VALID_STORIES))   # ← pack_stories: 이야기들을 BOS/EOS 로 이어 붙인 긴 숫자 스트림; make_chunks: 스트림을 일정 길이로 잘라 (N, BLOCK+1) 텐서
x, y = train_chunks[0, :-1], train_chunks[0, 1:]
"""
Next-Token Prediction : 다음 단어 예측 학습
입력(x) : 첫번째 토큰 부터 마지막 바로 전 토큰까지(:-1)
라벨(y) : 두번째 토큰 부터 맨 끝 토큰까지(1:)
즉, 모델에게 이 글자를 입력으로 줄테니, 한 칸씩 밀린 다음 글자가 뭔지 맞춰봐!
하고 가르치는 GPT 학습의 데이서 셋팅 방식.

긴스트림으로 묶어서(Packing)학습하면, 문장 마다 길이가 달라서 어쩔 수 없이 생기는 빈공간(PAD)을 최소화
문장마다 따로 길이를 맞출때(Padding)비해, 이렇게 통째로 패킹해서 학습하면 낭비되는 공간(패딩비율)이 얼마나 줄어드는지
효울성 수치로 비교해서 보여주는 모니터링 코드
"""
print('청크 수 train/valid:', len(train_chunks), len(valid_chunks), '| 청크 shape', tuple(train_chunks.shape))
print('입력:', decode(x.tolist()), '\n라벨:', decode(y.tolist()))   # ← decode: 숫자 id 리스트 → 문장(str)
pad_ratio = (train_chunks == PAD).float().mean().item()
print(f'패킹 후 패딩 비율 {pad_ratio:.4f}  (문장별 패딩이었다면 약 {1 - np.mean([len(encode(s)) for st,_ in TRAIN_STORIES for s in st]) / max(len(encode(s)) for st,_ in TRAIN_STORIES for s in st):.2f})')   # ← encode: 문장(str) → 숫자 id 리스트
"""
청크 수 train/valid: 316 80 | 청크 shape (316, 33)
입력: 서아는 학교에서 영어를 공부했다 서아는 단어를 외우고 문제를 풀었다 서아는 공부를 마치고 커피를 마셨다 서아는 집에 돌아가 일찍 잠들었다 도하는 시장에서 채소를 샀다 도하는 집에서 채소를 썰고 국을 끓였다 
라벨: 서아는 학교에서 영어를 공부했다 서아는 단어를 외우고 문제를 풀었다 서아는 공부를 마치고 커피를 마셨다 서아는 집에 돌아가 일찍 잠들었다 도하는 시장에서 채소를 샀다 도하는 집에서 채소를 썰고 국을 끓였다 도하는
패킹 후 패딩 비율 0.0016  (문장별 패딩이었다면 약 0.30)
"""

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    if mask is not None: scores = scores.masked_fill(mask, float('-inf'))
    w = torch.softmax(scores, dim=-1)
    return torch.matmul(w, v), w

class MultiHeadAttention(nn.Module):
    def __init__(self, em_dim: int, num_heads: int) -> None:
        super().__init__()
        self.h, self.dk, self.em_dim = num_heads, em_dim // num_heads, em_dim
        self.wq, self.wk, self.wv, self.dense = (nn.Linear(em_dim, em_dim) for _ in range(4))
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size(); return x.view(B, T, self.h, self.dk).transpose(1, 2)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B = q.size(0)
        q, k, v = self.split_heads(self.wq(q)), self.split_heads(self.wk(k)), self.split_heads(self.wv(v))
        out, w = scaled_dot_product_attention(q, k, v, mask)   # ← scaled_dot_product_attention: softmax(QKᵀ/√dk)V → (출력, 어텐션 가중치)
        return self.dense(out.transpose(1, 2).contiguous().view(B, -1, self.em_dim)), w

class GPTBlock(nn.Module):
    def __init__(self, D: int, H: int, ff: int, rate: float, pre_ln: bool = True) -> None:
        """
        GPT-1은 연산 후에 LayerNorm을 거치는 Post-LN을 썼으나,학습이 불안정하다는 단점 존재
        GPT-2스타일인 Pre-LN(LN을 먼저 거치고, 연산후 잔차 연결)을 기본값으로 채택.
        ReLU대신 부드러운 곡선 형태를 가진 GELU를 사용하여 성능 높였습니다.
        """
        super().__init__()
        self.pre_ln = pre_ln
        self.mha = MultiHeadAttention(D, H)
        self.ffn = nn.Sequential(nn.Linear(D, ff), nn.GELU(), nn.Linear(ff, D))
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.do1, self.do2 = nn.Dropout(rate), nn.Dropout(rate)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pre_ln:                                            # GPT-2: LN → sublayer → (+x)   (residual 경로에 LN 없음)
            h = self.ln1(x); a, _ = self.mha(h, h, h, mask)
            x = x + self.do1(a)
            return x + self.do2(self.ffn(self.ln2(x)))
        a, _ = self.mha(x, x, x, mask)                             # GPT-1: sublayer → (+x) → LN
        x = self.ln1(x + self.do1(a))
        return self.ln2(x + self.do2(self.ffn(x)))

class GPT(nn.Module):
    """
    토큰 임베딩 + 학습 가능한 위치 인코딩(pos_emb)
    : 앞서 본 트랜스포머의 수학적 고정상수(sin/cos)방식과 달리, GPT는 위치 정보 자체를 임베딩 테이블
    (nn.Embedding(block,D))로 선언하여 모델이 직접학습하게 만듦
    """
    def __init__(self, vocab: int, D: int = 64, H: int = 4, ff: int = 256, n_layers: int = 3, block: int = BLOCK, rate: float = 0.1, pre_ln: bool = True) -> None:
        super().__init__()
        self.block = block
        self.tok_emb, self.pos_emb = nn.Embedding(vocab, D, padding_idx=PAD), nn.Embedding(block, D)
        self.drop = nn.Dropout(rate)
        self.blocks = nn.ModuleList([GPTBlock(D, H, ff, rate, pre_ln) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D) if pre_ln else nn.Identity()
        self.lm_head = nn.Linear(D, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight# weight tying
        """
        입력 단어를 벡터로 바꿔주는 임베딩 층의 가중치와 마지막에 최종 단어를 예측하기 위해 점수를 내는 
        출력층(lm_head)의 가중치를 완전치를 완전히 똑같이 공유합니다. 파라미터 낭비를 줄이고 언어 모델의 성능을 극대화하는
        핵심 정석 기법입니다.
        """
        for n, p in self.named_parameters():
            if p.dim() > 1: nn.init.normal_(p, 0.0, 0.02)
            """
            특수 가중치 초기화
            모든 2차원 이상의 가중치를 평균0,표준편차0.02인정규분포로 초기화
            특히 잔차 연결이 거듭 될수록 값이 튀는걸 막기 위해, 블록 끝단 레이어들(dense.weight,ffn.2.weight)의 초기화 표준편차를
            줄여주는 고급 안정화기법이 포함되어 있습니다.
            """
            if n.endswith('dense.weight') or n.endswith('ffn.2.weight'):# GPT-2: residual 출력은 1/√(2N)
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layers))

    def forward(self, idx: torch.Tensor, inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx) if inputs_embeds is None else inputs_embeds
        x = self.drop(x + self.pos_emb(torch.arange(T, device=idx.device)))
        mask = create_look_ahead_mask(T).to(idx.device)# (T,T) 상삼각=가림 → 모든 head/batch 에 브로드캐스트   ← create_look_ahead_mask: 미래 위치 True → 못 보게 가림
        for blk in self.blocks: x = blk(x, mask)
        return self.lm_head(self.ln_f(x))  # (B, T, V)

def create_look_ahead_mask(T: int) -> torch.Tensor:
    """
    GPT는 자기보다 미래에 올 단어를 미리보고 치트키를 쓰면 안되므로!
    상삼각 행렬 형태로 미래를 가려주는 마스크를 걸어 순서대로만 단어를 예측하도록 통제
    """
    return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

gpt = GPT(V).to(device)
print('파라미터 수:', sum(p.numel() for p in gpt.parameters()), '| 임베딩과 lm_head 가중치 공유:', gpt.lm_head.weight is gpt.tok_emb.weight)
#파라미터 수: 156672 | 임베딩과 lm_head 가중치 공유: True

def run_eval(model: nn.Module, chunks: torch.Tensor, bs: int = 64) -> dict[str, float]:
    """
    검증 데이터(chunks)를 받아 모델이 다음 단어를 얼마나 잘 예측하는지 종합적으로 채점

    loss(cross_entropy오차) : 낮을수록 모델의 예측이 정답에 가까움
    ppl(perplexity) : 언어 모델의 성능을 평가하는 대표적 지표.
                     모델이 다음 단어를 고를 때 몇가지 후보중에 헷갈리고있는가를 뜻함. 낮을 수록 좋습니다(완벽하게 1.0수렴)
    top1/ top5 Accuracy : 모델이 1순위로 예측한 단어가 정답일 확률(top1),그리고 모델이 꼽은 상위5개 후보안에 정답이 포함되어 있을 확률(top5)을 측정
    """
    model.eval(); nll = n = top1 = top5 = 0
    with torch.no_grad():
        for i in range(0, len(chunks), bs):
            c = chunks[i:i + bs].to(device); x, y = c[:, :-1], c[:, 1:]
            logits = model(x); keep = y != PAD
            nll += F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=PAD, reduction='sum').item()
            top = logits.topk(5, -1).indices
            top1 += ((top[..., 0] == y) & keep).sum().item(); top5 += ((top == y[..., None]).any(-1) & keep).sum().item(); n += keep.sum().item()
    return {'loss': nll / n, 'ppl': math.exp(nll / n), 'top1': top1 / n, 'top5': top5 / n}

print(f'초기 상태 loss={run_eval(gpt, valid_chunks)["loss"]:.3f}  vs  ln(V)={math.log(V):.3f}   ← 거의 같아야 정상')
#초기 상태 loss=4.314  vs  ln(V)=4.263   ← 거의 같아야 정상

opt = torch.optim.AdamW(gpt.parameters(), lr=2e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=25 * math.ceil(len(train_chunks) / 32))
"""
AdamW : 일반 Adam에 가중치 감쇠(Weight Decay,규제화)를 올바르게 적용하여 오버핏을 막아주는 표준 옵티마이저
oneCycleLR : 학습 초반에는 학습률 빠르게 끌어올리고, 후반에는 부들버게 0으로 떨어뜨리는 강력한 학습률 스케줄러 중 하나.
             (학습 속도와 초종 성능을 극대화해줌)
"""

for epoch in range(25):
    gpt.train()
    perm = torch.randperm(len(train_chunks))
    """
    torch.randperm : 매 epoch마다 학습 데이터 chunk들의 순서를 무작위로 섞어서(Shuffle)
    모델이 순서대로 암기하는것 방지.
    ----------------------------------------------------------------------------
    배치 단위 학습 공식
    1. loss = ... : 입력 x를 넣고 나온 예측값과 정답 y의 오차를 구함(패딩은 ignore_index=PAD로 무시)
    2.opt.zero_grad() -> loss.backward() -> clip_grad_norm -> opt.step()
    : 기울기 초기화 -> 역전파(미분) -> 기울기 폭발 장지 클리핑 -> 가중치 업데이트
    3.sched.step() : 매 step마다 원사이클 스케줄러에 맞춰 학습률을 미세 조정. 주기적 평가
    """
    for i in range(0, len(perm), 32):
        c = train_chunks[perm[i:i + 32]].to(device); x, y = c[:, :-1], c[:, 1:]
        loss = F.cross_entropy(gpt(x).reshape(-1, V), y.reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0); opt.step(); sched.step()
    if (epoch + 1) % 5 == 0:
        #5 epoch마다 한번씩 검증(run_eval)를 돌려서 학습 손실,검증손실,퍼플렉서티(PPL),정확도 -> 모델 모니터링
        ev = run_eval(gpt, valid_chunks); print(f'epoch {epoch+1:2d} | train loss {loss.item():.4f} | valid loss {ev["loss"]:.4f} ppl {ev["ppl"]:.2f} top1 {ev["top1"]:.3f} top5 {ev["top5"]:.3f}')
    """
    epoch  5 | train loss 2.6148 | valid loss 2.4411 ppl 11.49 top1 0.584 top5 0.922
    epoch 10 | train loss 0.4190 | valid loss 0.4214 ppl 1.52 top1 0.801 top5 0.988
    epoch 15 | train loss 0.3934 | valid loss 0.3891 ppl 1.48 top1 0.816 top5 0.987
    epoch 20 | train loss 0.3608 | valid loss 0.3749 ppl 1.45 top1 0.824 top5 0.987
    epoch 25 | train loss 0.3637 | valid loss 0.3654 ppl 1.44 top1 0.832 top5 0.98
    """

@torch.no_grad()
def generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int, greedy: bool = False, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None, repetition_penalty: float = 1.0) -> torch.Tensor:
    """
     텍스트 생성 엔진

     @torch.no_grad() & model.eval()
     : 추론 단계이므로 가중치를 업데이트하지않고(기울기 계산 차단),Dropout등을 꺼서 안정적인 출력 만듦
    """
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.block:])[:, -1].clone()# 마지막 위치의 로짓 (B, V)
        """
        지금까지 입력된 문장 중 모델이 허용하는 최대 길이(model.block,여기선 32)만큼만 잘라서
        모델에 넣고, 오직 맨 마지막(가장 최근)토큰 자리에 올 다음 단어의 예측 점수(logits)만 쏙 빼옵니다.
        """
        if repetition_penalty != 1.0:
            """
            이미 방금전에 나왔던 단어가 또 반복되는 현상을 막기 위해, 이미 등장했던 토큰들의 로짓(점수)을
            깍아내려(Penalize)다양한 단어가 다채롭게 나오도록 유도
            """
            for b in range(idx.size(0)):
                for t in set(idx[b].tolist()):
                    logits[b, t] = logits[b, t] / repetition_penalty if logits[b, t] > 0 else logits[b, t] * repetition_penalty
        if greedy:
            """
            logits.argmax(...) : 무조건 확률이 가장 높은 단 하나의 정답(1위)만 집어서 골라냄
            항상 똑같은 결과만 나오고, 문장이 자칫 앵무새처럼 뻔하고 반복적으로 흘러갈 수 있음
            """
            nxt = logits.argmax(-1, keepdim=True)
        else:
            """
            확률적 샘플링 방법
            확률 분포에 기반해, 주사위 굴리듯 단어를 뽑아냄. 여기에 모델의 창의성과 품질을 제어하는 
            3가지 필터가 작동
            
            1.temperture : 로짓을 온도로 나누어준다. 온도가 낮으면 1위 후보로 쏠리고(보수적) 높으면 후보들간의 격차가 줄어들어 기발하고,
                           다양한 단어가 튀어나옴(창의적)
            2. top_k : 상위 k개만 남기기. 확률이 높은 상위 k개의 단어(예,3개)만 남기고, 나머지 엉뚱한 단어들은 전부 잘라내(inf)후보에서 배제
            3. top_p : 확률을 높은순으로 차례로 더했을때 누적확률이 정해진 기준(top_p=0.9)을 넘기기전 까지의 단어들만 후보풀에 남겨둠. 상황에 따라
                        단어 풀의 크기가 유동적으로 조절되는 똑똑한 샘플링기법 
            """
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[:, [-1]]
                logits[logits < kth] = float('-inf')
            if top_p:
                s_logits, s_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(s_logits, -1); cum = probs.cumsum(-1)
                s_logits[(cum - probs) > top_p] = float('-inf')               # 누적확률이 p 를 넘기 '전'까지는 유지(첫 토큰 항상 유지)
                logits = torch.full_like(logits, float('-inf')).scatter(1, s_idx, s_logits)
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            """최종 가공된 확률분포 속에서 주사위를 굴려(Multinomial)다음 단어 ID(nxt)를 최종 당첨 시킴"""
        idx = torch.cat([idx, nxt], dim=1)
        """
        새로 뽑힌 단어를 기존 문장뒤에 이어붙이고, 
        max_new_tokens만큼 반복하여 문장을 완성해 나갑니다. (AutoGressive 방식)
        """
    return idx


prompt = torch.tensor([[BOS] + encode('지수는 숨이 차서')]).to(device) # ← encode: 문장(str) → 숫자 id 리스트
#시작문장('지수는 숨이 차서')을 주고, 모델에게서 이어서 글을 써보라고 프롬프트를 던짐
for name, kw in {'greedy': dict(greedy=True), 'T=1.0': dict(temperature=1.0), 'top-k=3': dict(top_k=3), 'top-p=0.9': dict(top_p=0.9, temperature=0.8)}.items():
    """
    동일한 랜덤시드(torch.manual_seed(1)를 고정해둔 상태에서 생성 제어 옵션을 바꿀때 모델이 문장을 어떻게
    완성해내는지 비교, 
    decode(...)를 통해 생성된 숫자ID리스트를 사람이 읽을 수 있는 문장으로 변환하여 출력
    """
    torch.manual_seed(1); print(f'{name:<10}→', decode(generate(gpt, prompt, 18, **kw)[0].tolist()))   # ← decode: 숫자 id 리스트 → 문장(str)


from collections import Counter
def _ngrams(t: list[int], n: int) -> dict[tuple[int, ...], int]:
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def rouge_n(ref: list[int], hyp: list[int], n: int) -> float:
    """
    모델이 생성한 문장(hyp)과 실제 정답 문장(ref)사이에 단어(N_gram)얼마나 겹치는지 측정
    1.n_grams : 문장을 N-gram(1글자 또는 2글자 단위 묶음) 빈도수 딕셔너리로 만든다.
    2.ov : 정답과 생성문 사이에 겹치는(Intersection) N-gram 개수의 총합
    3.정밀도(p = ov / sum(h))와 재현율(rc = ov / sum(r))을 구한 뒤,
    평균(F1 Score, 2 * p * rc / (p + rc))을 최종 ROUGE 점수로 반환
    """
    r, h = _ngrams(ref, n), _ngrams(hyp, n); ov = sum((r & h).values())
    if not r or not h or ov == 0: return 0.0
    p, rc = ov / sum(h.values()), ov / sum(r.values()); return 2 * p * rc / (p + rc)



def lcs_len(a: list[int], b: list[int]) -> int:
    """
    lcs_len & rouge_l (ROUGE-L)
    최장 공통 부분 수열(Longest Common Subsequence, LCS)을 기반으로 합니다.
    단어가 연속되지 않더라도 순서가 유지된 채 얼마나 자연스럽게 일치하는지를 평가합니다.
    """
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1): cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(ref: list[int], hyp: list[int]) -> float:
    """
    lcs_len & rouge_l (ROUGE-L)
    최장 공통 부분 수열(Longest Common Subsequence, LCS)을 기반으로 합니다.
    단어가 연속되지 않더라도 순서가 유지된 채 얼마나 자연스럽게 일치하는지를 평가합니다.
    """
    l = lcs_len(ref, hyp)
    if l == 0: return 0.0
    p, r = l / len(hyp), l / len(ref); return 2 * p * r / (p + r)



def distinct_n(t: list[int], n: int = 2) -> float:
    """
    모델이 생성한 텍스트 안에 얼마나 다양한 단어/표현이 사용되었는지를 측정
    (중복을 제거한 2-gram 종류의 수) / (전체 2-gram 개수)
    이 값이 너무 낮으면 모델이 똑같은 단어만 반복하는 앵무새(Repetitive) 현상을 겪고 있다는 뜻이므로,
    텍스트 생성의 다양성을 평가하는 데 필수적
    """
    g = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]; return len(set(g)) / len(g) if g else 0.0

PROMPT_LEN, GEN_LEN = 4, 14
items = []
for st, _ in VALID_STORIES:
    ids = [BOS] + [w for s in st for w in encode(s)]   # ← encode: 문장(str) → 숫자 id 리스트
    items.append((ids[:PROMPT_LEN], ids[PROMPT_LEN:PROMPT_LEN + GEN_LEN]))

def eval_generation(**kw) -> dict[str, float]:
    """
    1.검증 데이터셋에서 프롬프트(질문/앞부분)만 쏙 빼서 모델에 집어넣습니다.
    2.generate 함수를 통해 뒤이어 나올 14개의 토큰(GEN_LEN)을 생성
    3.생성된 문장에서 EOS(문장 끝) 토큰이 나오면 그 뒤는 잘라냅니다
    4.실제 정답(ref), 다른 문서의 정답(other)과 비교하여 ROUGE-1, ROUGE-2, ROUGE-L, Distinct-2,
      기준선 점수를 골고루 채점한 뒤 전체 평균을 내어 보기 좋게 딕셔너리로 반환
     """
    torch.manual_seed(0)
    prompts = torch.tensor([p for p, _ in items]).to(device)
    out = generate(gpt, prompts, GEN_LEN, **kw)[:, PROMPT_LEN:].tolist()
    rows = []
    for k, ((_, ref), hyp) in enumerate(zip(items, out)):
        hyp = hyp[:hyp.index(EOS)] if EOS in hyp else hyp
        other = items[(k + 1) % len(items)][1]                     # 다른 문서의 정답 (불일치 기준선)
        rows.append([rouge_n(ref, hyp, 1), rouge_n(ref, hyp, 2), rouge_l(ref, hyp), distinct_n(hyp, 2), rouge_n(other, hyp, 1)])
        """
        rouge_n(other, hyp, 1)
        전혀 상관없는 다른 문서의 정답(other)과 모델이 생성한 문장을 비교하여 ROUGE 점수를 매긴다.
        최소한 무작위 글자에 비해서는 우리 모델이 진짜 정답과 얼마나 더 잘 맞닿아 있는지"를 
        보여주는 최하위 비교 기준선(Baseline) 역할
        """
    return dict(zip(['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Distinct-2', '기준선R1'], [round(float(v), 3) for v in np.mean(rows, 0)]))

for name, kw in {'Greedy': dict(greedy=True), 'Greedy+반복페널티1.2': dict(greedy=True, repetition_penalty=1.2),
                 'Top-k5 T=0.8': dict(top_k=5, temperature=0.8), 'Top-p0.9 T=1.0': dict(top_p=0.9)}.items():
    """
    Greedy(고지식하고 뻔한 답변), 
    Greedy+반복페널티(반복을 억제한 Greedy), 
    Top-k5 T=0.8(상위 5개 후보 내에서 온도 0.8로 샘플링), 
    Top-p0.9(누적 확률 90% 핵 샘플링) 등 앞서 구현했던 생성 제어 옵션별로 
    ROUGE와 다양성(Distinct-2) 점수가 어떻게 달라지는지 한눈에 비교
    ------------------------------------------------------------
    Teacher Forcing 정량 평가
    마지막 줄에서는 run_eval을 호출하여 정답을 깔끔하게 제공했을 때의 
    퍼플렉서티(PPL, 낮을수록 좋음)와 Top-1 / Top-5 정확도를 최종 출력하며 전체 실습 대장정을 마무리
    """
    print(f'{name:<20}', eval_generation(**kw))
    """
    greedy    → 지수는 숨이 차서 잠시 물을 마셨다 지수는 다시 힘을 내어 열 바퀴를 달렸다 지수는 운동을 마치고 샤워를 했다 하나는
    T=1.0     → 지수는 숨이 차서 잠시 물을 마셨다 지수는 다시 힘을 내어 열 바퀴를 달렸다 하나는 운동을 마치고 샤워를 했다 하나는
    top-k=3   → 지수는 숨이 차서 잠시 물을 마셨다 지수는 다시 힘을 내어 열 바퀴를 달렸다 도하는 운동을 마치고 샤워를 했다 하나는
    top-p=0.9 → 지수는 숨이 차서 잠시 물을 마셨다 지수는 다시 힘을 내어 열 바퀴를 달렸다 도하는 운동을 마치고 샤워를 했다 하나는
    Greedy               {'ROUGE-1': 0.899, 'ROUGE-2': 0.782, 'ROUGE-L': 0.899, 'Distinct-2': 1.0, '기준선R1': 0.256}
    Greedy+반복페널티1.2      {'ROUGE-1': 0.817, 'ROUGE-2': 0.605, 'ROUGE-L': 0.817, 'Distinct-2': 1.0, '기준선R1': 0.26}
    Top-k5 T=0.8         {'ROUGE-1': 0.883, 'ROUGE-2': 0.748, 'ROUGE-L': 0.883, 'Distinct-2': 1.0, '기준선R1': 0.261}
    Top-p0.9 T=1.0       {'ROUGE-1': 0.879, 'ROUGE-2': 0.74, 'ROUGE-L': 0.879, 'Distinct-2': 1.0, '기준선R1': 0.261}
    """
print('teacher forcing(정답 prefix 제공) PPL/Top-1/Top-5:', {k: round(v, 3) for k, v in run_eval(gpt, valid_chunks).items() if k != 'loss'})
"""
teacher forcing(정답 prefix 제공) PPL/Top-1/Top-5: {'ppl': 1.441, 'top1': 0.832, 'top5': 0.987}
"""

gpt.eval()
idx = train_chunks[:1, :-1].to(device)
emb = gpt.tok_emb(idx).detach().requires_grad_(True)# 임베딩 출력을 leaf 로 만들어 기울기를 직접 관찰
#모델의 가중치가 아니라, 입력 문장의 임베딩 값 자체에 기울기가 어떻게 맺히는지 직접 추적하기 위해 수동으로 잎(Leaf) 노드로 지정
t = 10
logits = gpt(idx, inputs_embeds=emb)
logits[0, t].logsumexp(-1).backward() # t 번째 위치 출력 하나에서 backward
#문장 전체의 오차가 아니라, 정확히 t번째(여기서는 10번째) 단어 위치에서 나온 예측 점수 하나만을 기준으로 역전파(backward)수행
g = emb.grad[0].norm(dim=-1)

print(f'위치 {t} 의 출력 → 입력 임베딩별 |grad|')
print('  ≤ t  :', [round(v, 3) for v in g[:t + 1].tolist()])
print('  > t  :', [round(v, 3) for v in g[t + 1:].tolist()], '← 전부 0 (미래 토큰으로는 기울기가 흐르지 않음)')
"""
-현재와 과거 위치 (`≤ t`): 임베딩 값들의 기울기(`|grad|`)가 0이 아닌 값들로 뚜렷하게 찍힙니다. 
                          (과거의 단어들이 현재 예측에 영향을 주었기 때문입니다.)
- 미래 위치 (`> t`): 기울기가 정확히 전부 `0.0`으로 떨어집니다.
- 의미: 룩어헤드 마스크(Look-ahead Mask) 덕분에 `t` 이후의 미래 토큰들은 현재의 예측에 단 1도 개입하지 않았음을 역전파 미분을 통해 완벽하게 증명
"""
"""
위치 10 의 출력 → 입력 임베딩별 |grad|
  ≤ t  : [0.053, 0.153, 0.123, 0.415, 0.429, 0.044, 0.034, 0.352, 0.522, 0.313, 1.655]
  > t  : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ← 전부 0 (미래 토큰으로는 기울기가 흐르지 않음)
"""

# causal 성질 (순전파): 미래 토큰을 바꿔도 과거 위치의 출력은 불변
x2 = idx.clone(); x2[0, -1] = (x2[0, -1] % (V - 7)) + 7
print('미래 토큰 변경해도 과거 로짓 불변:', torch.allclose(gpt(idx)[:, :-1], gpt(x2)[:, :-1], atol=1e-5))
#미래 토큰 변경해도 과거 로짓 불변: True
"""
1. 원본 입력 문장(`idx`)의 맨 마지막 토큰(가장 미래의 단어)을 엉뚱한 다른 단어(`x2`)로 강제로 바꿔치기합니다.
2. 원본 문장을 넣었을 때의 출력(`gpt(idx)`)과, 
    미래를 망가뜨린 문장을 넣었을 때의 출력(`gpt(x2)`)을 비교합니다. 
    (단, 마지막 토큰을 제외한 `[:-1]` 부분만 비교)
----------------------------------------------------
torch.allclose(...) 결과가 True
GPT는 순서대로 글을 읽어 나갈 때 과거의 단어들이 미래를 볼 수는 있어도, 미래의 단어가 변한다고 해서 
과거의 예측 결과가 소급해서 변하지 않는다는 시계열적 인과성(Causality)을 완벽하게 준수하고 있음을 직관적으로 보여줌
"""


