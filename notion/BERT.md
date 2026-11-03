# BERT

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
    EOS,BOS,CLS,SEP,
    SPECIALS,
    MASK,
    itos
)
 # ── NSP: 실제 다음 문장(1:양성)과 다른 이야기의 문장(0:음성)을 구분하는 문장쌍 만들기 ──
def build_nsp_pairs(stories: list[Story], seed: int = 0) -> list[tuple[str, str, int]]:
    """NSP용 문장쌍 생성: 실제 다음 문장은 1, 다른 이야기의 임의 문장은 0."""
    rng, pool, pairs = random.Random(seed), [(i, k) for i in range(len(stories)) for k in range(4)], []
    for si, (st, _) in enumerate(stories):
        for k in range(3):
            pairs.append((st[k], st[k + 1], 1))
            # 양성: 실제로 바로 다음에 오는 문장
            # 양성sample : 같은이야기안에서 실제로 바로나오는 문장쌍(st[k],st[k+1])가져옴:실제 다음 문장이 맞다
            sj, kj = rng.choice(pool)
            while sj == si or stories[sj][0][kj] in (st[k], st[k + 1]): sj, kj = rng.choice(pool) # 다른 이야기의 문장 (같은 글자 문장은 제외)
            pairs.append((st[k], stories[sj][0][kj], 0))# 음성(random)
    return pairs

# NSP는 BERT가 문장 간 관계를 이해하는 첫 번째 과제다.
# 실제로 이어지는 문장 쌍(label=1)과 무작위로 섞인 문장 쌍(label=0)을 만들면,
# 모델은 문장의 흐름과 논리적 연속성을 학습할 수 있다.
# 이 데이터는 나중에 [CLS] 표현을 기반으로 IsNext/NotNext를 분류하는 데 쓰인다.

def build_hard_pairs(stories: list[Story], kind: str) -> list[tuple[str, str, int]]:
    """순서는 반대지만 같은 이야기이거나, 같은 이야기의 비인접 문장인 어려운 음성 쌍."""
    pairs = []
    for st, _ in stories:
        for k in range(3):
            pairs.append((st[k], st[k + 1], 1))
        # reverse: 같은 이야기지만 문장 순서를 뒤집음 / skip: 다음 문장이 아니라 건너뜀
        # 이 둘은 랜덤 음성보다 훨씬 어려운 사례로, 모델이 단순 키워드 매칭이 아니라 문맥 흐름을 봐야 한다.
        negs = [(st[k + 1], st[k]) for k in range(3)] if kind == 'reverse' else [(st[0], st[2]), (st[0], st[3]), (st[1], st[3])]
        pairs += [(a, b, 0) for a, b in negs]
    return pairs

train_pairs, valid_pairs = build_nsp_pairs(TRAIN_STORIES, 0), build_nsp_pairs(VALID_STORIES, 1)
hard_rev, hard_skip = build_hard_pairs(VALID_STORIES, 'reverse'), build_hard_pairs(VALID_STORIES, 'skip')
"""
reverse(순서뒤집기) : (문장1,문장2) 대신 역순 (문장2,문장1)을 음성(0)으로 둔다.
                    문장들은 같은 이야기 속 단어들이라 단어 구성 비슷한데 순서 꼬여있어 모델이 헷갈리기 쉬움
skip(건너뛰기) : 바로 다음 문장이 아니라 한 두문장 건너뛴 문장을 오답으로 배치하여,문맥의 미세한 흐름까지 엄격하게 구별하도록 훈련
"""
print(len(train_pairs), len(valid_pairs), len(hard_rev), len(hard_skip))
print(valid_pairs[0]); print(valid_pairs[1])
# 실행 결과
"""
2880 720 720 720
('하나는 시장에서 채소를 샀다', '하나는 집에서 채소를 썰고 국을 끓였다', 1)
('하나는 시장에서 채소를 샀다', '하나는 저녁에 숙소로 돌아왔다', 0)
"""

 # BERT 입력: [CLS] 문장 A [SEP] 문장 B [SEP], segment 0/1로 문장 소속을 표시
def encode_pair(a: str, b: str) -> tuple[list[int], list[int]]:
    # BERT 입력 형식: [CLS] 문장A [SEP] 문장B [SEP]
    # segment id 0/1은 각 토큰이 어느 문장에 속하는지 알려 준다.
    # 이렇게 문장 경계를 명시해 주면 모델은 서로 다른 문장 정보를 분리해서 이해할 수 있다.
    A, B = encode(a), encode(b)   # ← encode: 문장(str) → 숫자 id 리스트
    return [CLS] + A + [SEP] + B + [SEP], [0] * (len(A) + 2) + [1] * (len(B) + 1)       # ids, segment ids

def collate(pairs: list[tuple[str, str, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    enc = [encode_pair(a, b) for a, b, _ in pairs]   # ← encode_pair: 문장 A,B → [CLS] A [SEP] B [SEP] 숫자 + 세그먼트 id
    # ids, segment ids는 (B,L), NSP 정답은 (B,)가 되도록 배치와 패딩을 만든다.
    #이렇게 만들어진 문장 쌍리스트들을 배치크기만큼 묶고, 길이가 부족한 빈자리는 PAD로 채워 네모난 텐서(ids,seg,lab)형태로 정돈하여 모델에 공급
    # 이를 통해 모델은 한 번에 여러 문장쌍을 효율적으로 처리할 수 있다.
    return pad_batch([e[0] for e in enc]), pad_batch([e[1] for e in enc]), torch.tensor([p[2] for p in pairs])# ← pad_batch: 길이가 다른 리스트들 → PAD(0)로 채워 (B, L) 텐서

ids, seg, lab = collate(valid_pairs[:2])
print('ids \n', ids, '\nsegment \n', seg, '\nlabel', lab)
# 실행 결과: ids는 [CLS] A [SEP] B [SEP]의 토큰 번호, segment의 0/1은 문장 A/B 구분이다.
"""
ids shape: torch.Size([2, 13]) | label: tensor([1, 0])
"""


 # ── MLM: 특수 토큰을 제외한 단어 15%에만 80/10/10 규칙 적용 ──
def mask_tokens_for_mlm(input_ids: torch.Tensor, special_ids: list[int], mask_id: int, vocab_size: int, mlm_prob: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """MLM 학습 입력과 정답 생성: 일반 토큰의 15%만 80/10/10 규칙으로 변경."""
    ids = input_ids.clone()
    labels = torch.full_like(ids, -100)
    # -100 = ignore_index: 선택되지 않은 위치는 손실에서 제외
    #정답 라벨 텐서를 100으로 채워 초기화
    """
    파이토치 CrossEntropyLoss에서는 ignore_index=-100설정시, 마스킹 되지 않은
    일반 단어들의 오차는 채점(Loss계산)에서 완전히 제외되도록 만들어 줍니다.
    오직 마스킹된자리(chose)만 정확도를 채점합니다.
    """
    special = torch.zeros_like(ids, dtype=torch.bool)
    for s in special_ids: special |= (input_ids == s)# 특수토큰 위치
    chosen = (torch.rand(ids.shape) < mlm_prob) & ~special # 15% 선택
    """
    [PAD],[CLS],[SEP]같은 특수 토큰들은 문맥의 핵심 뼈대 이므로 절대 무작위로 가리거나
    훼손하면 안됩니다. 따라서 ~special조건을 걸어 마스킹 후보 대상에서 제외합니다.
    
    15% 샘플링(chosen)
    특수 토큰을 제외한 전체 토큰 중 무작위로  15%(min_prob=-0.15)를 골라냅니다.
    """
    labels[chosen] = input_ids[chosen]# 정답 = 원래 토큰
    r = torch.rand(ids.shape)
    ids[chosen & (r < 0.8)] = mask_id# 80% [MASK]
    rand_pos = chosen & (r >= 0.8) & (r < 0.9)# 10% 랜덤 (위치마다 다른 토큰!)
    """
    80% (r < 0.8): 선택된 15%중 80%의 토큰을 진짜 [MASK]토큰으로 바꿔버립니다.
    10% (0.8 <=r<0.9): 10%의 토큰은 마스크처리하지않고, 단어장vocab_size안의 엉뚱한
    무작위 단어로 바꿔치기 합니다. (모델이 마스크만 쫓는게 아닌 문맥을 스스로 의심하고,교정하도록 만드는
    노이즈 주입 기법)
    나머지 10% (r>=0.9) : 아예 건드리지않고 원래 토큰 그대로 유지 
    """
    ids[rand_pos] = torch.randint(len(SPECIALS), vocab_size, ids.shape)[rand_pos]
    return ids, labels# 나머지 10%는 원본 유지: [MASK]만 보고 푸는 습관을 막음

# MLM의 핵심은 문맥을 보면서 '가려진 단어를 추론'하게 하는 것이다.
# 하지만 단순히 100% [MASK]만 넣는다면 모델이 마스크 위치만 보고 정답을 맞추는 훈련이 되어 버린다.
# 그래서 BERT는 80/10/10 규칙을 적용해, 모델이 실제 문맥을 읽고 단어 의미를 예측하도록 유도한다.
# 80%는 [MASK]로 바꾸고, 10%는 무작위 단어로 바꾸며, 나머지 10%는 그대로 두어 학습이 더 강건해진다.

SPECIAL_IDS = [PAD, CLS, SEP, BOS, EOS]
torch.manual_seed(0)
big = pad_batch([encode_pair(a, b)[0] for a, b, _ in train_pairs[:2000]])   # ← pad_batch: 길이가 다른 리스트들 → PAD(0)로 채워 (B, L) 텐서; encode_pair: 문장 A,B → [CLS] A [SEP] B [SEP] 숫자 + 세그먼트 id
mi, ml = mask_tokens_for_mlm(big, SPECIAL_IDS, MASK, V)   # ← mask_tokens_for_mlm: 15% 골라 80/10/10 으로 가리기 → (입력, 정답 라벨)
sel = ml != -100
nonspecial = ~torch.isin(big, torch.tensor(SPECIAL_IDS))# 마스킹 후보 = 특수토큰이 아닌 토큰
print(f'선택 비율(특수토큰 제외 기준) {sel.sum().item() / nonspecial.sum().item():.3f} (목표 0.15)')
print(f'  └ [MASK] {((mi == MASK) & sel).sum().item() / sel.sum().item():.3f} | 원본유지 {((mi == big) & sel).sum().item() / sel.sum().item():.3f} (목표 0.8 / ~0.1+우연)')
print('예시 원문:', decode(big[0].tolist())); print('예시 마스킹:', ' '.join(itos[i] for i in mi[0].tolist() if i != PAD))   # ← decode: 숫자 id 리스트 → 문장(str)
# 실행 결과: 선택 비율과 80/10/10 마스킹 비율이 목표에 가깝다.
"""
선택 비율(특수토큰 제외 기준) 0.148 (목표 0.15)
  └ [MASK] 0.807 | 원본유지 0.101 (목표 0.8 / ~0.1+우연)
예시 원문: 서아는 학교에서 영어를 공부했다 서아는 단어를 외우고 문제를 풀었다
예시 마스킹: [CLS] 서아는 [MASK] 영어를 공부했다 [SEP] 서아는 단어를 외우고 문제를 풀었다 [SEP]
"""


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    if mask is not None: scores = scores.masked_fill(mask, float('-inf'))
    #[양방향 어텐션]실제 Attention  계산
    w = torch.softmax(scores, dim=-1)
    return torch.matmul(w, v), w

class MultiHeadAttention(nn.Module):
    def __init__(self, em_dim: int, num_heads: int) -> None:
        super().__init__()
        self.h, self.dk, self.em_dim = num_heads, em_dim // num_heads, em_dim
        self.wq, self.wk, self.wv, self.dense = (nn.Linear(em_dim, em_dim) for _ in range(4))
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,D) 임베딩을 head별 묶음 (B,H,T,D/H)으로 나눈다.
        B, T, _ = x.size(); return x.view(B, T, self.h, self.dk).transpose(1, 2)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B = q.size(0)
        q, k, v = self.split_heads(self.wq(q)), self.split_heads(self.wk(k)), self.split_heads(self.wv(v))
        out, w = scaled_dot_product_attention(q, k, v, mask)   # ← scaled_dot_product_attention: softmax(QKᵀ/√dk)V → (출력, 어텐션 가중치)
        return self.dense(out.transpose(1, 2).contiguous().view(B, -1, self.em_dim)), w

# MultiHeadAttention은 한 번에 여러 관점의 어텐션을 보게 해 준다.
# 각 head가 서로 다른 의미적 관계를 보는 이유는, 같은 입력을 여러 선형 변환으로 만들어
# 서로 다른 '관심사'를 가진 attention map을 만들기 때문이다.
# BERT의 핵심은 이렇게 여러 head가 서로 다른 문맥적 패턴을 동시에 찾는 데 있다.

# MultiHeadAttention은 한 번에 여러 관점의 어텐션을 보게 해 준다.
# 각 head가 서로 다른 의미적 관계를 보는 이유는, 같은 입력을 여러 선형 변환으로 만들어
# 서로 다른 '관심사'를 가진 attention map을 만들기 때문이다.
# BERT의 핵심은 이렇게 여러 head가 서로 다른 문맥적 패턴을 동시에 찾는 데 있다.

class BertLayer(nn.Module):# Post-LN + GELU
    """
    GPT와의 차이점 : GPT는 연산 전에 LayerNorm을 먼저 거치는 Pre-LN을 사용했지만,
                   원조 BERT는 연산이 끝난후 LayerNorm을 거치는 Post-LN(LN(x+Sublayer(x))구조채택
    내부에 GELU 활성화 함수와 MHA(Multi-Head Attention),그리고 피드 포워드 네트워크(FFN)이 배치
    """
    def __init__(self, D: int, H: int, ff: int, rate: float) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(D, H)
        self.ffn = nn.Sequential(nn.Linear(D, ff), nn.GELU(), nn.Linear(ff, D))
        self.ln1, self.ln2, self.do = nn.LayerNorm(D), nn.LayerNorm(D), nn.Dropout(rate)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # self-attention → Add & Norm → FFN → Add & Norm
        # BERT는 self-attention으로 문맥을 보면서, FFN으로 비선형 변환을 더해 표현을 강화한다.
        a, _ = self.mha(x, x, x, mask) #[양방향 어텐션]어텐션 호출
        x = self.ln1(x + self.do(a))
        return self.ln2(x + self.do(self.ffn(x)))

class BertForPreTraining(nn.Module):
    def __init__(self, vocab: int, D: int = 64, H: int = 4, ff: int = 256, n_layers: int = 2, max_len: int = 32, rate: float = 0.1) -> None:
        super().__init__()
        # 토큰 의미 + 위치 정보 + 문장 A/B 구분 정보를 더해 BERT 입력 표현을 만든다.
        self.tok, self.pos, self.seg = nn.Embedding(vocab, D, padding_idx=PAD), nn.Embedding(max_len, D), nn.Embedding(2, D)
        """
        tok : 토큰임베딩. 단어 자체의 의미
        pos : 포지션임베딩. BERT는 GPT와 달리 고정된 상수가 아닌 학습 가능한 위치 임베딩(최대32)를사용
        seg : 세그먼트임베딩. 앞서 보았던 [CLS]A[SEP]B[SEP]구조에서 첫번째 문장인지 두번째 문장인지
            (0또는 1)구분해 주는 2가지 종류의 임베딩      
        """
        self.emb_ln, self.drop = nn.LayerNorm(D), nn.Dropout(rate)
        """-> 3가지 임베딩을 모두 더해 LayerNorm과 Dropout거쳐 인코더층으로 들어감"""
        # BERT는 미래를 가리는 causal mask 없이 PAD만 가리므로 양방향 문맥을 본다.
        self.layers = nn.ModuleList([BertLayer(D, H, ff, rate) for _ in range(n_layers)])
        self.pooler = nn.Linear(D, D)
        #pooler : 최종 인코더 출력에서 문장 수준 표현을 뽑아내는 프로젝션 헤드
        #-> 양방향 문맥을 반영한 최종 문장 표현을 만드는 역할
        self.mlm_dense, self.mlm_ln, self.mlm_bias = nn.Linear(D, D), nn.LayerNorm(D), nn.Parameter(torch.zeros(vocab))
        # MLM head는 token embedding 가중치를 재사용(weight tying)해 단어 사전 점수를 만든다.
        """
        MLM head
        : 마스킹된 15%의 단어들이 원래 무엇이었는지 단어장 크기(v)만큼 예측
        이때, weighTying이 적용되어 입력 토큰,입력 가중치(self.tok.weight)를 출력단에도 그대로사용
        """
        self.nsp = nn.Linear(D, 2)  # [CLS]의 문장쌍 표현으로 IsNext(1) / NotNext(0) 분류
        """
        문장 맨 앞에 위치한 [CLS]토큰의 최종 출력뒤, 두 문장이 실제로 이어지는 관계인지 (0또는1)판별
        """
        for p in [*self.tok.parameters(), *self.pos.parameters(), *self.seg.parameters()]: nn.init.normal_(p, 0, 0.02)

    def forward(self, ids: torch.Tensor, seg: torch.Tensor, inputs_embeds: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        T = ids.size(1)
        mask = (ids == PAD)[:, None, None, :] # (B,1,1,T): PAD만 가림, 미래 토큰은 가리지 않음 : [양방향 어텐션]입력마스크 생성
        # inputs_embeds는 마지막 gradient 확인 실험에서 토큰 임베딩을 직접 넣기 위해 사용한다.
        x = self.tok(ids) if inputs_embeds is None else inputs_embeds
        x = self.drop(self.emb_ln(x + self.pos(torch.arange(T, device=ids.device)) + self.seg(seg)))
        for layer in self.layers: x = layer(x, mask) # (B,T,D)
        # [CLS] 벡터는 문장쌍 전체를 대표하므로 NSP 분류에 사용한다.
        pooled = torch.tanh(self.pooler(x[:, 0])) # [CLS] 위치
        # MLM: 모든 위치의 단어 사전 점수. 입력 tok.weight를 출력에도 재사용하는 weight tying.
        mlm_logits = self.mlm_ln(F.gelu(self.mlm_dense(x))) @ self.tok.weight.t() + self.mlm_bias   # 임베딩 tying
        return mlm_logits, self.nsp(pooled) # (B,T,V), (B,2)

# BERT의 핵심은 입력 토큰들이 단순히 각자 의미를 가지는 것이 아니라,
# 서로의 문맥을 주고받으며 전체 표현을 만들고, 그 표현을 이용해 MLM과 NSP라는 두 과제를 동시에 해결하는 데 있다.
# 즉 단어 수준의 복원과 문장 수준의 관계 예측을 함께 학습함으로써, 문맥을 이해하는 인코더를 만드는 것이 핵심이다.

bert = BertForPreTraining(V).to(device)
ids, seg, lab = [t.to(device) for t in collate(train_pairs[:64])]
mi, ml = mask_tokens_for_mlm(ids.cpu(), SPECIAL_IDS, MASK, V); mi, ml = mi.to(device), ml.to(device)   # ← mask_tokens_for_mlm: 15% 골라 80/10/10 으로 가리기 → (입력, 정답 라벨)
mlm_logits, nsp_logits = bert(mi, seg)
l_mlm = F.cross_entropy(mlm_logits.reshape(-1, V), ml.reshape(-1), ignore_index=-100); l_nsp = F.cross_entropy(nsp_logits, lab)
print(f'초기 MLM loss {l_mlm.item():.3f} (~ ln V = {math.log(V):.3f}) | 초기 NSP loss {l_nsp.item():.3f} (~ ln 2 = {math.log(2):.3f})  ← sanity check')
print('파라미터 수:', sum(p.numel() for p in bert.parameters()))
# 실행 결과: 학습 전 loss는 무작위 추측 기준(ln V, ln 2)과 비슷해야 정상이다.
"""
MLM초기 : 아직 학습되지 않은 무작위 상태에서 단어장크기(V)중 하나를 맞혀야함
초기 crossentropy 오차 이론값인 ln(V)에 수렴해야 모델이 올바르게 초기화 된것
-----------------------------------------------------------------
NSP초기 : 이진분류(맞다/아니다,2개클래스)이므로, 무작위로 찍을때의 엔트로피 이론값 $\ln(2) \approx 0.693$ 부근에서 시작하는지확인하는 점검코드
"""
"""
초기 MLM loss 4.281 (~ ln V = 4.263) | 초기 NSP loss 0.717 (~ ln 2 = 0.693)
파라미터 수: 115465
"""


from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

@torch.no_grad()
def evaluate_bert(model: nn.Module, pairs: list[tuple[str, str, int]], bs: int = 256, seed: int = 123) -> dict:
    """검증 세트에서 MLM 성능과 NSP 확률을 계산한다. 마스킹 난수는 고정해 공정하게 비교한다."""
    model.eval(); torch.manual_seed(seed)# 평가용 마스킹을 고정
    """
    평가할때마다 무작위로 마스킹되는 위치가 달라지면 점수가 흔들림.
    seed를 고정해서 항상 똑같은 조건에서 MLM평가가 공정하게 이루어지도록 통제
    """
    nll = n_mask = correct = 0; probs, labels = [], []
    for i in range(0, len(pairs), bs):
        ids, seg, lab = collate(pairs[i:i + bs])
        mi, ml = mask_tokens_for_mlm(ids, SPECIAL_IDS, MASK, V) # ← mask_tokens_for_mlm: 15% 골라 80/10/10 으로 가리기 → (입력, 정답 라벨)
        mi, ml, seg, lab = mi.to(device), ml.to(device), seg.to(device), lab.to(device)
        mlm_logits, nsp_logits = model(mi, seg)
        keep = ml != -100  # 실제로 마스킹되어 MLM 정답이 있는 위치만 집계
        nll += F.cross_entropy(mlm_logits.reshape(-1, V), ml.reshape(-1), ignore_index=-100, reduction='sum').item()
        correct += ((mlm_logits.argmax(-1) == ml) & keep).sum().item(); n_mask += keep.sum().item()
        probs.append(torch.softmax(nsp_logits, -1)[:, 1].cpu()); labels.append(lab.cpu())
        """
        #NSP분류헤드(nsp_logits)에 softmax를 취해, 두 문장이 실제로 이어질 확률(pos_label=1)을 실수 형태의 확률값(porbs)로모아둔다.
        """
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return {'mlm_loss': nll / n_mask, 'mlm_ppl': math.exp(nll / n_mask), 'mlm_acc': correct / n_mask, 'probs': probs, 'labels': labels}
"""
MLM 측정 (mlm_loss, mlm_ppl, mlm_acc)
모델이 예측한 마스크 단어 점수(mlm_logits) 중 가장 확률이 높은 단어(argmax(-1))가 
실제 정답(ml)과 일치하는지 비교하여 MLM 정확도(mlm_acc)와 퍼플렉서티(mlm_ppl)를 산출합니다.
# 평가를 할 때는 단순히 한 번의 점수만 보는 것보다,
# 모델이 길을 잃는 지점이 어디인지까지 보는 것이 중요하다.
# 특히 NSP는 정답/오답이 불균형하거나 hard negative가 있을 때,
# Accuracy만으론 부족하므로 Precision, Recall, F1, ROC-AUC를 함께 보아야 한다.
"""

def nsp_report(ev: dict, name: str) -> dict:

    y, p = ev['labels'], ev['probs']; pred = (p > 0.5).astype(int)
    # NSP 확률 0.5를 기준으로 다음 문장(IsNext=1) 여부를 분류해 지표를 정리한다.
    #모델이 뱉어낸 연속적인 확률값 p -> 0.5넘으면 이어지는 문장1. 넘지못하면 엉뚱한문장0을로 최종판정
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average='binary', pos_label=1, zero_division=0)
    # Accuracy만으로 부족할 수 있어 정밀도·재현율·F1·ROC-AUC도 함께 본다.
    """
    - Accuracy (정확도): 전체 문장 쌍 중 맞힌 비율.
    - Precision (정밀도) & Recall (재현율): 참을 참이라 예측한 비율과, 실제 참인 것 중 모델이 찾아낸 비율.
    - F1-Score (F1 점수): 정밀도와 재현율의 조화평균 (불균형 데이터에서 모델의 실력을 판가름하는 핵심 지표).
    - ROC-AUC: 모델의 이진 분류 판정 능력을 종합적으로 평가하는 곡선 아래 면적 (1.0에 가까울수록 완벽한 분류기).
    """
    return {'평가셋': name, '쌍 수': len(y), 'Accuracy': accuracy_score(y, pred), 'Precision': pr, 'Recall': rc, 'F1': f1,
            'ROC-AUC': roc_auc_score(y, p) if len(set(y)) > 1 else float('nan')}


EPOCHS, BS = 12, 64
steps_total = EPOCHS * math.ceil(len(train_pairs) / BS)
opt = torch.optim.AdamW(bert.parameters(), lr=1e-3, weight_decay=0.01)

sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / (0.1 * steps_total), max(0.0, (steps_total - s) / (0.9 * steps_total))))   # warmup 10% → 선형 감소
#[LambdaLR선언부] 전체 학습 step중 처음 10%는 warmup으로 학습률을 올리고, 이후 90%는 선형으로 낮춘다.
for epoch in range(EPOCHS):
    bert.train(); random.shuffle(train_pairs); t_mlm = t_nsp = 0; nb_ = 0
    for i in range(0, len(train_pairs), BS):
        ids, seg, lab = collate(train_pairs[i:i + BS])
        mi, ml = mask_tokens_for_mlm(ids, SPECIAL_IDS, MASK, V)
        # 동적 마스킹: 배치마다 새로   ← mask_tokens_for_mlm: 15% 골라 80/10/10 으로 가리기 → (입력, 정답 라벨)
        """
        굳이 사전에 마스킹된 데이터셋을 어려개 만들필요 없이 학습할때마다 무작위로 마스킹 위치가 달라지므로,
        모델이 특정 패턴을 암기하지 못하고 더 Robust하게 문맥을 이해하는 능력 기르게 됨
        """
        mi, ml, seg, lab = mi.to(device), ml.to(device), seg.to(device), lab.to(device)
        mlm_logits, nsp_logits = bert(mi, seg)
        l_mlm = F.cross_entropy(mlm_logits.reshape(-1, V), ml.reshape(-1), ignore_index=-100)  # 선택된 MLM 위치만 채점
        l_nsp = F.cross_entropy(nsp_logits, lab)
        loss = l_mlm + l_nsp
        # MLM(빈칸 맞히기)과 NSP(다음 문장 판별)를 함께 학습한다.
        """
        마스크 된 단어를 맞히는 손실(l_mim)과 두문장의 연관성을 판별하는 손실(1_nsp)을 단순 합산 하여
        동시에 역전파(backward)를합니다.
        이를 통해 BERT인코더는 단어의 의미와 문장간의 관계를 동시학습하고, 양방향 문맥 표현 능력을 갖게된다.
        """
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0); opt.step(); sched.step()
        t_mlm += l_mlm.item(); t_nsp += l_nsp.item(); nb_ += 1
    if (epoch + 1) % 3 == 0:
        ev = evaluate_bert(bert, valid_pairs)
        print(f'epoch {epoch+1:2d} | train MLM {t_mlm/nb_:.3f} NSP {t_nsp/nb_:.3f} | valid MLM ppl {ev["mlm_ppl"]:.2f} top1 {ev["mlm_acc"]:.3f} | NSP acc {accuracy_score(ev["labels"], ev["probs"] > 0.5):.3f}')
# 실행 결과: epoch이 진행될수록 MLM perplexity는 낮아지고, NSP 정확도는 높아진다.
"""
epoch  3 | train MLM 2.915 NSP 0.351 | valid MLM ppl 15.42 top1 0.199 | NSP acc 0.900
epoch  6 | train MLM 1.788 NSP 0.295 | valid MLM ppl 4.78 top1 0.688 | NSP acc 0.922
epoch  9 | train MLM 1.139 NSP 0.264 | valid MLM ppl 2.88 top1 0.759 | NSP acc 0.929
epoch 12 | train MLM 0.949 NSP 0.217 | valid MLM ppl 2.51 top1 0.774 | NSP acc 0.939
"""

import pandas as pd

#[HARD Negative] 무작위 음성뿐 아니라 순서가 뒤집힌 문장, 같은 이야기의 비인접 문장에서도 NSP를 평가한다.
"""
- 표준 (`valid_pairs`): 완전히 무작위로 뽑힌 엉뚱한 문장 (모델 입장에서 난이도 '하')
- 역순 (`hard_rev`): 같은 이야기 안의 문장인데 순서만 거꾸로 뒤집힌 경우 (단어 구성은 똑같아서 난이도 '중상')
- 비인접 (`hard_skip`): 문장 사이를 몇 칸씩 건너뛴 경우 (문맥의 미세한 흐름을 모르면 속기 쉬운 난이도 '상')
"""
rows = [nsp_report(evaluate_bert(bert, p), n) for n, p in [('random negative (표준)', valid_pairs), ('hard: 역순', hard_rev), ('hard: 비인접', hard_skip)]]
print(pd.DataFrame(rows).round(4).to_string(index=False))
# 실행 결과: hard: 역순이 표준 random negative보다 더 어려워 Accuracy가 낮다.
"""
                 평가셋  쌍 수  Accuracy  Precision  Recall     F1  ROC-AUC
random negative (표준)  720    0.9389     0.8970  0.9917 0.9420   0.9646
            hard: 역순  720    0.9014     0.8400  0.9917 0.9096   0.9540
           hard: 비인접  720    0.9847     0.9781  0.9917 0.9848   0.9978
"""

ev = evaluate_bert(bert, valid_pairs)
cm = confusion_matrix(ev['labels'], (ev['probs'] > 0.5).astype(int), labels=[0, 1])
print('\n혼동행렬 (행=정답, 열=예측)  [NotNext, IsNext]'); print(cm)
"""
- [NotNext (0)]: 이어지지 않는 문장인데 맞게 판별했는지 / 착각했는지
- [IsNext (1)]: 실제로 이어지는 문장인데 맞게 판별했는지 / 놓쳤는지
-> 단순 Acc숫자 하나만 보는것 아닌 모델이 보수적으로 판정하는편인지,아니면 긍정오류(FalsePositive)가많은지 입체적으로 진단
"""
print(f"MLM(평가) loss {ev['mlm_loss']:.3f} | perplexity {ev['mlm_ppl']:.2f} | top-1 {ev['mlm_acc']:.3f}   (어휘 {V}개 무작위면 ppl={V})")
"""
(어휘 V개 무작위면 ppl = V): 만약 모델이 아무것도 모르는 상태에서 단어장(V) 속 단어들을 무작위로 찍는다면 
                           퍼플렉서티는 정확히 단어장의 크기(V)가 되어야 합니다.
"""
"""
실행 결과: 행은 정답, 열은 예측이며 MLM perplexity 2.73은 무작위 기준 71보다 훨씬 낮다.
혼동행렬 (행=정답, 열=예측)  [NotNext, IsNext]
[[319  41]
 [  3 357]]
MLM(평가) loss 0.922 | perplexity 2.51 | top-1 0.774   (어휘 71개 무작위면 ppl=71)
"""


def fill_mask(a: str, b: str, top: int = 3) -> list[tuple[str, float]]:
    """문장쌍에 있는 [MASK] 위치의 예측 단어와 확률 상위 top개를 반환한다."""
    ids, seg = encode_pair(a, b); ids_t = torch.tensor([ids]).to(device); seg_t = torch.tensor([seg]).to(device)
    # ← encode_pair: 문장 A,B → [CLS] A [SEP] B [SEP] 숫자 로 엮어주고, 세그먼트 id생성
    pos = [i for i, t in enumerate(ids) if t == MASK][0]  # [MASK] 하나의 위치를 찾아 그 자리 예측만 확인
    bert.eval()
    with torch.no_grad(): logits, _ = bert(ids_t, seg_t)
    #완성된 텐서를 학습된 BERT모델에 넣고 모든 토큰위치에서 예측점수(logits)를 받아온다
    probs = torch.softmax(logits[0, pos], -1)
    #그 중에서 정확히 [MASK]가 위치한 단어자리(pos)의 출력값만 빼낸뒤,softmax를 취해 전체 단어장 V대한
    #확률분포로 변환
    p, i = probs.topk(top)
    #확률이 가장 높은 상위 후보개수(top=3)만큼 단어ID(i)와 확률값(p)를 골라서 사람이 읽을 수 있는 다어로
    #디코딩(itos[j])한뒤 최종 리스트로 반환
    return [(itos[j], round(q, 3)) for j, q in zip(i.tolist(), p.tolist())]

# [MASK] 예측은 BERT의 가장 직관적인 '문맥 이해' 테스트다.
# 모델이 앞뒤 문맥을 보고 빈칸에 들어갈 단어를 맞히는 것은 단순 규칙 memorization이 아니라,
# 실제 문맥을 이해하고 있다는 신호로 볼 수 있다.

print(fill_mask('지수는 숨이 차서 [MASK] 물을 마셨다', '지수는 다시 힘을 내어 열 바퀴를 달렸다'))       # '잠시' 가 1순위여야 정상
print(fill_mask('하나는 [MASK] 채소를 썰고 국을 끓였다', '하나는 완성된 국을 식탁에 놓았다'))            # '집에서'
# 실행 결과: 문맥에 맞는 '잠시', '집에서'가 각각 1순위로 예측된다.
"""
[('잠시', 0.633), ('커피를', 0.077), ('내어', 0.052)]
[('집에서', 0.776), ('단어를', 0.034), ('채소를', 0.032)]
"""


bert.eval()
ids, seg, lab = [t.to(device) for t in collate(valid_pairs[:1])]
emb = bert.tok(ids).detach().requires_grad_(True)
# [CLS]만이 아니라 각 입력 토큰이 NSP 예측에 영향을 주는지 gradient 크기로 확인한다.
# BERT는 자기 자신의 문맥을 전부 참조하는 양방향 구조라서,
# [CLS]에서 나온 오차가 전체 시퀀스에 퍼져서 모든 토큰 임베딩에 기울기가 생긴다.
_, nsp_logits = bert(ids, seg, inputs_embeds=emb)
F.cross_entropy(nsp_logits, lab).backward() # NSP 손실만!
g = emb.grad[0].norm(dim=-1)  # 각 토큰 임베딩에 도달한 NSP gradient의 크기
print('토큰      :', [itos[i] for i in ids[0].tolist()])
print('|grad|    :', [round(v, 4) for v in g.tolist()])
print('→ 모든 위치(CLS 뿐 아니라 B 문장의 단어들까지)에 0 이 아닌 기울기:', bool((g > 0).all()))
"""
NSP(다음 문장 예측) 과제는 문장 맨 앞에 있는 [CLS] 토큰의 최종 출력값을 통째로 뽑아내어 판별하는 작업
마스크드 단어를 맞히는 MLM 손실은 아예 빼버리고, 오직 NSP 손실에서 발생한 오차만 뒤로(backward) 역전파
-------------------------------------------------------------------------------------------
# 실행 결과: CLS 외의 토큰도 NSP 예측에 영향을 주므로 gradient가 0이 아니다.
왜 그럴까?
1) BERT는 인코더 구조이므로 미래를 가리는 마스크가 없다
즉, 문장을 읽을 때 모든 단어가 자기 자신뿐만 아니라 앞뒤의 모든 다른 단어들을 양방향으로 동시에 참고(Attention)
2) 맨 앞에 있는 [CLS] 토큰 역시 층을 거쳐 올라가는 동안 문장 A와 문장 B에 있는 모든 단어들의 정보를 골고루 섞어(Mix)가진다.
3) 따라서 [CLS]를 이용해 NSP 정답을 맞히다가 틀렸을 때, 그 오차(기울기)를 거슬러 올라가 보면 문장 끝에 있는 
   단어 하나의 미세한 단서까지도 [CLS]에 영향을 주었기 때문에 모든 단어의 임베딩에 일제히 지분(기울기)이 나누어 들어가게 되는 것
즉,
양방향 어텐션 구조이기 때문에, 문장 맨 앞의 분류용 토큰([CLS]) 하나에서 발생한 오차조차도 
문장 전체(문장 A + 문장 B의 모든 단어)의 협업과 상호작용의 결과물이므로, 
역전파했을때 모든 토큰의 임베딩에 0이 아닌 기울기가 퍼져나간다
"""

"""
→ 모든 위치(CLS 뿐 아니라 B 문장의 단어들까지)에 0 이 아닌 기울기: True
"""


