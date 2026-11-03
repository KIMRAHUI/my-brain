# LoRA

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
import copy
import time

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
# ── LoRA: 동결된 기존 선형층 W0에 저랭크 업데이트 ΔW = scale × BA만 더한다 ──
# 핵심 아이디어는 모든 가중치를 다시 학습시키지 않고, 원래 가중치 W0는 고정한 채
# 작은 2개의 행렬 A, B만 학습해서 ΔW = s * B @ A 를 만들자는 것이다.
# 이렇게 되면 파라미터 수를 대폭 줄이면서도, 원래 가중치의 표현력을 유지할 수 있다.
# 특히 추론 시에는 ΔW를 W0에 병합해 계산 비용을 거의 그대로 유지할 수 있다.
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.base, self.r, self.scale, self.merged = base, r, alpha / r, False
        #[스케일 배율]self.scale = alpha / r하이퍼 파라미터 알파(a)와 랭크(r)의 비율로 학습 안정성과 LoRA 업데이트의 영향력을 조절

        for p in self.base.parameters(): p.requires_grad = False# W0 동결
        # A: (r, in)으로 입력을 작은 rank r 공간으로 압축, B: (out, r)로 다시 복원
        # 이 둘만 학습하고 W0는 동결해서 파라미터 효율성을 높인다.
        self.A = nn.Parameter(torch.empty(r, base.in_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))# B=0 → 초기 ΔW=0 ,초기 출력 동일 (B=0) 검증
        #self.B를 전부 0으로 초기화 ->  이로 인해 학습 시작 직전에는 추가된 변화량($\Delta W = B \times A$)이 정확히 0이 되므로,
        # 기존 모델의 출력값과 LoRA를 장착한 직후의 출력값이 100% 완벽하게 똑같음을 보장
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 학습 시: W0(x) + ΔW(x). 병합 뒤에는 base.weight에 ΔW가 이미 반영되어 있다.
        # A, B가 0으로 시작하므로 초기엔 base(x)와 정확히 동일한 결과를 낸다.
        """원래 모델의 결과물 self.base(x)에 입력값x가 A를 거쳐 B로 압축,복원된후 스케일이 곱해진 변화량을 더해서 내보냄"""
        y = self.base(x)
        if not self.merged:
            y = y + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale# x → A → B → ×scale
        return y

    @torch.no_grad()
    def merge(self) -> None:
        # 추론용: ΔW를 원래 가중치에 더해 어댑터의 추가 행렬곱을 없앤다.
        # 병합 후에는 LoRA 경로를 끄고 원래 선형층으로 계산할 수 있다.
        """
        학습끝난뒤 추론 Interface속도를 빠르게 만들기 위해서 따로 놀던 LoRA행렬을 기존 가중치에 미리 합침
        base.weight += scale * (B @ A)
        -> 추론할때 추가적인 연산 지연Latency 발생하지 않음
        """
        if not self.merged: self.base.weight += self.scale * (self.B @ self.A); self.merged = True
    @torch.no_grad()
    def unmerge(self) -> None:
        # 다시 학습하거나 다른 어댑터로 바꾸려면 합쳤던 ΔW를 원상 복구한다.
        if self.merged: self.base.weight -= self.scale * (self.B @ self.A); self.merged = False

def apply_lora(model: nn.Module, targets: tuple[str, ...] = ('wq', 'wv'), r: int = 4, alpha: int = 8, dropout: float = 0.0) -> nn.Module:
    """
    모델 전체에 파라피터를 먼저 동결,사용자가 지정한 타겟 레이어이름(예: 'wq', 'wv' 등 어텐션 가중치)을 찾아 기존 nn.Linear를
    LoRALinear로 동적 교체(Hot-swap)해 준다.
    결과적으로 거대한 모델 전체에서 오직 새로 추가된 작고 소중한 A, B 행렬들만 학습 가능(requires_grad=True)하게 한다.
    """
    for p in model.parameters(): p.requires_grad = False # 1) 전부 동결
    for _, module in list(model.named_modules()): # 2) 대상 Linear 를 LoRALinear 로 교체 (새 A,B 만 학습 가능)
        for cname, child in list(module.named_children()):
            # 이름이 target이고 실제 Linear인 경우에만 LoRA wrapper로 교체한다.
            if cname in targets and isinstance(child, nn.Linear):
                setattr(module, cname, LoRALinear(child, r, alpha, dropout))
    return model

def count_params(model: nn.Module) -> tuple[int, int]:
    # 전체 크기와 실제 optimizer가 갱신할 trainable 크기를 함께 센다.
    tot = sum(p.numel() for p in model.parameters()); trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, trn


torch.manual_seed(0)
base = nn.Linear(32, 32); lora = LoRALinear(base, r=4, alpha=8)
x = torch.randn(5, 32)
print('초기 출력 동일 (B=0):', torch.allclose(base(x), lora(x)))
# 실행 결과: True
"""
초기 출력 동일 (B=0): True
"""


def lora_manual_backward(x: torch.Tensor, W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, dy: torch.Tensor, s: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """y = xWᵀ + s(xAᵀ)Bᵀ에 대해 dX, dA, dB를 연쇄법칙으로 직접 계산해서 구한 동 미분 공식(dx, dA, dB)과, 파이토치가 알아서 미분해 주는 정답(torch.autograd.grad)을 비교"""
    dB = s * dy.t() @ (x @ A.t())                 # (out, r)
    dA = s * (dy @ B).t() @ x                     # (r, in)
    dx = dy @ W + s * (dy @ B) @ A                # (N, in)
    return dx, dA, dB

N, i, o, r, s = 6, 8, 5, 3, 2.0
x = torch.randn(N, i, dtype=torch.double, requires_grad=True); W = torch.randn(o, i, dtype=torch.double)
A = torch.randn(r, i, dtype=torch.double, requires_grad=True); B = torch.randn(o, r, dtype=torch.double, requires_grad=True)
y = x @ W.t() + s * (x @ A.t()) @ B.t(); dy = torch.randn(N, o, dtype=torch.double)
gx, gA, gB = torch.autograd.grad(y, (x, A, B), dy)                            # autograd 정답
mx, mA, mB = lora_manual_backward(x.detach(), W, A.detach(), B.detach(), dy, s)
print('수동 backward == autograd (dx, dA, dB):', torch.allclose(gx, mx), torch.allclose(gA, mA), torch.allclose(gB, mB))

# 초기(B=0) 에서의 기울기: dA=0, dB≠0
lora = LoRALinear(nn.Linear(32, 32), r=4, alpha=8)
lora(torch.randn(10, 32)).pow(2).sum().backward()
print(f'초기 |dA| = {lora.A.grad.norm().item():.4f} (0!),  |dB| = {lora.B.grad.norm().item():.4f} (0 아님)')
# 실행 결과: B=0이면 dA 식 안의 B가 0이므로 A는 첫 step에서 gradient를 받지 않는다.
"""
수동 backward == autograd (dx, dA, dB): True True True -> LoRA의 행렬 연산과 스케일링 인자(s)가 수학적 미분 관점에서 한 치의 오차없음 확인
초기 |dA| = 0.0000 (0!),  |dB| = 49.8886 (0 아님)
행렬 A의 기울기(|dA|)는 정확히 0.0000이 나오고, 행렬 B의 기울기(|dB|)는 0이 아닌 값이 나온다.
왜그럴까?
- 학습 초기 상태에서 B 행렬은 전부 0으로 초기화되어있다.(`self.B = 0`).
- 따라서(dy \cdot B)계산 결과 자체가 통째로 0이 되어버리기 때문에, A 쪽으로 전달되는 기울기가 완전히 차단되어 `dA = 0`이 되는 것
- 반면B에 대한 기울기(`dB`)는 A와 입력값 x의 곱(`x @ A.t()`)이 살아있기 때문에 0이 아닌 정상적인 기울기가 확인된다.
"""


class MultiHeadAttention(nn.Module):
    def __init__(self, D: int, H: int) -> None:
        super().__init__()
        self.h, self.dk, self.D = H, D // H, D
        self.wq, self.wk, self.wv, self.dense = (nn.Linear(D, D) for _ in range(4))
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # MiniBert의 양방향 self-attention: PAD만 가리고 미래 토큰은 가리지 않는다.
        B, T, _ = x.size()
        sp = lambda t: t.view(B, T, self.h, self.dk).transpose(1, 2)
        q, k, v = sp(self.wq(x)), sp(self.wk(x)), sp(self.wv(x))
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if mask is not None: s = s.masked_fill(mask, float('-inf'))
        return self.dense((torch.softmax(s, -1) @ v).transpose(1, 2).contiguous().view(B, T, self.D))

class EncLayer(nn.Module):
    #MHA와 FFN(GELU 활성화 포함)을 품고 있으며, 잔차 연결(Residual Connection)과 LayerNorm을 거치는 인코더 블록
    def __init__(self, D: int, H: int, ff: int, rate: float = 0.1) -> None:
        super().__init__()
        self.mha, self.ffn = MultiHeadAttention(D, H), nn.Sequential(nn.Linear(D, ff), nn.GELU(), nn.Linear(ff, D))
        self.ln1, self.ln2, self.do = nn.LayerNorm(D), nn.LayerNorm(D), nn.Dropout(rate)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # attention과 FFN 각각에 잔차 연결 + LayerNorm을 적용하는 Post-LN 블록
        x = self.ln1(x + self.do(self.mha(x, mask)))
        return self.ln2(x + self.do(self.ffn(x)))

class MiniBert(nn.Module):                                      # 백본 (사전학습 대상)
    def __init__(self, vocab: int, D: int = 64, H: int = 4, ff: int = 256, n_layers: int = 3, max_len: int = 24) -> None:
        super().__init__()
        self.tok, self.pos, self.ln = nn.Embedding(vocab, D, padding_idx=PAD), nn.Embedding(max_len, D), nn.LayerNorm(D)
        self.layers = nn.ModuleList([EncLayer(D, H, ff) for _ in range(n_layers)])
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # 토큰 임베딩과 학습 가능한 위치 임베딩을 더한 뒤 인코더 블록을 통과한다.
        mask = (ids == PAD)[:, None, None, :]
        x = self.ln(self.tok(ids) + self.pos(torch.arange(ids.size(1), device=ids.device)))
        for l in self.layers: x = l(x, mask)
        return x

# ── ① MLM 사전학습 (단일 문장) ──
def mlm_mask(ids: torch.Tensor, p: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """
    특수 토큰(PAD, CLS)은 철저히 보호하면서,
    전체 토큰 중 15%를 무작위로 골라 80%는 [MASK],
    10%는 무작위 단어,
    10%는 원본으로 유지하여 모델에게 "빈칸 채우기" 학습 과제를 부여
    """
    # 단일 문장 MLM도 특수 토큰을 제외하고 15%를 80/10/10 방식으로 바꾼다.
    ids, labels = ids.clone(), torch.full_like(ids, -100)
    chosen = (torch.rand(ids.shape) < p) & (ids != PAD) & (ids != CLS)
    labels[chosen] = ids[chosen]; r = torch.rand(ids.shape)
    ids[chosen & (r < 0.8)] = MASK; rnd = chosen & (r >= 0.8) & (r < 0.9)
    ids[rnd] = torch.randint(len(SPECIALS), V, ids.shape)[rnd]
    return ids, labels

sent_ids = [[CLS] + encode(s) for st, _ in TRAIN_STORIES for s in st]   # ← encode: 문장(str) → 숫자 id 리스트
backbone = MiniBert(V).to(device); mlm_head = nn.Linear(64, V).to(device)
opt = torch.optim.AdamW([*backbone.parameters(), *mlm_head.parameters()], lr=2e-3)
for step in range(250):
    """
    - `AdamW` 옵티마이저: 백본 모델과 MLM 예측 헤드(`mlm_head`)의 파라미터들을 동시에 묶어서 최적화
    - 미니 배치 학습 (250 스텝): 문장 데이터셋에서 무작위로 64개씩 샘플링(`random.sample`)하고, 
     실시간으로 MLM 마스킹을 거친 뒤 크로스 엔트로피 오차를 계산하여 역전파(`loss.backward()`)를 수행
    """
    batch = pad_batch(random.sample(sent_ids, 64)); mi, ml = mlm_mask(batch)# ← pad_batch: 길이가 다른 리스트들 → PAD(0)로 채워 (B, L) 텐서; mlm_mask: 15% 골라 80/10/10 으로 가리기 → (입력, 정답 라벨)
    logits = mlm_head(backbone(mi.to(device)))
    loss = F.cross_entropy(logits.reshape(-1, V), ml.to(device).reshape(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
    if (step + 1) % 50 == 0: print(f'MLM 사전학습 step {step+1} loss {loss.item():.3f}')
pretrained_state = copy.deepcopy(backbone.state_dict())
"""
사전 학습이 완료된 백본의 가중치 상태(state_dict)를 딥카피(Deep Copy)하여 메모리에 안전하게 박제
이 사전 학습된 가중치는 나중에 파인튜닝(Fine-Tuning)이나 LoRA 등의 어댑터 실험을 진행할 때, 
아무것도 모르는 백지 상태의 모델과 비교하는 기준점(Baseline) 혹은 출발점으로 재사용
"""
# 실행 결과: loss가 전반적으로 내려가며, 복사본은 아래 네 가지 미세조정의 같은 출발점이 된다.
"""
MLM 사전학습 step 50 loss 1.270
MLM 사전학습 step 100 loss 0.508
MLM 사전학습 step 150 loss 0.340
MLM 사전학습 step 200 loss 0.594
MLM 사전학습 step 250 loss 0.286
"""


# ── ② 다운스트림 분류: 문장 → 이야기 속 몇 번째 문장인가(4-class), 소량 라벨(40개)만 사용 ──
"""
문장 하나를 입력받았을 때, 이 문장이 이야기 4문장 중 몇 번째 위치(0번~3번)에 오는 문장인지 맞히는 4개 클래스 분류(Classification) 문제
-----------------------------------------------------
[CLS]토큰 활용
Classifier 클래스는 백본을 통과한 출력 중 
맨 앞의 [CLS] 토큰 위치(backbone(ids)[:, 0])의 벡터를 뽑아내어 최종 분류 헤드(nn.Linear(64, n_cls))에 연결
"""
from sklearn.metrics import accuracy_score, f1_score
import io, pandas as pd
def sent_dataset(stories: list[Story]) -> list[tuple[list[int], int]]: return [([CLS] + encode(s), k) for st, _ in stories for k, s in enumerate(st)]
few_train = random.Random(0).sample(sent_dataset(TRAIN_STORIES), 40); val_set = sent_dataset(VALID_STORIES)

class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, n_cls: int = 4) -> None:
        super().__init__(); self.backbone, self.head = backbone, nn.Linear(64, n_cls)
    def forward(self, ids: torch.Tensor) -> torch.Tensor: return self.head(self.backbone(ids)[:, 0])        # [CLS] 위치로 4개 문장 위치를 분류

def make_model(method: str, r: int = 4, alpha: int = 8) -> nn.Module:
    """
    1. `full` (전체 파인튜닝):
    - 모델의 모든 가중치를 열어두고 처음부터 끝까지 전부 학습시킵니다. 데이터가 적을 때 과적합(Overfitting) 위험이 큼
    2. `freeze_head` (선형 프로빙, Linear Probing):
        - 사전 학습된 백본은 완전히 꽁꽁 얼려두고(`requires_grad = False`),
        오직 마지막에 새로 붙은 분류 헤드(`head`)만 학습. 학습 파라미터가 가장 적습니다.
    3. `freeze_top1` (상위 1개 층만 학습):
        - 백본 전체를 동결하되, 가장 마지막 인코더 층(`layers[-1]`)과 분류 헤드만 열어두어 고차원 특징을 미세하게 적응시킨다.
    4. `lora` (LoRA 어댑터 적용):
        - 백본 가중치는 단단히 동결하고, 어텐션 가중치(`wq`, `wv`)에
        LoRA 행렬(A, B)을 주입하여 저랭크 파라미터들만 학습시키다.
    """
    bb = MiniBert(V).to(device); bb.load_state_dict(pretrained_state)# 같은 사전학습 가중치에서 시작
    model = Classifier(bb).to(device)
    if method == 'freeze_head':
        # Linear probing: 백본은 고정하고 새 분류 head만 학습
        for p in model.backbone.parameters(): p.requires_grad = False
    elif method == 'freeze_top1':
        # 마지막 인코더 블록과 head만 열어 제한적으로 적응
        for p in model.backbone.parameters(): p.requires_grad = False
        for p in model.backbone.layers[-1].parameters(): p.requires_grad = True
    elif method == 'lora':
        # 백본 W0는 전부 동결, 각 층 wq/wv의 A,B 어댑터와 분류 head만 학습
        apply_lora(model.backbone, ('wq', 'wv'), r, alpha) # head 는 새로 만든 파라미터라 학습 가능 유지
    return model.to(device)

def evaluate_cls(model: nn.Module, data: list[tuple[list[int], int]]) -> tuple[float, float]:
    model.eval(); preds, golds = [], []
    with torch.no_grad():
        for i in range(0, len(data), 256):
            ch = data[i:i + 256]; preds += model(pad_batch([c[0] for c in ch]).to(device)).argmax(-1).tolist(); golds += [c[1] for c in ch]   # ← pad_batch: 길이가 다른 리스트들 → PAD(0)로 채워 (B, L) 텐서
    return accuracy_score(golds, preds), f1_score(golds, preds, average='macro')

LR = {'full': 5e-4, 'freeze_head': 3e-3, 'freeze_top1': 2e-3, 'lora': 3e-3}
results, models = [], {}
for method in ['full', 'freeze_head', 'freeze_top1', 'lora']:
    torch.manual_seed(0); random.seed(0)
    model = make_model(method); tot, trn = count_params(model)
    # 동결 파라미터의 합을 기록해 학습 후에도 바뀌지 않았는지 검증한다.
    frozen_before = {n: p.detach().double().sum().item() for n, p in model.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR[method])
    t0 = time.time()
    for epoch in range(15):
        model.train(); random.shuffle(few_train)
        for i in range(0, len(few_train), 8):
            ch = few_train[i:i + 8]
            loss = F.cross_entropy(model(pad_batch([c[0] for c in ch]).to(device)), torch.tensor([c[1] for c in ch]).to(device))   # ← pad_batch: 길이가 다른 리스트들 → PAD(0)로 채워 (B, L) 텐서
            opt.zero_grad(); loss.backward(); opt.step()
    train_s = time.time() - t0
    changed = [n for n, v in frozen_before.items() if v != {k: p.detach().double().sum().item() for k, p in model.named_parameters()}[n]]
    assert not changed, f'동결했는데 변한 파라미터: {changed[:3]}'# 동결 가중치 불변 검증
    buf = io.BytesIO(); torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, buf)   # 학습된 파라미터만 저장
    acc, f1 = evaluate_cls(model, val_set); models[method] = model
    results.append({'방식': method, '전체 파라미터': tot, '학습 파라미터': trn, '학습 %': round(100 * trn / tot, 2), '옵티마이저 상태(×2)': 2 * trn,
                    '학습시간(s)': round(train_s, 2), '저장용량(KB)': round(buf.tell() / 1024, 1), 'val acc': round(acc, 4), 'val macroF1': round(f1, 4)})
print(pd.DataFrame(results).to_string(index=False))
# 실행 결과: LoRA는 전체의 약 2%만 학습하면서 이 토이 과제에서 full과 같은 성능을 냈다.
"""
         방식  전체 파라미터  학습 파라미터   학습 %  옵티마이저 상태(×2)  학습시간(s)  저장용량(KB)  val acc  val macroF1
       full   156420   156420 100.00        312840     0.45     626.8   1.0000       1.0000
freeze_head   156420      260   0.17           520     0.11       2.7   0.9292       0.9292
freeze_top1   156420    50244  32.12        100488     0.21     201.8   1.0000       1.0000
       lora   159492     3332   2.09          6664     0.28      17.5   1.0000       1.0000
"""


m = models['lora']; m.eval()
x = pad_batch([c[0] for c in val_set[:32]]).to(device)   # ← pad_batch: 길이가 다른 리스트들 → PAD(0)로 채워 (B, L) 텐서
with torch.no_grad(): before = m(x)
for mod in m.modules():
    if isinstance(mod, LoRALinear): mod.merge()                         # W0 ← W0 + s·BA  (어댑터 경로 제거)
with torch.no_grad(): after = m(x)
print('merge 후 로짓 동일:', torch.allclose(before, after, atol=1e-5), '| max diff', (before - after).abs().max().item())
"""
병합하기 전, 기존 가중치와 LoRA 어댑터 경로를 따로 거쳐 계산된 예측 결과값(before)을 저장
모델 내부의 모든 LoRALinear 모듈을 순회하며 .merge()를 호출
-> 이 메서드는 따로 놀던 LoRA 행렬(B \times A)에 스케일(s)을 곱해 기존 베이스 가중치(W_0)에 미리 덧셈으로 흡수
어댑터 경로가 사라진 상태에서 다시 예측 결과값(after)을 뽑아냄
---------------------------------------------------------------
torch.allclose(before, after, atol=1e-5) 결과가 True로 떨어지며 최대 오차(max diff)가 거의 0에 가깝게 출력
학습할 때는 어댑터와 베이스를 따로 연산하지만, 학습이 끝나고 서비스를 서빙(Inference)할 때는 
둘을 하나로 합쳐버려도(Merge) 성능이나 출력 결과에 일말의 오차도 없으며, 
추가적인 연산 지연(Latency) 없이 원래 모델과 똑같은 속도로 빠르게 추론할 수 있음 확인.
"""
for mod in m.modules():
    if isinstance(mod, LoRALinear): mod.unmerge()
lora_keys = [n for n, _ in m.named_parameters() if n.endswith('.A') or n.endswith('.B')]
"""
모델 전체 파라미터 이름 중 끝이 .A 또는 .B로 끝나는 LoRA 전용 파라미터들을 골라냄
모델은 총 3개의 인코더 층(layers 3개)을 가지고 있고, 
각 층마다 어댑터 타겟으로 wq, wv 두 개의 선형 레이어를 지정(3개 층 × 2개 레이어 = 6개).
각 레이어마다 행렬 A와 B가 각각 존재하므로 총 텐서 개수는 6 × 2 = 12개
백본 모델의 수많은 파라미터는 꽁꽁 얼려둔 채, 오직 이 12개의 소형 어댑터 텐서들만 쏙쏙 골라내어 효율적으로 미세조정(Fine-Tuning)을 완료
"""
print('LoRA 어댑터 텐서 수:', len(lora_keys), '(레이어 3개 × {wq, wv} × {A, B} = 12)')
# 실행 결과: 병합 전후의 로짓은 수치 오차 범위에서 같고, A/B 텐서는 정확히 12개다.
"""
merge 후 로짓 동일: True | max diff 1.9073486328125e-06
LoRA 어댑터 텐서 수: 12 (레이어 3개 × {wq, wv} × {A, B} = 12)
"""



# !pip install -q "transformers" "peft" "datasets" "accelerate" "evaluate" "scikit-learn"
# 선택 실습: 이 아래는 외부 모델 다운로드가 필요한 Hugging Face/PEFT 예시다.
# 현재 토이 실습 실행을 막지 않도록 문자열로 보관한다. 필요하면 별도 노트북 셀에서 꺼내 실행한다.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

model_name = 'klue/bert-base'                                         # 과제 13 의 분류 모델 (klue/roberta-base 도 동일)
tok = AutoTokenizer.from_pretrained(model_name)
base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,                 # 분류 헤드(classifier)도 자동으로 학습 대상(modules_to_save)에 포함
    #다운스트림 태스크가 시퀀스 분류(Sequence Classification)임을 명시. 이 설정 덕분에 백본의 어텐션 가중치뿐만 아니라, 
    #새로 붙은 분류 헤드(Classifier Head)도 학습 대상(modules_to_save)에 자동으로 포함되어 안전하게 학습
    r=8, lora_alpha=16, lora_dropout=0.05, #랭크는 8, 스케일링 배율을 결정하는 알파는 16으로 주어 저랭크 행렬의 용량과 표현력을 적절히 조절
    target_modules=['query', 'value'],# BERT 계열 어텐션 이름 (GPT-2 계열은 ['c_attn'], LLaMA/Gemma 는 ['q_proj','v_proj'])
)
model = get_peft_model(base, lora_cfg)
#기존의 거대한 베이스 모델(base)을 PEFT 래퍼(Wrapper)로 감싸는 순간, 
#백본의 모든 파라미터는 자동으로 꽁꽁 얼려지고(requires_grad=False), 지정한 타겟 모듈에만 소형 어댑터(A, B)가 장착
model.print_trainable_parameters()
# trainable params: ... || all params: ... || trainable%: ...
# model.print_trainable_parameters()를 실행하면 전체 파라미터 중 단 1~2% 미만의 파라미터만 학습 가능하도록 줄어든 비율 확인 가능
args = TrainingArguments(output_dir='lora_out', per_device_train_batch_size=16, learning_rate=2e-4,# LoRA 는 Full(2e-5) 보다 큰 lr
                         num_train_epochs=3, eval_strategy='epoch', save_strategy='no', report_to='none',
                         lr_scheduler_type='linear', warmup_ratio=0.1, fp16=True)
# trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
#                   data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
# trainer.train()
# model.save_pretrained('lora_adapter') # 어댑터(+헤드)만 저장 → 수 MB
# merged = PeftModel.from_pretrained(AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3), 'lora_adapter').merge_and_unload()
"""
"""
save_pretrained & merge_and_unload
- `model.save_pretrained('lora_adapter')`:
    - 수백 MB~수 GB에 달하는 모델 전체를 저장할 필요 없이, 
      오직 수 MB 짜리 소형 어댑터 가중치 파일들만 싹 모아서 가볍게 저장->공유와 배포 편해진다.
- `merge_and_unload()`:
    - 수동으로 구현했던 `.merge()`와 똑같은 기능
      학습이 끝난 어댑터 가중치를 원본 베이스 모델에 깔끔하게 합쳐버리고(`Merge`), 
      PEFT 래퍼를 벗겨내어(`Unload`) 추론 속도 지연(Latency)이 전혀 없는 
      순수한 원본 형태의 고성능 모델로 되돌려 서빙(Inference)할 수 있게 만들어 준다.
"""

