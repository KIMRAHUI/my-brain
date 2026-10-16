# Transformer Architecture Summary

왜 Transfomer인가?

RNN과 Seq2Seq의 한계

Seq2Seq는 인코더 RNN이 입력 문장을 다 읽은 뒤 마지막 hidden state하나에 
문장 전체를 압축하고, 디코고 RNN이 그 벡터 하나만 보고 번역문을 생성

3가지 문제

병목(bottleneck) : 문장의 길고짧든 고정크기 하나에 모든 의미를 담아야한다

장기 의존성 : 첫 단어의 정보가 마지막까지 가려면 수십번 곱셈을 거쳐야해 기울기소실or폭주

병렬화 불가 : t번째 hidden은 t-1번째가 끝나야 계산가능 GPU를 놀리게 된다.

Transformer

BiLSTM

Attention의 시작 :Bahdanau

BiLSTM의 모든 시점(HiddenState)보존한뒤,
Bahdanau스타일의 Additive Attention(가산어텐션)적용해 문맥벡터(ContextVector)구하고,
분류를 수행하는구조

import torch
import torch.nn as nn

#Embedding → BiLSTM → (Bahdanau식) Attention Pooling → 분류기
class BiLSTMWithAttention(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size=128, num_layer=4, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 양방향 LSTM: 정방향/역방향 hidden(각 hidden_size)을 이어붙여 출력 차원 = hidden_size*2
        # (주석은 괄호 '밖'에 둬야 합니다. 함수 인자 사이에 문자열을 넣으면 SyntaxError)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,       # (batch, seq, feature) 순서
            bidirectional=True,
        )

        # u_t = tanh(W_att · h_t)
        self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        # score_t = W_c · u_t  → 토큰마다 스칼라 1개 (out_features=1 필수)
        self.context_vector = nn.Linear(hidden_size * 2, 1, bias=False)

        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    # forward는 __init__과 같은 들여쓰기 레벨 (클래스의 메서드)
    def forward(self, x, return_attn=False):
        pad_mask = (x == 0)                          # (B, T)
        x = self.embedding(x)                        # (B, T, E)
        out, _ = self.lstm(x)                        # (B, T, 2H)  모든 시점의 hidden

        u = torch.tanh(self.attention(out))          # (B, T, 2H)
        ️score = self.context_vector(u)               # (B, T, 1)
        score = score.masked_fill(pad_mask.unsqueeze(-1), float("-inf"))
        a = torch.softmax(score, dim=1)              # 시퀀스 축(dim=1)으로 softmax

        context = (a * out).sum(dim=1)               # (B, 2H)  가중합 = 컨텍스트 벡터

        context = self.dropout(self.layer_norm(context))
        dense = torch.relu(self.fc1(context))        # 원본은 활성화 없이 fc1→fc2 (선형 2개 = 선형 1개)
        logits = self.fc2(dense)                     # (B, num_classes)
        return (logits, a.squeeze(-1)) if return_attn else logits


# 모델 생성 및 테스트 (클래스 바깥, 들여쓰기 없음)
if __name__ == "__main__":
    model = BiLSTMWithAttention(vocab_size=100, embedding_dim=32)
    inp = torch.tensor([[5, 2, 7, 4, 0, 0]])         # 배치 1, 길이 6 (뒤 2개는 PAD)
    logits, attn = model(inp, return_attn=True)
    print("logits", logits.shape, "attn", attn.shape)
    print("attn weights", [round(v, 3) for v in attn[0].tolist()], "합 =", round(attn.sum().item(), 4))

Flow

Q · K · V

Q · K · V 는 같은 입력 x에 서로다른 Linear(Wq, Wk, Wv)를 곱해서 만듭니다.
같은 단어라도 질문할때의 모습, 색인될때의 모습,내용을 줄때 모습을 따로 배우게하려는것입니다.

Query : 내가 지금 찾는것. “이 단어를 이해하려면 무엇이 필요하지?”

설정 : 하이퍼파라미터와 특수토큰

PAD, BOS, EOS = 0, 1, 2          # 특수 토큰
NUM_DIGITS = 10                  # 실제 데이터 토큰: 3..12 (숫자 0~9)
VOCAB = 3 + NUM_DIGITS           # 전체 어휘 수 = 13

D_MODEL = 128    # 모든 층의 hidden 차원 (논문: 512)
N_HEADS = 4      # 헤드 수 (논문: 8) → 헤드당 d_k = 32
D_FF = 512       # FFN 중간 차원 (논문: 2048, 보통 4*d_model)
N_LAYERS = 3     # 인코더/디코더 층 수 (논문: 6)
DROPOUT = 0.1
MAX_LEN = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TokenEmbedding - 임베딩과 룩업테이블

모델은 정수를 다룰수 없으니 토큰id를 학습 가능한 d_model차원 벡터로 바꿉니다.
nn.Embedding은 vocab X d_model크기의 표에서 id번째 행을 꺼내오는 조회표 





"""
InputEmbedding * √d_ / OutputEmbedding * √d_
토큰 id를 벡터로 바꾸는 입구. 인코더와 디코더에 하나씩 있음
"""
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.d_model = d_model

    def forward(self, x):                      # x: (B, T) 정수
        return self.emb(x) * math.sqrt(self.d_model)   # (B, T, D)

Positional Encoding - 순서 정보 주입

Attention매커니즘을 단어들이 어떤 순서로 배열되었는지 원래는 알지 못합니다.
즉. 컴퓨터 입장에서 단어뭉치가 똑같아서 순서에 따른 주객 관계를 구별하지 못합니다.
그래서 모델에게 이 단어는 몇번째 위치한 단어다 → 일종의 좌표(표식)을 덧붙여 주어야합니다.
이 역할을 하는게 문장 임베딩(Vector)에 더해주는 Positional Encoding입니다.
→ 모델이 단어의 순서와 문맥을 정확히 파악 할 수 있게 만들어 줍니다.

PE(pos, 2i) = sin(pos / 100002i/d)    PE(pos, 2i+1) = cos(pos / 100002i/d)

앞쪽 차원(i가 작음)은 빠르게, 뒤쪽 차원은 느리게 진동합니다. 
0번 차원의 주기는 2π ≈ 6.3칸, 마지막 차원의 주기는 2π·10000칸입니다. 
시계의 초침, 분침, 시침처럼 조합하면 위치마다 고유한 값이 됩니다.

PE(pos+k)는 PE(pos)를 회전시킨 것(선형변환)으로 쓸 수 있어서, 
모델이 "k칸 떨어진 관계"를 쉽게 배웁니다.

학습하지 않는 고정값이라 register_buffer로 등록합니다. 
파라미터는 아니지만 model.to(device)를 할 때 함께 이동하고 state_dict에도 저장됩니다.

"""
Positional Embedding
임베딩 바로 위에서 위치 정보를 더함
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()              # (L, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))             # (D/2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))                   # (1, L, D) 학습X

    def forward(self, x):                      # x: (B, T, D)
        return self.dropout(x + self.pe[:, : x.size(1)])



Scaled Dot - Product Attention - Transformer의 핵심

Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V

Q·Kᵀ 모든 Query와 모든 Key의 내적. 
결과는 (Tq × Tk) 점수표이고, i행 j열은 "i번 토큰이 j번 토큰을 얼마나 원하나"입니다.

÷ √d_k 점수 크기 보정 (아래 설명).

mask 보면 안 되는 칸을 −∞로.

softmax(dim=−1) 각 행을 합이 1인 가중치로.

× V 가중치로 Value들을 가중평균.

Q             (20, 64)      문장 길이 20, 차원 64
Kᵀ            (64, 20)      k.transpose(-1, -2)
Q·Kᵀ          (20, 20)      ← 20개 토큰 × 20개 토큰의 관계 지도
softmax(…)·V  (20, 20) @ (20, 64) = (20, 64)   ← 원래 차원 유지, 내용은 문맥이 섞인 값
배치 포함      output (3, 20, 64), weights (3, 20, 20)

숫자가 왜 커질까?

점수가 커지면 왜 문제가 될까? (Softmax의 함정)

그래서 왜 \sqrt{d_k}로 나누나요?

mask는 왜 쓸까?

 가리는데 왜 -\infty (마이너스 무한대)를 쓸까?

왜 -1e9나 dtype.min 같은 걸 쓸까? 



"""
Multi-Head Attention
Cross-Attention
Masked Self-Attention
3개의 Attention블록 모두 안에서 이 함수를 사용
"""
def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    # q: (B, H, Tq, dk), k: (B, H, Tk, dk), v: (B, H, Tk, dk)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)        # (B, H, Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)                          # 각 행의 합 = 1
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v, attn                                     # (B, H, Tq, dk)

Multi-Head Attention

Attention 한번 - 한 가지 관점의 관계만 잡는다.
d_model을 H조각으로 나눠 H개의 attention을 병렬로 돌리면, 어떤 헤드는 바로 옆단어를,
다른 헤드는 주어와 동사의 관계를 보는 식으로 역할이 나뉨.

x          (B, T, 128)
W_q(x)     (B, T, 128)
view       (B, T, 4, 32)          ← 128을 4헤드 × 32로 나눔
transpose  (B, 4, T, 32)          ← 헤드 축을 앞으로 → 헤드마다 독립 계산
attention  (B, 4, T, 32)   weights (B, 4, T, T)
transpose  (B, T, 4, 32)
view       (B, T, 128)            ← 헤드 결과 이어붙이기(concat)
W_o        (B, T, 128)            ← 헤드 정보 섞기

핵심은 view와 transpose이다. 
실제로 행렬을 4개로 만든게 아닌 하나의 큰 Linear결과를 헤드별로 잘라 보는것 아닌, 계산량은 단일 헤드와
동일하다. → permute(0, 2, 1, 3)는 여기서 transpose(1, 2)와 같은 연산

왜 contiguous( )가 필요할까?

"""
Multi-Head Attention
Cross-Attention
Masked Self-Attention

같은 MultiHeadAttention 클래스를 세 곳에서 씁니다. 
Q, K, V를 어디서 가져오는지만 다릅니다.
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = None                      # 시각화용으로 마지막 가중치 저장

    def _split(self, x):                      # (B, T, D) → (B, H, T, dk)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def _merge(self, x):                      # (B, H, T, dk) → (B, T, D)
        B, H, T, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * dk)

    def forward(self, query, key, value, mask=None):
        # self-attention이면 query=key=value=x
        # cross-attention이면 query=디코더 상태, key=value=인코더 출력(memory)
        q = self._split(self.w_q(query))
        k = self._split(self.w_k(key))
        v = self._split(self.w_v(value))
        out, self.attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        return self.w_o(self._merge(out))



Position-wise-Feed-Forward

FFN(x) = W₂ · ReLU(W₁·x + b₁) + b₂

Attention이 토큰사이에서 정보를 섞는다면,
FFN은 토큰 각각을 독립적으로 가공한다. 모든 위치에 같은 가중치를 쓰고 (Position - wise)
128 → 512 → 128로 넓혔다 좁히면서 비선형성을 더해준다.
Attention만으로 학습가능한 가중치가 적고 비선형도 softmax뿐이라, 표현력을 FFN이채운다.
층 하나의 파라미터 중 약 2/3이 여기 있다.

"""
Feed-Forward

각 층의 Attention 다음에 오는 블록입니다
"""
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

FFN에서 표현력이 뭘까?

비선형성을 왜 키워야할까?

왜 파라미터 2/3이 여기에 몰림?



LayerNorm과 Residual (Add&Norm)

딥러닝 모델이 수십,수백층을 아주 깊게 쌓아올려도 학습안깨지고, 
무사히 성공하게 지탱해주는 거대한 안전 장치에 대한 것

층 정규화(Layer Norm): 숫자들을 균일하게 다스린다!

잔차 연결(Residual / SkipConnection) : 변화량만 더해준다.

Pre-LN vs Post-LN (최신 모델들의 표준)

# 강의 노트북 방식 (원 논문, Post-LN): 더한 "다음"에 정규화
out1 = self.layernorm1(x + self.dropout1(self.mha(x, x, x, mask)[0]))
out2 = self.layernorm2(out1 + self.dropout2(self.ffn(out1)))

# 이 페이지 방식 (Pre-LN, GPT-2 이후 표준): 정규화한 것을 서브층에 넣고 원본 x에 더함
x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
x = x + self.dropout(self.ffn(self.norm2(x)))
# → 잔차 경로(x)가 어떤 정규화도 거치지 않고 끝까지 이어져 gradient가 안정적
# → 대신 마지막 층 뒤에 LayerNorm을 한 번 더 붙여야 함 (Encoder/Decoder의 self.norm)

"""
모든 서브층을 감싸는 Add & Norm 5곳입니다.
"""
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class ResidualConnection(nn.Module):
    """x + Dropout(Sublayer(LayerNorm(x)))   (Pre-LN)"""
    def __init__(self, d_model, dropout=DROPOUT):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))



Encoder

인코더 층 하나는 Self-Attention ➔ FFN 순서로 이어져 있으며, 
각각의 서브층은 Add & Norm (Residual + LayerNorm)으로 안전하게 감싸져 있습니다.

입력 (B, S, D)  →  memory (B, S, D)   ← shape은 그대로, 내용만 문맥화
강의 노트북 확인값 EncoderLayer(32, 4, 64)에 (3, 10, 32) 입력 → (3, 10, 32)

인코더 층은 어떤 순서로 동작하고, 토큰들은 왜 앞뒤 구분 없이 양방향으로 볼까?

인코더에서는 왜 다른마스크는 안쓰고 오직 PAD(빈칸)만 가릴까?

똑같은 인코더층을 왜 굳이 N번씩 겹겹이 쌓고 그것을 왜 memory라 부를까?

lambda y: self.self_attn(y, y, y, mask)는 정확히 왜 쓰나요

Decoder

디코더에는 3개의 서브층이 있다.
Masked Selt-Attention , Cross-Attention, FFN

Masked Self-Attention은 왜 쓰고, 일반 Self-Attention과 무엇이 다른가?

Cross-Attention은 정확히 어떤 역할을 하고, 왜 이름이 '크로스'인가?

디코더에 있는 FFN은 인코더의 FFN과 다른가?

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=DROPOUT):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.res1 = ResidualConnection(d_model, dropout)
        self.res2 = ResidualConnection(d_model, dropout)
        self.res3 = ResidualConnection(d_model, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.res1(x, lambda y: self.self_attn(y, y, y, tgt_mask))
        x = self.res2(x, lambda y: self.cross_attn(y, memory, memory, src_mask))
        x = self.res3(x, self.ffn)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)



mask 만들기

mask값1은 봐도된다! 0은 가린다! 
(B, 1, Tq, Tk)로 맞춰두면 헤드 차원(1)에 자동으로 broadcast됩니다.

src_mask (B, 1, 1, S)   ← 모든 query 행에 같은 PAD 가림을 broadcast
tgt_mask (B, 1, T, T)   ← PAD 가림 AND 하삼각(tril)

# 규약 A — 이 페이지: 1(True) = 봐도 됨, 0(False) = 가림
src_mask = (src != PAD)                                  # 진짜 토큰이 True
scores = scores.masked_fill(src_mask == 0, float("-inf"))

# 규약 B — 강의 노트북: True = 가림
pad_mask = (seq == PAD)                                  # PAD가 True
scores = scores.masked_fill(pad_mask == True, -1e9)

# 두 마스크를 합칠 때 연산자도 반대가 됩니다
tgt_mask_A = pad_A & causal_A        # A: 둘 다 "봐도 됨"이어야 봄  → AND
tgt_mask_B = pad_B | look_ahead_B    # B: 하나라도 "가림"이면 가림   → OR

# 서로 바꾸기
mask_B = ~mask_A

# 주의: nn.MultiheadAttention 의 key_padding_mask 와 bool attn_mask 는 규약 B(True=가림),
#       F.scaled_dot_product_attention 의 bool attn_mask 는 규약 A(True=참여)입니다.

"""
Causal mask가 있기 때문에 
디코더는 학습 때 정답 문장 전체를 한 번에 넣고도 
각 위치가 자기 이전만 보게 됩니다. 한 번의 병렬 연산으로 모든 위치를 학습하는 비결입니다. 
강의 노트북 출력에서 1행은 [1.0, 0, 0, …], 2행은 [0.52, 0.48, 0, …]처럼 
상삼각이 정확히 0인 것을 볼 수 있습니다
"""
def make_src_mask(src):
    # PAD 토큰은 key로서 보지 않는다.  (B, 1, 1, S)
    return (src != PAD).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt):
    # PAD 가림 & 미래 가림(하삼각 행렬)  → (B, 1, T, T)
    T = tgt.size(1)
    pad = (tgt != PAD).unsqueeze(1).unsqueeze(2)                     # (B,1,1,T)
    causal = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # (T,T)
    return pad & causal

Transformer 전체 조립 

encode와 decode를 따로 두는 이유는 
추론때 인코더는 한번만! 디코더는 토큰마다 반복 호출하기 위함

 Transformer 전체 조립 클래스에서 encode와 decode 함수를 왜 따로 분리해 두었을까?


전체 조립 구조에서 
최종 단계의 nn.Linear(d_model, tgt_vocab)(Generator)은 무엇을 하는 곳인가요?

디코더를 다 거쳐 나온 최종 출력에 왜 선형층(Generator)을 붙일까?

