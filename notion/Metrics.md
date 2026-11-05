# 🔥Metrics

import math
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics as skm

from Common_data import PAD, TRAIN_STORIES

# ── 평가 지표의 본질: 숫자 하나로 모델이 얼마나 잘했는지 정량화하는 법 ──
# 1) 분류 문제에서는 '정답/오답'을 표로 정리하는 혼동 행렬이 가장 기본이다.
#    - 행: 실제 클래스, 열: 모델의 예측 클래스
#    - 대각선: 정답 맞춘 개수, 나머지: 오분류 수
# 2) Precision / Recall / F1은 혼동 행렬에서 각 클래스별로 계산되며,
#    - Precision = 실제로 맞춘 값 / 모델이 긍정이라고 한 값
#    - Recall = 실제로 맞춘 값 / 실제 정답이 긍정인 값
#    - F1 = Precision과 Recall의 조화평균으로 불균형 데이터에서 더 안정적이다.
# 3) average='macro'는 클래스별 점수를 같은 가중치로 평균,
#    average='micro'는 전체 TP/FP/FN을 합쳐 한 번에 계산,
#    average='weighted'는 실제 클래스 비중에 따라 가중 평균한다.
#    이렇게 각 평균 방식은 어떤 평가를 중요하게 보느냐에 따라 서로 다른 해석을 준다.
# 4) 여기서 구현하는 핵심은 '외부 라이브러리 없이 NumPy만으로 계산하고, sklearn 결과와 일치하는지 검증'하는 것이다.
#    - 수치 오차가 허용 범위 안에 들어오는지 확인하는 과정은 모델 평가 코드를 신뢰할 수 있게 만드는 핵심 검증 단계다.


# ── 분류 지표: 혼동 행렬 → Precision / Recall / F1 ──
def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int) -> np.ndarray:
    """행=실제 정답, 열=모델 예측인 (클래스 수, 클래스 수) 혼동 행렬을 만든다.

    각 샘플을 순회하면서 실제 클래스 t와 예측 클래스 p를 찾고, 해당 위치 cm[t, p]를 1씩 누적한다.
    이 표는 분류 모델의 '오분류 패턴'을 한눈에 보여 주므로, 어떤 클래스가 자주 헷갈리는지를 빠르게 파악할 수 있다.
    예를 들어, 클래스 0이 클래스 1로 많이 잘못 분류되었다면 cm[0, 1] 값이 커진다.
    """
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def prf_from_cm(cm: np.ndarray, average: str = 'macro') -> tuple[float, float, float]:
    """혼동 행렬만으로 macro / micro / weighted 평균 P, R, F1을 계산한다.

    핵심 아이디어는 각 클래스별로 다음을 계산하는 것이다:
    - tp: 해당 클래스에서 맞춘 수
    - fp: 다른 클래스인데 그 클래스로 잘못 예측한 수
    - fn: 해당 클래스인데 다른 클래스로 놓친 수

    이 세 값은 혼동 행렬에서 간단히 뽑아낼 수 있으며, 이후 각 평균 방식에 따라
    클래스별 점수를 합치거나 가중 평균해서 최종 P/R/F1을 얻는다.
    """
    tp = np.diag(cm).astype(float)   # 대각선: 각 클래스를 맞힌 수
    fp = cm.sum(0) - tp              # 해당 클래스로 잘못 예측한 수
    fn = cm.sum(1) - tp              # 해당 클래스인데 놓친 수
    sup = cm.sum(1)                  # 각 클래스의 실제 샘플 수(support)

    if average == 'micro':
        # 모든 클래스를 합쳐 TP/FP/FN을 한 번만 계산한다.
        # 단일 라벨 다중 분류에서는 전체 정확도(accuracy)와 동일한 의미를 가질 때가 많다.
        P = tp.sum() / max(tp.sum() + fp.sum(), 1)
        R = tp.sum() / max(tp.sum() + fn.sum(), 1)
        return P, R, 2 * P * R / max(P + R, 1e-12)

    # where로 분모가 0인 클래스를 안전하게 0점 처리한다.
    # 이는 소수 클래스가 전혀 등장하지 않는 경우에도 NaN이 발생하지 않게 보장한다.
    P = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    R = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    F = np.divide(2 * P * R, P + R, out=np.zeros_like(tp), where=(P + R) > 0)

    # macro는 클래스마다 같은 비중, weighted는 support가 큰 클래스에 더 큰 비중을 준다.
    # 불균형 데이터에서는 weighted가 다수 클래스 중심으로 보이므로,
    # 소수 클래스의 성능이 하락하는 걸 macro가 더 명확히 보여준다.
    w = np.ones_like(tp) / len(tp) if average == 'macro' else sup / sup.sum()
    return (P * w).sum(), (R * w).sum(), (F * w).sum()


rng = np.random.RandomState(0)
# 0/1/2 클래스 비율이 0.6/0.3/0.1인 불균형 분류 상황을 만들고, 25%에는 오답을 섞는다.
y_true = rng.choice(3, 300, p=[0.6, 0.3, 0.1])
y_pred = np.where(rng.rand(300) < 0.75, y_true, rng.choice(3, 300))
cm = confusion(y_true, y_pred, 3)
print('혼동행렬(행=정답)\n', cm)
print('accuracy', (np.diag(cm).sum() / cm.sum()).round(4), '| sklearn', round(skm.accuracy_score(y_true, y_pred), 4))
for avg in ['macro', 'micro', 'weighted']:
    mine = np.round(prf_from_cm(cm, avg), 4)
    ref = np.round(skm.precision_recall_fscore_support(y_true, y_pred, average=avg)[:3], 4)
    print(f'{avg:9s} 직접 P,R,F1 = {mine} | sklearn = {ref} | 일치 {np.allclose(mine, ref, atol=1e-4)}')
# 실행 결과: 직접 구현한 P/R/F1이 sklearn과 모두 일치한다. 불균형 데이터에서는 macro F1이 낮은 클래스 성능도 반영한다.
"""
혼동행렬(행=정답)
[[156  15   7]
 [  4  75  11]
 [  2   4  26]]
accuracy 0.8567 | sklearn 0.8567
macro     직접 P,R,F1 = [0.7839 0.8407 0.8057] | sklearn = [0.7839 0.8407 0.8057] | 일치 True
micro     직접 P,R,F1 = [0.8567 0.8567 0.8567] | sklearn = [0.8567 0.8567 0.8567] | 일치 True
weighted  직접 P,R,F1 = [0.8738 0.8567 0.862 ] | sklearn = [0.8738 0.8567 0.862 ] | 일치 True
"""

# ── 이진 분류: ROC-AUC ──
# ROC-AUC는 '양성 샘플의 점수가 음성 샘플의 점수보다 높은 확률'을 측정한다.
# 즉, 모델이 실제로 긍정인 샘플을 더 높은 점수로 매기느냐를 살피는 지표다.
# 이 지표는 클래스 불균형이 심해도 threshold를 바꿔가며 전체적인 분리력을 보기에 좋다.
# AUC = 1.0은 완벽, 0.5는 무작위 수준, 0.0은 반대로 예측하는 수준을 의미한다.


def roc_auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    """양성 점수가 음성 점수보다 높을 확률(Mann-Whitney U)로 AUC를 계산한다.

    모든 양성-음성 점수 쌍을 비교한다.
    - pos[i] > neg[j]이면 양성 점수가 음성 점수보다 높은 경우이므로 1점
    - pos[i] == neg[j]이면 동점이므로 0.5점
    을 부여하고, 전체 가능한 쌍의 수로 나누어 AUC를 계산한다.
    이 방식은 각 점수의 순위를 직접 비교하므로, threshold를 전혀 지정하지 않고도
    분류기의 전체 성능을 요약하는 데 적합하다.
    """
    y, score = np.asarray(y), np.asarray(score)
    pos, neg = score[y == 1], score[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum()   # 양성 점수가 더 큰 모든 쌍
    eq = (pos[:, None] == neg[None, :]).sum()  # 동점은 0.5점
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def roc_curve_points(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """점수 임계값을 높은 순서로 이동하며 (FPR, TPR) 좌표를 만든다.

    score를 내림차순으로 정렬하고, 각 threshold를 넘는 샘플들의 집합을 기준으로
    TPR = TP / P, FPR = FP / N 을 계산한다.
    이렇게 계산된 곡선은 모델이 얼마나 잘 양성을 구분하는지를 시각적으로 보여 준다.
    """
    order = np.argsort(-score)
    y = np.asarray(y)[order]
    tpr = np.cumsum(y) / y.sum()
    fpr = np.cumsum(1 - y) / (1 - y).sum()
    return np.r_[0, fpr], np.r_[0, tpr]


yb = rng.randint(0, 2, 400)
sc = yb * 0.8 + rng.randn(400) * 0.9  # 정답과 약하게 관계 있는 확률 점수
print('AUC 직접', round(roc_auc_rank(yb, sc), 5), '| sklearn', round(skm.roc_auc_score(yb, sc), 5))
f, t = roc_curve_points(yb, sc)
area = np.trapz(t, f) if hasattr(np, 'trapz') else np.trapezoid(t, f)
print('곡선 아래 면적(사다리꼴)', round(area, 5))
print('점수를 완전 무작위로 -> AUC ~', round(roc_auc_rank(yb, rng.rand(400)), 3), '(0.5 = 무능)')
# 실행 결과: 랭크 방식 AUC와 ROC 곡선 적분 AUC가 sklearn과 일치한다.
"""
AUC 직접 0.72154 | sklearn 0.72154
곡선 아래 면적(사다리꼴) 0.72154
점수를 완전 무작위로 -> AUC ~ 0.525 (0.5 = 무능)
"""

# ── 언어 모델 지표: Cross Entropy, Perplexity, Top-k Accuracy ──
# 언어 모델은 다중 클래스 분류 문제의 한 형태로 볼 수 있다.
# 각 위치마다 다음 토큰이 될 수 있는 후보 단어 집합이 있고, 모델은 각 후보마다 점수(logit)를 뽑아낸다.
# 이 때 정답 토큰의 확률을 평가하는 것이 CE(Cross Entropy), 이를 지수화한 것이 PPL(Perplexity)이다.
# Top-k Accuracy는 정답이 상위 k개 후보에 포함되었는지의 비율로, 단순히 가장 큰 1개만 맞춘 것보다
# 모델이 여러 후보를 갖고있을 때의 유연성을 더 반영한다.


def perplexity_from_logits(logits: torch.Tensor, targets: torch.Tensor, ignore: int = PAD) -> tuple[float, float]:
    """정답 토큰의 음의 로그 확률 평균(CE)과 exp(CE)인 PPL을 반환한다.

    log_softmax(logits, -1)를 통해 각 토큰이 정답일 확률의 로그값을 구하고,
    실제 정답 위치를 gather로 뽑아 그 값의 음수로 NLL(Negative Log Likelihood)을 계산한다.
    이후 PAD 위치는 무시하고 나머지 위치만 평균하여 CE를 구한다.
    PPL = e^(CE)이며, 값이 작을수록 모델이 정답을 잘 예측한다.
    """
    lp = torch.log_softmax(logits, -1)
    nll = -lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # 정답 토큰 위치의 -log p
    m = targets != ignore                                    # PAD는 평가에서 제외
    ce = nll[m].mean()
    return ce.item(), math.exp(ce.item())


def topk_acc(logits: torch.Tensor, targets: torch.Tensor, k: int = 5, ignore: int = PAD) -> float:
    """정답이 예측 상위 k개 후보 안에 들어간 비율을 계산한다.

    logits.topk(k, -1).indices는 각 위치에서 점수가 높은 상위 k개 인덱스를 의미한다.
    정답 토큰이 그 안에 포함되면 정답이라고 간주하고, PAD 위치는 제외한다.
    top1은 보통 정확도와 연결되고, top5는 큰 후보 집합에서 모델이 정답을 '근처'에 놓쳤는지 판단한다.
    """
    m = targets != ignore
    top = logits.topk(k, -1).indices
    return (top == targets.unsqueeze(-1)).any(-1)[m].float().mean().item()


Vv = 71
lg = torch.zeros(4, 10, Vv)
tg = torch.randint(1, Vv, (4, 10))
ce, ppl = perplexity_from_logits(lg, tg)
print(f'균등 예측: CE={ce:.4f} (ln V={math.log(Vv):.4f}), PPL={ppl:.2f} (V={Vv})')
good = F.one_hot(tg, Vv).float() * 8  # 정답 위치에만 큰 logit을 준 이상적인 예측
ce, ppl = perplexity_from_logits(good, tg)
print(f'정답에 큰 로짓: CE={ce:.4f}, PPL={ppl:.3f}, top1/top5 = {topk_acc(good, tg, 1):.2f}/{topk_acc(good, tg, 5):.2f}')
print('F.cross_entropy 와 일치:', abs(F.cross_entropy(good.reshape(-1, Vv), tg.reshape(-1)).item() - ce) < 1e-6)
# 실행 결과: 균등 예측은 CE=ln(V), PPL=V이고 정답 logit을 높이면 PPL이 1에 가까워진다.
"""
균등 예측: CE=4.2627 (ln V=4.2627), PPL=71.00 (V=71)
정답에 큰 로짓: CE=0.0232, PPL=1.023, top1/top5 = 1.00/1.00
F.cross_entropy 와 일치: True
"""

# ── 생성 문장 지표: BLEU (정밀도 중심) ──
# BLEU는 기계 번역과 생성형 모델에서 가장 널리 쓰이는 평가 지표다.
# 핵심은 '정답 문장과 얼마나 n-gram이 겹치냐'를 보되, 후보가 너무 짧거나 같은 단어만 반복하면 점수를 깎는 것이다.
# 이것이 BLEU가 단순 정밀도보다 더 현실적인 평가를 가능하게 하는 이유다.


def ngrams(toks: list[str], n: int) -> dict[tuple[str, ...], int]:
    # 특정 길이 n의 연속 단어 묶음(ngram) 빈도를 센다.
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu(cand: list[str], refs: list[list[str]], max_n: int = 4) -> float:
    """클리핑 n-gram 정밀도, 길이 벌점(BP), 기하평균을 쓰는 문장 단위 BLEU다.

    1) n-gram 정밀도: 후보 문장에서 나온 n-gram 중 정답 문장에 등장한 것만 인정한다.
       여기서 clipping을 적용하므로, 후보가 같은 단어를 반복해도 정답 빈도만큼만 점수를 얻는다.
    2) Brevity Penalty(BP): 후보가 너무 짧으면 벌점을 준다.
       구문을 너무 짧게 생성한 모델은 정밀도가 높아도 실제로는 낮게 평가된다.
    3) 기하평균: 1~4그램 정밀도를 모두 곱하는 대신 로그합의 평균을 취해, 작은 n-gram 점수도 반영된다.
    """
    logs = []
    for n in range(1, max_n + 1):
        c, mx = ngrams(cand, n), Counter()
        for r in refs:
            for g, k in ngrams(r, n).items():
                mx[g] = max(mx[g], k)  # 여러 정답 중 해당 n-gram의 최대 허용 횟수
        clipped = sum(min(k, mx[g]) for g, k in c.items())  # 반복 꼼수는 정답 횟수까지만 인정
        total = max(sum(c.values()), 1)
        logs.append(math.log(clipped / total) if clipped > 0 else -float('inf'))
    ref_len = min((abs(len(r) - len(cand)), len(r)) for r in refs)[1]
    bp = 1.0 if len(cand) > ref_len else math.exp(1 - ref_len / max(len(cand), 1))
    return bp * math.exp(sum(logs) / max_n)


from nltk.translate.bleu_score import sentence_bleu

ref = '그 는 아침 에 학교 에 가서 영어 를 공부 했다'.split()
cand = '그 는 아침 에 학교 에 가서 수학 을 공부 했다'.split()
print('직접 BLEU-4', round(bleu(cand, [ref]), 5), '| nltk', round(sentence_bleu([ref], cand), 5))
print('반복 꼼수 후보 "그 그 그 그 그 그" unigram 정밀도 =', 1 / 6, '(clipping 덕분에 1.0 이 아님)')
print('짧은 후보 BP 효과:', round(bleu('그 는 아침 에'.split(), [ref], 1), 4), '(unigram 정밀도 1.0 이어도 길이 벌점)')
# 실행 결과: BLEU는 N-gram 불일치와 너무 짧은 후보에 모두 감점한다.
"""
직접 BLEU-4 0.63156 | nltk 0.63156
반복 꼼수 후보 "그 그 그 그 그 그" unigram 정밀도 = 0.16666666666666666
짧은 후보 BP 효과: 0.1738
"""

# ── 생성 문장 지표: ROUGE (재현율 중심)와 Distinct-N (다양성) ──
# ROUGE는 생성형 모델의 답변이 정답 문장을 얼마나 덮는지(Recall)를 보려고 만든 지표다.
# 매번 같은 단어만 반복문으로 내뱉는 모델은 정밀도는 높아 보여도 Recall은 낮아지고,
# 반대로 너무 많이 엉뚱한 단어를 추가해도 Precision이 낮아져 F1이 낮게 나온다.
# Distinct-N은 문장 안에서 n-gram이 얼마나 다양하게 나타나는지를 본다.
# 값이 낮으면 반복 생성이 심하다는 신호이며, 생성형 모델의 '창의성/충실도'를 간접적으로 판단하는 데 쓰인다.


def rouge_n(cand: list[str], ref: list[str], n: int = 1) -> tuple[float, float, float]:
    """클리핑 n-gram 겹침으로 ROUGE-N의 Precision, Recall, F1을 계산한다.

    overlap = 후보와 정답이 겹치는 n-gram 수
    Precision = overlap / 후보의 n-gram 수
    Recall = overlap / 정답의 n-gram 수
    F1 = (2PR)/(P+R)
    이 구조는 BLEU와 대칭적이지만, ROUGE는 Recall을 더 중시하는 경향이 있다.
    """
    c, r = ngrams(cand, n), ngrams(ref, n)
    overlap = sum(min(k, r[g]) for g, k in c.items())
    P = overlap / max(sum(c.values()), 1)
    R = overlap / max(sum(r.values()), 1)
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def lcs_len(a: list[str], b: list[str]) -> int:
    """순서는 유지하되 중간 단어는 건너뛸 수 있는 최장 공통 부분 수열(LCS) 길이.

    ROUGE-L은 단어들의 순서를 유지한 채 얼마나 길게 겹치는지를 보는데,
    이는 단순 n-gram보다 의미적으로 더 유연하다. 예를 들어 '영어를 공부했다'와 '공부를 했다'처럼
    순서가 비슷하지만 일부 단어가 빠져 있어도 LCS 길이를 통해 얼마나 덮이는지 확인할 수 있다.
    """
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(cand: list[str], ref: list[str]) -> tuple[float, float, float]:
    """ROUGE-L: LCS 길이를 바탕으로 Precision, Recall, F1을 계산한다."""
    l = lcs_len(cand, ref)
    P, R = l / max(len(cand), 1), l / max(len(ref), 1)
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def distinct_n(toks: list[str], n: int) -> float:
    """생성한 n-gram 중 중복되지 않은 n-gram의 비율. 높을수록 반복이 적다.

    불필요하게 같은 단어가 반복되는 생성 모델은 distinct_n 값이 낮아진다.
    예를 들어 '서아는 서아는 ...'처럼 반복되면 n-gram 집합의 유니크 비율이 낮아지며,
    반대로 정상 문장은 다양한 조합을 가지므로 distinct_n이 높게 나온다.
    """
    g = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(g)) / max(len(g), 1)


ref = '서아는 학교에서 영어를 공부하고 집에 돌아가 잠들었다'.split()
cand = '서아는 학교에서 수학을 공부하고 일찍 잠들었다'.split()
print('ROUGE-1 (P,R,F)', np.round(rouge_n(cand, ref, 1), 3))
print('ROUGE-2 (P,R,F)', np.round(rouge_n(cand, ref, 2), 3))
print('ROUGE-L (P,R,F)', np.round(rouge_l(cand, ref), 3))
try:
    from rouge_score import rouge_scorer
    score = rouge_scorer.RougeScorer(['rouge1']).score(' '.join(ref), ' '.join(cand))['rouge1'].fmeasure
    print('rouge_score 기본 토크나이저(한글 입력):', score, '← 0 이면 한글이 지워진 것')
except ImportError:
    print('(rouge_score 미설치 - Colab에서 !pip install rouge-score 후 확인)')
rep = '서아는 서아는 서아는 서아는 서아는 서아는'.split()
div = '서아는 학교에서 영어를 공부하고 집에 갔다'.split()
print('Distinct-1/2 반복문:', distinct_n(rep, 1), distinct_n(rep, 2), '| 정상문:', distinct_n(div, 1), distinct_n(div, 2))
# 실행 결과: ROUGE는 n-gram/LCS 겹침을, Distinct-N은 반복 여부를 보여 준다.
"""
ROUGE-1 (P,R,F) [0.667 0.571 0.615]
ROUGE-2 (P,R,F) [0.2   0.167 0.182]
ROUGE-L (P,R,F) [0.667 0.571 0.615]
(rouge_score 미설치 - Colab에서 !pip install rouge-score 후 확인)
Distinct-1/2 반복문: 0.16666666666666666 0.2 | 정상문: 1.0 1.0
"""


# 표면 일치 지표가 반복·동의어·완전 오답에 어떻게 반응하는지 한 표로 비교한다.
ref = '서아는 학교에서 영어를 공부했다'.split()
cands = {
    '정확히 같음': '서아는 학교에서 영어를 공부했다',
    '반복 꼼수': '서아는 서아는 학교에서 학교에서 영어를 영어를 공부했다',
    '동의어 치환': '서아는 학원에서 영어를 학습했다',
    '엉뚱한 문장': '민수는 카페에서 커피를 마셨다',
}
rows = []
for name, c in cands.items():
    c = c.split()
    rows.append({
        '후보': name,
        'R1-F': round(rouge_n(c, ref, 1)[2], 3),
        'R2-F': round(rouge_n(c, ref, 2)[2], 3),
        'RL-F': round(rouge_l(c, ref)[2], 3),
        'BLEU-2': round(bleu(c, [ref], 2), 3),
        'Distinct-1': round(distinct_n(c, 1), 3),
    })
print(pd.DataFrame(rows).to_string(index=False))
# 실행 결과: 표면 단어가 달라진 동의어 문장은 의미가 비슷해도 BLEU/ROUGE가 크게 떨어진다.
"""
    후보  R1-F  R2-F  RL-F  BLEU-2  Distinct-1
정확히 같음 1.000 1.000 1.000   1.000       1.000
 반복 꼼수 0.727 0.667 0.727   0.535       0.571
동의어 치환 0.500 0.000 0.500   0.000       1.000
엉뚱한 문장 0.000 0.000 0.000   0.000       1.000
"""

# ── 의미 유사도: 문장 동시출현 → PPMI → SVD 토이 임베딩 ──
# 표면적 단어 일치만 본 ROUGE/BLEU는 동의어, 같은 맥락의 다른 표현, 의미적 유사도 문제를 잘 잡지 못한다.
# 그래서 단어 간 의미 공간을 만들어 문장의 의미적 유사도를 측정하는 접근이 필요하다.
# 여기서는 아주 간단한 토이 버전으로, 문장 안의 동시출현 정보(PPMI)를 이용해 단어 임베딩을 만들고,
# 각 문장 단어 벡터가 얼마나 비슷한 위치에 있는지 코사인 유사도로 판단한다.
# 이 방식은 BERTScore의 핵심 아이디어를 아주 축소해서 연습하는 형태다.

sents = [s.split() for st, _ in TRAIN_STORIES for s in st]
vocab = sorted({w for s in sents for w in s})
idx = {w: i for i, w in enumerate(vocab)}
co = np.zeros((len(vocab), len(vocab)))
for s in sents:
    for a in s:
        for b in s:
            if a != b:
                co[idx[a], idx[b]] += 1
tot = co.sum()
pw, pc = co.sum(1, keepdims=True) / tot, co.sum(0, keepdims=True) / tot
ppmi = np.maximum(np.log((co / tot + 1e-12) / (pw * pc + 1e-12)), 0)
U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
emb = U[:, :16] * S[:16] ** 0.5
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

# PPMI 기반 단어 임베딩을 만든 뒤, 각 문장을 구성하는 단어 벡터들이 얼마나 비슷한 위치에 있는지를 보자.
# 여기서는 정답 문장과 후보 문장의 단어별 코사인 유사도를 모두 계산하며,
# 가장 가까운 정답 단어로 후보 단어를 매칭해 Precision / Recall / F1을 만든다.
# 표면적 단어 일치만 보는 ROUGE와 달리, 의미적으로 비슷한 문장도 상대적으로 높은 점수를 받을 수 있다.
# 이 방식은 BERTScore의 핵심 아이디어를 아주 단순화한 구현이다.

def bertscore_like(cand: list[str], ref: list[str]) -> tuple[float, float, float]:
    """실제 BERTScore의 토이판: 각 단어를 가장 유사한 상대 문장 단어와 매칭한다."""
    c = np.stack([emb[idx[w]] if w in idx else np.zeros(emb.shape[1]) for w in cand])
    r = np.stack([emb[idx[w]] if w in idx else np.zeros(emb.shape[1]) for w in ref])
    sim = c @ r.T  # 정규화된 벡터의 모든 쌍 코사인 유사도, (|cand|, |ref|)
    P, R = sim.max(1).mean(), sim.max(0).mean()
    return P, R, 2 * P * R / (P + R + 1e-12)


ref = '서아는 학교에서 영어를 공부했다'.split()
for name, c in {
    '정확히 같음': '서아는 학교에서 영어를 공부했다',
    '다른 주제': '서아는 카페에서 커피를 마셨다',
    '같은 주제 다른 문장': '서아는 학교에서 수학을 공부했다',
}.items():
    c = c.split()
    print(f'{name:12s} BERTScore-like F1 = {bertscore_like(c, ref)[2]:.3f} | ROUGE-1 F = {rouge_n(c, ref, 1)[2]:.3f}')
# 실행 결과: 토이 임베딩 기반 점수는 같은 주제의 다른 문장을 ROUGE보다 조금 더 높게 본다.
"""
정확히 같음       BERTScore-like F1 = 1.000 | ROUGE-1 F = 1.000
다른 주제        BERTScore-like F1 = 0.689 | ROUGE-1 F = 0.250
같은 주제 다른 문장  BERTScore-like F1 = 0.857 | ROUGE-1 F = 0.750
"""

# ── 불확실성: bootstrap 95% 신뢰구간 ──
# 단일 정확도 숫자는 표본 크기와 분산을 고려하지 않는다.
# 붓스트랩은 주어진 정오답 결과를 기반으로 같은 크기의 새로운 표본을 여러 번 재추출하여
# 모델 성능의 분포를 근사한다. 이렇게 얻은 하한/상한은 '정확도가 이 범위 안에 있다'는 통계적 불확실성을 보여준다.
# 특히 두 모델의 차이를 비교할 때, 95% 신뢰구간이 0을 포함하면 차이가 우연일 가능성이 높고,
# 포함하지 않으면 그 차이가 통계적으로 유의미하다고 볼 수 있다.


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, np.ndarray]:
    """정오답 배열을 복원추출해 정확도의 95% bootstrap 신뢰구간을 구한다.

    데이터셋이 한 번만 주어졌더라도, 복원추출을 여러 번 반복하면 실제 성능의 분포를 모사할 수 있다.
    여기서 2.5%, 97.5% 백분위수는 95% 신뢰구간을 의미하므로,
    '정확도가 0.91 ± 0.04' 같은 형태로 불확실성을 함께 기술할 수 있다.
    """
    rs = np.random.RandomState(seed)
    correct = np.asarray(correct, dtype=float)
    n = len(correct)
    stats = np.array([correct[rs.randint(0, n, n)].mean() for _ in range(n_boot)])
    return correct.mean(), np.percentile(stats, [2.5, 97.5])


rs = np.random.RandomState(1)
a = (rs.rand(120) < 0.91).astype(int)
b = (rs.rand(120) < 0.93).astype(int)  # 같은 120개 테스트에 대한 두 모델의 정오답
for name, c in [('모델 A', a), ('모델 B', b)]:
    m, (lo, hi) = bootstrap_ci(c)
    print(f'{name}: acc {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]')
diffs = []
for _ in range(2000):
    ix = rs.randint(0, 120, 120)  # 같은 인덱스를 A/B에 동시에 적용하는 paired bootstrap
    diffs.append(b[ix].mean() - a[ix].mean())
print('B-A 차이 95% CI', np.round(np.percentile(diffs, [2.5, 97.5]), 3), '-> 0 포함이면 차이가 있다고 단정 불가')
# 실행 결과: B의 점추정 정확도는 높지만, 차이 신뢰구간에 0이 있어 우월하다고 단정할 수 없다.
"""
모델 A: acc 0.908  95% CI [0.850, 0.958]
모델 B: acc 0.917  95% CI [0.867, 0.958]
B-A 차이 95% CI [-0.058  0.075] -> 0 포함이면 차이가 있다고 단정 불가



