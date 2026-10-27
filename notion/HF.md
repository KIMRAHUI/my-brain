# HF

전체 데이터 이동 경로 (Data Flow Map)

허깅페이스 코드는 복잡해 보이지만, 
결국 아래 5단계 컨베이어 벨트를 따라 데이터가 이동하는 과정입니다.

[1. 원문 텍스트] "이 영화 정말 최고예요!" (문자열)
       │
       ▼  AutoTokenizer (쪼개기 + 번호 붙이기 + 포장)
[2. 토크나이저 출력] input_ids, attention_mask  
                   ──▶ Shape: (Batch, Seq_Len)
       │
       ▼  AutoModel (임베딩 변환 + 셀프 어텐션 문맥 파악)
[3. 은닉 상태(Hidden State)] 문맥이 담긴 숫자 덩어리 
                            ──▶ Shape: (Batch, Seq_Len, 768)
       │
       ▼  Classification Head ([CLS] 토큰 추출 + 분류기 통과)
[4. 로짓(Logits)] 날것의 점수 (예: [-1.8, +2.4])  
                 ──▶ Shape: (Batch, Num_Labels)
       │
       ▼  Softmax (추론 시) / CrossEntropy (학습 시)
[5. 최종 결과] "긍정 98.5%" 또는 Loss 계산 후 가중치 업데이트(역전파)

단계별 코드 & 데이터 이동 시뮬레이터

가상의 입력 문장 2개(Batch = 2)가 코드 한 줄을 지날 때마다 
어떻게 변하는지 시뮬레이션해 보겠습니다.

입력 문장 A: "영화 최고예요" (짧은 문장)

입력 문장 B: "스토리가 너무 지루해요" (긴 문장)

1.Tokenizer: 글자를 숫자 박스로 규격화하기:

String ➔ Tensor (Batch, Seq_Len).

모델은 글자를 읽지 못하고, 길이가 다른 두 문장을 한 번에 계산(행렬 곱)할 수 없습니다. 
따라서 토크나이저가 
1) 단어를 쪼개고, 2) 번호(ID)를 붙이고, 3) 짧은 문장에 빈칸([PAD])을 채워 길이를 맞춥니다.

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
inputs = tokenizer(
    ["영화 최고예요", "스토리가 너무 지루해요"],
    padding=True,       # 가장 긴 문장 길이에 맞춰 빈칸(0) 채우기
    truncation=True,    # 너무 길면 자르기
    return_tensors="pt" # PyTorch 텐서로 포장해서 반환
)

🔍 [시뮬레이터] inputs 변수 내부의 실제 데이터 이동 모습:

토큰 분해 및 특수 토큰 부착:

문장 A
: ['[CLS]', '영화', '최고', '##예요', '[SEP]', '[PAD]'] (길이 6으로 맞춤)

문장 B
: ['[CLS]', '스토리', '##가', '너무', '지루', '[SEP]'] (길이 6)

원리: 
문장 맨 앞의 [CLS](101번)는 나중에 문장 전체의 요약본이 담길 '대표 빈 바구니’

최종 생성되는 텐서 (inputs 딕셔너리):

inputs['input_ids'] (단어 번호표) - Shape: (2, 6)
┌                                      ┐
│ [ 101, 3776, 4120, 2180,  102,    0] │  ◀ 문장 A (끝에 0은 빈칸 [PAD])
│ [ 101, 5812, 2116, 3760, 6233,  102] │  ◀ 문장 B
└                                      ┘

inputs['attention_mask'] (진짜 단어 vs 가짜 빈칸 표시기) - Shape: (2, 6)
┌                    ┐
│ [ 1, 1, 1, 1, 1, 0] │  ◀ 마지막 0번 위치는 "빈칸이니 어텐션 계산할 때 무시해!"라고 알려줌
│ [ 1, 1, 1, 1, 1, 1] │  ◀ 모두 진짜 단어
└                    ┘

2.Base Model (BERT): 번호표를 '문맥 벡터'로 진화시키기:

Tensor (2, 6) ➔ Tensor (2, 6, 768).

이제 번호표(input_ids)가 모델 안으로 들어갑니다. 
여기서 가장 헷갈리기 쉬운 차원의 확장이 일어납니다. 
숫자 1개였던 토큰 번호가 768개의 실수로 이루어진 벡터로 변합니다.

from transformers import AutoModel

model = AutoModel.from_pretrained("klue/bert-base")
outputs = model(**inputs) 
# inputs 안의 input_ids, attention_mask를 풀어서 전달

🔍 [시뮬레이터] 모델 내부에서 일어나는 2단계 변화:

Embedding 층 통과 (단순 사전 찾기):

12개의 Transformer Encoder 층 통과 (Self-Attention):

최종 출력 (outputs.last_hidden_state):

Shape: (2, 6, 768)
문장 A의 출력:
  0번 위치 [CLS] : [ 0.85, -0.12, ..., 0.34 ] (768개) ◀ ★ 문장 A 전체의 요약본!
  1번 위치 '영화' : [ 0.11,  0.54, ..., -0.21 ] (768개)
  ...
  5번 위치 [PAD] : [ 0.01,  0.02, ...,  0.00 ] (768개)

3.Classification Head: 768개 숫자를 정답 점수(Logits)로 압축하기:

Tensor (2, 6, 768) ➔ Tensor (2, 2).

문장 분류(감성 분석 등)를 할 때는 AutoModel 대신 끝에 분류기(선형 레이어)가 달린 AutoModelForSequenceClassification을 사용합니다.

from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F

cls_model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)
cls_outputs = cls_model(**inputs)

logits = cls_outputs.logits
probs = F.softmax(logits, dim=-1)

🔍 [시뮬레이터] 768차원이 2차원(부정/긍정)으로 줄어드는 과정:

[CLS] 벡터만 뽑아내기 (Pooling):

Linear Layer 통과 (행렬 곱):

Logits ➔ Softmax 확률 변환:

1) cls_outputs.logits (날것의 점수, 범위 제한 없음) - Shape: (2, 2)
   문장 A ("영화 최고예요")      : [ -1.85 (부정 점수),  +2.42 (긍정 점수) ]
   문장 B ("스토리가 너무 지루해요"): [ +1.94 (부정 점수),  -1.12 (긍정 점수) ]

2) probs (Softmax 통과 후: 합이 1.0(100%)이 되도록 예쁘게 변환) - Shape: (2, 2)
   문장 A : [ 0.014 (1.4%),  0.986 (98.6%) ] ──▶ 1번(긍정)이 더 크므로 "긍정(1)" 판정!
   문장 B : [ 0.955 (95.5%), 0.045 (4.5%)  ] ──▶ 0번(부정)이 더 크므로 "부정(0)" 판정!

4.Dataset.map & Trainer: 대량 데이터를 자동으로 흘려보내기:

학습(Fine-Tuning) 루프의 원리.

허깅페이스 코드에서 가장 직관적으로 안 와닿는 부분이 dataset.map()과 Trainer입니다. 
이 둘은 앞서 본 1~3단계를 수만 개의 데이터에 대해 자동으로 반복해 주는 장치입니다.

# 1) 모든 데이터에 토크나이저 일괄 적용
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

# 2) Trainer로 학습 실행
trainer = Trainer(
    model=cls_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)
trainer.train()

🔍 [시뮬레이터] map과 Trainer 내부에서 벌어지는 일:

dataset.map()의 역할 (사전 준비):

trainer.train() 실행 시 1 스텝(Step)의 흐름:

한눈에 보는 핵심 요약표 (헷갈릴 때 보는 치트시트)

코드를 보다가 "지금 이 변수가 뭘 담고 있는 거지?" 싶을 때 아래 표에서 변수명을 찾아보세요.

💡 직접 눈으로 확인해보는 팁 (코랩 실습용)

코랩 노트북에서 흐름이 막히는 셀이 있다면, 아래 시뮬레이터 출력 코드 4줄을 새 코드 셀에 복사해서 실행해 보세요. 값들이 이동하는 모습이 바로 출력됩니다.

# 데이터의 모양과 실제 값을 눈으로 찍어보는 시뮬레이터 코드
sample_inputs = tokenizer(["허깅페이스 구조가 이제 눈에 보여요!"], return_tensors="pt")
print("1. 토크나이저 통과 후 ID:", sample_inputs["input_ids"], "| Shape:", sample_inputs["input_ids"].shape)
print("2. 분해된 실제 토큰 글자:", tokenizer.convert_ids_to_tokens(sample_inputs["input_ids"][0]))

with torch.no_grad():
    sample_out = cls_model(**sample_inputs)
print("3. 모델 통과 후 Logits 점수:", sample_out.logits, "| Shape:", sample_out.logits.shape)
print("4. Softmax 변환 후 최종 확률:", torch.softmax(sample_out.logits, dim=-1))





