# AdaLoRA (Adaptive Low-Rank Adaptation)

경우에 따라 양자화 모델에 적용안될 가능성이 있으니 주의

AdaLoRA는 LoRA가 모든 레이어에 균일하게 저차원 어댑터(rank)를 할당하는 것과 달리, 중요한 가중치 행렬 계층에는 더 많은 매개변수가 할당되게, 덜 중요한 가중치 행렬 계층에는 더 적은 매개변수가 할당되게 어댑터의 rank를 동적으로 재조정하여 효율성을 높히는 기법입니다.



원리:

장점:

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import AdaLoraConfig, get_peft_model

# 베이스 모델 불러오기
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    return_dict=True,
    device_map="auto"
)

# AdaLoRA 설정
config = AdaLoraConfig(
    lora_alpha=32,               
    lora_dropout=0.1,             
    target_modules=["query", "value"]  
    task_type="SEQ_CLS",          
    init_r=16,                    # 초기 rank
    target_r=4,                   # 목표 평균 rank
    tinit=10,                     # 워밍업 스텝 수
    tfinal=20,                    # 최종 미세조정 스텝 수
    total_step=100,               # 전체 학습 스텝
    beta1=0.85,                   # EMA 감도 (작을수록 민감)
    beta2=0.85,                   # EMA 불확실도 (작을수록 민감)
    orth_reg_weight=0.1,          # 직교 규제 계수 (클수록 규제 강화)
)

# PEFTModel 생성
peft_model = get_peft_model(model, config)


X-LoRA (Mixture of LoRA Experts)

X-LoRA는 다양한 테스크에서 이미 학습된 LoRA 가중치 여러개를 토큰, 레이어, 시퀀스 단위로 동적으로 혼합하는 전문가 기법(Mixture of Experts, MoE)입니다. 
이는 입력 프롬프트의 컨텍스트(d-model)에 따라 어떤 테스크에 적절한 어텝터를  활성화할지 동적으로 결정함으로써, 단일 모델이 다양한 태스크와 도메인에 적응할 수 있다는 장점을 제공합니다.

허깅페이스

원리:

장점:

import torch
from transformers import AutoModelForCausalLM
from peft import XLoraConfig, get_peft_model

# 베이스 모델 설정
model_name = "base_model"
model = AutoModelForCausalLM.from_pretrained(model_name)

# d-model 크기를 가져오기 위해 AutoConfig 생성
model_cfg  = AutoConfig.from_pretrained(model_name)

# 저장된 LoRA 경로
lora1 = './saved_model1'
lora2 = './saved_model2'

# X-LoRA 설정
xlora_cfg = XLoraConfig(
    task_type="CAUSAL_LM",                # 언어모델링 태스크 지정
    hidden_size=model_cfg.hidden_size,    # 모델 히든 사이즈
    xlora_depth=2,                        # 게이팅 네트워크 깊이
    xlora_size=512,                       # 게이팅 히든 유닛 수
    xlora_dropout_p=0.1,                  # 드롭아웃 확률
    adapters={                            # 사용할 LoRA 어댑터 체크포인트들
        "adapter_a": lora1 ,
        "adapter_b": lora2 ,
    },
    enable_softmax=True,                  # 소프트맥스 게이팅 사용
    softmax_temperature=0.8,              # 소프트맥스 온도 조절
    layerwise_scalings=True,              # 레이어별 스케일링 활성화
)

# PEFT 모델 생성
xlora_model = get_peft_model(model, xlora_cfg)





IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

IA3는 Transformer의 어텐션 블록과 피드포워드 네트워크 내부 활성화(activation) 함수에 학습 가능한 스케일링 벡터를 삽입하여, 전체 모델 가중치는 동결한 채로 이 벡터들만 학습하는 PEFT 기법입니다. 이를 통해 모델은 특성 벡터에 GELU, RELU등을 얼마나 활성화 시킬지 만 학습하여 매우 적은 파라미터 학습으로 미세조정이 가능합니다.

원리:

피드포워드 중간층에 있는 활성화 함수에 곱해지는 스케일링 게이트(벡터)를 생성

어텐션 모듈의 Q, K, V 프로젝션 출력에 각각 곱해지는 스케일링 게이트(벡터)를 생성

이 세 종류(IA³)의 스케일링 벡터만 학습하며, 나머지 모든 가중치는 동결

학습 후에는 스케일링 벡터를 기본 가중치에 곱하여 병합(merge)

장점:

파라미터 효율성 - 전체 모델의 약 0.01%라는 매우 적은 파라미터만 학습하므로 메모리와 연산 효율성이 매우 높음

유연한 어댑터 관리 - 원본 가중치를 동결시키기 때문에 여러 IA3 어댑터를 경량 체크포인트 형태로 병렬로 보유 가능

높은 성능 - 전체 파라미터를 미세조정하는 것과 비슷한 수준의 downstream 성능을 제공



import torch
from transformers import  AutoModelForCausalLM
from peft import IA3Config, get_peft_model

# 베이스 모델 준비
model_name = "gemma3"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",)

# IA3Config 정의
ia3_config = IA3Config(
    peft_type="IA3",                   
    task_type="CAUSAL_LM",                            
    target_modules=["q_proj", "v_proj", "k_proj", "down_proj"], # 스케일링 적용될 계층 (출력에 처리)
    feedforward_modules=["down_proj"],                          # 입력에 스케일링 처리시(활성화 이후)
    modules_to_save=["lm_head"],                                # IA3 레이어 외 같이 저장할 레이어
)

# PEFT 모델 생성
ia3_model = get_peft_model(model, ia3_config)






Prompt Tuning

Prompt Tuning은 사전 학습된 언어 모델의 가중치를 전부 동결한 채, 입력 시퀀스 앞에 소프트 프롬프트(가상 토큰)만을 추가해서 파인튜닝하는 방법입니다. 모델은 가상 토큰에 대한 임베딩 행렬만 학습하며 해당 임베딩 행렬은 추론시 테스크에 맞는 지침서 역할을 하게 됩니다.

원리

모델 입력 시퀀스 앞에 정해진 길이 L 만큼의 가상 토큰(virtual tokens) 시퀀스를 붙임

가상 토큰이 가지는 임베딩 행렬 P를 만들고 P값을 랜덤 또는 기존 토큰중  L개의 임베딩 값으로 설정

모델의 기존 파라미터는 동결하고, 오직 추가된 가상 토큰의 임베딩 행렬 P만 학습

추론 시 학습된 임베딩 행렬 P를 기존 모델에 입력된 임베딩 시퀀스 앞에 붙여 추론 진행

장점

매우 적은 파라미터만 업데이트 → 메모리·연산량 절약

원본 모델 가중치를 건드리지 않으므로 여러 태스크에 동일 모델 재사용 가능

다운스트림 태스크별로 프롬프트만 바꿔치기해 빠른 전환(switching)

기억 상실(catastrophic forgetting) 없이, 모델 본연의 지식 유지

import torch
from transformers import AutoModelForCausalLM
from peft import PromptTuningConfig, get_peft_model

# 베이스 모델 로드
model_name = "base_model"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Prompt Tuning 설정
pt_config = PromptTuningConfig(
    peft_type="PROMPT_TUNING",          # 고정
    task_type="CAUSAL_LM",              # 시퀀스-투-시퀀스 언어모델링
    num_virtual_tokens=10,              # 학습할 가상 토큰 개수
    prompt_tuning_init="TEXT",          # 초기화 방식: TEXT or RANDOM
    prompt_tuning_init_text="입력된 텍스트를 간략하게 요약", # TEXT 초기화 시 사용할 문자열
    tokenizer_name_or_path=model_name   # 토크나이저 경로(이름)
)

# PEFT 모델 래핑
pt_model = get_peft_model(model, pt_config)





Prefix Tuning

사전 학습된 언어 모델의 모든 파라미터를 동결한 채, 각 블록의 어텐션 계층에 가상 토큰(prefix)에 해당하는 임베딩을 삽입해 학습하는 기법입니다. Prompt Tuning과 비슷하지만 입력 계층에서 가상 임베딩을 적용하는게 아닌 모든 어텐션 계층에서 가상 임베딩을 적용한다는 차이점이 있습니다.

Prompt Tuning과 Prefix Tuning의 차이

import torch
from transformers import AutoModelForCausalLM
from peft import PrefixTuningConfig, get_peft_model

# 베이스 모델  로드
model_name = "base_mode" 
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",)

# PrefixTuningConfig 정의
config = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",                # 기법 지정
    task_type="CAUSAL_LM",                    # 인과 언어 모델링 과제
    num_virtual_tokens=20,                    # 학습할 가상 토큰 수
    token_dim=model.config.hidden_size,       # 모델 히든 차원 (d_model)
    num_transformer_submodules=1,             # GPT는 decoder submodule만 1개
    num_attention_heads=model.config.n_head,  # 모델 헤드 개수
    num_layers=model.config.n_layer,          # 모델 레이어 개수
)

# PEFT 모델 생성
prefix_model = get_peft_model(model, config)



P-Tuning

사전 학습된 언어 모델의 입력 시퀀스에 소프트 프롬프트(가상 토큰)을 추가해서 간단한 MLP 또는 RNN 구조인 프롬프트 인코더 계층을 추가하여 파인튜닝하는 방법입니다. Prompt Tuning과는 다르게 가상 토큰을 기존 임베딩 행렬에 직접 추가하고, 임베딩 행렬은 대신 추가된 프롬프트 인코더가 학습된다는 것 입니다.

Prompt Tuning과 P-Tuning 의 차이

import torch
from transformers import AutoModelForCausalLM
from peft import PromptEncoderConfig, get_peft_model

# 베이스 모델  로드
model_name = "base_mode" 
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",)

# PrefixTuningConfig 정의
config = PromptEncoderConfig(
    task_type="CAUSAL_LM",                    
    num_virtual_tokens=20,                    # 가상 토큰 수
    encoder_reparameterization_type="MLP",    # 프롬프트 인코더 레이어타입
		encoder_hidden_size=128,                  # 프롬프트 인코더 히든 사이즈
		num_transformer_submodules=1,             # GPT는 decoder submodule만 1개
)

# PEFT 모델 생성
prefix_model = get_peft_model(model, config)

