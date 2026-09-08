# Generative Model

개요

생성 모델(Generative Model)의 기초 개념부터 인코더-디코더 구조, 그리고 VAE, GAN, DCGAN, cGAN, Pix2Pix에 이르기까지 핵심 머신러닝 학습 자료의 내용을 과도한 생략이나 누락 없이 학습합니다.

01. 생성 모델링이란?

생성 모델링(Generative Modeling): 주어진 데이터셋과 유사한 데이터를 생성하도록 모델을 훈련하는 머신러닝의 한 분야입니다.

핵심 목적: 샘플 x를 관측할 확률을 모델링하는 것이 목표이며 P(x)를 추정합니다. 원본 데이터와 동일한 규칙으로 생성된 것처럼 보이는 새로운 특성을 만듭니다.

생성 모델 vs 판별 모델:

02. 표현 학습 (Representation Learning)

정의: 원본 데이터(raw data)에서 중요한 특징(feature)을 자동으로 학습하여 모델이 더 효과적으로 학습할 수 있도록 하는 기법입니다. (수동 특징 엔지니어링과 대비됨)

주요 기법: 신경망의 은닉층(CNN 초반 층은 가장자리, 후반 층은 객체 형태 학습), 오토인코더(입력 압축을 통한 의미 있는 잠재 표현 학습)

인코더-디코더 기법:

03. 주요 생성 모델 아키텍처

① AutoEncoder (오토인코더)

개요 및 구조: 입력 데이터를 저차원 표현(latent space)으로 압축하고, 다시 원본과 유사한 데이터를 복원하는 비지도 학습 신경망 모델입니다.

구조: Encoder (압축, Bottleneck z) \rightarrow Decoder (복원, x \approx x')

손실 함수: 입력과 출력의 차이를 최소화 (L = \Vert{}x - x'\Vert{}^2)

한계: 결정론적 특성으로 인해 새로운 데이터를 임의로 샘플링하여 생성하는 데 한계가 있습니다.

② VAE (Variational AutoEncoder)

개요: 입력 데이터를 저차원 잠재 공간의 확률 분포(정규분포)로 인코딩한 후 샘플링하여 데이터를 재구성하는 모델입니다. (2013)

핵심 기술:

손실 함수: 재구성 손실(Reconstruction Loss) + 잠재 공간 정규화 손실 (KL Divergence를 통해 사전 분포와 추정 분포 간의 차이 조절)

③ GAN (Generative Adversarial Network)

개요: 생성자(Generator)와 판별자(Discriminator) 두 신경망이 서로 경쟁(Minimax Game)하며 학습하는 적대적 모델입니다. (2014)

주요 이슈:

④ DCGAN (Deep Convolutional GAN)

개요: GAN의 학습 불안정 문제를 개선하기 위해 CNN 구조와 아키텍처 제약을 도입한 모델입니다.

주요 제약:

04. 조건부 생성 모델 (cGAN & Pix2Pix)

① cGAN (Conditional GAN)

개요: 생성자와 판별자 모두에 추가 조건 정보(클래스 레이블, 텍스트 등 y)를 함께 입력하여 원하는 속성이나 조건에 맞는 데이터를 제어하며 생성할 수 있는 모델입니다.

장점: 조건 기반 제어를 통해 결과물의 다양성과 제어력이 높습니다.

② Pix2Pix (Image-to-Image Translation)

개요: cGAN을 기반으로 한 이미지-투-이미지 번역 모델입니다. (예: 스케치 \rightarrow 사진, 지도 \rightarrow 항공사진 변환)

핵심 구조:

장점: 하나의 통합된 모델로 다양한 이미지 변환 문제에 적용 가능하며, 수작업 설계 없이 고해상도의 세밀한 텍스처를 생성합니다.

Python 코드로 보는 VAE 재파라미터화 트릭 가상 구현 예시

import torch

# 가상의 평균(mu)과 로그 분산(log_var) 출력값
mu = torch.tensor([0.0, 1.0])
log_var = torch.tensor([-0.5, 0.2])

# Log variance를 실제 분산의 표준편차(sigma)로 변환 (수치적 안정성 확보)
sigma = torch.exp(0.5 * log_var)

# 재파라미터화 트릭 (Reparameterization Trick) 구현
epsilon = torch.randn_like(sigma)  # 표준 정규분포 N(0, I)에서 노이즈 샘플링
z = mu + sigma * epsilon          # 미분이 가능한 형태로 잠재 변수 z 생성

print(f"변환된 잠재 변수 z: {z}")

공식 문서 및 참고 링크

Generative Deep Learning 2nd Edition (David Foster) - 공식 GitHub 저장소

Auto-Encoding Variational Bayes (arXiv:1312.6114)

Generative Adversarial Networks (arXiv:1406.2661)

Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (arXiv:1511.06434)

Conditional Generative Adversarial Nets (arXiv:1411.1784)

Image-to-Image Translation with Conditional Adversarial Networks (arXiv:1611.07004)

스탠포드 cs231n : http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf

서울대 시각적 이해를 위한 머신러닝 : YouTube 재생 목록 링크

