---
notion-url: https://app.notion.com/p/Generative-Model-3d077fa64f7f803f87c1dd236453e81b
title: Generative Model
date: '2026-09-03 05:08:00.000'
from_notion: https://app.notion.com/p/Generative-Model-3d077fa64f7f803f87c1dd236453e81b
author: rahui kim
last_edited_time: '2026-09-03 05:13:00.000'
---
## 개요

## 01. 생성 모델링이란?

- **생성 모델링(Generative Modeling)**: 주어진 데이터셋과 유사한 데이터를 생성하도록 모델을 훈련하는 머신러닝의 한 분야입니다.

- **핵심 목적**: 샘플 $ x $를 관측할 확률을 모델링하는 것이 목표이며 $ P(x) $를 추정합니다. 원본 데이터와 동일한 규칙으로 생성된 것처럼 보이는 새로운 특성을 만듭니다.

- **생성 모델 vs 판별 모델**:

	- **판별 모델 (Discriminative Model)**: 샘플 $ x $가 주어졌을 때 레이블 $ y $의 확률을 모델링 ($ P(y\Vert{}x) $ 추정). 예: "이 사진 속 동물이 강아지인가요?" $ \rightarrow $ 강아지 가능성 `0.83`

	- **생성 모델 (Generative Model)**: 데이터셋과 유사하지만 기존에 존재하지 않는 새로운 이미지를 생성 ($P(x)$ 추정).

## 02. 표현 학습 (Representation Learning)

- **정의**: 원본 데이터(raw data)에서 중요한 특징(feature)을 자동으로 학습하여 모델이 더 효과적으로 학습할 수 있도록 하는 기법입니다. (수동 특징 엔지니어링과 대비됨)

- **주요 기법**: 신경망의 은닉층(CNN 초반 층은 가장자리, 후반 층은 객체 형태 학습), 오토인코더(입력 압축을 통한 의미 있는 잠재 표현 학습)

- **인코더-디코더 기법**:

	1. 훈련 데이터셋을 잠재공간으로 인코딩

	1. 잠재공간에서 샘플링

	1. 샘플링한 데이터를 디코딩하여 원래 도메인으로 되돌리기

## 03. 주요 생성 모델 아키텍처

### ① AutoEncoder (오토인코더)

- **개요 및 구조**: 입력 데이터를 저차원 표현(latent space)으로 압축하고, 다시 원본과 유사한 데이터를 복원하는 비지도 학습 신경망 모델입니다.

- **구조**: `Encoder` (압축, Bottleneck $ z $) $ \rightarrow $ `Decoder` (복원, $ x \approx x' $)

- **손실 함수**: 입력과 출력의 차이를 최소화 ($ L = \Vert{}x - x'\Vert{}^2 $)

- **한계**: 결정론적 특성으로 인해 새로운 데이터를 임의로 샘플링하여 생성하는 데 한계가 있습니다.

### ② VAE (Variational AutoEncoder)

- **개요**: 입력 데이터를 저차원 잠재 공간의 확률 분포(정규분포)로 인코딩한 후 샘플링하여 데이터를 재구성하는 모델입니다. (2013)

- **핵심 기술**:

	- **재파라미터화 트릭 (Reparameterization Trick)**: 직접 샘플링 시 미분이 불가능하므로, 표준 정규분포에서 샘플링한 변수 $ \epsilon $을 이용해 $ z = \mu + \sigma \cdot \epsilon $ 형태로 변환하여 **역전파(Backpropagation)가 가능**하게 만듭니다.

	- **Log Variance**: 분산의 수치적 안정성과 양수 제약을 위해 로그 스케일로 예측 후 $ \exp(\cdot) $ 적용

- **손실 함수**: 재구성 손실(Reconstruction Loss) + 잠재 공간 정규화 손실 (KL Divergence를 통해 사전 분포와 추정 분포 간의 차이 조절)

### ③ GAN (Generative Adversarial Network)

- **개요**: 생성자(Generator)와 **판별자(Discriminator)** 두 신경망이 서로 경쟁(Minimax Game)하며 학습하는 적대적 모델입니다. (2014)

- **주요 이슈**:

	- **학습 불안정성**: 두 네트워크의 균형 실패 시 학습 붕괴

	- **모드 붕괴 (Mode Collapse)**: 생성자가 특정 패턴(예: 개 이미지만)의 가짜만 무한 반복 생성하는 현상

### ④ DCGAN (Deep Convolutional GAN)

- **개요**: GAN의 학습 불안정 문제를 개선하기 위해 CNN 구조와 아키텍처 제약을 도입한 모델입니다.

- **주요 제약**:

	- 풀링 층 대신 **스트라이드 컨볼루션(Strided Convolution)** 사용

	- **배치 정규화(Batch Normalization)** 적용 (생성자 출력층, 판별자 입력층 제외)

	- 완전 연결층(Fully Connected Layer) 제거

	- 활성화 함수: 생성자는 **Tanh** (출력 범위 $ [-1, 1] $ 제한), 판별자는 **LeakyReLU** 사용

## 04. 조건부 생성 모델 (cGAN & Pix2Pix)

### ① cGAN (Conditional GAN)

- **개요**: 생성자와 판별자 모두에 추가 조건 정보(클래스 레이블, 텍스트 등 $ y $)를 함께 입력하여 원하는 속성이나 조건에 맞는 데이터를 제어하며 생성할 수 있는 모델입니다.

- **장점**: 조건 기반 제어를 통해 결과물의 다양성과 제어력이 높습니다.

### ② Pix2Pix (Image-to-Image Translation)

- **개요**: cGAN을 기반으로 한 이미지-투-이미지 번역 모델입니다. (예: 스케치 $ \rightarrow $ 사진, 지도 $ \rightarrow $ 항공사진 변환)

- **핵심 구조**:

	- **U-Net 기반 생성자**: 인코더-디코더 구조에 스킵 연결(Skip Connections)을 활용하여 저수준의 텍스처와 위치 정보를 그대로 전달합니다.

	- **PatchGAN 판별자**: 전체 이미지를 한 번에 평가하는 대신 $ N \times N $ 패치 단위로 나누어 진짜/가짜를 판단하며, 세밀한 국소적 텍스처 포착에 유리합니다.

- **장점**: 하나의 통합된 모델로 다양한 이미지 변환 문제에 적용 가능하며, 수작업 설계 없이 고해상도의 세밀한 텍스처를 생성합니다.

## Python 코드로 보는 VAE 재파라미터화 트릭 가상 구현 예시


```python
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
```

## 공식 문서 및 참고 링크

- [Generative Deep Learning 2nd Edition (David Foster) - 공식 GitHub 저장소](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FdavidADSP%2FGenerative_Deep_Learning_2nd_Edition)

- [Auto-Encoding Variational Bayes (arXiv:1312.6114)](https://arxiv.org/abs/1312.6114)

- [Generative Adversarial Networks (arXiv:1406.2661)](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F1406.2661)

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (arXiv:1511.06434)](https://arxiv.org/abs/1511.06434)

- [Conditional Generative Adversarial Nets (arXiv:1411.1784)](https://arxiv.org/abs/1411.1784)

- [Image-to-Image Translation with Conditional Adversarial Networks (arXiv:1611.07004)](https://arxiv.org/abs/1611.07004)

- 스탠포드 cs231n : [http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)

- 서울대 시각적 이해를 위한 머신러닝 : [YouTube 재생 목록 링크](https://www.youtube.com/watch?v=MuTibPS09wQ&list=PL0E_1UqNACXDTwuxUzCl5AeEjXBfWxCwc&index=21)

