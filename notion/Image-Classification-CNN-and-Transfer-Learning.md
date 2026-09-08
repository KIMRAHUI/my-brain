# Image-Classification-CNN-and-Transfer-Learning

개요

본 문서는 컴퓨터 비전의 기초이자 핵심 과제인 **이미지 분류(Image Classification)를 위한 합성곱 신경망(CNN)의 연산 구조, 대표적인 CNN 백본 모델의 변천사, 그리고 현대 딥러닝 실무의 핵심 방법론인 전이학습(Transfer Learning을 다룹니다. 합성곱(Conv)과 풀링(Pooling) 계층의 수치 연산 원리부터 LeNet에서 EfficientNet에 이르는 9대 대표 모델의 구조적 혁신, 그리고 가중치 동결(Freezing)과 역전파(Backpropagation)를 활용한 전이학습의 4단계 실행 프로세스를 체계적으로 정리하였습니다.

Part 1. CNN 핵심 연산 및 특징 추출 메커니즘 복습

합성곱 신경망(CNN, Convolutional Neural Network)의 정의

합성곱 계층(Convolutional Layer)의 수치 연산과 특성 맵(Feature Map)

풀링 계층(Pooling Layer)의 다운샘플링 역할

Part 2. 주요 CNN 백본 모델 변천사 및 아키텍처 혁신

초기 CNN과 딥러닝 붐의 태동 (1998~2014)

구조적 다변화와 극단적 심층화 (2014~2016)

채널 어텐션과 모바일 경량화 및 최적화 (2017~2019)

Part 3. 전이학습(Transfer Learning) 패러다임과 작동 원리

처음부터 학습(Train from Scratch) vs 전이학습(Transfer Learning)

전이학습의 2대 전략 비교

전이학습의 성공 조건 (Andrew Ng의 3대 원칙)

전이학습 4단계 실행 프로세스 및 역전파(Backpropagation) 원리

공식 문서 및 참고 링크

PyTorch Torchvision Models & Pretrained Weights Documentation

CS231n Convolutional Neural Networks - Transfer Learning Notes

Deep Residual Learning for Image Recognition (ResNet Paper)

EfficientNet: Rethinking Model Scaling for CNNs (ICML 2019 Paper)

Andrew Ng - When to use Transfer Learning (YouTube Lecture)

LG CNS Insight Blog - 전이학습(Transfer Learning) 개념과 원리

