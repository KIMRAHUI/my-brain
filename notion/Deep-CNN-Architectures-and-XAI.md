# Deep-CNN-Architectures-and-XAI

개요

본 문서는 컴퓨터 비전 딥러닝 혁명을 이끈 5대 대표 CNN 아키텍처(LeNet, AlexNet, VGGNet, GoogLeNet, ResNet)의 세부 구조와 연산 최적화 메커니즘, 그리고 딥러닝 모델의 블랙박스 한계를 극복하는 설명 가능한 AI(XAI: CAM, Grad-CAM) 기법을 다룹니다. AlexNet의 듀얼 GPU 병렬화부터 GoogLeNet의 1×1 병목 연산 절감, ResNet의 잔차 학습(Residual Learning) 4대 메커니즘, 그리고 역전파 그래디언트 기반 히트맵 시각화 원리까지 체계적으로 정리하였습니다.

Part 1. 고전 및 초기 심층 CNN (LeNet, AlexNet, VGGNet)

LeNet-5 (1998) - 현대 CNN의 시초

AlexNet (2012) - 딥러닝 혁명의 기폭제

VGGNet (2014) - 단순성과 3\times3 소형 필터 중첩의 미학

Part 2. 연산 최적화와 극심층 아키텍처 (GoogLeNet, ResNet)

GoogLeNet / Inception Net (2014) - 멀티스케일 병렬 처리와 1\times1 병목 계층

ResNet (2015) - 잔차 학습(Residual Learning)과 스킵 연결(Skip Connection)

CNN 모델 발전 계보 (Timeline)

Part 3. 설명 가능한 AI (XAI) - CAM & Grad-CAM

XAI(eXplainable AI)의 필요성과 분류

CAM (Class Activation Mapping)

Grad-CAM (Gradient-weighted Class Activation Mapping)

CAM vs Grad-CAM 핵심 비교표

공식 문서 및 참고 링크

AlexNet: ImageNet Classification with Deep CNNs (NIPS 2012 Paper)

Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet Paper)

Going Deeper with Convolutions (GoogLeNet Inception Paper)

Deep Residual Learning for Image Recognition (ResNet Paper)

Learning Deep Features for Discriminative Localization (CAM Original Paper)

Grad-CAM: Visual Explanations from Deep Networks (ICCV 2017 Paper)

PyTorch Grad-CAM Official Open-Source Library

