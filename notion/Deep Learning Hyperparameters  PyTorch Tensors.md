# Deep Learning Hyperparameters & PyTorch Tensors

순전파 (Forward Propagation)

개념: 입력 데이터(X)가 신경망의 여러 은닉층(Hidden Layers)을 거치면서 가중치(W)와 편향(b)에 의해 계산되고, 활성화 함수(Activation Function)를 지나 **최종 예측값(\hat{y})에 도달하는 과정입니다.

동작 흐름:

목적: 현재 모델의 가중치 상태로 입력 데이터에 대한 예측 및 오차를 측정하는 것

역전파 (Backpropagation)

개념: 순전파에서 계산된 손실(오차)을 신경망의 반대 방향(출력층 \rightarrow 입력층)으로 전파하면서, 각 가중치가 오차에 얼마나 기여했는지(기울기/Gradient)를 계산하는 과정입니다.

동작 흐름:

목적: 계산된 기울기(Gradient)를 바탕으로 오차를 최소화하는 최적의 가중치를 찾아가는 것

학습 기본 설정 및 최적화 제어 (Hyperparameters & Training Control)

순전파 및 역전파 연산 (NumPy Operations & Linear Algebra)

PyTorch Tensor 기본 속성 및 메서드 (PyTorch Tensor Operations)

참고 공식 문서 및 학습 자료

PyTorch Official Documentation - Tensor Basics

PyTorch Official Tutorial - Deep Learning with PyTorch: A 60 Minute Blitz

PyTorch Official Guide - Automatic Differentiation with torch.autograd

CS231n: Convolutional Neural Networks for Visual Recognition - Backpropagation Guide

