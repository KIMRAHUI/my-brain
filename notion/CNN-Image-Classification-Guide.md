# CNN-Image-Classification-Guide

개요

본 문서는 PyTorch를 활용한 딥러닝 이미지 분류의 기초가 되는 CNN(Convolutional Neural Network)의 아키텍처 구조, 이미지 데이터의 수치화 및 텐서 변환, 합성곱(Convolution)과 풀링(Pooling) 레이어의 작동 원리, 주요 이미지 증강(Data Augmentation) 기법별 세부 파라미터와 주의점, 커스텀 모델 클래스 설계 기법, 그리고 학습(Training) 및 평가(Evaluation) 루프의 전체 구현 프로세스를 정리하였습니다.

각 파트는 실행 가능한 예제 코드의 뼈대가 되며, 나중에 이 노트만 봐도 전체 파이프라인의 개념부터 구현 세부 사항까지 체계적으로 다시 학습할 수 있도록 구성했습니다.

Part 1. 이미지 데이터와 CNN 기초

이미지 데이터의 수치화(RGB 채널과 픽셀 행렬) 구조 해석

합성곱(Convolution)과 풀링(Pooling) 레이어의 역할

다차원 특성 맵을 1차원으로 펼치는 Flatten과 전결합층 연결

Part 2. 주요 이미지 증강(Data Augmentation) 기법, 파라미터 및 주의점

전처리 및 데이터 증강 주요 클래스 (torchvision.transforms / v2)

Part 3. 모델 구현 및 학습 루프 (Cnn_Image_Classification.ipynb)

PyTorch를 이용한 커스텀 CNN 모델 클래스 설계

이미지 분류 모델의 학습(Training) 및 평가(Evaluation) 루프 구현

공식 문서 및 참고 링크

PyTorch Official - Image Classification Tutorial

Torchvision Datasets & Transforms Documentation

PyTorch Transforms v2 API Reference

Scikit-Learn Model Evaluation Metrics Guide

