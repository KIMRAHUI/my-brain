# Data-Normalization-and-Batch-Normalization

개요

본 문서는 딥러닝 및 컴퓨터 비전 모델의 학습 수렴 속도와 안정성을 결정짓는 데이터 정규화(Data Normalization)와 배치 정규화(Batch Normalization)의 개념, 연산 구조, 필요성을 다룹니다. 모델 입력단에서 픽셀 스케일을 일관되게 맞추는 데이터 정규화부터, 심층 신경망 내부의 고질적 난제인 내부 공변량 변화(Internal Covariate Shift, ICS)를 제어하는 배치 정규화의 연산 메커니즘, 그리고 학습 가능한 파라미터(\gamma, \beta) 를 통해 신경망 고유의 비선형 표현력을 복원하는 원리까지  정리하였습니다.

Part 1. 데이터 정규화 (Data Normalization)

데이터 정규화의 개념과 화이트 밸런스 비유

데이터 정규화가 필요한 핵심 이유

수식 및 PyTorch 전처리 코드

Part 2. 배치 정규화 (Batch Normalization)

배치 정규화의 정의와 내부 공변량 변화(ICS)

배치 정규화 도입의 3대 이점

Part 3. 배치 정규화 연산 절차 및 표현력 복원 (\gamma, \beta)

배치 정규화의 3단계 연산 프로세스

학습 가능한 파라미터(\gamma, \beta)가 필수적인 이유 (표현력 복원)

공식 문서 및 참고 링크

PyTorch nn.BatchNorm2d Official Documentation

Torchvision Transforms.Normalize Documentation

Batch Normalization: Accelerating Deep Network Training (Original Paper, ICML 2015)

How Does Batch Normalization Help Optimization? (NeurIPS 2018 Paper)

CS231n Convolutional Neural Networks - Normalization Notes

