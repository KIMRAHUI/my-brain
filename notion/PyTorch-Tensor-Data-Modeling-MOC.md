# PyTorch-Tensor-Data-Modeling-MOC

개요

본 문서는 PyTorch의 핵심 데이터 구조인 텐서(Tensor)의 행렬 구조 해석, 생성 함수, CPU/GPU 장치(Device) 관리 및 디바이스 불일치 에러 해결법, 데이터 타입 변환, NumPy 배열과의 연동, 그리고 브로드캐스팅(Broadcasting) 원리를 심층적으로 다룹니다.

각 노트는 원문 내용에 더해 실행 가능한 예제 코드, 예상 출력, 왜 그런 결과가 나오는지에 대한 원리 설명, 자주 하는 실수까지 포함해서 나중에 이 노트만 봐도 처음부터 다시 학습할 수 있도록 구성했습니다.

Part 1. 텐서(Tensor) 다루기

Tensor-Shape-Interpretation — 텐서의 행과 열(Shape) 해석 방법

Tensor-Attributes-Cheatsheet — 텐서 기본 속성 및 메서드 치트시트

Tensor-Creation-Functions — 특수한 텐서 생성 함수

Dtype-Conversion-NumPy-Interop — 데이터 타입 변환 및 NumPy 연동

Device-Management-CPU-GPU — CPU/GPU 장치 관리 및 에러 해결

Tensor-Operations-Broadcasting — 텐서 연산 및 브로드캐스팅

Part 2. 모델 만들기 (torch.nn)

Autograd-Gradient-Computation — Autograd와 Gradient 계산 (backward, requires_grad, no_grad)

Torch-nn-Basics — torch.nn 기초 (Linear, 활성화 함수, 손실 함수, 텐서 vs 레이어 차원)

nn-Module-Model-Building — nn.Module 상속, nn.Sequential, 복잡한 모델 설계, 모델 정보 확인

Model-Device-Management — 모델을 CPU/GPU로 옮기기

Part 3. 모델 학습시키기

Training-Loop-and-Optimizer — Training Loop, Loss 계산, Optimizer, 모델 저장/불러오기

In-place-Methods-and-Functional — in-place 메소드, torch.nn.functional

참고

References — 공식 문서 및 참고 링크

전체 개념

