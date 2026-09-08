# References

공식 문서 및 참고 링크

1) PyTorch Official Documentation - torch.Tensor

링크: https://docs.pytorch.org/docs/stable/tensors.html

관련 노트: Tensor-Shape-Interpretation, Tensor-Attributes-Cheatsheet, Tensor-Creation-Functions

이 문서에는 torch.Tensor 클래스의 모든 속성과 메서드가 정리되어 있습니다. 여기서 다루지 않은 torch.arange(), torch.linspace(), torch.eye() 같은 추가 생성 함수들도 확인할 수 있습니다.

2) PyTorch Official Documentation - CUDA semantics (Device Management)

링크: https://docs.pytorch.org/docs/stable/notes/cuda.html

관련 노트: Device-Management-CPU-GPU

GPU 메모리 관리, 여러 개의 GPU를 동시에 쓰는 방법(DataParallel, DistributedDataParallel), CUDA 스트림 등 이 노트에서 다루지 않은 심화 내용이 담겨 있습니다.

3) NumPy Official Documentation - Broadcasting

링크: https://numpy.org/doc/stable/user/basics.broadcasting.html

관련 노트: Tensor-Operations-Broadcasting

PyTorch의 브로드캐스팅 규칙은 NumPy의 규칙을 그대로 따르기 때문에, NumPy 공식 문서의 그림과 예제가 PyTorch를 이해하는 데도 그대로 도움이 됩니다.

4) PyTorch Official Documentation - Autograd mechanics

링크: https://docs.pytorch.org/docs/stable/notes/autograd.html

관련 노트: Autograd-Gradient-Computation

requires_grad, 계산 그래프(computational graph), leaf tensor, in-place 연산이 autograd에 미치는 영향 등 자동 미분의 내부 동작 원리를 더 깊이 다룹니다. in-place 연산과 관련된 에러를 이해하는 데도 도움이 됩니다.

5) PyTorch Official Documentation - torch.autograd (API)

링크: https://docs.pytorch.org/docs/stable/autograd.html

관련 노트: Autograd-Gradient-Computation

torch.autograd.grad, backward()의 전체 파라미터(예: retain_graph, create_graph)와 torch.autograd.Function을 활용해 직접 미분 연산을 정의하는 방법까지 확인할 수 있습니다.

6) PyTorch Official Documentation - torch.nn

링크: https://docs.pytorch.org/docs/stable/nn.html

관련 노트: Torch-nn-Basics, nn-Module-Model-Building

이 시리즈에서 다룬 nn.Linear, nn.ReLU, nn.Sequential, 손실 함수 외에도 nn.Conv2d, nn.LSTM, nn.BatchNorm2d, nn.Dropout 등 CNN·RNN·정규화 관련 레이어 전체 목록을 확인할 수 있습니다.

7) PyTorch Official Documentation - torch.nn.functional

링크: https://docs.pytorch.org/docs/stable/nn.functional.html

관련 노트: In-place-Methods-and-Functional

F.relu, F.mse_loss, F.max_pool2d 외에 objective 함수·convolution 함수 등 함수형(functional) API 전체 목록이 정리되어 있습니다.

8) PyTorch Official Documentation - torch.optim

링크: https://docs.pytorch.org/docs/stable/optim.html

관련 노트: Training-Loop-and-Optimizer

이 시리즈에서 다룬 SGD 외에 Adam, AdamW, RMSprop 같은 다른 옵티마이저와, learning rate를 학습 도중에 조절하는 lr_scheduler(학습률 스케줄러) 사용법을 확인할 수 있습니다.

9) PyTorch Official Tutorial - Saving and Loading Models

링크: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

관련 노트: Training-Loop-and-Optimizer

state_dict() 저장/불러오기 외에도 모델 전체(구조+파라미터) 저장, 체크포인트에 optimizer 상태까지 함께 저장해서 학습을 이어서 진행하는(resume training) 방법을 다룹니다.

관련 노트

PyTorch-Tensor-Data-Modeling-MOC

Autograd-Gradient-Computation

Torch-nn-Basics

nn-Module-Model-Building

Training-Loop-and-Optimizer

In-place-Methods-and-Functional

