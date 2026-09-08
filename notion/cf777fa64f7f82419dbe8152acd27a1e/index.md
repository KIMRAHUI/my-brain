---
notion-url: https://app.notion.com/p/References-cf777fa64f7f82419dbe8152acd27a1e
title: References
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/References-cf777fa64f7f82419dbe8152acd27a1e
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- 링크: [https://docs.pytorch.org/docs/stable/tensors.html](https://docs.pytorch.org/docs/stable/tensors.html)

- 관련 노트: [Tensor-Shape-Interpretation](https://app.notion.com/p/Tensor-Shape-Interpretation.md), [Tensor-Attributes-Cheatsheet](https://app.notion.com/p/Tensor-Attributes-Cheatsheet.md), [Tensor-Creation-Functions](https://app.notion.com/p/Tensor-Creation-Functions.md)

- 이 문서에는 `torch.Tensor` 클래스의 모든 속성과 메서드가 정리되어 있습니다. 여기서 다루지 않은 `torch.arange()`, `torch.linspace()`, `torch.eye()` 같은 추가 생성 함수들도 확인할 수 있습니다.

[//]: # (heading_4 is not supported)

- 링크: [https://docs.pytorch.org/docs/stable/notes/cuda.html](https://docs.pytorch.org/docs/stable/notes/cuda.html)

- 관련 노트: [Device-Management-CPU-GPU](https://app.notion.com/p/Device-Management-CPU-GPU.md)

- GPU 메모리 관리, 여러 개의 GPU를 동시에 쓰는 방법(`DataParallel`, `DistributedDataParallel`), CUDA 스트림 등 이 노트에서 다루지 않은 심화 내용이 담겨 있습니다.

[//]: # (heading_4 is not supported)

- 링크: [https://numpy.org/doc/stable/user/basics.broadcasting.html](https://numpy.org/doc/stable/user/basics.broadcasting.html)

- 관련 노트: [Tensor-Operations-Broadcasting](https://app.notion.com/p/Tensor-Operations-Broadcasting.md)

- PyTorch의 브로드캐스팅 규칙은 NumPy의 규칙을 그대로 따르기 때문에, NumPy 공식 문서의 그림과 예제가 PyTorch를 이해하는 데도 그대로 도움이 됩니다.

[//]: # (heading_4 is not supported)

- 링크: [https://docs.pytorch.org/docs/stable/notes/autograd.html](https://docs.pytorch.org/docs/stable/notes/autograd.html)

- 관련 노트: [Autograd-Gradient-Computation](https://app.notion.com/p/Autograd-Gradient-Computation.md)

- `requires_grad`, 계산 그래프(computational graph), leaf tensor, in-place 연산이 autograd에 미치는 영향 등 자동 미분의 내부 동작 원리를 더 깊이 다룹니다. in-place 연산과 관련된 에러를 이해하는 데도 도움이 됩니다.

[//]: # (heading_4 is not supported)

- 링크: [https://docs.pytorch.org/docs/stable/autograd.html](https://docs.pytorch.org/docs/stable/autograd.html)

- 관련 노트: [Autograd-Gradient-Computation](https://app.notion.com/p/Autograd-Gradient-Computation.md)

- `torch.autograd.grad`, `backward()`의 전체 파라미터(예: `retain_graph`, `create_graph`)와 `torch.autograd.Function`을 활용해 직접 미분 연산을 정의하는 방법까지 확인할 수 있습니다.

[//]: # (heading_4 is not supported)

- 링크: [https://docs.pytorch.org/docs/stable/nn.html](https://docs.pytorch.org/docs/stable/nn.html)

- 관련 노트: [Torch-nn-Basics](https://app.notion.com/p/Torch-nn-Basics.md), [nn-Module-Model-Building](https://app.notion.com/p/nn-Module-Model-Building.md)

- 이 시리즈에서 다룬 `nn.Linear`, `nn.ReLU`, `nn.Sequential`, 손실 함수 외에도 `nn.Conv2d`, `nn.LSTM`, `nn.BatchNorm2d`, `nn.Dropout` 등 CNN·RNN·정규화 관련 레이어 전체 목록을 확인할 수 있습니다.

[//]: # (heading_4 is not supported)

- 링크: [https://docs.pytorch.org/docs/stable/nn.functional.html](https://docs.pytorch.org/docs/stable/nn.functional.html)

- 관련 노트: [In-place-Methods-and-Functional](https://app.notion.com/p/In-place-Methods-and-Functional.md)

- `F.relu`, `F.mse_loss`, `F.max_pool2d` 외에 objective 함수·convolution 함수 등 함수형(functional) API 전체 목록이 정리되어 있습니다.

[//]: # (heading_4 is not supported)

- 링크: [https://docs.pytorch.org/docs/stable/optim.html](https://docs.pytorch.org/docs/stable/optim.html)

- 관련 노트: [Training-Loop-and-Optimizer](https://app.notion.com/p/Training-Loop-and-Optimizer.md)

- 이 시리즈에서 다룬 `SGD` 외에 `Adam`, `AdamW`, `RMSprop` 같은 다른 옵티마이저와, learning rate를 학습 도중에 조절하는 `lr_scheduler`(학습률 스케줄러) 사용법을 확인할 수 있습니다.

[//]: # (heading_4 is not supported)

- 링크: [https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

- 관련 노트: [Training-Loop-and-Optimizer](https://app.notion.com/p/Training-Loop-and-Optimizer.md)

- `state_dict()` 저장/불러오기 외에도 모델 전체(구조+파라미터) 저장, 체크포인트에 optimizer 상태까지 함께 저장해서 학습을 이어서 진행하는(resume training) 방법을 다룹니다.

---

[//]: # (heading_4 is not supported)

- [PyTorch-Tensor-Data-Modeling-MOC](https://app.notion.com/p/PyTorch-Tensor-Data-Modeling-MOC.md)

- [Autograd-Gradient-Computation](https://app.notion.com/p/Autograd-Gradient-Computation.md)

- [Torch-nn-Basics](https://app.notion.com/p/Torch-nn-Basics.md)

- [nn-Module-Model-Building](https://app.notion.com/p/nn-Module-Model-Building.md)

- [Training-Loop-and-Optimizer](https://app.notion.com/p/Training-Loop-and-Optimizer.md)

- [In-place-Methods-and-Functional](https://app.notion.com/p/In-place-Methods-and-Functional.md)

