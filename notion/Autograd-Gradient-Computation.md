# Autograd-Gradient-Computation

Autograd란?

텐서(Tensor)에서 이루어지는 모든 연산에 대해 자동 미분을 위한 순전파(Forward) 그래프가 만들어지고, 이를 바탕으로 역전파(Backward) 그래프가 자동으로 정의됩니다. 딥러닝의 핵심인 역전파(Back Propagation) 는 순전파(Forward Propagation) 동안 수행된 연산 기록을 바탕으로 그래디언트(기울기)를 구하는 과정이며, PyTorch는 이 과정을 autograd 기능으로 자동화합니다.

requires_grad 속성

PyTorch에서 텐서를 생성할 때 requires_grad 속성을 설정하면 해당 텐서와 관련된 연산이 기록되어, 나중에 그래디언트를 계산할 수 있습니다.

# 생성 시 설정
x = torch.tensor([2.0], requires_grad=True)
print(x.requires_grad)  # True

# 이미 만든 텐서의 속성을 나중에 바꾸기
x = torch.tensor([2.0])
x.requires_grad_()        # True로 변경 (밑줄 in-place 메서드)
x.requires_grad_(False)   # 다시 False로 변경

기본값은 requires_grad=False입니다.

딥러닝 모델의 학습 가능한 파라미터(가중치, 편향)에는 보통 requires_grad=True가 자동으로 설정되어 있습니다.

방법 1: torch.autograd.grad를 이용한 기울기 계산

import torch
from torch.autograd import grad

x1 = torch.tensor(2, requires_grad=True, dtype=torch.float16)
x2 = torch.tensor(3, requires_grad=True, dtype=torch.float16)
x3 = torch.tensor(1, requires_grad=True, dtype=torch.float16)
x4 = torch.tensor(4, requires_grad=True, dtype=torch.float16)

z1 = x1 * x2
z2 = x3 * x4
f = z1 + z2   # f = (x1 * x2) + (x3 * x4)

df_dx = grad(outputs=f, inputs=[x1, x2, x3, x4])
print(df_dx)
# (tensor(3.), tensor(2.), tensor(4.), tensor(1.))

torch.autograd.grad는 함수 f의 결과(outputs)에 대해 지정된 입력 변수들(inputs) 각각의 기울기를 반환합니다.

반환값은 튜플 형태이며, 기울기 값을 직접 확인할 수 있습니다.

장점: 특정 입력 변수만 선택적으로 기울기를 계산할 수 있습니다.

방법 2: .backward()를 이용한 기울기 계산

import torch
import torch.optim as optim

x1 = torch.tensor(2, requires_grad=True, dtype=torch.float16)
x2 = torch.tensor(3, requires_grad=True, dtype=torch.float16)
x3 = torch.tensor(1, requires_grad=True, dtype=torch.float16)
x4 = torch.tensor(4, requires_grad=True, dtype=torch.float16)

z1 = x1 * x2
z2 = x3 * x4
f = z1 + z2

f.backward()  # 모든 requires_grad=True 변수의 기울기 자동 계산

print(x1.grad)  # tensor(3., dtype=torch.float16)
print(x2.grad)  # tensor(2., dtype=torch.float16)

.backward()는 requires_grad=True로 설정된 모든 변수의 기울기를 자동으로 계산하고, 각 변수의 .grad 속성에 저장합니다.

장점: 코드가 간결하며, 다수의 변수에 대해 빠르게 기울기를 계산할 수 있습니다.

torch.autograd.grad vs .backward() 비교

기울기 초기화(zero_grad)가 필요한 이유

PyTorch에서는 .backward()를 호출할 때마다 기울기가 기존 값에 누적됩니다. 초기화 없이 반복하면 기울기가 계속 더해져서 잘못된 값이 됩니다.

초기화 없이 반복한 경우 (기울기가 누적되어 잘못됨):

opt = optim.SGD(params=[x1, x2, x3, x4], lr=0.001)

for i in range(3):
    z1 = x1 * x2
    z2 = x3 * x4
    f = z1 + z2
    f.backward()  # 기존 기울기에 계속 누적됨
    print(f"Iteration {i+1}: x1.grad = {x1.grad}")

zero_grad()로 초기화한 경우 (정상 동작):

for i in range(3):
    opt.zero_grad()  # 기울기 초기화
    z1 = x1 * x2
    z2 = x3 * x4
    f = z1 + z2
    f.backward()
    print(f"Iteration {i+1}: x1.grad = {x1.grad}")

초기화 없음: 기울기가 계속 누적되어 잘못된 업데이트로 이어질 수 있음.

초기화 있음: 매 반복마다 정확한 기울기를 계산하고 학습을 안정적으로 수행 가능.

그래서 실제 학습 루프에서는 항상 optimizer.zero_grad()를 backward() 전후에 호출합니다.

torch.no_grad() — Gradient 계산이 필요하지 않을 때

왜 필요한가?

PyTorch는 순전파 연산 정보를 기록하고 역전파로 gradient를 계산합니다. 그런데 검증(validation)이나 추론(inference)처럼 gradient 계산이 필요 없는 상황에서도 연산 정보를 계속 기록하면 불필요한 메모리와 연산 자원을 소모하게 됩니다. 이를 방지하기 위해 연산 기록을 일시적으로 비활성화하는 torch.no_grad()를 사용합니다.

x = torch.tensor([2.0], requires_grad=True)

# 연산 기록이 활성화된 경우
y1 = x ** 3
print(y1)  # tensor([8.], grad_fn=<PowBackward0>)

# torch.no_grad()로 연산 기록 비활성화
with torch.no_grad():
    y2 = x ** 3
print(y2)  # tensor([8.])  ← grad_fn이 없음

# 블록을 벗어나면 연산 기록이 자동으로 다시 활성화됨

torch.no_grad()는 with 블록 내부에서만 작동하며, 블록이 끝나면 자동으로 연산 기록이 재활성화됩니다.

주로 검증(validation) 또는 추론(inference) 과정에서 사용됩니다.

requires_grad=False vs torch.no_grad() 비교

자주 하는 실수

backward()를 호출하기 전에 optimizer.zero_grad()를 빼먹으면 기울기가 이전 배치 값과 누적되어 학습이 이상하게 진행됩니다.

검증/테스트 루프에서 torch.no_grad()를 빼먹으면 불필요하게 메모리를 많이 사용하고 속도도 느려집니다.

requires_grad=False인 텐서에 .backward()를 호출하려 하면 grad_fn이 없어서 에러가 납니다. 손실(loss) 텐서는 반드시 grad_fn을 가지고 있어야(즉, 연산 기록이 살아있어야) .backward()가 가능합니다.

관련 노트

PyTorch-Tensor-Data-Modeling-MOC

Tensor-Attributes-Cheatsheet

Training-Loop-and-Optimizer

nn-Module-Model-Building

