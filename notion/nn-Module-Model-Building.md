# nn-Module-Model-Building

torch.nn.Module이란?

torch.nn은 PyTorch에서 신경망을 구축하는 데 필요한 다양한 구성 요소와 기능을 제공하는 핵심 모듈입니다. 계층(layer), 손실 함수(loss function), 활성화 함수(activation function), 정규화 기법(BatchNorm, LayerNorm 등) 같은 신경망 관련 도구들을 포함하고 있습니다.

주요 특징

신경망 구축과 학습 지원: 딥러닝 모델을 구축하고 학습하는 데 필요한 다양한 도구 제공

다양한 구성 요소: 각 계층, 활성화 함수, 손실 함수, 정규화 기법 등을 손쉽게 사용 가능

모듈 상속으로 사용자 정의 모델 설계 가능: torch.nn.Module을 상속받아 사용자 정의 모델을 설계할 수 있음

모델 정의: nn.Module 상속

모델 클래스를 정의할 때는 반드시 torch.nn.Module을 상속받아야 합니다. 이렇게 하면 PyTorch의 다양한 기능(파라미터 자동 관리, .to(device), .parameters() 등)을 활용할 수 있고, 계층 구조도 손쉽게 설계할 수 있습니다.

다음 구조의 모델을 예로 만들어보겠습니다.

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(8, 4)
        self.linear1 = nn.Linear(4, 6)
        self.linear2 = nn.Linear(6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear0(x))
        x = self.relu(self.linear1(x))
        y = self.linear2(x)
        return y

model = MyModel()

input_tensor = torch.randn(2, 8)  # 배치 크기: 2, 입력 차원: 8
result = model(input_tensor)
print(result)

입력 텐서의 크기: [2, 8] (배치 크기 2, 입력 차원 8)

출력 텐서의 크기: [2, 3] (배치 크기 2, 출력 차원 3)

사용자 정의 모델 설계의 기본 패턴

torch.nn.Module을 상속받아 클래스를 정의

__init__()에서 모델 구성 요소(레이어, 활성화 함수 등)를 정의하고, 맨 처음에 super().__init__()을 반드시 호출

forward() 메서드에서 입력 텐서가 각 레이어를 어떤 순서로 통과하는지 연산 과정을 구현

nn.Module의 계층 구조 (모델 안에 모델 넣기)

PyTorch에서 nn.Module은 계층 구조를 지원합니다. 즉, 모델 내부에 다른 nn.Module을 서브모듈로 포함할 수 있습니다.

class MyModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = MyModel()      # 서브모듈로 MyModel 포함
        self.layer = nn.Linear(3, 4)

    def forward(self, x):
        x = self.block(x)  # MyModel 통과
        y = self.layer(x)  # 추가 레이어 통과
        return y

model2 = MyModel2()
input_tensor = torch.randn(2, 8)
result = model2(input_tensor)
print(result)

이렇게 모델을 조립식으로 구성할 수 있으면, 반복되는 블록(예: Transformer의 인코더 블록)을 재사용하기 매우 편리해집니다.

nn.Sequential 활용하기 — 레이어가 순차적으로 연결된 모델

앞서 정의한 MyModel처럼, 모델에 입력된 텐서가 첫 번째 레이어를 통과하고 그 출력이 다음 레이어의 입력이 되는 식으로 순차적으로만 연결된 간단한 구조라면, nn.Sequential을 이용해 더 간편하게 만들 수 있습니다.

사용법은 간단합니다. 순차적으로 연결할 레이어(또는 활성화 함수) 객체들을 콤마로 구분해서 nn.Sequential()의 입력으로 넣어 주면 됩니다.

model = nn.Sequential(
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 6),
    nn.ReLU(),
    nn.Linear(6, 3),
)

nn.Sequential 객체에 텐서를 입력하면 각 레이어의 출력이 다음 레이어의 입력으로 자동으로 전달됩니다.

따로 __init__()과 forward() 메서드를 구현할 필요가 없어 코드가 간결해집니다.

위 코드는 앞서 만든 MyModel 클래스로 만든 모델과 구조가 완전히 동일합니다.

nn.Sequential 객체가 모델 안에 포함된 경우

nn.Sequential 역시 nn.Module이 부모 클래스이므로, nn.Sequential 객체도 모델 안에 서브모듈로 포함될 수 있습니다.

class MyModelWithSequential(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
        )
        self.layer = nn.Linear(3, 4)

    def forward(self, x):
        x = self.block(x)
        y = self.layer(x)
        return y

이 방식은 "순차적인 블록"과 "그 외 별도 레이어"를 조합할 때 특히 유용합니다.

복잡한 모델 설계하기 — 입력/출력이 여러 개, 병렬 구조

입력과 출력이 여러 개이며, 병렬 구조를 포함하는 복잡한 모델도 torch.nn.Module을 활용해 효율적으로 구현할 수 있습니다.

import torch
import torch.nn as nn

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0_0 = nn.Linear(4, 8)   # input0 → Linear(4, 8)
        self.linear0_1 = nn.Linear(6, 4)   # input1 → Linear(6, 4)
        self.linear1_0 = nn.Linear(14, 8)  # concat된 출력 → Linear(14, 8)
        self.linear1_1 = nn.Linear(14, 2)  # concat된 출력 → Linear(14, 2)
        self.relu = nn.ReLU()

    def forward(self, input0, input1, input2):
        # 첫 번째 병렬 레이어
        h0_0 = self.relu(self.linear0_0(input0))
        h0_1 = self.relu(self.linear0_1(input1))

        # 컨캐터네이트(concat) 연산 — 여러 텐서를 이어 붙임
        h1 = torch.cat([h0_0, h0_1, input2], dim=1)

        # 출력 레이어 두 개 (다중 출력)
        output0 = self.linear1_0(h1)
        output1 = self.linear1_1(h1)

        return output0, output1

model = ComplexModel()

x0 = torch.randn(2, 4)  # input0: 크기 [2, 4]
x1 = torch.randn(2, 6)  # input1: 크기 [2, 6]
x2 = torch.randn(2, 2)  # input2: 크기 [2, 2]

y0, y1 = model(x0, x1, x2)

print(y0.size())  # torch.Size([2, 8])
print(y1.size())  # torch.Size([2, 2])

구조 해설

병렬 레이어: input0은 Linear(4, 8)로 처리 → [batch, 8] / input1은 Linear(6, 4)로 처리 → [batch, 4]

컨캐터네이트: [batch, 8] + [batch, 4] + [batch, 2] (input2를 그대로 이어붙임) → [batch, 14]

출력 레이어: [batch, 14] → Linear(14, 8) → output0 [batch, 8] / [batch, 14] → Linear(14, 2) → output1 [batch, 2]

이처럼 forward() 메서드는 인자를 여러 개(input0, input1, input2) 받을 수 있고, 반환값도 여러 개(output0, output1)로 만들 수 있습니다. 이 유연함 덕분에 다중 입력/다중 출력 모델(예: 멀티태스크 러닝)을 자유롭게 설계할 수 있습니다.

다중 출력 예시 — Auxiliary Output 구조

Input_1 (10차원) ───> Dense_1 (16차원) ────────────┐
                                                    ├─> Concatenate ──> Dense_3 (32차원) ──> Dense_4 (32차원) ──> Main_Output (1차원)
Input_2 (5차원)  ───> Dense_2 (16차원) ────────────┘
                                                    └────────────> Aux_Output (1차원)

class ComplexModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.Dense1 = nn.Linear(10, 16)
        self.Dense2 = nn.Linear(5, 16)
        self.Dense3 = nn.Linear(32, 32)
        self.Dense4 = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.aux_output = nn.Linear(16, 1)
        self.main_output = nn.Linear(32, 1)

    def forward(self, input1, input2):
        x1 = self.relu(self.Dense1(input1))
        out1 = self.aux_output(x1)              # 보조 출력(auxiliary output)

        x2 = self.relu(self.Dense2(input2))
        x2 = torch.concat((x1, x2), dim=1)       # 두 갈래를 이어붙임
        x2 = self.relu(self.Dense3(x2))
        x2 = self.relu(self.Dense4(x2))
        out2 = self.main_output(x2)              # 주 출력(main output)

        return out1, out2

model = ComplexModel2()
input1 = torch.ones((1, 10))
input2 = torch.ones((1, 5))
output1, output2 = model(input1, input2)
print(output1, output2)

이런 구조는 학습을 안정시키기 위해 중간 지점에서도 손실(loss)을 하나 더 계산하는 "보조 출력(auxiliary output)" 기법에 자주 사용됩니다.

모델 정보 확인하기

PyTorch 모델에 담긴 다양한 정보를 확인하면 모델 디버깅과 분석에 큰 도움이 됩니다.

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(8, 4)
        self.linear1 = nn.Linear(4, 6)
        self.linear2 = nn.Linear(6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear0(x))
        x = self.relu(self.linear1(x))
        y = self.linear2(x)
        return y

model = MyModel()

1) 모델 전체 구조 출력

print(model)

print(model)을 실행하면 모델 안에 어떤 서브모듈(레이어, 활성화 함수)이 어떤 이름으로 등록되어 있는지 트리 형태로 보여줍니다.

2) 서브 모듈 확인 — children() / named_children()

list(model.children())
# [Linear(in_features=8, out_features=4, bias=True),
#  Linear(in_features=4, out_features=6, bias=True),
#  Linear(in_features=6, out_features=3, bias=True),
#  ReLU()]

for named_child in model.named_children():
    print(named_child)
# ('linear0', Linear(in_features=8, out_features=4, bias=True))
# ('linear1', Linear(in_features=4, out_features=6, bias=True))
# ('linear2', Linear(in_features=6, out_features=3, bias=True))
# ('relu', ReLU())

.children()은 모델 바로 아래에 있는 서브모듈들을 순서대로 보여줍니다.

.named_children()은 서브모듈의 속성 이름(예: linear0)과 함께 보여줘서, 어떤 속성이 어떤 레이어인지 짝지어 확인할 수 있습니다.

3) 파라미터 정보 확인 — parameters() / named_parameters()

# 파라미터 값(텐서)만 확인
for param in model.parameters():
    print(param)

# 파라미터 이름과 함께 확인
for named_param in model.named_parameters():
    print(named_param)
# ('linear0.weight', Parameter containing: tensor([...], requires_grad=True))
# ('linear0.bias', Parameter containing: tensor([...], requires_grad=True))
# ...

각 레이어의 weight(가중치)와 bias(편향) 텐서 값을 직접 출력해서 확인할 수 있습니다.

named_parameters()를 쓰면 어떤 레이어의 어떤 파라미터인지(linear0.weight처럼) 이름까지 함께 확인할 수 있어 디버깅에 유용합니다.

4) 모델이 다른 모델을 포함한 경우(중첩 모델)의 정보 확인

class MyModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = MyModel()          # 서브모듈로 MyModel 포함
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        x = self.block(x)
        return self.linear(x)

model2 = MyModel2()

print(model2)
# MyModel2 안에 block(MyModel 전체)과 linear가 트리 구조로 출력됨

for named_child in model2.named_children():
    print(named_child)
# ('block', MyModel(...))
# ('linear', Linear(in_features=3, out_features=4, bias=True))

for named_param in model2.named_parameters():
    print(named_param)
# ('block.linear0.weight', ...) 처럼 서브모듈 이름이 접두사로 붙어서 나옴

MyModel2는 MyModel을 서브모듈로 포함한 모델입니다. 이렇게 중첩된 구조에서도 named_parameters()는 block.linear0.weight처럼 전체 경로를 이름으로 알려주기 때문에, 아무리 모델이 복잡하게 중첩되어도 각 파라미터가 어디 소속인지 정확히 추적할 수 있습니다.

실습 예제: California Housing 회귀 모델

California Housing 데이터는 입력 피처가 8가지이고 타깃은 주택 가격 1가지이므로, 모델의 입력 차원과 출력 차원은 각각 8과 1이 되어야 합니다.

class CaliforniaHousingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(8, 16)
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear0(x))
        x = self.relu(self.linear1(x))
        output = self.linear2(x)
        return output

model = CaliforniaHousingModel()
for child in model.children():
    print(child)

실행 결과:

Linear(in_features=8, out_features=16, bias=True)
Linear(in_features=16, out_features=32, bias=True)
Linear(in_features=32, out_features=1, bias=True)
ReLU()

회귀(regression) 문제이므로 마지막 레이어에는 활성화 함수를 적용하지 않습니다. 분류 문제와 달리 회귀는 예측값이 임의의 실수 범위를 가질 수 있어야 하기 때문입니다.

자주 하는 실수

__init__()에서 super().__init__()을 호출하는 걸 빼먹으면, nn.Module의 기본 기능(파라미터 자동 등록 등)이 동작하지 않아 다양한 에러가 발생합니다. 항상 __init__()의 첫 줄에 작성하는 습관을 들이세요.

forward() 메서드 이름은 정확히 forward여야 합니다. 다른 이름으로 정의하면 model(x)처럼 호출했을 때 자동으로 실행되지 않습니다.

회귀 모델의 마지막 레이어에 실수로 ReLU()나 Sigmoid() 같은 활성화 함수를 적용하면 예측값의 범위가 제한되어(예: 음수가 아예 안 나옴) 잘못된 학습이 될 수 있습니다.

관련 노트

PyTorch-Tensor-Data-Modeling-MOC

Torch-nn-Basics

Model-Device-Management

Training-Loop-and-Optimizer

