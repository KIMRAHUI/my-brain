# Model-Device-Management

개요

Device-Management-CPU-GPU 노트에서 텐서의 연산 장치를 어떻게 지정하는지 배웠습니다. 이 노트에서는 모델(nn.Module) 자체의 연산 장치를 어떻게 다루는지 정리합니다.

PyTorch에서 기본 장치는 CPU

PyTorch에서는 텐서뿐만 아니라 모델도 어떤 장치에서 사용할지 설정할 수 있습니다. 텐서와 마찬가지로 모델도 기본적으로 CPU에 만들어집니다.

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
        output = self.linear2(x)
        return output

model = MyModel()  # model이 CPU에 만들어짐

모델이 어느 장치에 있는지 확인하기

그런데 모델이 CPU에 있는지 GPU에 있는지를 모델 객체에서 바로 확인할 수는 없습니다. 부모 클래스인 nn.Module에는 device라는 속성이 따로 없기 때문입니다. 그 대신 모델의 파라미터 텐서에서 device 속성을 확인해 보면 됩니다.

for name, param in model.named_parameters():
    print(f'{name}: {param.device}')

출력 예시:

linear0.weight: cpu
linear0.bias: cpu
linear1.weight: cpu
linear1.bias: cpu
linear2.weight: cpu
linear2.bias: cpu

결과를 보면 모든 파라미터 텐서가 CPU에 있다고 나옵니다. 즉, 모든 파라미터가 CPU에 있으므로 "모델이 CPU에 있다"라고 할 수 있는 것입니다.

CPU에 있는 모델을 GPU로 옮기기

CPU에 있는 모델을 GPU로 옮기려면, 텐서를 GPU로 옮길 때처럼 .to() 메서드에 'cuda'를 입력하면 됩니다.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to('cuda')

# 변수에 to() 메소드 결과를 할당해도 좋습니다.
# model = model.to('cuda')

모델에서 .to() 메서드를 쓸 때는 텐서와 달리 결과를 따로 변수에 할당하지 않고 그냥 model.to('cuda')만 실행해도 모델 자체가 GPU로 옮겨집니다. (내부적으로 모델은 참조로 관리되기 때문입니다.)

물론 model = model.to('cuda')처럼 변수에 다시 할당해도 문제없이 동작합니다.

for name, param in model.named_parameters():
    print(f'{name}: {param.device}')

다시 파라미터 텐서들의 device 속성을 출력해 보면 아래처럼 나옵니다.

linear0.weight: cuda:0
linear0.bias: cuda:0
...

여기서 숫자 0은 모델이 0번 인덱스의 GPU에 있다는 뜻입니다.

to() 대신 cuda() / cpu() 메서드 사용하기

.to() 메서드 대신 .cuda()나 .cpu() 메서드를 사용할 수도 있습니다.

model.cpu()   # 모델을 CPU로 이동

for name, param in model.named_parameters():
    print(f'{name}: {param.device}')  # cpu

model.cuda()  # 모델을 (기본) GPU로 이동

for name, param in model.named_parameters():
    print(f'{name}: {param.device}')  # cuda:0

.to("cuda")와 .cuda()는 결과가 동일합니다. 다만 .to(device) 방식이 device 변수 하나로 CPU/GPU를 자동 전환할 수 있어 더 권장됩니다. (자세한 이유는 Device-Management-CPU-GPU 참고)

텐서의 연산 장치와 모델의 연산 장치가 일치해야 하는 이유

모델에 텐서를 입력하면 입력된 텐서와 모델 파라미터 텐서가 여러 연산(행렬 곱셈 등)을 수행하게 됩니다. 텐서 간 연산이 이루어지려면 모든 텐서가 같은 연산 장치에 있어야 한다고 했었죠. 따라서 모델에 텐서를 입력할 때도 입력 텐서와 모델이 동일한 장치에 있어야 정상적으로 연산이 가능합니다.

에러 재현: 모델은 GPU, 입력은 CPU

model.to('cuda')
tensor_cpu = torch.randn(2, 8)
model(tensor_cpu)  # 에러 발생!

모델 파라미터 텐서는 GPU에 있는 반면, 모델에 입력된 텐서는 CPU에 있어서 텐서 간 연산이 불가능하다는 에러가 발생합니다. (에러 메시지는 Device-Management-CPU-GPU에서 다룬 디바이스 불일치 에러와 동일한 원리입니다.)

해결: 입력 텐서도 같은 장치로 옮기기

model.to('cuda')
tensor_gpu = torch.randn(2, 8).to('cuda')
model(tensor_gpu)  # 정상 작동

여러 개의 GPU를 사용하는 경우

GPU가 여러 대인 환경에서는 텐서와 모델이 동일한 번호의 GPU에 있어야 연산이 가능합니다. .to() 메서드를 사용할 때 GPU 인덱스까지 정확히 동일하게 지정해야 합니다.

model.to('cuda:0')
tensor_gpu = torch.randn(2, 8).to('cuda:0')
model(tensor_gpu)  # 정상 작동 (둘 다 0번 GPU)

단순히 "GPU에 있다"는 것만으로는 부족하고, cuda:0과 cuda:1처럼 GPU 번호까지 정확히 맞아야 합니다.

실전 패턴: 학습 루프에서의 device 처리

실무에서는 아래처럼 모델과 입력 데이터를 모두 동일한 device 변수로 관리하는 패턴을 거의 항상 사용합니다.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel()
model.to(device)

for x_batch, y_batch in dataloader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    pred = model(x_batch)
    ...

자세한 학습 루프 패턴은 Training-Loop-and-Optimizer 노트를 참고하세요.

자주 하는 실수

모델은 .to(device)로 옮겼는데, 배치마다 반복해서 가져오는 입력 데이터 텐서를 device로 옮기는 걸 깜빡하는 경우가 정말 많습니다. 학습 루프 안에서 매번 입력 텐서를 device로 옮기고 있는지 확인하세요.

모델 객체 자체에는 .device라는 속성이 없다는 걸 모르고 model.device처럼 접근하려다 에러를 만나는 경우가 있습니다. 항상 named_parameters()를 통해 간접적으로 확인해야 합니다.

멀티 GPU 환경에서 cuda:0과 cuda:1을 섞어서 쓰다가 디바이스 불일치 에러를 만나는 경우가 있습니다. 여러 GPU를 쓸 계획이라면 처음부터 GPU 인덱스를 명시적으로 관리하는 것이 좋습니다.

관련 노트

PyTorch-Tensor-Data-Modeling-MOC

Device-Management-CPU-GPU

nn-Module-Model-Building

Training-Loop-and-Optimizer

