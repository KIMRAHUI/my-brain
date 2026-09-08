---
notion-url: https://app.notion.com/p/Model-Device-Management-35677fa64f7f823f808181c0b23422e8
title: Model-Device-Management
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Model-Device-Management-35677fa64f7f823f808181c0b23422e8
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
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
```

[//]: # (heading_4 is not supported)


```python
for name, param in model.named_parameters():
    print(f'{name}: {param.device}')
```


```plain text
linear0.weight: cpu
linear0.bias: cpu
linear1.weight: cpu
linear1.bias: cpu
linear2.weight: cpu
linear2.bias: cpu
```

[//]: # (heading_4 is not supported)


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to('cuda')

# 변수에 to() 메소드 결과를 할당해도 좋습니다.
# model = model.to('cuda')
```

- 모델에서 `.to()` 메서드를 쓸 때는 텐서와 달리 결과를 따로 변수에 할당하지 않고 그냥 `model.to('cuda')`만 실행해도 모델 자체가 GPU로 옮겨집니다. (내부적으로 모델은 참조로 관리되기 때문입니다.)

- 물론 `model = model.to('cuda')`처럼 변수에 다시 할당해도 문제없이 동작합니다.


```python
for name, param in model.named_parameters():
    print(f'{name}: {param.device}')
```


```plain text
linear0.weight: cuda:0
linear0.bias: cuda:0
...
```

[//]: # (heading_4 is not supported)


```python
model.cpu()   # 모델을 CPU로 이동

for name, param in model.named_parameters():
    print(f'{name}: {param.device}')  # cpu

model.cuda()  # 모델을 (기본) GPU로 이동

for name, param in model.named_parameters():
    print(f'{name}: {param.device}')  # cuda:0
```

- `.to("cuda")`와 `.cuda()`는 결과가 동일합니다. 다만 `.to(device)` 방식이 device 변수 하나로 CPU/GPU를 자동 전환할 수 있어 더 권장됩니다. (자세한 이유는 [Device-Management-CPU-GPU](https://app.notion.com/p/Device-Management-CPU-GPU.md) 참고)

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
model.to('cuda')
tensor_cpu = torch.randn(2, 8)
model(tensor_cpu)  # 에러 발생!
```

[//]: # (heading_4 is not supported)


```python
model.to('cuda')
tensor_gpu = torch.randn(2, 8).to('cuda')
model(tensor_gpu)  # 정상 작동
```

[//]: # (heading_4 is not supported)


```python
model.to('cuda:0')
tensor_gpu = torch.randn(2, 8).to('cuda:0')
model(tensor_gpu)  # 정상 작동 (둘 다 0번 GPU)
```

- 단순히 "GPU에 있다"는 것만으로는 부족하고, `cuda:0`과 `cuda:1`처럼 GPU 번호까지 정확히 맞아야 합니다.

[//]: # (heading_4 is not supported)


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel()
model.to(device)

for x_batch, y_batch in dataloader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    pred = model(x_batch)
    ...
```

[//]: # (heading_4 is not supported)

- 모델은 `.to(device)`로 옮겼는데, 배치마다 반복해서 가져오는 입력 데이터 텐서를 device로 옮기는 걸 깜빡하는 경우가 정말 많습니다. 학습 루프 안에서 매번 입력 텐서를 device로 옮기고 있는지 확인하세요.

- 모델 객체 자체에는 `.device`라는 속성이 없다는 걸 모르고 `model.device`처럼 접근하려다 에러를 만나는 경우가 있습니다. 항상 `named_parameters()`를 통해 간접적으로 확인해야 합니다.

- 멀티 GPU 환경에서 `cuda:0`과 `cuda:1`을 섞어서 쓰다가 디바이스 불일치 에러를 만나는 경우가 있습니다. 여러 GPU를 쓸 계획이라면 처음부터 GPU 인덱스를 명시적으로 관리하는 것이 좋습니다.

---

[//]: # (heading_4 is not supported)

- [PyTorch-Tensor-Data-Modeling-MOC](https://app.notion.com/p/PyTorch-Tensor-Data-Modeling-MOC.md)

- [Device-Management-CPU-GPU](https://app.notion.com/p/Device-Management-CPU-GPU.md)

- [nn-Module-Model-Building](https://app.notion.com/p/nn-Module-Model-Building.md)

- [Training-Loop-and-Optimizer](https://app.notion.com/p/Training-Loop-and-Optimizer.md)

