---
notion-url: https://app.notion.com/p/Tensor-Creation-Functions-43d77fa64f7f83ed9b8a81b6717cd695
title: Tensor-Creation-Functions
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Tensor-Creation-Functions-43d77fa64f7f83ed9b8a81b6717cd695
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
import torch

t = torch.rand(2, 3)
print(t)
```


```plain text
tensor([[0.5734, 0.1298, 0.8821],
        [0.4456, 0.9012, 0.2233]])
```


```python
min_val, max_val = -5, 5
t_ranged = min_val + (max_val - min_val) * torch.rand(2, 3)
```

- 원리: `torch.rand()`가 만드는 값은 항상 [0, 1) 구간이므로, 여기에 `(max_val - min_val)`을 곱해서 구간의 "폭"을 늘리고, `min_val`을 더해서 시작점을 옮기는 것입니다.

[//]: # (heading_4 is not supported)


```python
t = torch.randn(2, 3)
print(t)
```


```plain text
tensor([[-0.3211,  1.4523, -0.8821],
        [ 0.0456, -1.2012,  0.7233]])
```

[//]: # (heading_4 is not supported)

 | 구분 | `torch.rand()` | `torch.randn()` | 
 | ---- | ---- | ---- | 
 | 분포 | 균등 분포 (Uniform) | 정규 분포 (Normal/Gaussian) | 
 | 값의 범위 | 0 이상 1 미만 | 이론상 -∞ ~ +∞ (대부분 -3~3 사이) | 
 | 평균 | 약 0.5 | 0 | 
 | 주 용도 | 단순 랜덤 샘플링, 데이터 시뮬레이션 | 가중치 초기화, 노이즈 생성 | 

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)


```python
t = torch.zeros(2, 3)
print(t)
```


```plain text
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

- 주로 마스크(mask) 초기값, 손실(loss) 누적용 변수 초기화, 신경망의 편향(bias) 초기값 등에 사용됩니다.

[//]: # (heading_4 is not supported)


```python
t = torch.ones(2, 3)
print(t)
```


```plain text
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

- 마스킹 연산이나 "전체를 그대로 유지"하는 초기값이 필요할 때 자주 사용됩니다.

[//]: # (heading_4 is not supported)


```python
t = torch.full((2, 3), 7)
print(t)
```


```plain text
tensor([[7, 7, 7],
        [7, 7, 7]])
```

- `zeros`나 `ones`가 아닌 다른 상수(예: 패딩값으로 자주 쓰이는 -1이나 특정 토큰 ID)로 텐서를 채워야 할 때 사용합니다.

[//]: # (heading_4 is not supported)

- `torch.rand()`와 `torch.randn()`을 이름이 비슷해서 혼동하는 경우가 많습니다. 이름 끝에 `n`이 붙으면 "Normal(정규분포)"라고 기억하면 헷갈리지 않습니다.

- `torch.full((2, 3), 7)`에서 shape 인자는 튜플 `(2, 3)`로 감싸서 넣어야 합니다. `torch.zeros(2, 3)`처럼 괄호 없이 숫자만 나열하는 다른 함수들과 문법이 다르니 주의가 필요합니다.

- 매번 실행할 때마다 `rand`/`randn`의 결과가 달라지는 것이 당연합니다. 실험을 재현 가능하게 하려면 `torch.manual_seed(42)`처럼 시드를 고정해야 합니다.


```python
torch.manual_seed(42)
t = torch.rand(2, 3)  # 항상 동일한 값이 나옴
```

---

## 관련 노트

- [PyTorch-Tensor-Data-Modeling-MOC](https://app.notion.com/p/PyTorch-Tensor-Data-Modeling-MOC.md)

- [Tensor-Shape-Interpretation](https://app.notion.com/p/Tensor-Shape-Interpretation.md)

- [Tensor-Operations-Broadcasting](https://app.notion.com/p/Tensor-Operations-Broadcasting.md)

