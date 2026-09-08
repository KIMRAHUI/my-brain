---
notion-url: https://app.notion.com/p/Tensor-Shape-Interpretation-d5b77fa64f7f83aaa5c0815699ac44cf
title: Tensor-Shape-Interpretation
date: '2026-08-31 01:15:00.000'
from_notion: https://app.notion.com/p/Tensor-Shape-Interpretation-d5b77fa64f7f83aaa5c0815699ac44cf
author: rahui kim
last_edited_time: '2026-08-31 01:15:00.000'
---
[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

[//]: # (heading_4 is not supported)

- **순서 기억하는 법:** 우리말 "행렬"이라는 단어 자체가 "행"이 먼저 오고 "열"이 나중에 옵니다. 그대로 순서를 외우면 됩니다. (행 → 세로 → 먼저 / 열 → 가로 → 나중)

- **헷갈릴 때 비유:** 엑셀 시트를 떠올리면 쉽습니다. "3행 5열" 이라고 하면 위에서 아래로 3번째 줄, 왼쪽에서 오른쪽으로 5번째 칸을 뜻하죠. 텐서도 동일한 논리입니다.

[//]: # (heading_4 is not supported)


```python
import torch

t = torch.rand(2, 3)
print(t)
```


```plain text
tensor([[0.1234, 0.5678, 0.9012],
        [0.3456, 0.7890, 0.2345]])
```

- **앞의 숫자 **`**2**`** (행 / 세로):** 바깥쪽 대괄호 `[ ]` 안에 있는 **큰 묶음(줄, row)이 총 몇 개인지**를 의미합니다. 위 예시에서는 `[0.1234, ...]` 줄과 `[0.3456, ...]` 줄, 이렇게 2개의 줄이 있으므로 첫 번째 숫자가 2입니다.

- **뒤의 숫자 **`**3**`** (열 / 가로):** **각 줄마다 요소(숫자)가 몇 개씩** 들어있는지를 의미합니다. `[0.1234, 0.5678, 0.9012]`처럼 한 줄에 숫자가 3개씩 있으므로 두 번째 숫자가 3입니다.

 | 구분 | 영어 명칭 | 방향 | 텐서 모양 `(x, y)`에서의 위치 | 확인하는 법 | 
 | ---- | ---- | ---- | ---- | ---- | 
 | 행 | Row | 세로 (위 ↕ 아래) | 앞쪽 (`x`) | 바깥쪽 대괄호 안의 줄 개수 | 
 | 열 | Column | 가로 (좌 ↔ 우) | 뒤쪽 (`y`) | 각 줄 안에 들어있는 요소 개수 | 

[//]: # (heading_4 is not supported)


```python
t3 = torch.rand(4, 2, 3)
print(t3.shape)  # torch.Size([4, 2, 3])
```

- 가장 바깥쪽 대괄호 안에 **"2행 3열짜리 표"가 4개** 들어있다고 읽으면 됩니다.

- 실무에서는 흔히 `(배치 크기, 행, 열)` = `(Batch, Row, Column)` 순서로 해석하는 경우가 많습니다. (딥러닝 이미지 데이터라면 `(Batch, Channel, Height, Width)` 처럼 4차원까지 늘어나기도 합니다.)

- 즉 **가장 왼쪽 숫자로 갈수록 "더 큰 묶음", 가장 오른쪽 숫자로 갈수록 "가장 안쪽의 개별 요소 개수"** 라는 규칙은 차원이 늘어나도 동일하게 적용됩니다.

[//]: # (heading_4 is not supported)

- `torch.rand(3, 2)`와 `torch.rand(2, 3)`을 헷갈려서 shape가 안 맞는 에러(`RuntimeError: mat1 and mat2 shapes cannot be multiplied`)를 자주 만납니다. 코드를 짜기 전에 항상 "몇 행 몇 열이 필요한지" 먼저 종이에 적어보는 습관을 들이면 좋습니다.

- `.shape`와 `.size()`는 결과가 동일합니다. `.shape`는 속성(attribute), `.size()`는 메서드(method)라는 차이만 있을 뿐 값은 같습니다.

---

[//]: # (heading_4 is not supported)

- [PyTorch-Tensor-Data-Modeling-MOC](https://app.notion.com/p/PyTorch-Tensor-Data-Modeling-MOC.md)

- [Tensor-Attributes-Cheatsheet](https://app.notion.com/p/Tensor-Attributes-Cheatsheet.md)

- [Tensor-Creation-Functions](https://app.notion.com/p/Tensor-Creation-Functions.md)

- [Tensor-Operations-Broadcasting](https://app.notion.com/p/Tensor-Operations-Broadcasting.md)

