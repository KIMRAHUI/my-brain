---
notion-url: https://app.notion.com/p/Computer-Vision-Newcomer-Guide-3ca77fa64f7f80749836d77244e74736
title: Computer Vision Newcomer Guide
date: '2026-08-28 08:59:00.000'
from_notion: https://app.notion.com/p/Computer-Vision-Newcomer-Guide-3ca77fa64f7f80749836d77244e74736
author: rahui kim
last_edited_time: '2026-08-28 09:07:00.000'
---
# Computer Vision Newcomer Guide

## 1. 데이터 및 파이프라인 (Data & Pipeline)

- **데이터셋 경로(Path) 하드코딩 금지**

	- `C:/Users/...` 형태의 경로는 서버나 동료의 환경에서 에러를 유발합니다. `pathlib`을 활용해 운영체제에 독립적인 안전한 경로를 설정하고, 외부 설정을 통해 주입받도록 구성합니다.

- **학습 전 **`**Dataset.__getitem__**`** 시각화 디버깅 습관화**

	- 데이터 로더가 정상 작동하는지 맹신하지 말고, 배치를 꺼내어 이미지와 바운딩 박스/마스크가 픽셀 단위로 정확히 일치하는지 눈으로 직접 확인해야 합니다.

- **메모리 누수(Memory Leak) 방지**

	- 학습 루프 내에서 PyTorch 텐서를 그대로 리스트에 누적하면 VRAM이나 RAM이 고갈됩니다. 반드시 파이썬 기본 자료형(`.item()` 등)으로 변환하여 로깅해야 합니다.


```python
import cv2
from pathlib import Path
from torch.utils.data import Dataset

# 1. Pathlib을 이용한 안전한 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"

assert IMAGE_DIR.exists(), f"이미지 디렉토리가 존재하지 않습니다: {IMAGE_DIR}"

# 2. 데이터셋 구현 및 시각화 검증 스크립트
class SafeVisionDataset(Dataset):

  def __init__(self, image_dir, annotation_dict):
    self.image_dir = Path(image_dir)
    self.image_ids = list(annotation_dict.keys())
    self.annotations = annotation_dict

  def __len__(self):
    return len(self.image_ids)

  def __getitem__(self, idx):
    image_id = self.image_ids[idx]
    ann = self.annotations[image_id]

    image_path = self.image_dir / ann["file_name"]
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = ann["boxes"]  # [x1, y1, x2, y2] 형식 가정

    # 디버깅 시각화 검증
    debug_img = image.copy()
    for box in boxes:
      x1, y1, x2, y2 = map(int, box)
      cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image, boxes
```

### 1-1. 어노테이션 파일 파싱 및 데이터 구조화 꿀팁

- **판단 우선순위:** 가장 먼저 어노테이션 파일의 형식(JSON, XML, CSV, TXT 등)과 내부 스키마(키 값 구조)를 파악하세요.

- **딕셔너리 기반 캐싱:** 이미지 파일명이나 ID를 키(`key`)로, 바운딩 박스나 클래스 라벨을 값(`value`)으로 갖는 딕셔너리를 구축하면 데이터 접근 속도와 직관성이 극대화됩니다.

- **데이터 클래스(DataClass) 활용:** 파이썬의 `dataclasses` 또는 `collections.namedtuple`을 사용하면 딕셔너리의 오타 실수를 줄이고 자동 완성 기능을 활용할 수 있습니다.


```python
from dataclasses import dataclass
from typing import List

@dataclass
class Annotation:
    image_id: str
    boxes: List[list]
    labels: List[int]
```

## 2. 모델링 및 학습 (Modeling & Training)

### 2-1. 모델 입력 및 출력 파악 전략

- **더미 텐서(Dummy Tensor) 활용:** 실제 데이터를 넣기 전에 `torch.randn` 등을 이용해 임의의 입력 텐서를 만들고 모델에 통과시켜 에러 없이 출력이 나오는지 확인하세요.


```python
import torch
dummy_input = torch.randn(1, 3, 224, 224) # [Batch, Channel, Height, Width] 예시
output = model(dummy_input)
print(type(output), output.shape if hasattr(output, 'shape') else [o.shape for o in output])
```

- **출력 형태(Shape)와 타입 검증:** 모델의 출력 결과가 딕셔너리인지, 튜플인지, 혹은 단일 텐서인지 확인하고 각 차원이 무엇을 의미하는지(`[batch_size, num_classes]` 등) 주석으로 명시하세요.

- **디버깅 시각화:** 리서치팀 모델의 출력을 받아 정답(Annotation)과 비교하는 시각화 코드를 가장 먼저 짜두면, 모델의 입출력 구조가 올바르게 파악되었는지 즉시 검증할 수 있습니다.

### 2-2. 오버피팅 테스트 (Sanity Check) 선행

- 전체 데이터를 투입하기 전, 아주 소수의 샘플(1~2장)만으로 모델이 과적합되어 Loss가 0으로 수렴하는지 먼저 확인합니다. 이 과정에서 문제가 발생한다면 파이프라인이나 모델 구현에 결함이 있는 것입니다.


```python
import torch
import torch.nn as nn
import torch.optim as optim

def run_sanity_check(model, train_loader, device):
  model.train()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.CrossEntropyLoss()

  # 단일 배치 추출 후 강제 오버피팅 테스트
  images, targets = next(iter(train_loader))
  images, targets = images.to(device), targets.to(device)

  for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f"Sanity Check Epoch {epoch+1:02d} | Loss: {loss.item():.6f}")
    if loss.item() < 1e-4:
      print("검증 완료: 모델이 단일 배치를 완벽하게 학습합니다.")
      break
```

### 2-3. Gradient Flow 및 NaN 추적

- TensorBoard나 Wandb를 활용해 학습 초기 `grad_norm`을 모니터링하고, 필요 시 레이어 출력값에 예외 처리를 걸어 수치 불안정성을 추적합니다.

### 2-4. Pre-trained 모델 가중치 로드 상태 검증

- 백본 로드 시 가중치가 의도대로 적용되었는지, 동결(Freeze) 상태가 올바르게 설정되었는지 `requires_grad` 속성을 통해 코드로 직접 검증합니다.

## 3. 코드 작성 및 협업 (Code & Collaboration)

- **변수명 및 텐서 Shape 주석 명시**

	- 차원 변환이 잦은 딥러닝 코드 특성상, 주요 연산 구간마다 텐서의 형태를 주석으로 남겨 가독성을 높입니다.

- **체계적인 실험 관리 (Experiment Tracking)**

	- 모호한 이름의 모델 파일 대신 Wandb나 MLflow를 도입하여 하이퍼파라미터, 손실 곡선, 결과물을 기록합니다.

- **단위별 명확한 Git Commit**

	- 변경 사항을 뭉뚱그려 올리지 않고, 기능 추가나 버그 수정 단위로 명확한 커밋 메시지를 작성합니다.


```python
import torch

def forward_pass_with_tracking(model, x, criterion, labels):
  # x shape: [Batch, Channel, Height, Width] -> [32, 3, 224, 224]
  B, C, H, W = x.shape

  features = model.backbone(x)
  # features shape: [32, 512, 7, 7]

  features = features.view(B, -1)
  # flatten shape: [32, 25088]

  out = model.classifier(features)
  # out shape: [32, Num_Classes] -> [32, 10]

  # 메모리 누수 방지 로깅 (.item() 사용)
  loss_val = criterion(out, labels)
  running_loss = loss_val.item()

  return out, running_loss
```

## 4. 실무 비주얼 프리패스 태도 (Mindset)

- **에러 발생 시 시도한 내역 정리하여 문의하기**

	- 단순히 에러 화면을 공유하기보다, 원인을 조사하고 시도해 본 해결 방법을 함께 제시합니다.

- **오픈소스 및 레퍼런스 구현체 분석 역량 갖추기**

	- 논문 기반의 공식 GitHub 구현체나 README의 추론(Inference) 코드를 먼저 실행해 보며 구조를 파악하는 습관을 들입니다.

<br/>

