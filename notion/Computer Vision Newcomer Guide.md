# Computer Vision Newcomer Guide

Computer Vision Newcomer Guide

1. 데이터 및 파이프라인 (Data & Pipeline)

신입 시절 가장 사고가 많이 나고, 선배들의 손을 가장 많이 타는 영역입니다.

데이터셋 경로(Path) 하드코딩 금지

학습 전 Dataset.__getitem__ 시각화 디버깅 습관화

메모리 누수(Memory Leak) 방지

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

1-1. 어노테이션 파일 파싱 및 데이터 구조화 꿀팁

판단 우선순위: 가장 먼저 어노테이션 파일의 형식(JSON, XML, CSV, TXT 등)과 내부 스키마(키 값 구조)를 파악하세요.

딕셔너리 기반 캐싱: 이미지 파일명이나 ID를 키(key)로, 바운딩 박스나 클래스 라벨을 값(value)으로 갖는 딕셔너리를 구축하면 데이터 접근 속도와 직관성이 극대화됩니다.

데이터 클래스(DataClass) 활용: 파이썬의 dataclasses 또는 collections.namedtuple을 사용하면 딕셔너리의 오타 실수를 줄이고 자동 완성 기능을 활용할 수 있습니다.

from dataclasses import dataclass
from typing import List

@dataclass
class Annotation:
    image_id: str
    boxes: List[list]
    labels: List[int]

2. 모델링 및 학습 (Modeling & Training)

모델을 다룰 때 감으로 접근하지 않고, 공학적 검증 단계를 거쳐야 합니다.

2-1. 모델 입력 및 출력 파악 전략

더미 텐서(Dummy Tensor) 활용: 실제 데이터를 넣기 전에 torch.randn 등을 이용해 임의의 입력 텐서를 만들고 모델에 통과시켜 에러 없이 출력이 나오는지 확인하세요.

import torch
dummy_input = torch.randn(1, 3, 224, 224) # [Batch, Channel, Height, Width] 예시
output = model(dummy_input)
print(type(output), output.shape if hasattr(output, 'shape') else [o.shape for o in output])

출력 형태(Shape)와 타입 검증: 모델의 출력 결과가 딕셔너리인지, 튜플인지, 혹은 단일 텐서인지 확인하고 각 차원이 무엇을 의미하는지([batch_size, num_classes] 등) 주석으로 명시하세요.

디버깅 시각화: 리서치팀 모델의 출력을 받아 정답(Annotation)과 비교하는 시각화 코드를 가장 먼저 짜두면, 모델의 입출력 구조가 올바르게 파악되었는지 즉시 검증할 수 있습니다.

2-2. 오버피팅 테스트 (Sanity Check) 선행

전체 데이터를 투입하기 전, 아주 소수의 샘플(1~2장)만으로 모델이 과적합되어 Loss가 0으로 수렴하는지 먼저 확인합니다. 이 과정에서 문제가 발생한다면 파이프라인이나 모델 구현에 결함이 있는 것입니다.

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

2-3. Gradient Flow 및 NaN 추적

TensorBoard나 Wandb를 활용해 학습 초기 grad_norm을 모니터링하고, 필요 시 레이어 출력값에 예외 처리를 걸어 수치 불안정성을 추적합니다.

2-4. Pre-trained 모델 가중치 로드 상태 검증

백본 로드 시 가중치가 의도대로 적용되었는지, 동결(Freeze) 상태가 올바르게 설정되었는지 requires_grad 속성을 통해 코드로 직접 검증합니다.

3. 코드 작성 및 협업 (Code & Collaboration)

개발 역량만큼이나 팀원들과 협업하는 방식이 평가에 큰 영향을 줍니다.

변수명 및 텐서 Shape 주석 명시

체계적인 실험 관리 (Experiment Tracking)

단위별 명확한 Git Commit

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

4. 실무 비주얼 프리패스 태도 (Mindset)

기술 외적으로 신뢰를 얻는 태도적 노하우입니다.

에러 발생 시 시도한 내역 정리하여 문의하기

오픈소스 및 레퍼런스 구현체 분석 역량 갖추기

AI 모델 커스텀 및 이식 마스터 가이드: 어떤 복잡한 아키텍처라도 10분 만에 정복하는 실전 노하우

복잡한 딥러닝 오픈소스 모델(예: Torchvision Detection 등)을 가져와 내 프로젝트나 서비스에 맞게 이식해야 할 때, 코드를 처음부터 끝까지 분석하는 것은 시간 낭비입니다. 어떤 모델을 마주하더라도 뼈대와 출력을 분리해서 파악하는 핵심 관점과 교체 노하우를 정리했습니다.

1. 가장 먼저 봐야 하는 3대 관점 (입력, 백본, 출력)

데이터 인터페이스(Input/Target Formats):

사전 학습된 뼈대 (Backbone):

최종 예측기 (Head / Predictor):

2. 실전 모델 교체 및 커스텀 노하우 (Step-by-Step)

하드코딩 금지, 동적 속성 추출 활용하기

클래스 개수 계산 공식 (num_classes = 실제 객체 수 + 1)

멀티태스크 예측기 각각의 차원 개별 맞춤

3. AI 엔지니어의 핵심 마인드셋

거대한 오픈소스 모델은 '블록 조립 키트'로 바라보아야 합니다.

모든 소스 코드를 이해하려 하지 말고, "외부에서 데이터를 받아오는 입구(Dataset)", "특징을 추출하는 몸통(Backbone)", "원하는 결과로 뱉어내는 출구(Predictor Head)" 이 3가지 연결고리만 찾아내어 내 서비스 요구사항에 맞게 끼워 맞추는 능력이 곧 엔지니어의 핵심 역량입니다.

