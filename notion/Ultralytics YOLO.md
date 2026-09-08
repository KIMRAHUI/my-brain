# Ultralytics YOLO

참고: Ultralytics YOLO 실전 활용 (라이브러리 기반)

[!info] 이 노트는
03~05 노트에서 직접 구현한 YOLO 아키텍처와 달리, 실무에서는 ultralytics 패키지로
pretrained 모델을 바로 불러와 추론/학습한다는 것을 보여주는 실전 예제입니다.
"이론은 직접 구현으로, 실무는 라이브러리로" 라는 관점 차이를 이해하는 것이 핵심.

참고 링크

Ultralytics YOLOv3 문서: https://docs.ultralytics.com/ko/models/yolov3/

훈련 가이드: https://docs.ultralytics.com/ko/modes/train/#train-settings

Colab 튜토리얼: https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb

1. Pretrained 모델 로드 (직접 구현 vs 3줄 로드)

from ultralytics import YOLO

# COCO로 사전학습된 YOLOv3u 모델을 곧바로 로드
# (참고: "u"는 ultralytics가 anchor-free 방식으로 개선한 버전을 의미)
model = YOLO("yolov3u.pt")

# 모델 구조/파라미터 수 등 정보 출력
model.info()

[!tip] 05_YOLOv3와 비교
우리가 직접 구현한 Darknet53 + FPN Neck + 3 Head 전체가 YOLO("yolov3u.pt")
한 줄에 캡슐화되어 있음. 프로젝트에서 시간이 부족하면 이 라이브러리를 우선 사용하고,
커스터마이징이 필요할 때만 직접 구현부를 참고하는 전략이 효율적.

2. 비디오 프레임 단위 추론 + 박스 시각화

from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow
import torch

model = YOLO("yolov3u.pt")
model.info()

# 비디오 캡처 초기화
capture = cv2.VideoCapture("/content/dog.mp4")
if not capture.isOpened():
    print("비디오 파일을 열 수 없습니다. 파일 경로를 확인하세요.")

# 한 프레임을 입력받아 예측 결과(result 객체)를 반환하는 함수
def predict(frame, iou=0.7, conf=0.25):
    results = model(
        source=frame,
        device="0" if torch.cuda.is_available() else "cpu",
        iou=iou,     # NMS IoU threshold
        conf=conf,   # confidence threshold
        verbose=False
    )
    result = results[0]   # 배치 중 첫 번째 결과 (단일 프레임이므로 결과도 1개)
    return result

# 예측 결과를 프레임 위에 사각형으로 그리는 함수
def draw_box(result, frame):
    if not hasattr(result, "boxes") or len(result.boxes) == 0:
        return frame
    for boxes in result.boxes:
        # boxes.data: (x1, y1, x2, y2, score, class) 형태
        data = boxes.data.squeeze().cpu().numpy()
        if data.ndim == 0 or data.size != 6:
            continue
        x1, y1, x2, y2, score, cls = data
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return frame

# 첫 프레임으로 파이프라인 테스트
ret, frame = capture.read()
if not ret or frame is None:
    print("프레임을 읽어오지 못했습니다.")
else:
    result = predict(frame)
    result_frame = draw_box(result, frame)
    cv2_imshow(result_frame)

capture.release()
cv2.destroyAllWindows()

3. 비디오 전체 처리 + 결과 저장 (VideoWriter)

from ultralytics import YOLO
import cv2, torch

model = YOLO("yolov3u.pt")

input_path = "/content/dog.mp4"
capture = cv2.VideoCapture(input_path)

# 원본 비디오의 속성(해상도, FPS)을 그대로 유지해 출력 비디오 생성
width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = capture.get(cv2.CAP_PROP_FPS)

output_path = "/content/output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def predict(frame, iou=0.7, conf=0.25):
    results = model(source=frame, device="0" if torch.cuda.is_available() else "cpu",
                     iou=iou, conf=conf, verbose=False)
    return results[0]

def draw_box(result, frame):
    if not hasattr(result, "boxes") or len(result.boxes) == 0:
        return frame
    for boxes in result.boxes:
        data = boxes.data.squeeze().cpu().numpy()
        if data.ndim == 0 or data.size != 6:
            continue
        x1, y1, x2, y2, score, cls = data
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return frame

# 비디오 전체를 프레임 단위로 순회하며 처리
frame_count = 0
while True:
    # 영상 끝에 도달하면 처음으로 되감기 (루프 재생)
    if capture.get(cv2.CAP_PROP_POS_FRAMES) >= capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = capture.read()
    if not ret or frame is None:
        break

    result = predict(frame)
    result_frame = draw_box(result, frame)
    out_writer.write(result_frame)   # 결과 프레임을 출력 비디오 파일에 기록
    frame_count += 1

[!warning] 주의
위 코드는 capture.get(...) >= capture.get(...)로 무한 루프(재생 반복)를 만듭니다.
실제 프로젝트에서는 break 조건을 명확히 설정해야 무한 실행을 방지할 수 있습니다.

4. 공식 저장소로 직접 학습 (CLI 방식)

# ultralytics의 원조 YOLOv3 저장소 클론
!git clone https://github.com/ultralytics/yolov3
!cd yolov3; pip install -qr requirements.txt

import torch
from IPython.display import Image, clear_output

clear_output()
print(f"Setup complete. Using torch {torch.__version__} "
      f"({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

%cd yolov3
# --img: 입력 이미지 크기, --batch: 배치 크기, --epochs: 학습 epoch
# --data: 데이터셋 설정(yaml), --weights: 초기 가중치, --nosave: 중간 체크포인트 저장 안 함
!python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov3.pt --nosave

[!tip] 데이터 커스터마이징
--data coco128.yaml 부분을 나만의 데이터셋 yaml로 교체하면 바로 커스텀 학습 가능.
yaml에는 train/val 경로, nc(클래스 수), names(클래스 이름 리스트)를 정의.

🎯 프로젝트 대비 체크리스트

pip install ultralytics만으로 pretrained YOLO를 불러와 추론할 수 있는가?

model(source=..., iou=..., conf=...) 파라미터의 의미(NMS threshold, confidence threshold)를 아는가?

비디오 입력 시 cv2.VideoCapture + cv2.VideoWriter로 프레임 단위 처리 파이프라인을 짤 수 있는가?

CLI 학습(train.py --data custom.yaml)으로 커스텀 데이터셋 학습을 실행할 수 있는가?

"직접 구현 이해" ↔ "라이브러리로 실전 적용" 두 갈래를 프로젝트 상황에 맞게 선택할 수 있는가?

관련 노트

[[05_YOLOv3]] — 이 노트에서 라이브러리로 감싸진 내부 구조(Darknet-53, FPN neck)

[[00_Object-Detection-MOC]] — 전체 Object Detection 개념 지도

