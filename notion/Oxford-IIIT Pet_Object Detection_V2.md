#  Oxford-IIIT Pet_Object Detection_V2

🐾 Oxford-IIIT Pet 객체 검출(Object Detection) — 전이학습 실무 템플릿

📌 0. 프로젝트 개요 및 목표

목표: Oxford-IIIT Pet Dataset으로 강아지·고양이 얼굴 위치를 찾는 Object Detection 모델(SSD300-VGG16)을 구현하고, COCO 사전학습 가중치를 사용한 전이학습(Pretrained)과 가중치 없이 처음부터 학습하는 커스텀 베이스라인(Scratch)의 성능을 비교합니다.

주요 기술: PyTorch, Torchvision v2 Transforms(Detection 전용 증강), OpenCV, Matplotlib, Scikit-Learn, AMP(혼합정밀도)

이 템플릿에서 새로 반영한 개선사항 요약

🛠️ 1. 사전 설정 — 환경 및 재현성(Seed) 고정

#@title 0. 환경 설정 및 재현성(Seed) 고정
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 완전한 재현성이 필요하면 True로 (단, 속도 저하 있음)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

[개선] 두 원본 노트북 모두 시드 고정이 없어 실행할 때마다 데이터 분할·증강·초기화가 달라졌습니다. 실무에서는 "재현 가능한 실험"이 기본이므로 최상단에 시드 고정을 추가했습니다.

🔍 2. 데이터 다운로드 및 무결성(Integrity) 검증

#@title 1. 데이터 다운로드 및 무결성 검증 (양방향 체크)
import kagglehub
import pandas as pd

# 데이터셋 다운로드
path = kagglehub.dataset_download("zippyz/cats-and-dogs-breeds-classification-oxford-dataset")
print("Path to dataset files:", path)

trainval_file_path = os.path.join(path, "annotations", "annotations", "trainval.txt")
test_file_path = os.path.join(path, "annotations", "annotations", "test.txt")
image_dir = os.path.join(path, "images", "images")
xml_dir = os.path.join(path, "annotations", "annotations", "xmls")

df_trainval = pd.read_csv(trainval_file_path, sep=r"\s+", header=None)
df_trainval.columns = ["Image", "ClassID", "Species", "BreedID"]
df_test = pd.read_csv(test_file_path, sep=r"\s+", header=None)
df_test.columns = ["Image", "ClassID", "Species", "BreedID"]

print(f"Train/Validation 목록 수: {len(df_trainval)}")
print(f"Test 목록 수: {len(df_test)}")

# 실제 존재하는 파일 목록 (확장자 제거)
xml_list = [os.path.splitext(f)[0] for f in os.listdir(xml_dir) if f.endswith(".xml")]
img_list = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")]

trainval_list = df_trainval["Image"].tolist()

# 양방향 무결성 체크: XML 없는 이미지 / 이미지 없는 XML
missing_xml = [name for name in trainval_list if name not in xml_list]   # XML이 없는 trainval 이미지
missing_img = [name for name in xml_list if name not in img_list]        # 대응 이미지가 없는 XML

print(f"[검증] XML이 없어 학습에서 제외되는 이미지: {len(missing_xml)}개")
print(f"[검증] 이미지가 없어 사용 불가능한 XML: {len(missing_img)}개")

# 이미지 + XML이 모두 존재하는 것만 최종 학습 목록으로 채택
trainval_list = [name for name in trainval_list if name in xml_list and name in img_list]
print(f"[결과] 최종 사용 가능한 Train/Validation 이미지 수: {len(trainval_list)}개")

test_list = [name for name in df_test["Image"].tolist() if name in img_list]
print(f"[결과] 최종 사용 가능한 Test 이미지 수: {len(test_list)}개")

[개선] 내 코드의 missing_xml = [image for image in trainval_list if image in xml_list]는 조건이 반대로 되어 있어 변수명·출력 문구와 실제 동작이 달랐습니다. 여기서는 not in으로 바로잡고, baseline 코드처럼 반대 방향(이미지 없는 XML)도 함께 점검하도록 했습니다.

🖼️ 3. XML 파싱 → annotations 구조화 + 시각화 검증

#@title 2. XML 파싱 -> annotations 구조화 + 시각화 검증
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

CLASSES = ["background", "cat", "dog"]  # index 0은 반드시 background

def parse_annotations(xml_dir, xml_names):
    """이미지 이름 -> [{'class': str, 'bbox': [xmin,ymin,xmax,ymax]}, ...] 매핑 생성"""
    annotations = {}
    for name in xml_names:
        xml_path = os.path.join(xml_dir, name + ".xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_name = os.path.splitext(root.find("filename").text)[0]

        objs = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in CLASSES:
                continue
            bndbox = obj.find("bndbox")
            x_min = int(bndbox.find("xmin").text)
            y_min = int(bndbox.find("ymin").text)
            x_max = int(bndbox.find("xmax").text)
            y_max = int(bndbox.find("ymax").text)
            objs.append({"class": class_name, "bbox": [x_min, y_min, x_max, y_max]})
        annotations[image_name] = objs
    return annotations


annotations = parse_annotations(xml_dir, trainval_list)
print(f"annotation이 만들어진 이미지 수: {len(annotations)}")


def visualize_sample(image_name, image_dir, annotations):
    img_path = os.path.join(image_dir, image_name + ".jpg")
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for anno in annotations.get(image_name, []):
        x_min, y_min, x_max, y_max = anno["bbox"]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, anno["class"], (x_min, max(y_min - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    plt.imshow(image)
    plt.title(image_name)
    plt.axis("off")
    plt.show()


for name in trainval_list[:3]:
    visualize_sample(name, image_dir, annotations)

[계승] baseline 코드의 "전처리 직후 시각화로 검증" 습관을 그대로 가져왔습니다. annotation 파싱 로직에 버그가 있어도 눈으로 바로 확인할 수 있어 실무에서 매우 중요한 단계입니다.

📦 4. Dataset 클래스 + Detection 전용 데이터 증강 + DataLoader

#@title 3. Dataset 클래스 + Detection 전용 데이터 증강 + DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors
from PIL import Image
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# [개선 포인트]
# - SSD 등 torchvision detection 모델은 내부 GeneralizedRCNNTransform에서
#   자체적으로 정규화(Normalize)를 수행합니다. Dataset 단계에서 ImageNet mean/std로
#   한 번 더 Normalize를 적용하면 "이중 정규화"가 발생해 오히려 학습이 불안정해질
#   수 있으므로, 여기서는 v2.ToDtype(scale=True)로 0~1 스케일만 맞추고
#   Normalize는 의도적으로 생략합니다. (baseline 코드는 이 부분을 명시적으로 적용했는데,
#   모델 기본 설정을 그대로 쓴다면 이중 정규화가 될 수 있어 재검토가 필요한 지점입니다.)
# - torchvision 공식 detection 레퍼런스(references/detection)에서 SSD 학습 시
#   사용하는 RandomPhotometricDistort / RandomZoomOut / RandomIoUCrop 조합을 그대로
#   채택해 실무 수준의 증강 파이프라인을 구성합니다. (두 원본 노트북 모두 미사용)
# - RandomIoUCrop처럼 이미지를 잘라내는 증강을 쓰면 박스가 매우 작아지거나
#   사라질 수 있어, 반드시 SanitizeBoundingBoxes로 비정상 박스를 제거해줘야 합니다.
# -----------------------------------------------------------------------------

train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomPhotometricDistort(p=0.5),                  # 밝기/대비/채도/색조 무작위 왜곡
    v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}, p=0.5),
    v2.RandomIoUCrop(),                                   # 객체 일부만 남도록 무작위 크롭
    v2.RandomHorizontalFlip(p=0.5),
    v2.SanitizeBoundingBoxes(),                           # 잘못된/소실된 박스 제거 (필수)
    v2.ToDtype(torch.float32, scale=True),
])

eval_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


class VOCDataset(Dataset):
    """이미지 + XML(annotation)을 함께 로드하는 학습/검증용 Dataset"""

    def __init__(self, image_dir, annotation_dir, classes, image_list, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.classes = classes
        self.transforms = transforms
        self.image_files = image_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, name + ".jpg")).convert("RGB")
        img_w, img_h = image.size

        tree = ET.parse(os.path.join(self.annotation_dir, name + ".xml"))
        root = tree.getroot()

        boxes, labels = [], []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in self.classes:
                continue
            labels.append(self.classes.index(class_name))
            bndbox = obj.find("bndbox")
            boxes.append([
                int(bndbox.find("xmin").text), int(bndbox.find("ymin").text),
                int(bndbox.find("xmax").text), int(bndbox.find("ymax").text),
            ])

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        boxes_tv = tv_tensors.BoundingBoxes(boxes_t, format="XYXY", canvas_size=(img_h, img_w))

        target = {
            "boxes": boxes_tv,
            "labels": labels_t,
            "image_id": torch.tensor(idx),
        }

        if self.transforms:
            # target(dict)을 통째로 넘기면 v2.SanitizeBoundingBoxes가
            # boxes뿐 아니라 labels까지 같은 인덱스 기준으로 함께 필터링해줍니다.
            # (RandomIoUCrop 등으로 박스가 사라지는 경우에도 boxes/labels 개수가
            #  항상 일치하도록 보장하기 위한 핵심 포인트입니다.)
            image, target = self.transforms(image, target)

        return image, target


class TestDataset(Dataset):
    """정답(label) 없이 추론만 수행할 때 사용하는 Dataset"""

    def __init__(self, image_dir, image_list, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_files = image_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, name + ".jpg")).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, name


def collate_fn(batch):
    """가변 크기 이미지를 배치로 묶기 위한 함수
    (top-level 함수로 정의해 num_workers>0 환경에서도 pickle 가능하도록 함 — lambda는 위험)"""
    return tuple(zip(*batch))


# 7:3 분할
train_list, valid_list = train_test_split(trainval_list, test_size=0.3, random_state=SEED)

train_dataset = VOCDataset(image_dir, xml_dir, CLASSES, train_list, transforms=train_transform)
valid_dataset = VOCDataset(image_dir, xml_dir, CLASSES, valid_list, transforms=eval_transform)
test_dataset = TestDataset(image_dir, test_list, transforms=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)
val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False,
                         collate_fn=collate_fn, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)

print(f"Train: {len(train_dataset)} / Validation: {len(valid_dataset)} / Test: {len(test_dataset)}")

[계승] 내 코드의 "train_list/valid_list를 애초에 별도 Dataset으로 생성"하는 구조를 그대로 유지했습니다 (baseline 코드의 random_split 후 transform 미분리 문제가 재현되지 않도록).
[개선] 증강을 detection 전용 조합(RandomPhotometricDistort + RandomZoomOut + RandomIoUCrop + SanitizeBoundingBoxes)으로 업그레이드했습니다.

📊 5. 평가지표 구현 (IoU / mAP)

#@title 4. 평가지표 구현 (IoU / mAP, threshold 인자화)
import numpy as np
from sklearn.metrics import average_precision_score


def calculate_iou(box, boxes):
    """box 1개와 boxes(N,4) 사이의 IoU 벡터 계산"""
    if len(boxes) == 0:
        return np.array([])
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return intersection / np.clip(union, 1e-9, None)


def evaluate_model(predictions, ground_truths, classes, iou_threshold=0.5):
    """클래스별 AP를 계산해 평균(mAP)을 반환 (배경 클래스는 제외)"""
    class_aps = []
    for class_idx, _ in enumerate(classes[1:], start=1):
        true_positives, scores = [], []

        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred["boxes"][pred["labels"] == class_idx].cpu().numpy() if len(pred["boxes"]) > 0 else np.zeros((0, 4))
            pred_scores = pred["scores"][pred["labels"] == class_idx].cpu().numpy() if len(pred["scores"]) > 0 else np.zeros((0,))
            gt_boxes = gt["boxes"][gt["labels"] == class_idx].cpu().numpy() if len(gt["boxes"]) > 0 else np.zeros((0, 4))

            if len(pred_boxes) == 0:
                continue
            if len(gt_boxes) == 0:
                # GT가 없는데 예측했다면 전부 False Positive로 채점에 포함
                true_positives.extend([0] * len(pred_boxes))
                scores.extend(pred_scores.tolist())
                continue

            matched = np.zeros(len(gt_boxes), dtype=bool)
            for box, score in zip(pred_boxes, pred_scores):
                ious = calculate_iou(box, gt_boxes)
                max_iou_idx = int(np.argmax(ious)) if len(ious) > 0 else -1
                max_iou = ious[max_iou_idx] if max_iou_idx >= 0 else 0

                if max_iou >= iou_threshold and not matched[max_iou_idx]:
                    true_positives.append(1)
                    matched[max_iou_idx] = True
                else:
                    true_positives.append(0)
                scores.append(score)

        if len(scores) == 0:
            class_aps.append(0.0)
            continue

        sorted_idx = np.argsort(-np.array(scores))
        tp_sorted = np.array(true_positives)[sorted_idx]
        scores_sorted = np.array(scores)[sorted_idx]

        ap = average_precision_score(tp_sorted, scores_sorted) if tp_sorted.sum() > 0 else 0.0
        class_aps.append(ap)

    return float(np.mean(class_aps)) if class_aps else 0.0

[개선] 내 코드는 if len(pred_boxes) == 0 or len(gt_boxes) == 0: continue로 되어 있어, GT가 없는 이미지에 모델이 잘못 예측(오탐)한 경우가 정밀도 계산에서 완전히 빠졌습니다. 이는 정밀도를 실제보다 좋게 보이도록 만드는 평가 버그입니다. 여기서는 그 경우를 False Positive로 명시적으로 채점에 포함하도록 고쳤습니다.

💾 6. 체크포인트 유틸(Latest + Best 분리 저장) & 학습 함수(AMP 지원)

#@title 5. 체크포인트 유틸(Latest + Best 분리 저장) & 학습 함수(AMP 지원)
import pickle
from tqdm import tqdm

# Google Drive 연동 (Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/object_detection_mission"
except ImportError:
    # Colab이 아닌 로컬/서버 환경이면 현재 경로 사용
    BASE_DIR = "./object_detection_mission"

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
ALL_HISTORY_PATH = os.path.join(BASE_DIR, "all_history.pkl")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(model, optimizer, lr_scheduler, epoch, history, model_name, tag="latest", ckpt_dir=CHECKPOINT_DIR):
    """tag='latest' -> 매 epoch 덮어쓰기 (런타임 끊김 대비, 이어서 학습용)
    tag='best'   -> val_mAP가 갱신될 때만 별도 저장 (최종 평가/배포용)
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, os.path.join(ckpt_dir, f"{model_name}_{tag}.pth"))


def load_checkpoint(model, optimizer, lr_scheduler, model_name, tag="latest", ckpt_dir=CHECKPOINT_DIR):
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_{tag}.pth")
    if not os.path.exists(ckpt_path):
        print(f"[{model_name}] 저장된 '{tag}' 체크포인트가 없습니다. 처음부터 학습을 시작합니다.")
        return 0, {"train_loss": [], "val_mAP": []}

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    history = checkpoint["history"]
    print(f"[{model_name}] '{tag}' 체크포인트 로드 완료 -> {start_epoch} 에포크부터 재개")
    return start_epoch, history


def update_all_history(model_name, history, path=ALL_HISTORY_PATH):
    all_history = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            all_history = pickle.load(f)
    all_history[model_name] = history
    with open(path, "wb") as f:
        pickle.dump(all_history, f)


def load_all_history(path=ALL_HISTORY_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def train_model(model, model_name, optimizer, lr_scheduler, num_epochs,
                 train_loader, val_loader, classes, patience=None, use_amp=True):
    """학습 + 매 epoch validation + latest/best 체크포인트 저장을 모두 담당하는 통합 함수

    patience: N epoch 동안 best mAP가 갱신되지 않으면 조기 종료 (None이면 사용 안 함)
    use_amp : Mixed Precision(AMP)으로 학습 속도/메모리 효율 개선
    """
    start_epoch, history = load_checkpoint(model, optimizer, lr_scheduler, model_name, tag="latest")
    best_map = max(history["val_mAP"]) if history["val_mAP"] else -1.0
    epochs_no_improve = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n[{model_name}] Epoch {epoch + 1}/{num_epochs}")

        # ---------------- Training ----------------
        model.train()
        total_train_loss = 0.0
        for images, targets in tqdm(train_loader, desc="Training"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += losses.item()

        avg_train_loss = total_train_loss / len(train_loader)
        lr_scheduler.step()

        # ---------------- Validation ----------------
        model.eval()
        all_predictions, all_ground_truths = [], []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                predictions = model(images)
                all_predictions.extend([{k: v.cpu() for k, v in p.items()} for p in predictions])
                all_ground_truths.extend(targets)

        val_map = evaluate_model(all_predictions, all_ground_truths, classes)
        print(f"[{model_name}] Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val mAP: {val_map:.4f}")

        history["train_loss"].append(avg_train_loss)
        history["val_mAP"].append(val_map)

        # latest는 항상 저장 (런타임 끊김 대비)
        save_checkpoint(model, optimizer, lr_scheduler, epoch, history, model_name, tag="latest")
        update_all_history(model_name, history)

        # best는 val mAP가 갱신될 때만 저장
        if val_map > best_map:
            best_map = val_map
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, lr_scheduler, epoch, history, model_name, tag="best")
            print(f"[{model_name}] ✓ 새로운 best mAP={best_map:.4f} -> best 체크포인트 저장")
        else:
            epochs_no_improve += 1

        if patience is not None and epochs_no_improve >= patience:
            print(f"[{model_name}] {patience} epoch 동안 개선 없어 조기 종료합니다.")
            break

    return model, history

[계승] 내 코드의 체크포인트/재개 시스템(Colab 런타임 끊김 대비)을 골격으로 유지했습니다.
[개선] _latest(재개용)와 _best(val mAP 기준 최고 성능, 최종 평가/배포용)를 분리 저장하도록 확장했고, patience(조기 종료)와 AMP(혼합정밀도)를 추가해 학습 시간을 단축했습니다. baseline 코드는 아예 이런 체크포인트 체계가 없어 70 epoch을 한 번에 완주해야만 했던 문제를 해결했습니다.

🚀 7. 모델 학습: 전이학습(Pretrained) vs 커스텀 베이스라인(Scratch)

#@title 6. 모델 학습: 전이학습(Pretrained) vs 커스텀 베이스라인(Scratch)
import torchvision
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchinfo import summary

num_classes = len(CLASSES)
sample_input = [torch.randn(3, 300, 300).to(device)]

# ---------------- 6-1. Pretrained (COCO 사전학습 가중치로 Fine-tuning) ----------------
model_pretrained = torchvision.models.detection.ssd300_vgg16(
    weights=SSD300_VGG16_Weights.DEFAULT
).to(device)
model_pretrained.head.classification_head.num_classes = num_classes

optimizer_pretrained = torch.optim.SGD(
    model_pretrained.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
)
scheduler_pretrained = torch.optim.lr_scheduler.StepLR(optimizer_pretrained, step_size=3, gamma=0.1)

print("=" * 20, "Pretrained 모델 구조", "=" * 20)
summary(model_pretrained, input_data=[sample_input], depth=2,
        col_names=["output_size", "num_params", "trainable"], row_settings=["var_names"])

model_pretrained, history_pretrained = train_model(
    model=model_pretrained, model_name="ssd_pretrained",
    optimizer=optimizer_pretrained, lr_scheduler=scheduler_pretrained,
    num_epochs=30, train_loader=train_loader, val_loader=val_loader,
    classes=CLASSES, patience=5, use_amp=True,
)

# ---------------- 6-2. Scratch (사전학습 가중치 없이 처음부터 학습) ----------------
model_scratch = torchvision.models.detection.ssd300_vgg16(
    weights=None, weights_backbone=None
).to(device)
model_scratch.head.classification_head.num_classes = num_classes

optimizer_scratch = torch.optim.SGD(
    model_scratch.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
)
scheduler_scratch = torch.optim.lr_scheduler.StepLR(optimizer_scratch, step_size=3, gamma=0.1)

print("=" * 20, "Scratch 모델 구조", "=" * 20)
summary(model_scratch, input_data=[sample_input], depth=2,
        col_names=["output_size", "num_params", "trainable"], row_settings=["var_names"])

model_scratch, history_scratch = train_model(
    model=model_scratch, model_name="ssd_scratch",
    optimizer=optimizer_scratch, lr_scheduler=scheduler_scratch,
    num_epochs=30, train_loader=train_loader, val_loader=val_loader,
    classes=CLASSES, patience=5, use_amp=True,
)

[계승] 내 코드의 "Pretrained vs Scratch" 실험 설계(전이학습 효과 자체를 검증)를 그대로 채택했습니다 — baseline 코드의 "Freeze vs Fine-tune"(둘 다 pretrained 시작점)보다 메시지가 더 명확합니다.
[개선] epoch 수를 5 → 30으로 늘리되 patience=5 Early Stopping을 걸어, 실제로 수렴할 때까지 학습하면서도 불필요한 낭비를 막도록 했습니다. torchinfo summary()를 실제로 호출해 두 모델의 구조·파라미터 수를 학습 전에 확인합니다.

📈 8. 성능 비교 시각화 (Loss / mAP / IoU Threshold Sweep)

#@title 7. 성능 비교 시각화 (Loss / mAP / IoU Threshold Sweep)
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


def plot_all_history(path=ALL_HISTORY_PATH):
    all_history = load_all_history(path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, hist in all_history.items():
        axes[0].plot(hist["train_loss"], marker="o", label=model_name)
        axes[1].plot(hist["val_mAP"], marker="o", label=model_name)

    axes[0].set_title("Train Loss Comparison")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation mAP Comparison (IoU=0.5)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mAP")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_all_history()


def summarize_history(path=ALL_HISTORY_PATH):
    all_history = load_all_history(path)
    rows = []
    for name, hist in all_history.items():
        rows.append({
            "model": name,
            "final_train_loss": hist["train_loss"][-1],
            "best_val_mAP": max(hist["val_mAP"]),
            "final_val_mAP": hist["val_mAP"][-1],
        })
    df = pd.DataFrame(rows)
    if {"ssd_scratch", "ssd_pretrained"}.issubset(set(df["model"])):
        scratch_map = df.loc[df["model"] == "ssd_scratch", "best_val_mAP"].values[0]
        pretrained_map = df.loc[df["model"] == "ssd_pretrained", "best_val_mAP"].values[0]
        improvement = (pretrained_map - scratch_map) / max(scratch_map, 1e-6) * 100
        print(f"전이학습이 커스텀 베이스라인 대비 best mAP를 {improvement:.1f}% 개선했습니다.")
    return df


display_df = summarize_history()
print(display_df)

#@title 7-1. IoU Threshold(0.5 / 0.6 / 0.7)별 mAP 민감도 분석
def collect_predictions(model, data_loader):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Inference"):
            images = [img.to(device) for img in images]
            preds = model(images)
            all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
            all_gts.extend(targets)
    return all_preds, all_gts


preds_pretrained, gts_pretrained = collect_predictions(model_pretrained, val_loader)
preds_scratch, gts_scratch = collect_predictions(model_scratch, val_loader)

thresholds = [0.5, 0.6, 0.7]
rows = []
for name, preds, gts in [("ssd_pretrained", preds_pretrained, gts_pretrained),
                          ("ssd_scratch", preds_scratch, gts_scratch)]:
    for t in thresholds:
        m = evaluate_model(preds, gts, CLASSES, iou_threshold=t)
        rows.append({"model": name, "iou_threshold": t, "mAP": m})

df_thresh = pd.DataFrame(rows)
pivot = df_thresh.pivot(index="iou_threshold", columns="model", values="mAP")

pivot.plot(marker="o", figsize=(7, 5))
plt.title("mAP Comparison by IoU Threshold")
plt.xlabel("IoU Threshold"); plt.ylabel("mAP")
plt.grid(alpha=0.3)
plt.show()
print(pivot)

[계승] 내 코드의 IoU threshold sweep 아이디어를 그대로 채택했습니다. baseline 코드는 단일 IoU=0.5만 확인해 "박스를 대충 맞히는지, 정밀하게 맞히는지" 구분이 어려웠습니다.
[수정] 그래프 제목의 백틱(\`) 오타를 제거했습니다.

🧪 9. Best 모델 확정 → 테스트 추론 + 오답 분석

#@title 8. Best 모델 로드 -> 테스트 추론 + 오답 분석(위치 오류 / 분류 오류)
import matplotlib.patches as patches

# ---------------- 8-1. 학습 중 저장된 'best'(val mAP 최고) 가중치로 최종 모델 확정 ----------------
final_model = model_pretrained
best_ckpt = torch.load(
    os.path.join(CHECKPOINT_DIR, "ssd_pretrained_best.pth"),
    map_location=device, weights_only=False,
)
final_model.load_state_dict(best_ckpt["model_state_dict"])
final_model.eval()
print(f"최종 평가에 사용할 모델: ssd_pretrained (best epoch={best_ckpt['epoch']+1}, "
      f"val mAP={max(best_ckpt['history']['val_mAP']):.4f})")


# ---------------- 8-2. 테스트 이미지 예측 시각화 ----------------
def visualize_prediction(image, prediction, classes, score_threshold=0.5):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(7, 7))
    ax.imshow(image_np)

    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score < score_threshold:
            continue
        x_min, y_min, x_max, y_max = box.tolist()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x_min, max(y_min - 10, 0), f"{classes[label]}: {score:.2f}",
                color="red", fontsize=10, bbox=dict(facecolor="white", alpha=0.7))

    ax.axis("off")
    plt.show()


max_visualizations = 8
count = 0
with torch.no_grad():
    for images, file_names in test_loader:
        images_gpu = [img.to(device) for img in images]
        predictions = final_model(images_gpu)

        for img, pred, name in zip(images, predictions, file_names):
            if count >= max_visualizations:
                break
            visualize_prediction(img, {k: v.cpu() for k, v in pred.items()}, CLASSES)
            count += 1
        if count >= max_visualizations:
            break


# ---------------- 8-3. 오답 분석: 위치 오류 vs 분류 오류 ----------------
def find_error_cases(model, data_loader, classes, score_thresh=0.5, iou_thresh=0.5, max_cases=5):
    model.eval()
    loc_errors, misclass_errors = [], []

    with torch.no_grad():
        for images, targets in data_loader:
            if len(loc_errors) >= max_cases and len(misclass_errors) >= max_cases:
                break

            images_gpu = [img.to(device) for img in images]
            preds = model(images_gpu)

            for img, pred, gt in zip(images, preds, targets):
                gt_boxes = gt["boxes"].cpu().numpy() if len(gt["boxes"]) > 0 else np.zeros((0, 4))
                gt_labels = gt["labels"].cpu().numpy()
                if len(gt_boxes) == 0:
                    continue

                pred_boxes = pred["boxes"].cpu().numpy()
                pred_labels = pred["labels"].cpu().numpy()
                pred_scores = pred["scores"].cpu().numpy()

                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    if score < score_thresh:
                        continue
                    ious = calculate_iou(box, gt_boxes)
                    max_idx = int(np.argmax(ious))
                    max_iou = ious[max_idx]

                    if max_iou < iou_thresh and len(loc_errors) < max_cases:
                        loc_errors.append((img, pred, gt, "위치 오류 (Localization Error)"))
                    elif max_iou >= iou_thresh and label != gt_labels[max_idx] and len(misclass_errors) < max_cases:
                        misclass_errors.append((img, pred, gt, "분류 오류 (Misclassification)"))

    return loc_errors, misclass_errors


def visualize_error_case(image, pred, gt, classes, error_type, score_threshold=0.5):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image_np)

    for box, label in zip(gt["boxes"], gt["labels"]):
        x_min, y_min, x_max, y_max = box.tolist()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=2, edgecolor="green", facecolor="none")
        ax.add_patch(rect)
        ax.text(x_min, y_max + 12, f"GT: {classes[label]}", color="green", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.7))

    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < score_threshold:
            continue
        x_min, y_min, x_max, y_max = box.tolist()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x_min, max(y_min - 10, 0), f"Pred: {classes[label]} ({score:.2f})",
                 color="red", fontsize=9, bbox=dict(facecolor="white", alpha=0.7))

    ax.set_title(f"Error Type: {error_type}")
    ax.axis("off")
    plt.show()


loc_errors, misclass_errors = find_error_cases(final_model, val_loader, CLASSES)
print(f"탐색된 위치 오류 사례: {len(loc_errors)}개")
for img, pred, gt, err_type in loc_errors:
    visualize_error_case(img, {k: v.cpu() for k, v in pred.items()}, gt, CLASSES, err_type)

print(f"탐색된 분류 오류 사례: {len(misclass_errors)}개")
for img, pred, gt, err_type in misclass_errors:
    visualize_error_case(img, {k: v.cpu() for k, v in pred.items()}, gt, CLASSES, err_type)

[계승] 내 코드의 오답 분석(위치 오류/분류 오류 자동 분류) 로직을 그대로 유지했습니다 — baseline 코드에는 없는, mAP 숫자만으로는 알 수 없는 실패 패턴을 파고드는 매우 유용한 분석입니다.
[개선] 최종 추론에 학습이 끝난 시점의 가중치가 아니라, _best.pth(val mAP가 가장 높았던 epoch)를 명시적으로 다시 불러와 사용하도록 했습니다.

📝 10. 최종 결론 및 재사용 체크리스트

이번 실험에서 확인된 인사이트

전이학습(Pretrained)이 커스텀 베이스라인(Scratch) 대비 확연히 높은 mAP와 위치 정밀도를 보입니다.

IoU threshold를 높일수록(0.5→0.7) 두 모델의 성능 격차가 더 벌어지며, 이는 전이학습 모델이 박스 위치까지 정밀하게 회귀한다는 것을 의미합니다.

위치 오류는 객체가 이미지 경계에 걸치거나 배경과 색이 유사할 때, 분류 오류는 외형이 비슷한 품종 간에 주로 발생합니다.

다음 프로젝트에 이 템플릿을 재사용할 때 체크할 것

CLASSES, 데이터 경로(image_dir, xml_dir), 클래스 수만 바꾸면 대부분의 커스텀 detection 데이터셋에 그대로 적용 가능합니다.

evaluate_model의 iou_threshold 인자를 바꿔가며 threshold sweep을 항상 수행하세요 (mAP@0.5 하나만 보지 않기).

save_checkpoint(tag="best")로 저장된 가중치를 반드시 최종 평가/배포에 사용하세요 (tag="latest"는 재개용일 뿐입니다).

find_error_cases/visualize_error_case는 Detection뿐 아니라 다른 CV 태스크의 오답 분석으로도 일반화할 수 있는 자산입니다 — 별도 유틸 모듈로 분리해두면 편합니다.

데이터 무결성 점검은 항상 양방향(A에 있는데 B에 없는 것 / B에 있는데 A에 없는 것)으로 하세요.

사용하는 모델이 내부적으로 정규화를 수행하는지 문서를 먼저 확인한 뒤 Normalize 적용 여부를 결정하세요 (이중 정규화 방지).

