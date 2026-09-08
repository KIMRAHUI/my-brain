#  Oxford-IIIT Pet_Object Detection

🐾 Oxford-IIIT Pet 객체 검출(Object Detection) 및 전이학습 프로젝트

📌 0. 프로젝트 개요 및 목표

목표: Oxford-IIIT Pet Dataset을 활용하여 강아지와 고양이의 위치를 찾는 객체 검출 모델을 구현하고, COCO 사전학습 가중치를 사용한 전이학습(Pretrained)과 가중치 없이 처음부터 학습하는 커스텀 베이스라인(Scratch)의 성능을 비교합니다.

주요 기술: PyTorch, Torchvision v2 Transforms, OpenCV, Matplotlib, Scikit-Learn

🛠️ 1. 데이터 탐색 및 준비

캐글 허브를 통해 데이터셋을 다운로드하고 파일 경로를 설정한 뒤, 이미지와 XML 어노테이션 구조를 파악합니다.

Python



#@title 데이터 탐색, 불러오기 및 환경 설정
import kagglehub
import torch
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 데이터셋 다운로드
path = kagglehub.dataset_download("zippyz/cats-and-dogs-breeds-classification-oxford-dataset")
print("Path to dataset files:", path)

# 파일 및 디렉토리 경로 지정
trainval_file_path = os.path.join(path, "annotations", "annotations", "trainval.txt")
test_file_path = os.path.join(path, "annotations", "annotations", "test.txt")

image_dir = os.path.join(path, "images", "images")
xml_dir = os.path.join(path, "annotations", "annotations", "xmls")

# 데이터프레임 읽기
df_trainval = pd.read_csv(trainval_file_path, sep=r"\s+", header=None)
df_trainval.columns = ["Image", "ClassID", "Species", "BreedID"]

df_test = pd.read_csv(test_file_path, sep=r"\s+", header=None)
df_test.columns = ["Image", "ClassID", "Species", "BreedID"]

print(f"Train/Validation 데이터 수: {len(df_trainval)}")
print(f"Test 데이터 수: {len(df_test)}")

# XML 파일이 없는 데이터 제외 처리
trainval_list = df_trainval['Image'].tolist()
xml_list = [os.path.splitext(file)[0] for file in os.listdir(xml_dir) if file.endswith(".xml")]
missing_xml = [image for image in trainval_list if image not in xml_list]
trainval_list = [image for image in trainval_list if image in xml_list]
print(f'xml 파일 없는 Train 이미지 수 제외 완료: {len(missing_xml)}개 제외')

📦 2. Dataset 및 DataLoader 구성 (데이터 증강 포함)

torchvision.transforms.v2를 활용하여 이미지뿐만 아니라 바운딩 박스 좌표까지 안전하게 연동 변환되도록 합니다.

Train 데이터에는 과적합 방지를 위해 무작위 수평 뒤집기(RandomHorizontalFlip)와 색상 변조(ColorJitter)를 적용합니다.

Python



#@title Dataset 및 DataLoader 구현
from torchvision.transforms import v2
from torchvision import tv_tensors
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. Transform 정의
train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToDtype(torch.float32, scale=True),
])

eval_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# 2. 커스텀 Dataset 클래스
class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, classes, image_list, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.classes = classes
        self.transforms = transforms
        self.image_files = image_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx] + ".jpg"
        annotation_file = self.image_files[idx] + ".xml"
        image_path = os.path.join(self.image_dir, image_file)
        annotation_path = os.path.join(self.annotation_dir, annotation_file)

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        boxes = []
        labels = []
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in self.classes:
                continue
            labels.append(self.classes.index(class_name))

            bndbox = obj.find("bndbox")
            x_min = int(bndbox.find("xmin").text)
            y_min = int(bndbox.find("ymin").text)
            x_max = int(bndbox.find("xmax").text)
            y_max = int(bndbox.find("ymax").text)
            boxes.append([x_min, y_min, x_max, y_max])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # v2.BoundingBoxes로 감싸주면 Augmentation 시 좌표가 자동 변환됨
        boxes_tv = tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(img_h, img_w))

        if self.transforms:
            image, boxes_tv = self.transforms(image, boxes_tv)

        target = {"boxes": boxes_tv, "labels": labels}
        return image, target


class TestDataset(Dataset):
    def __init__(self, image_dir, image_list, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_files = image_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx] + ".jpg"
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, self.image_files[idx]


# 3. 데이터 분할 및 DataLoader 생성
train_list, valid_list = train_test_split(trainval_list, test_size=0.3, random_state=42)
classes = ["background", "dog", "cat"]

train_dataset = VOCDataset(image_dir, xml_dir, classes, train_list, transforms=train_transform)
valid_dataset = VOCDataset(image_dir, xml_dir, classes, valid_list, transforms=eval_transform)
test_dataset = TestDataset(image_dir, df_test['Image'].tolist(), transforms=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

print(f"Train 데이터셋 크기: {len(train_dataset)}")
print(f"Validation 데이터셋 크기: {len(valid_dataset)}")

📊 3. 평가지표 구현 (IoU & mAP)

객체 검출의 정확도를 측정하기 위해 교집합 영역을 계산하는 IoU와 클래스별 정밀도를 평균 내는 mAP 함수를 구현합니다.

Python



#@title IoU 및 mAP 평가 함수 구현
import numpy as np
from sklearn.metrics import average_precision_score

def calculate_iou(box, boxes):
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

    return intersection / union

def evaluate_model(predictions, ground_truths, classes, iou_threshold=0.5):
    class_aps = []
    for class_idx, class_name in enumerate(classes[1:], start=1):
        true_positives = []
        scores = []
        num_ground_truths = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred["boxes"][pred["labels"] == class_idx].cpu().numpy() if len(pred["boxes"]) > 0 else np.zeros((0, 4))
            pred_scores = pred["scores"][pred["labels"] == class_idx].cpu().numpy() if len(pred["scores"]) > 0 else np.zeros((0,))
            gt_boxes = gt["boxes"][gt["labels"] == class_idx].cpu().numpy() if len(gt["boxes"]) > 0 else np.zeros((0, 4))

            num_ground_truths += len(gt_boxes)

            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue

            matched = np.zeros(len(gt_boxes), dtype=bool)
            for box, score in zip(pred_boxes, pred_scores):
                ious = calculate_iou(box, gt_boxes)
                max_iou_idx = np.argmax(ious) if len(ious) > 0 else -1
                max_iou = ious[max_iou_idx] if max_iou_idx >= 0 else 0

                if max_iou >= iou_threshold and not matched[max_iou_idx]:
                    true_positives.append(1)
                    matched[max_iou_idx] = True
                else:
                    true_positives.append(0)
                scores.append(score)

        if len(scores) == 0:
            class_aps.append(0)
            continue

        sorted_indices = np.argsort(-np.array(scores))
        true_positives = np.array(true_positives)[sorted_indices]
        scores = np.array(scores)[sorted_indices]

        ap = average_precision_score(true_positives, scores) if len(scores) > 0 else 0
        class_aps.append(ap)

    return np.mean(class_aps)

💾 4. 체크포인트 및 학습/이력 관리 유틸리티

Colab 환경에서 런타임 끊김 현상에 대비해 에포크마다 모델 가중치와 학습 이력을 Google Drive에 자동 저장하고 불러오는 시스템을 구축합니다.

Python



#@title 구글 드라이브 연동 및 체크포인트 관리 함수
from google.colab import drive
drive.mount('/content/drive')

BASE_DIR = "/content/drive/MyDrive/object_detection_mission"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
ALL_HISTORY_PATH = os.path.join(BASE_DIR, "all_history.pkl")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

import pickle

def save_checkpoint(model, optimizer, lr_scheduler, epoch, history, model_name, ckpt_dir=CHECKPOINT_DIR):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "history": history,
    }
    torch.save(checkpoint, os.path.join(ckpt_dir, f"{model_name}_latest.pth"))
    print(f"[{model_name}] Epoch {epoch} 체크포인트 저장 완료")

def load_checkpoint(model, optimizer, lr_scheduler, model_name, ckpt_dir=CHECKPOINT_DIR):
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_latest.pth")
    if not os.path.exists(ckpt_path):
        print(f"[{model_name}] 저장된 체크포인트가 없습니다. 처음부터 학습을 시작합니다.")
        return 0, {"train_loss": [], "val_mAP": []}

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    history = checkpoint["history"]
    print(f"[{model_name}] 체크포인트 로드 완료 -> {start_epoch} 에포크부터 재개")
    return start_epoch, history

def update_all_history(model_name, history, path=ALL_HISTORY_PATH):
    if os.path.exists(path):
        with open(path, "rb") as f:
            all_history = pickle.load(f)
    else:
        all_history = {}
    all_history[model_name] = history
    with open(path, "wb") as f:
        pickle.dump(all_history, f)

def load_all_history(path=ALL_HISTORY_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

# 통합 학습 함수
from tqdm import tqdm
def train_model(model, model_name, optimizer, lr_scheduler, num_epochs, train_loader, val_loader, classes):
    start_epoch, history = load_checkpoint(model, optimizer, lr_scheduler, model_name)

    for epoch in range(start_epoch, num_epochs):
        print(f"\n[{model_name}] Epoch {epoch + 1}/{num_epochs} 시작")
        model.train()
        total_train_loss = 0

        for images, targets in tqdm(train_loader, desc="Training"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_train_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        lr_scheduler.step()

        # Validation 평가
        model.eval()
        all_predictions, all_ground_truths = [], []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                predictions = model(images)
                all_predictions.extend(predictions)
                all_ground_truths.extend(targets)

        mAP = evaluate_model(all_predictions, all_ground_truths, classes)
        print(f"[{model_name}] Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val mAP: {mAP:.4f}")

        history["train_loss"].append(avg_train_loss)
        history["val_mAP"].append(mAP)

        save_checkpoint(model, optimizer, lr_scheduler, epoch, history, model_name)
        update_all_history(model_name, history)

    return model, history

🚀 5. 모델 학습 (전이학습 vs 커스텀 베이스라인)

5-1. 전이학습 모델 (ssd_pretrained) 학습

Python



#@title 전이학습(Pretrained) SSD300-VGG16 학습
import torchvision
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchinfo import summary

num_classes = len(classes)

model_pretrained = torchvision.models.detection.ssd300_vgg16(
    weights=SSD300_VGG16_Weights.DEFAULT
).to(device)
model_pretrained.head.classification_head.num_classes = num_classes

optimizer_pretrained = torch.optim.SGD(
    model_pretrained.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
)
scheduler_pretrained = torch.optim.lr_scheduler.StepLR(optimizer_pretrained, step_size=3, gamma=0.1)

# 모델 구조 요약 확인
sample_input = [torch.randn(3, 300, 300).to(device)]
summary(model_pretrained, input_data=[sample_input], depth=2, col_names=["output_size", "num_params", "trainable"], row_settings=["var_names"])

# 학습 실행
model_pretrained, history_pretrained = train_model(
    model=model_pretrained,
    model_name="ssd_pretrained",
    optimizer=optimizer_pretrained,
    lr_scheduler=scheduler_pretrained,
    num_epochs=5,
    train_loader=train_loader,
    val_loader=val_loader,
    classes=classes,
)

5-2. 커스텀 베이스라인 모델 (ssd_scratch) 학습

Python



#@title 커스텀 베이스라인(Scratch) SSD300-VGG16 학습
model_scratch = torchvision.models.detection.ssd300_vgg16(
    weights=None, weights_backbone=None
).to(device)
model_scratch.head.classification_head.num_classes = num_classes

optimizer_scratch = torch.optim.SGD(
    model_scratch.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005
)
scheduler_scratch = torch.optim.lr_scheduler.StepLR(optimizer_scratch, step_size=3, gamma=0.1)

# 학습 실행
model_scratch, history_scratch = train_model(
    model=model_scratch,
    model_name="ssd_scratch",
    optimizer=optimizer_scratch,
    lr_scheduler=scheduler_scratch,
    num_epochs=5,
    train_loader=train_loader,
    val_loader=val_loader,
    classes=classes,
)

📈 6. 모델 성능 비교 및 분석 시각화

학습이 완료된 두 모델의 손실 함수(Loss)와 검증 mAP를 비교하고, IoU Threshold 변화에 따른 모델 강건성을 평가합니다.

Python



#@title 성능 비교 시각화 및 요약
import warnings
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
    if "ssd_scratch" in df["model"].values and "ssd_pretrained" in df["model"].values:
        scratch_map = df.loc[df["model"] == "ssd_scratch", "best_val_mAP"].values[0]
        pretrained_map = df.loc[df["model"] == "ssd_pretrained", "best_val_mAP"].values[0]
        improvement = (pretrained_map - scratch_map) / max(scratch_map, 1e-6) * 100
        print(f"전이학습이 커스텀 베이스라인 대비 best mAP를 {improvement:.1f}% 개선했습니다.")
    return df

summarize_history()

IoU Threshold별 성능 민감도 분석

Python



#@title Threshold(0.5, 0.6, 0.7)에 따른 mAP 변화 비교
def collect_predictions(model, data_loader):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Inference"):
            images = [img.to(device) for img in images]
            preds = model(images)
            all_preds.extend(preds)
            all_gts.extend(targets)
    return all_preds, all_gts

preds_pretrained, gts_pretrained = collect_predictions(model_pretrained, val_loader)
preds_scratch, gts_scratch = collect_predictions(model_scratch, val_loader)

thresholds = [0.5, 0.6, 0.7]
rows = []
for name, preds, gts in [("ssd_pretrained", preds_pretrained, gts_pretrained), ("ssd_scratch", preds_scratch, gts_scratch)]:
    for t in thresholds:
        m = evaluate_model(preds, gts, classes, iou_threshold=t)
        rows.append({"model": name, "iou_threshold": t, "mAP": m})

df_thresh = pd.DataFrame(rows)
pivot = df_thresh.pivot(index="iou_threshold", columns="model", values="mAP")

pivot.plot(marker="o", figsize=(7, 5))
plt.title("mAP Comparison by IoU Threshold")
plt.xlabel("IoU Threshold"); plt.ylabel("mAP")
plt.grid(alpha=0.3)
plt.show()
print(pivot)

🧪 7. 테스트 데이터 추론 및 오답 분석 (Error Analysis)

모델의 전반적인 정량적 평가를 넘어 위치 오류(Localization Error)와 분류 오류(Misclassification)를 구체적으로 탐색합니다.

Python



#@title 오답 분석 함수 및 시각화 코드
import matplotlib.patches as patches

def find_error_cases(model, data_loader, classes, score_thresh=0.5, iou_thresh=0.5, max_cases=3):
    model.eval()
    loc_errors = []
    misclass_errors = []

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

    # Ground Truth (초록색)
    for box, label in zip(gt["boxes"], gt["labels"]):
        x_min, y_min, x_max, y_max = box.tolist()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="green", facecolor="none")
        ax.add_patch(rect)
        ax.text(x_min, y_max + 12, f"GT: {classes[label]}", color="green", fontsize=9, bbox=dict(facecolor="white", alpha=0.7))

    # Prediction (빨간색)
    for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < score_threshold:
            continue
        x_min, y_min, x_max, y_max = box.tolist()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x_min, y_min - 10, f"Pred: {classes[label]} ({score:.2f})", color="red", fontsize=9, bbox=dict(facecolor="white", alpha=0.7))

    ax.set_title(f"Error Type: {error_type}")
    ax.axis("off")
    plt.show()

# 오답 분석 실행
loc_errors, misclass_errors = find_error_cases(model_pretrained, val_loader, classes)
print(f"탐색된 위치 오류 사례: {len(loc_errors)}개")
for img, pred, gt, err_type in loc_errors:
    visualize_error_case(img, pred, gt, classes, err_type)

print(f"탐색된 분류 오류 사례: {len(misclass_errors)}개")
for img, pred, gt, err_type in misclass_errors:
    visualize_error_case(img, pred, gt, classes, err_type)

📝 8. 최종 결론 및 복습 포인트 (Key Takeaways)

전이학습의 압도적 우위:

IoU Threshold에 따른 강건성 차이:

데이터 증강의 역할:

오답 분석 인사이트:

