---
notion-url: https://app.notion.com/p/cGAN-FashionMNIST-3d477fa64f7f80ff96f2fbfa55537731
title: cGAN (FashionMNIST)
date: '2026-09-07 23:55:00.000'
from_notion: https://app.notion.com/p/cGAN-FashionMNIST-3d477fa64f7f80ff96f2fbfa55537731
author: rahui kim
last_edited_time: '2026-09-07 23:56:00.000'
---
# cGAN (FashionMNIST) 코드 리뷰 & 재사용 템플릿

> 강사님 공유 코드(baseline) vs 김라희님 코드 비교 정리
노션에 그대로 붙여넣기용 — 표/코드블록 포함

---

## 1. 한눈에 보는 비교표

 | 항목 | 강사님 코드 (baseline) | 내 코드 (라희님) | 코멘트 | 
 | ---- | ---- | ---- | ---- | 
 | Label embedding (G) | `num_classes`(10) 차원 | `label_embed_dim=50`으로 확장 | 표현력↑ 시도, 좋은 선택 | 
 | Label embedding (D) | 스칼라 1개 → 전체 broadcast | 28×28 **spatial map**으로 embedding | 훨씬 정교한 조건부 정보 전달 | 
 | Discriminator 깊이 | Conv 2단 (64→128) | Conv 3단 (64→128→256) + `Dropout2d` | 과적합/모드붕괴 대응 강화 | 
 | Loss 구성 | `BCELoss` + D 출력에 `Sigmoid` | `BCEWithLogitsLoss` + D는 logit만 출력 | **수치적으로 더 안정적** (라희님 선택이 더 좋음) | 
 | Label smoothing | 없음 | real=0.9 | GAN 안정화 기법, 잘 적용함 | 
 | 가중치 초기화 | 없음 (PyTorch 기본값) | `weights_init_normal` (평균0, 표준편차0.02) | DCGAN 논문 권장 사항 반영 | 
 | 데이터 증강 | Flip+Rotation+Affine+RandomResizedCrop (강함) | Rotation(±8°)+Affine(이동5%) (약함) | 라희님이 "의류 형태 왜곡 방지" 의도로 약하게 설계 → 근거 있는 설계 | 
 | 모델 구조 확인 | 없음 | `torchinfo.summary()` | 실무에서 매우 좋은 습관 | 
 | 체크포인트 저장 | **매 에폭** 저장 (`generator_tr_epoch_{n}.pth`) | 마지막 1개만 저장 | 강사님 방식이 장애 대응(중단 시 복구)에 유리 | 
 | 학습 시간 측정 | 없음 | `time.time()` 으로 측정 | 좋은 습관 | 
 | 결과 저장 | `plt.show()`만 (휘발성) | png/pkl 파일로 저장 | 재현성 측면에서 라희님이 훨씬 우수 | 
 | 평가지표(FID/IS) | Inception-v3 직접 로드 + 수식 직접 구현 | `torchmetrics` 라이브러리 사용 | 각각 장단점 있음 (아래 3번 참고) | 
 | Discriminator 조건 레이블 | 가짜 이미지에 **랜덤 레이블** 부여 | 가짜 이미지에 **실제 배치의 레이블** 재사용 | 설계 철학 차이 (아래 설명) | 

---

## 2. 내가 잘한 점 (자신 있게 가져가도 되는 부분)

### ✅ ① BCEWithLogitsLoss + logit 출력 구조


```python
# 라희님 방식 (권장)
self.adv_layer = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 1))  # sigmoid 없음
criterion = nn.BCEWithLogitsLoss()

# 강사님 방식 (동작은 하지만 덜 안정적)
self.model = nn.Sequential(..., nn.Sigmoid())
adversarial_loss = nn.BCELoss()
```

### ✅ ② Label을 spatial map으로 임베딩 (Discriminator)

### ✅ ③ 가중치 초기화 (`weights_init_normal`)

### ✅ ④ 실험 재현성 확보 (파일 저장 습관)

### ✅ ⑤ 데이터 증강을 "생각해서" 약하게 설계

### ✅ ⑥ torchinfo로 모델 구조 사전 검증

---

## 3. 강사님 코드 보고 개선하면 좋은 점

### 🔧 ① BatchNorm2d 파라미터 버그 (중요! 확인 필요)


```python
# 라희님 코드
nn.BatchNorm2d(128, 0.8)   # ⚠️ 두 번째 위치인자는 momentum이 아니라 eps!
```


```python
# 올바른 수정
nn.BatchNorm2d(128, momentum=0.8)
```

### 🔧 ② 체크포인트를 매 epoch 저장하는 습관


```python
if (epoch + 1) % 5 == 0:
    torch.save(generator.state_dict(), f"{CKPT_DIR}/generator_epoch_{epoch+1}.pth")
```

### 🔧 ③ 가짜 이미지 레이블을 랜덤으로 뽑는 방식 고려


```python
gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
gen_imgs = generator(z, gen_labels)
```

### 🔧 ④ 두 그룹(real/fake) 판별 결과를 D(x), D(G(x))로 명확히 로깅


```python
# 강사님 방식: 함수화 → 재사용 용이
def train_one_epoch(generator, discriminator, optimizer_G, optimizer_D, ...):
    ...
    return {"g_loss": ..., "d_loss": ..., "d_real": ..., "d_fake": ...}
```

### 🔧 ⑤ FID/IS를 직접 구현해본 경험도 가치 있음

---

## 4. 노션 템플릿용 재사용 코드 모음

> 다음 프로젝트에서 그대로 복붙해서 쓸 수 있는 조각들입니다.

### 📌 4-1. 환경 세팅 (seed 고정 + 디바이스 + 폴더)


```python
import os, random, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "/content/outputs"
CKPT_DIR = "/content/checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
```

### 📌 4-2. cGAN Generator 템플릿 (라희님 구조 기반, 버그 수정 반영)


```python
class Generator(nn.Module):
    """조건부 이미지 생성자: (noise, label) -> image"""
    def __init__(self, latent_dim, num_classes, label_embed_dim, image_size):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, label_embed_dim)
        self.init_size = image_size // 4

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + label_embed_dim, 128 * self.init_size ** 2),
            nn.BatchNorm1d(128 * self.init_size ** 2),
            nn.ReLU(inplace=True),
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, momentum=0.8),   # 키워드 인자로 명시!
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32, momentum=0.8),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), dim=-1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        return self.conv_blocks(out)
```

### 📌 4-3. cGAN Discriminator 템플릿 (spatial label map + logit 출력)


```python
class Discriminator(nn.Module):
    """조건부 판별자: (image, label) -> logit (Sigmoid 없음, BCEWithLogitsLoss와 함께 사용)"""
    def __init__(self, num_classes, image_size):
        super().__init__()
        self.image_size = image_size
        self.label_emb = nn.Embedding(num_classes, image_size * image_size)

        def conv_block(in_ch, out_ch, use_bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 3, 2, 1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            return layers

        self.model = nn.Sequential(
            *conv_block(2, 64, use_bn=False),
            *conv_block(64, 128),
            *conv_block(128, 256),
        )
        self.adv_layer = nn.Sequential(nn.Flatten(), nn.Linear(256 * 4 * 4, 1))

    def forward(self, img, labels):
        bs = img.size(0)
        label_map = self.label_emb(labels).view(bs, 1, self.image_size, self.image_size)
        d_in = torch.cat((img, label_map), dim=1)
        return self.adv_layer(self.model(d_in))
```

### 📌 4-4. 가중치 초기화 템플릿


```python
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
```

### 📌 4-5. 학습 루프 템플릿 (함수화 + 랜덤 gen_label + 체크포인트 주기 저장)


```python
def train_one_epoch(generator, discriminator, optimizer_G, optimizer_D,
                     criterion, train_loader, device, latent_dim, num_classes,
                     label_smooth_real=1.0):
    g_sum = d_sum = d_real_sum = d_fake_sum = 0.0
    n_batches = 0
    generator.train(); discriminator.train()

    for imgs, labels in train_loader:
        bs = imgs.size(0)
        imgs, labels = imgs.to(device), labels.to(device)
        valid = torch.full((bs, 1), label_smooth_real, device=device)
        fake = torch.zeros((bs, 1), device=device)

        # 랜덤 클래스로 가짜 이미지 생성 (클래스 균형 강제)
        z = torch.randn(bs, latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (bs,), device=device)
        gen_imgs = generator(z, gen_labels)

        # --- Discriminator ---
        optimizer_D.zero_grad()
        real_pred = discriminator(imgs, labels)
        d_real_loss = criterion(real_pred, valid)
        fake_pred = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = criterion(fake_pred, fake)
        d_loss = 0.5 * (d_real_loss + d_fake_loss)
        d_loss.backward()
        optimizer_D.step()

        # --- Generator ---
        optimizer_G.zero_grad()
        g_pred = discriminator(gen_imgs, gen_labels)
        g_loss = criterion(g_pred, torch.ones((bs, 1), device=device))
        g_loss.backward()
        optimizer_G.step()

        g_sum += g_loss.item(); d_sum += d_loss.item()
        d_real_sum += torch.sigmoid(real_pred).mean().item()
        d_fake_sum += torch.sigmoid(fake_pred).mean().item()
        n_batches += 1

    return {"g_loss": g_sum/n_batches, "d_loss": d_sum/n_batches,
            "d_real": d_real_sum/n_batches, "d_fake": d_fake_sum/n_batches}


def train_cgan(num_epochs, generator, discriminator, optimizer_G, optimizer_D,
                criterion, train_loader, device, latent_dim, num_classes,
                idx_to_class, ckpt_dir, save_every=5):
    history = {"g_loss": [], "d_loss": [], "d_real": [], "d_fake": []}
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        stats = train_one_epoch(generator, discriminator, optimizer_G, optimizer_D,
                                 criterion, train_loader, device, latent_dim, num_classes,
                                 label_smooth_real=0.9)
        for k in history: history[k].append(stats[k])
        print(f"[{epoch}/{num_epochs}] G:{stats['g_loss']:.4f} D:{stats['d_loss']:.4f} "
              f"D(real):{stats['d_real']:.3f} D(fake):{stats['d_fake']:.3f}")

        if epoch % save_every == 0 or epoch == num_epochs:
            torch.save(generator.state_dict(), f"{ckpt_dir}/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{ckpt_dir}/discriminator_epoch_{epoch}.pth")

    print(f"총 학습 시간: {(time.time()-start)/60:.1f}분")
    return history
```

### 📌 4-6. 클래스별 생성 이미지 시각화 템플릿


```python
import matplotlib.pyplot as plt

@torch.no_grad()
def visualize_by_class(generator, latent_dim, num_classes, device, idx_to_class,
                        n_row=3, save_path=None):
    generator.eval()
    labels = torch.arange(num_classes, device=device).repeat(n_row)
    z = torch.randn(n_row * num_classes, latent_dim, device=device)
    imgs = generator(z, labels)
    imgs = (imgs + 1) / 2.0  # [-1,1] -> [0,1]

    fig, axs = plt.subplots(n_row, num_classes, figsize=(num_classes, n_row))
    for i in range(n_row):
        for j in range(num_classes):
            idx = i * num_classes + j
            axs[i, j].imshow(imgs[idx, 0].cpu(), cmap="gray")
            axs[i, j].axis("off")
            if i == 0:
                axs[i, j].set_title(idx_to_class[j], fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    generator.train()
```

### 📌 4-7. FID / Inception Score 평가 템플릿 (torchmetrics 사용 — 실무 권장)


```python
# pip install torchmetrics -q
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

def to_uint8_rgb(x):
    """[-1,1] 1채널 -> [0,255] 3채널 uint8 (Inception 입력 형식)"""
    x = ((x + 1) / 2).clamp(0, 1)
    x = (x * 255).to(torch.uint8)
    return x.repeat(1, 3, 1, 1)

def evaluate_fid_is(generator, test_loader, latent_dim, num_classes, device,
                     n_eval=1000, chunk=100):
    real_imgs, collected = [], 0
    for imgs, _ in test_loader:
        real_imgs.append(imgs); collected += imgs.size(0)
        if collected >= n_eval: break
    real_batch = torch.cat(real_imgs)[:n_eval]

    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_eval, latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (n_eval,), device=device)
        fake_batch = generator(z, gen_labels).cpu()
    generator.train()

    fid = FrechetInceptionDistance(feature=64, normalize=False).to(device)
    inception_score = InceptionScore(normalize=False).to(device)

    for i in range(0, n_eval, chunk):
        fid.update(to_uint8_rgb(real_batch[i:i+chunk]).to(device), real=True)
    for i in range(0, n_eval, chunk):
        fake_chunk = to_uint8_rgb(fake_batch[i:i+chunk]).to(device)
        fid.update(fake_chunk, real=False)
        inception_score.update(fake_chunk)

    fid_score = fid.compute().item()
    is_mean, is_std = inception_score.compute()
    print(f"FID: {fid_score:.3f} (낮을수록 좋음)")
    print(f"Inception Score: {is_mean.item():.3f} ± {is_std.item():.3f} (높을수록 좋음)")
    return fid_score, is_mean.item(), is_std.item()
```

### 📌 4-8. 학습 결과 그래프 & 히스토리 저장 템플릿


```python
def plot_and_save_history(history, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history["d_loss"], label="D loss")
    plt.plot(history["g_loss"], label="G loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("cGAN Training Loss")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history["d_real"], label="D(real)")
    plt.plot(history["d_fake"], label="D(fake)")
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="균형점 (0.5)")
    plt.xlabel("Epoch"); plt.ylabel("평균 D 출력 확률")
    plt.title("Discriminator Confidence")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/discriminator_confidence.png", dpi=150, bbox_inches="tight")
    plt.show()

def save_history_pickle(history, hyperparams, output_dir, filename="training_report.pkl"):
    with open(f"{output_dir}/{filename}", "wb") as f:
        pickle.dump({"history": history, "hyperparameters": hyperparams}, f)
```

---

## 5. 요약 한 줄씩

- **Loss 설계(BCEWithLogitsLoss)**, **spatial label embedding**, **재현성(파일 저장 습관)** → 라희님이 강사님보다 더 나은 실무적 선택

- **BatchNorm 인자 버그**, **체크포인트 주기 저장**, **학습 루프 함수화** → 강사님 코드에서 배워올 부분

- 다음 프로젝트에서는 위 4번 템플릿을 그대로 붙여넣고 하이퍼파라미터만 바꿔서 실험하면 됩니다.

