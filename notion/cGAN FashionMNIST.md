#  cGAN (FashionMNIST)

cGAN (FashionMNIST) 코드 리뷰 & 재사용 템플릿

강사님 공유 코드(baseline) vs 김라희님 코드 비교 정리
노션에 그대로 붙여넣기용 — 표/코드블록 포함

1. 한눈에 보는 비교표

2. 내가 잘한 점 (자신 있게 가져가도 되는 부분)

✅ ① BCEWithLogitsLoss + logit 출력 구조

Sigmoid를 Discriminator 안에 넣지 않고, loss 함수에서 한 번에 처리하도록 설계한 것은 정답에 가까운 선택입니다. Sigmoid+BCE를 따로 쓰면 극단값 근처에서 그래디언트가 소실되거나 log(0) 문제가 생기기 쉬운데, BCEWithLogitsLoss는 내부적으로 log-sum-exp trick을 써서 이를 방지합니다.

# 라희님 방식 (권장)
self.adv_layer = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 1))  # sigmoid 없음
criterion = nn.BCEWithLogitsLoss()

# 강사님 방식 (동작은 하지만 덜 안정적)
self.model = nn.Sequential(..., nn.Sigmoid())
adversarial_loss = nn.BCELoss()

✅ ② Label을 spatial map으로 임베딩 (Discriminator)

스칼라 broadcast보다 훨씬 정보량이 많습니다. 클래스 조건을 위치 단위로 줄 수 있어 Discriminator가 "이 위치에서 이 클래스면 이런 형태여야 한다"는 판단을 더 세밀하게 할 수 있습니다.

✅ ③ 가중치 초기화 (weights_init_normal)

DCGAN 논문에서 제안한 초기화 방식을 직접 구현해 적용한 것은 강사님 코드에는 없는 부분입니다. GAN은 초기화에 민감하므로 이 습관은 계속 가져가면 좋습니다.

✅ ④ 실험 재현성 확보 (파일 저장 습관)

OUTPUT_DIR, CKPT_DIR을 나누고 loss curve, 샘플 이미지, 학습 히스토리(pickle), 모델 요약(torchinfo)까지 전부 파일로 남긴 것은 실무 수준의 습관입니다. 강사님 코드는 Colab 화면 출력에 의존하기 때문에 세션이 끊기면 결과가 날아갑니다.

✅ ⑤ 데이터 증강을 "생각해서" 약하게 설계

강사님 코드는 Flip까지 포함한 강한 증강을 썼는데, 옷 종류에 따라 좌우 반전이 의미를 왜곡할 수 있습니다(예: 신발 좌/우, 가방 방향). 라희님은 이걸 인지하고 회전+이동만 약하게 적용했고, 마크다운에 근거까지 남겼습니다. 왜 이렇게 설계했는지 이유를 기록하는 습관 자체가 매우 좋은 태도입니다.

✅ ⑥ torchinfo로 모델 구조 사전 검증

학습 시작 전에 파라미터 수, 레이어별 출력 shape를 확인하는 것은 버그를 조기에 잡는 좋은 습관입니다.

3. 강사님 코드 보고 개선하면 좋은 점

🔧 ① BatchNorm2d 파라미터 버그 (중요! 확인 필요)

# 라희님 코드
nn.BatchNorm2d(128, 0.8)   # ⚠️ 두 번째 위치인자는 momentum이 아니라 eps!

nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, ...) 순서라서, 의도는 momentum=0.8이었겠지만 실제로는 eps=0.8이 들어갑니다. 강사님 코드는 이 부분을 momentum=0.8로 명시적 키워드 인자로 넘겨서 의도한 대로 동작합니다.

# 올바른 수정
nn.BatchNorm2d(128, momentum=0.8)

→ 키워드 인자를 명시하는 습관을 들이면 이런 실수를 방지할 수 있습니다. (다만 결과적으로 eps가 커져서 정규화가 과하게 완만해지는 효과가 생겼는데, 이게 우연히 학습을 안정시켰을 수도 있어서 흥미로운 케이스입니다 — 수정 후 재학습해서 비교해보면 좋습니다.)

🔧 ② 체크포인트를 매 epoch 저장하는 습관

강사님은 매 에폭마다 저장해서 학습이 중간에 끊겨도 어느 시점이든 복구 가능합니다. 라희님은 마지막 1개만 저장하는데, Colab처럼 세션이 끊기기 쉬운 환경에서는 주기적으로(예: 5epoch마다) 저장하는 편이 안전합니다.

if (epoch + 1) % 5 == 0:
    torch.save(generator.state_dict(), f"{CKPT_DIR}/generator_epoch_{epoch+1}.pth")

🔧 ③ 가짜 이미지 레이블을 랜덤으로 뽑는 방식 고려

강사님은 가짜 이미지 생성 시 torch.randint로 매번 랜덤 레이블을 뽑아 Generator가 모든 클래스를 골고루 학습하도록 강제합니다.

gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
gen_imgs = generator(z, gen_labels)

라희님 코드는 실제 배치의 labels를 그대로 재사용합니다(generator(noise, labels)). 배치가 랜덤 셔플이라 장기적으로는 비슷하게 수렴하지만, 의도적으로 클래스 균형을 강제하고 싶다면 강사님 방식이 더 명시적이고 안전합니다.

🔧 ④ 두 그룹(real/fake) 판별 결과를 D(x), D(G(x))로 명확히 로깅

강사님 코드는 d_real, d_fake 평균을 별도 함수로 분리해 반환값을 dict로 관리(train_one_epoch 함수화)하는데, 이렇게 학습 로직을 함수로 분리해두면 나중에 하이퍼파라미터 스윕(여러 설정 반복 실험)할 때 재사용하기 훨씬 편합니다. 라희님 코드는 학습 루프가 셀 안에 통째로 들어있어서, 다른 설정으로 다시 실험하려면 코드를 복붙해야 합니다.

# 강사님 방식: 함수화 → 재사용 용이
def train_one_epoch(generator, discriminator, optimizer_G, optimizer_D, ...):
    ...
    return {"g_loss": ..., "d_loss": ..., "d_real": ..., "d_fake": ...}

🔧 ⑤ FID/IS를 직접 구현해본 경험도 가치 있음

torchmetrics를 쓰는 게 실무적으로는 더 안전하고 빠르지만(라희님 선택 good), 강사님처럼 Inception-v3 feature 추출 → mean/cov → Fréchet distance를 직접 구현해보면 지표의 원리를 훨씬 깊게 이해할 수 있습니다. 한 번쯤은 두 방식을 나란히 실행해서 값이 비슷하게 나오는지 검증해보는 것도 좋은 공부가 됩니다.

4. 노션 템플릿용 재사용 코드 모음

다음 프로젝트에서 그대로 복붙해서 쓸 수 있는 조각들입니다.

📌 4-1. 환경 세팅 (seed 고정 + 디바이스 + 폴더)

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

📌 4-2. cGAN Generator 템플릿 (라희님 구조 기반, 버그 수정 반영)

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

📌 4-3. cGAN Discriminator 템플릿 (spatial label map + logit 출력)

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

📌 4-4. 가중치 초기화 템플릿

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

📌 4-5. 학습 루프 템플릿 (함수화 + 랜덤 gen_label + 체크포인트 주기 저장)

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

📌 4-6. 클래스별 생성 이미지 시각화 템플릿

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

📌 4-7. FID / Inception Score 평가 템플릿 (torchmetrics 사용 — 실무 권장)

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

📌 4-8. 학습 결과 그래프 & 히스토리 저장 템플릿

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

5. 요약 한 줄씩

Loss 설계(BCEWithLogitsLoss), spatial label embedding, 재현성(파일 저장 습관) → 라희님이 강사님보다 더 나은 실무적 선택

BatchNorm 인자 버그, 체크포인트 주기 저장, 학습 루프 함수화 → 강사님 코드에서 배워올 부분

다음 프로젝트에서는 위 4번 템플릿을 그대로 붙여넣고 하이퍼파라미터만 바꿔서 실험하면 됩니다.

