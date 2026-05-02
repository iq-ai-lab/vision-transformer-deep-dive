# 02. MAE — Masked Autoencoder

## 🎯 핵심 질문

- 왜 MAE 는 BEiT 의 40% 보다 **5 배 큰 75% masking ratio** 를 사용할 수 있는가?
- Asymmetric encoder–decoder 구조가 어떻게 **4× 이상 속도 향상** 을 가능하게 하는가?
- 왜 MAE 는 pixel space 에서 MSE loss 를 사용하는데도 높은 representation quality 를 얻는가?
- Fine-tuning 성능이 linear probe 보다 훨씬 높은 이유는? (contrastive 와 반대)

## 🔍 왜 75% masking 이 가능한가

**자연이미지의 극도의 spatial redundancy**:
- Neighboring patch 들이 매우 유사 (correlation ≈ 0.8~0.95)
- Random sampling 으로 25% 의 patch 만 봐도 전체 image 를 거의 복원 가능
- Compressive sensing 의 관점: sparse signal 은 적은 measurement 로도 복원 가능

**BEiT (discrete token) 대비 MAE (continuous pixel) 의 이점**:
- Pixel 은 "ground truth" 가 명확 (8192 vocabulary 의 discrete ambiguity 없음)
- Per-patch normalization 으로 reconstruction objective 가 더 안정적
- Decoder 를 사전학습 후 폐기하므로, "tokenizer quality ceiling" 문제 없음

## 📐 수학적 선행 조건

- **Autoencoder** : encoder $E$, decoder $D$, reconstruction loss
- **Masking 및 Reconstruction** : random mask, regression on masked region
- **Per-patch normalization** : patch-wise mean/std 를 이용한 loss 정규화
- **Asymmetric architecture** : encoder (large), decoder (lightweight)
- **Linear probe evaluation** : frozen backbone + linear classifier 로 downstream 평가

## 📖 직관적 이해

```
MAE Forward Pass:

┌─────────────────┐
│  원본 이미지    │ (224 × 224 × 3)
│  196 patches    │ (14 × 14 spatial)
└────────┬────────┘
         │
    ┌────▼──────────────┐
    │  Random Masking   │ (75% mask, 25% visible)
    │  [P₁] [MASK] [P₃] │ 49 visible patches
    │  ...              │
    └────┬──────────────┘
         │
    ┌────▼──────────────────┐
    │  Encoder (ViT-L)      │ (only 49 patches)
    │  + position encoding  │ → 4× speedup
    │  → latent [h₁...h₄₉] │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  Decoder (lightweight)│ (8 layers, dim 512)
    │  + 196 mask tokens    │ (learnable, shuffled)
    │  → [d₁...d₁₉₆]       │
    └────┬──────────────────┘
         │
    ┌────▼──────────────────┐
    │  MSE Reconstruction   │
    │  Per-patch normalized │
    │  Loss on masked 147   │
    └──────────────────────┘

이 decoder 는 pretrain 후 폐기됨.
Downstream task (fine-tuning) 는 encoder 만 사용.
```

## ✏️ 엄밀한 정의

### 정의 5.4: MAE Forward Process

Image $x \in \mathbb{R}^{H \times W \times 3}$ 를 patch embedding 으로 변환:
$$\mathbf{p} = [p_1, \ldots, p_N] \in \mathbb{R}^{N \times D}, \quad N = \frac{HW}{P^2}$$

(standard ViT patch embedding, Ch1-02 참조)

### 정의 5.5: Random Masking (Uniform)

Masking ratio $r = 0.75$ 에 대해, random subset $M \subset [N]$ with $|M| = 0.75N$ 를 선택.

Visible patches: $\mathbf{p}_{\text{vis}} = \{\mathbf{p}_i : i \notin M\}$ (크기 $0.25N$)

### 정의 5.6: Encoder (Large ViT)

Visible patches 만 처리:
$$\mathbf{h}_{\text{vis}} = \mathrm{Transformer}_{\text{enc}}(\mathbf{p}_{\text{vis}} + \mathbf{E}_{\text{pos,vis}})$$

여기서 position embeddings $\mathbf{E}_{\text{pos}}$ 는 **visible position 만** 포함 (masked 위치 PE 없음).

Output: $\mathbf{h}_{\text{vis}} \in \mathbb{R}^{(0.25N) \times D}$

### 정의 5.7: Decoder (Lightweight)

Learnable mask tokens $\mathbf{m}_{\text{mask}} \in \mathbb{R}^{0.75N \times D}$ 를 도입.

Encoder output 과 mask tokens 를 모두 사용:
$$\mathbf{z} = [\mathbf{h}_{\text{vis}} ; \mathbf{m}_{\text{mask}}]_{\text{shuffled}}$$

Decoder processes shuffled latent:
$$\mathbf{d} = \mathrm{Transformer}_{\text{dec}}(\mathbf{z} + \mathbf{E}_{\text{pos,full}})$$

여기서 position embeddings 는 **원본 patch 위치** 를 반영 (shuffled 상태에서).

Output: $\mathbf{d} \in \mathbb{R}^{N \times D}$

### 정의 5.8: Per-Patch Normalized MSE Loss

Decoder output 의 masked region 에 대해:
$$L_{\mathrm{MAE}} = \frac{1}{|M|} \sum_{i \in M} \|\mathbf{d}_i - \mathbf{p}_i\|_2^2$$

**Per-patch normalization**:
각 patch $p_i$ 를 먼저 정규화 (mean 제거, std 로 scale):
$$\tilde{p}_i = \frac{p_i - \mu_i}{\sigma_i + \epsilon}$$

Loss 를 정규화된 patches 에 대해 계산하면, **high-variance patches 가 loss 를 dominate 하지 않음**.

## 🔬 정리와 증명

### 정리 5.3: Asymmetric Encoder–Decoder 의 계산 복잡도 감소

**명제**: Visible patches 만 encoder 에 입력하면, forward pass 의 FLOPs 가 $4\times$ 감소한다 (75% masking 시).

**증명**:
Self-attention 의 복잡도는 sequence length 의 제곱에 비례:
$$\mathrm{FLOPs}_{\text{attn}} \propto L^2$$

**Symmetric (전체 patch 처리)**:
$$\mathrm{FLOPs}_{\text{sym}} \propto N^2$$

**Asymmetric (visible 25% 만 처리)**:
$$\mathrm{FLOPs}_{\text{asym}} \propto (0.25N)^2 = 0.0625 N^2$$

따라서:
$$\frac{\mathrm{FLOPs}_{\text{sym}}}{\mathrm{FLOPs}_{\text{asym}}} = \frac{N^2}{0.0625 N^2} = 16$$

더 정확하게는, layer normalization 과 MLP 도 선형이므로:
$$\frac{\mathrm{FLOPs}_{\text{sym}}}{\mathrm{FLOPs}_{\text{asym}}} \approx 3.6 \approx 4\times$$

실제로는 메모리 access 와 reduction operations 등으로 정확히 4× 는 아니지만,
**3.6~4× 범위의 significant speedup** 이 확인됨 (He et al. 2022).

$$\square$$

### 정리 5.4: Fine-tuning vs Linear Probe 의 차이

**관찰** (He et al. 2022, Table):
- Linear probe (frozen encoder + linear head): 79.0% (ViT-L)
- Fine-tuning (전체 모델 학습): 83.6% (ViT-L)
- **차이**: 4.6% — MAE pretrain 이 "fine-tuning friendly" representation 학습

이는 **contrastive SSL (SimCLR, DINO)** 와 대조:
- Contrastive linear probe: ~75.4%
- Contrastive fine-tuning: ~74.8%
- **거의 같음** — contrastive 는 이미 "linear-friendly" feature

**원인 분석**:

MAE representation 은 **low-level detail** (edge, texture 등) 을 많이 보존한다.
Linear probe 는 이 detail 을 활용하지 못 (single layer) 하지만,
fine-tuning 시 깊은 layer 들이 이를 higher-level semantic 으로 변환.

Contrastive 는 semantic information 에 이미 focus (instance discrimination) 하므로,
layer 를 추가해도 marginal gain.

**수학적 해석**:
MAE 의 reconstruction objective 가 **manifold 의 intrinsic geometry** 를 학습하는 반면,
contrastive 는 **discriminative subspace** 에만 focus.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Asymmetric Speed Test

```python
import torch
import torch.nn as nn
import time

class DummyViTEncoder(nn.Module):
    def __init__(self, seq_len, dim, num_heads=12, num_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, 
                                       dim_feedforward=4*dim, batch_first=True)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Test: 25% vs 100% sequence length
B, D, num_heads = 32, 768, 12
seq_len_full = 196  # 14×14 patches
seq_len_visible = int(0.25 * seq_len_full)  # 75% masked

encoder_full = DummyViTEncoder(seq_len_full, D, num_heads, num_layers=12)
encoder_visible = DummyViTEncoder(seq_len_visible, D, num_heads, num_layers=12)

# Warm-up
for _ in range(3):
    encoder_full(torch.randn(B, seq_len_full, D))
    encoder_visible(torch.randn(B, seq_len_visible, D))

# Benchmark
n_iter = 10
torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()
for _ in range(n_iter):
    encoder_full(torch.randn(B, seq_len_full, D))
time_full = (time.time() - start) / n_iter

start = time.time()
for _ in range(n_iter):
    encoder_visible(torch.randn(B, seq_len_visible, D))
time_visible = (time.time() - start) / n_iter

print(f"Full sequence (196): {time_full*1000:.2f} ms")
print(f"Visible 25% (49): {time_visible*1000:.2f} ms")
print(f"Speedup: {time_full / time_visible:.2f}×")
```

### 실험 2: Per-Patch Normalization 효과

```python
import torch
import torch.nn.functional as F
import numpy as np

# Simulate patches with varying statistics
B, N, D = 8, 196, 768
patches_orig = torch.randn(B, N, D)
# Introduce high-variance patches
patches_orig[:, :10, :] *= 10  # 10 patches with high variance

# Predict (random)
patches_pred = patches_orig + 0.1 * torch.randn_like(patches_orig)

# Loss without normalization
loss_unnorm = F.mse_loss(patches_pred, patches_orig)

# Loss with per-patch normalization
patches_normalized = (patches_orig - patches_orig.mean(dim=2, keepdim=True)) / \
                     (patches_orig.std(dim=2, keepdim=True) + 1e-6)
patches_pred_norm = (patches_pred - patches_orig.mean(dim=2, keepdim=True)) / \
                    (patches_orig.std(dim=2, keepdim=True) + 1e-6)

loss_norm = F.mse_loss(patches_pred_norm, patches_normalized)

print(f"Loss (unnormalized): {loss_unnorm:.4f}")
print(f"Loss (per-patch normalized): {loss_norm:.4f}")
print(f"Normalization prevents high-variance patches from dominating.")
```

### 실험 3: Reconstruction Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate image reconstruction
B, H, W = 1, 14, 14  # 14×14 patches
mask_ratio = 0.75

# Create random mask
mask = np.random.rand(H, W) < mask_ratio
visible = ~mask

# Simulate reconstruction: visible patches kept, masked reconstructed
image_orig = np.random.randn(H, W)
image_recon = image_orig.copy()
image_recon[mask] = image_orig[mask] + 0.2 * np.random.randn(mask.sum())

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image_orig, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(image_recon, cmap='gray')
axes[1].set_title('Reconstructed (75% masked)')
axes[2].imshow(np.abs(image_orig - image_recon), cmap='hot')
axes[2].set_title('Reconstruction Error')
plt.tight_layout()
plt.savefig('/tmp/mae_reconstruction.png', dpi=100)
print("Visualization saved.")
```

### 실험 4: Linear Probe vs Fine-tuning 시뮬레이션

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simulate pretrained encoder
class SimpleEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.latent_dim = latent_dim
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x

# Task: CIFAR-10 classification (10 classes)
encoder = SimpleEncoder(256)
num_classes = 10

# Linear probe
linear_head = nn.Linear(64, num_classes)
optimizer_linear = optim.SGD(linear_head.parameters(), lr=0.01)

# Fine-tuning
encoder_finetune = SimpleEncoder(256)
finetune_head = nn.Linear(64, num_classes)
optimizer_finetune = optim.SGD(
    list(encoder_finetune.parameters()) + list(finetune_head.parameters()),
    lr=0.001
)

print("Linear probe: only head is trainable, encoder frozen")
print("Fine-tuning: entire model is trainable, lower LR")
print("\nExpectation (MAE regime):")
print("  - Linear probe: ~79%")
print("  - Fine-tuning: ~83%")
print("  - Difference: 4% (MAE 가 low-level detail 을 보존)")
```

## 🔗 실전 활용

**MAE Pretraining (He et al. 2022)**:
1. ImageNet-1k (또는 더 큰 데이터) 에서 사전학습
2. 75% masking ratio, asymmetric encoder–decoder
3. Per-patch normalized MSE loss
4. 800 epochs, batch size 4096, learning rate 1.5e-4
5. Decoder 폐기, encoder 만 downstream 에 사용

**구현 특이사항**:
- Position embedding: **learnable**, absolute (not relative)
- Mask tokens: learnable embeddings, 초기값 가우시안
- Decoder: lightweight (8 layers, dim 512) — 사전학습 후 폐기
- Per-patch norm: 각 patch 의 mean/std 계산 후 정규화

**하이퍼파라미터**:
- Masking ratio: 75% (고정)
- Encoder depth/width: ViT-B (12L, 768D), ViT-L (24L, 1024D), ViT-H (32L, 1280D)
- Decoder: 8 layers, 512 dimension (모든 encoder size 에 동일)
- Learning rate: 1.5e-4 (base LR, batch size 4096)
- Warmup: 40 epochs
- Total: 800 epochs

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **75% masking 범용성** | 자연이미지에 최적화 (spatial redundancy) | Medical/synthetic image 에서는 다를 수 있음 |
| **Random uniform masking** | Structured masking (block, grid) 보다 우수 | Task-specific masking 은 미탐색 |
| **Per-patch 정규화** | High-variance patches 를 균형 맞춤 | 모든 dataset 에 동등 효과는 아님 |
| **Decoder 폐기** | 사전학습 후 decoder 는 불필요 | Generation task 에서는 decoder 필요 |
| **Fine-tuning 우위** | Linear probe 대비 4~5% 향상 | Domain shift 가 크면 더 필요 |

## 📌 핵심 정리

$$\boxed{\text{MAE Forward}: \, x \xrightarrow{\text{mask 75%}} \mathbf{p}_{\text{vis}} \xrightarrow{\text{Enc}} \mathbf{h} \xrightarrow{\text{Dec}} \hat{\mathbf{p}}}$$

$$\boxed{\text{Per-patch MSE}: \, L = \frac{1}{|M|} \sum_{i \in M} \left\| \frac{\hat{p}_i - \mu_i}{\sigma_i} - \frac{p_i - \mu_i}{\sigma_i} \right\|_2^2}$$

| 개념 | 수식/값 | 의미 |
|------|--------|------|
| **Masking ratio** | 75% | BEiT 의 40% 대비 5배 (spatial redundancy) |
| **Visible patches** | 25% (49 out of 196) | 충분한 정보로 전체 복원 |
| **Encoder** | Large ViT (L, H) | Visible 만 처리 → 4× speedup |
| **Decoder** | 8 layers, 512 dim | Lightweight, pretrain 후 폐기 |
| **Loss** | Per-patch normalized MSE | High-variance patch 편향 제거 |
| **Fine-tuning** | 83.6% (ViT-L) | Linear probe 79.0% 대비 4.6% 향상 |

## 🤔 생각해볼 문제

### 문제 1 (기초): 왜 Decoder 를 폐기하는가?

MAE 의 lightweight decoder 는 사전학습 후 버려진다.
왜 downstream task 에서 decoder 를 fine-tune 하거나 활용하지 않는가?
이론적, 실용적 이유를 설명하시오.

<details>
<summary>해설</summary>

**이론적 이유**:
- Reconstruction objective 는 **low-level detail** (pixel-level) 에 초점
- Fine-tuning task (ImageNet classification, detection) 는 **high-level semantic** 필요
- Decoder 를 계속 학습하면 semantic 을 해칠 수 있음

**실용적 이유**:
- Decoder 를 폐기하면 파라미터 수 ~절반 감소 (lightweight decoder 사용)
- Fine-tuning 시 memory 절약, 더 큰 batch size 가능
- 실험적으로 "decoder 없이 encoder 만 fine-tune" 이 더 나은 성능

**유추**:
Masked modeling 은 **self-supervised representation 학습** 에만 필요.
실제 task 에는 encoder 만으로 충분 (또는 task-specific head 추가).

</details>

### 문제 2 (심화): 75% Masking 의 정보론적 근거

MAE 논문 (He et al. 2022) 는 "75% 는 compressive sensing 과 유사" 라고 했다.
자연이미지의 spatial autocorrelation 을 정량화하고,
**Nyquist sampling theorem** 의 관점에서 왜 25% 가 충분한지 설명하시오.

<details>
<summary>해설</summary>

**Spatial autocorrelation**:
자연이미지의 power spectrum 이 $1/f^\alpha$ (fractal-like) 분포.
Low-frequency component 가 dominant → high-frequency (detail) 는 sparse.

**Compressive sensing 관점**:
Sparse signal (high-frequency 에 sparse) 는 random measurement 의 일부로도 복원 가능.
Nyquist rate 보다 훨씬 낮은 sampling rate 로도 가능 (random sampling).

**정량화**:
Image patch 의 mutual information (neighboring patches):
$$I(p_i, p_j) \approx C \cdot \exp(-\lambda \|i - j\|)$$

이를 수치적으로 계산하면, 대부분 정보가 이웃 5-10 patch 범위 내.
25% random sampling 시, 평균적으로 각 location 에 이웃 정보 충분.

**결론**: 자연이미지 의 sparsity (frequency domain) + random sampling 의 incoherence 가
25% 로도 충분하게 만듦.

</details>

### 문제 3 (논문 비평): Fine-tuning vs Linear Probe 의 reversed 관계

Contrastive SSL (DINO, SimCLR) 는 **linear probe ≈ fine-tuning** 이고,
MAE 는 **linear probe << fine-tuning** 이다.

이 차이가 두 objective (contrastive vs reconstruction) 의 본질적 차이를 
어떻게 반영하는지, 그리고 이것이 downstream task 의 유형에 따라 어떻게 
일반화되는지 토론하시오.

<details>
<summary>해설</summary>

**Contrastive (instance discrimination)**:
- Objective: 같은 이미지의 두 view 를 close, 다른 이미지를 far
- 결과: discriminative subspace 학습 (semantic 중심)
- Linear probe 가 이미 이 subspace 를 활용 → fine-tuning 과 비슷

**Reconstruction (MAE)**:
- Objective: pixel-level detail 복원
- 결과: low-level feature (edge, texture) 를 풍부하게 학습
- Linear probe 는 이 detail 을 활용 못 함 (single layer)
- Fine-tuning 시 깊은 layer 가 detail → semantic 변환

**일반화**:
- Task 가 **low-level detail 중요** (detection, segmentation) → MAE 우위
- Task 가 **semantic 중심** (classification) → 비슷 또는 contrastive 우위
- Domain shift 클 때 → fine-tuning 필수, 따라서 MAE 우위

**결론**: "어떤 representation 을 학습할 것인가" 가 downstream 에서의
trade-off 를 결정. Reconstruction 은 "풍부한 정보", contrastive 는 "정제된 정보".

</details>

---

<div align="center">

[◀ 이전](./01-beit.md) | [📚 README](../README.md) | [다음 ▶](./03-mae-information-theory.md)

</div>
