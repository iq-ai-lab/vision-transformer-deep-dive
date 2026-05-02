# 05. MaskFeat · MVP — Advanced Targets

## 🎯 핵심 질문

- 왜 pixel space 재구성 대신 **mid-level feature (HOG, CLIP)** 를 target 으로 삼는가?
- MaskFeat 의 HOG (Histogram of Oriented Gradients) 가 pixel 보다 **semantic 한 representation** 을 만드는 이유는?
- MVP (Masked Video with Pretrained CLIP) 의 **pretrained feature target** 이 갖는 ceiling 의존성은?
- Target 의 abstraction level 과 downstream task performance 의 trade-off 는?

## 🔍 왜 Target 을 변경하는가

**기존 MIM (BEiT, MAE, SimMIM) 의 한계**:
- **BEiT**: discrete token (8192 vocab) → tokenizer quality ceiling
- **MAE**: pixel space → low-level detail 에 focus (semantic 정보 약함)
- **SimMIM**: pixel space → 계산 비용, 저수준 noise 에 sensitive

**새로운 접근 (MaskFeat, MVP)**:
Target 을 다른 feature space 로 변경하자:
1. **HOG (hand-crafted)**: 중간 수준의 edge/texture 정보 → semantic 중간단계
2. **CLIP feature (pretrained)**: 고수준의 semantic 정보 → downstream task 와 better align

**직관**: 
"Pixel 복원" 대신 "의미있는 feature 복원" 을 학습하면,
결과적으로 더 semantic 한 representation 이 나온다.

## 📐 수학적 선행 조건

- **HOG 특징추출**: oriented gradient histogram 의 계산
- **Pretrained features**: CLIP ViT 의 patch embedding
- **Feature regression**: L2 loss (discrete classification 아닌 continuous)
- **Abstraction levels**: pixel (low) → HOG (mid) → CLIP (high)

## 📖 직관적 이해

```
Target Space 의 Abstraction Levels:

Pixel space (MAE):
│
├─ Intensity 분포
├─ Local contrast
└─ Fine details (noise-prone)
   └─→ Low-level, spatial detail focused

HOG space (MaskFeat):
│
├─ Edge orientations
├─ Gradient directions (불변성 있음)
└─ Mid-level texture structure
   └─→ Mid-level, structural focus

CLIP feature space (MVP):
│
├─ Object categories
├─ Semantic meanings
└─ High-level concepts
   └─→ High-level, semantic focus

Expected downstream performance:
┌────────────────────┐
│ Pixel target (MAE) │ ← linear probe: good (low-level)
│                    │   fine-tune: better
└────────────────────┘

┌────────────────────┐
│ HOG target         │ ← linear probe: decent
│ (MaskFeat)         │   fine-tune: very good
└────────────────────┘

┌────────────────────┐
│ CLIP target (MVP)  │ ← linear probe: excellent
│                    │   fine-tune: excellent
└────────────────────┘

Tradeoff: Target 이 high-level 일수록,
pretrained encoder 의 quality 에 의존 증가.
```

## ✏️ 엄밀한 정의

### 정의 5.16: HOG Feature Extraction

Image patch $p_i \in \mathbb{R}^{P \times P \times 3}$ (예: 16×16 patch) 에 대해:

**Step 1: Gradient 계산**
$$G_x = \nabla_x I, \quad G_y = \nabla_y I$$

**Step 2: Magnitude & Orientation**
$$M = \sqrt{G_x^2 + G_y^2}, \quad \theta = \mathrm{atan2}(G_y, G_x)$$

**Step 3: Orientation Binning**
$\theta \in [0, 2\pi)$ 를 $B$ 개 bin (보통 $B=9$) 로 양자화:
$$b_i = \lfloor \frac{\theta_i}{2\pi / B} \rfloor$$

**Step 4: Histogram**
각 bin 의 magnitude 누적:
$$H_b = \sum_{(x,y): b_{x,y} = b} M_{x,y}$$

**HOG feature**: $\mathbf{f}_{\mathrm{HOG}} = \mathrm{normalize}([H_0, \ldots, H_{B-1}]) \in \mathbb{R}^B$

(추가 spatial binning, gradient weighting 등 표준 HOG 사용)

### 정의 5.17: MaskFeat Training Objective

Masked region $M$ 에 대해:

**Target extraction** (offline, pretrain 전):
각 patch $p_i$ 에 대해 HOG feature 계산:
$$\mathbf{t}_i = \mathrm{HOG}(p_i) \in \mathbb{R}^{d_{\mathrm{HOG}}}$$

(일반적으로 $d_{\mathrm{HOG}} \approx 256$ 또는 더 작게 압축)

**Prediction head** (online, training 중):
Encoder output $\mathbf{h}_i$ 로부터 HOG feature 예측:
$$\hat{\mathbf{t}}_i = W_{\mathrm{hog}} \mathbf{h}_i + b_{\mathrm{hog}}$$

여기서 $W_{\mathrm{hog}} \in \mathbb{R}^{d_{\mathrm{HOG}} \times D}$ (D = encoder dim)

**Loss** (masked region 만):
$$L_{\mathrm{MaskFeat}} = \frac{1}{|M|} \sum_{i \in M} \|\hat{\mathbf{t}}_i - \mathbf{t}_i\|_2^2$$

### 정의 5.18: MVP (Masked Video with Pretrained CLIP)

Video frame 의 patch 에 대해:

**Target extraction** (frozen CLIP ViT):
각 patch 에 대해 CLIP ViT 의 feature 계산:
$$\mathbf{t}_i = \mathrm{CLIP}_{\mathrm{ViT}}(p_i) \in \mathbb{R}^d$$

(CLIP ViT-L: 1024D, ViT-B: 768D)

**Prediction** (learnable head):
$$\hat{\mathbf{t}}_i = W_{\mathrm{clip}} \mathbf{h}_i$$

**Loss**:
$$L_{\mathrm{MVP}} = \frac{1}{|M|} \sum_{i \in M} \|\hat{\mathbf{t}}_i - \mathbf{t}_i\|_2^2$$

## 🔬 정리와 증명

### 정리 5.10: Target 의 Abstraction Level 과 Representation Quality

**명제**: Target feature 의 abstraction level 이 높을수록,
downstream semantic task 에 대한 representation quality 가 높지만,
target encoder 의 quality 에 의존하는 ceiling 이 존재한다.

**증명 스케치**:

**Target 추상도와 성능의 관계**:

$$\text{Downstream performance} = \min(\text{Learned capability}, \text{Target quality ceiling})$$

- **Pixel target** (low abstraction):
  - Ceiling: image 의 theoretical maximum (고수준 semantic 정보 손실)
  - Advantage: target 자체에 noise 없음 (ground truth)
  
- **HOG target** (mid abstraction):
  - Ceiling: HOG 의 표현력 (edge/texture 는 잘 capture, high-level semantic 약함)
  - Advantage: noise 에 robust, 의미있는 level
  
- **CLIP target** (high abstraction):
  - Ceiling: CLIP 의 semantic alignment (매우 높음)
  - Advantage: semantic task 와 직접 align
  - Disadvantage: CLIP 자체의 오류가 model 로 propagate

**수식화**:

Target feature 의 task-relevance 를 $\tau \in [0, 1]$ 라 하면:
$$\text{Downstream acc} \approx \tau \times f(\text{pretrain data quality})$$

- Pixel: $\tau_{\text{pixel}} \approx 0.6$ (low-level 만)
- HOG: $\tau_{\mathrm{hog}} \approx 0.75$ (mid-level)
- CLIP: $\tau_{\mathrm{clip}} \approx 0.9$ (semantic align)

따라서:
$$\text{Acc}_{\mathrm{clip}} > \text{Acc}_{\mathrm{hog}} > \text{Acc}_{\mathrm{pixel}}$$

**단, CLIP target 의 경우**:
$$\text{Acc}_{\mathrm{clip}} \leq \text{CLIP 의 ImageNet accuracy}$$

CLIP 가 못하는 task 에서는 MVP 도 제한됨.

$$\square$$

### 정리 5.11: HOG 의 불변성과 Robustness

**명제**: HOG feature 는 pixel-level perturbation 에 대해
pixel 이나 naive feature 보다 robust 하다.

**증명**:

HOG 는 gradient 에 기반하므로:

1. **Global brightness change**: gradient 무변화
   $$\nabla(I + c) = \nabla I$$

2. **Small spatial jitter**: gradient direction 은 대체로 보존
   $$\nabla I(x + \delta x) \approx \nabla I(x) \quad \text{(for small } \delta x \text{)}$$

3. **Local contrast change**: normalized histogram 이므로 robust
   $$H(k \cdot I) = H(I) \quad \text{(after norm)}$$

반면 **pixel target** 은:
- Brightness shift 에 취약
- Spatial jitter 에 민감
- Quantization artifact 에 민감

따라서 HOG target 으로 훈련하면:
- 더 stable 한 loss landscape
- 더 robust 한 representation

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: HOG Feature 추출

```python
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def compute_hog(patch, bins=9):
    """
    Compute HOG (Histogram of Oriented Gradients) for a patch.
    patch: [H, W, 3] uint8 image patch
    returns: [bins] normalized histogram
    """
    # Convert to grayscale
    if patch.ndim == 3:
        patch_gray = patch.mean(axis=2).astype(np.float32)
    else:
        patch_gray = patch.astype(np.float32)
    
    # Compute gradients
    Gx = ndimage.sobel(patch_gray, axis=1)
    Gy = ndimage.sobel(patch_gray, axis=0)
    
    # Magnitude & orientation
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) + np.pi  # [0, 2π)
    
    # Histogram
    hist, _ = np.histogram(orientation.flatten(), bins=bins,
                           range=(0, 2*np.pi),
                           weights=magnitude.flatten())
    
    # Normalize
    hist = hist / (hist.sum() + 1e-6)
    
    return hist

# Test on sample patch
patch = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
hog_feature = compute_hog(patch, bins=9)

print(f"Patch shape: {patch.shape}")
print(f"HOG feature shape: {hog_feature.shape}")
print(f"HOG feature: {hog_feature}")
print(f"Sum of bins: {hog_feature.sum():.3f}")
```

### 실험 2: CLIP Feature Extraction (Mock)

```python
import torch
import torch.nn as nn

# Mock CLIP ViT (real version would use open_clip)
class MockCLIPViT(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        # Simulate CLIP ViT patch embedding
        self.patch_embed = nn.Conv2d(3, output_dim, kernel_size=16, stride=16)
    
    def forward(self, x):
        # x: [B, C, H, W]
        # Output: [B, N, D] or [B, D] (depending on pooling)
        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x

# Test
B, C, H, W = 4, 3, 224, 224
mock_clip = MockCLIPViT(output_dim=768)
images = torch.randn(B, C, H, W)
clip_features = mock_clip(images)

print(f"Input shape: {images.shape}")
print(f"CLIP feature shape: {clip_features.shape}")
print(f"Per-patch CLIP feature dim: {clip_features.shape[-1]}")
```

### 실험 3: Target Abstraction Level 비교

```python
import numpy as np

# Simulate downstream task performance with different targets
targets = ['Pixel', 'HOG', 'CLIP']
target_abstraction = [0.3, 0.6, 0.85]  # abstraction level
target_ceiling = [0.85, 0.90, 0.95]   # theoretical ceiling

# Linear probe performance
linear_probe_acc = [0.79, 0.81, 0.85]

# Fine-tuning performance
finetune_acc = [0.83, 0.87, 0.92]

# Plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Linear probe
axes[0].bar(targets, linear_probe_acc, alpha=0.7)
axes[0].set_ylabel('Linear Probe Accuracy')
axes[0].set_title('Linear Probe by Target Type')
axes[0].set_ylim(0.75, 0.90)
axes[0].grid(True, axis='y', alpha=0.3)

# Fine-tuning
axes[1].bar(targets, finetune_acc, alpha=0.7)
axes[1].set_ylabel('Fine-tuning Accuracy')
axes[1].set_title('Fine-tuning by Target Type')
axes[1].set_ylim(0.80, 0.95)
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/target_abstraction.png', dpi=100)

print("Target abstraction 이 높을수록 semantic task 에서 성능 우위")
```

### 실험 4: HOG Robustness 검증

```python
import numpy as np
from scipy import ndimage

def compute_hog(patch, bins=9):
    if patch.ndim == 3:
        patch_gray = patch.mean(axis=2).astype(np.float32)
    else:
        patch_gray = patch.astype(np.float32)
    
    Gx = ndimage.sobel(patch_gray, axis=1)
    Gy = ndimage.sobel(patch_gray, axis=0)
    
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) + np.pi
    
    hist, _ = np.histogram(orientation.flatten(), bins=bins,
                           range=(0, 2*np.pi),
                           weights=magnitude.flatten())
    hist = hist / (hist.sum() + 1e-6)
    return hist

# Original patch
patch_orig = np.random.rand(16, 16, 3) * 255

# Perturbations
patch_bright = patch_orig + 30  # brightness shift
patch_jitter = np.roll(patch_orig, 1, axis=0)  # spatial jitter
patch_noise = patch_orig + np.random.randn(16, 16, 3) * 10  # noise

# HOG comparison
hog_orig = compute_hog(patch_orig)
hog_bright = compute_hog(patch_bright)
hog_jitter = compute_hog(patch_jitter)
hog_noise = compute_hog(patch_noise)

# Distance (L2)
dist_bright = np.linalg.norm(hog_orig - hog_bright)
dist_jitter = np.linalg.norm(hog_orig - hog_jitter)
dist_noise = np.linalg.norm(hog_orig - hog_noise)

print(f"HOG robustness test:")
print(f"  Brightness shift: L2={dist_bright:.4f} (invariant)")
print(f"  Spatial jitter: L2={dist_jitter:.4f} (relatively robust)")
print(f"  Noise: L2={dist_noise:.4f} (somewhat sensitive)")
print(f"\nHOG 는 global perturbation 에는 불변, local 에는 준-robust")
```

## 🔗 실전 활용

**MaskFeat (Wei et al. 2022)**:
1. ImageNet 또는 larger scale data
2. 50-75% masking ratio
3. Offline HOG extraction (모든 patch에 미리 계산)
4. Encoder → head 로 HOG target 예측
5. MSE loss
6. 300 epochs

**MVP (Masked Video with Pretrained)**:
1. Video dataset (e.g., Kinetics-400, ImageNet video subset)
2. Tube masking (같은 spatial 위치, 여러 frame 연속 mask)
3. Frozen CLIP ViT 로 target feature 추출
4. Encoder → projection head 로 CLIP feature 예측
5. MSE loss
6. Masked video 기반 pretraining

**구현 체크리스트 (MaskFeat)**:
- HOG 계산: skimage.feature.hog 또는 custom implementation
- Target 저장: offline 으로 모든 patch 의 HOG 계산 후 저장 (메모리/속도 tradeoff)
- Prediction head: linear projection (D → HOG_dim)
- Masking: standard random uniform

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **HOG 가정** | Edge/gradient 가 의미있는 feature | Texture-less image 에서는 약함 |
| **Offline target** | HOG 를 offline 으로 미리 계산 | Update 불가능, 고정 target |
| **CLIP 의존성** | MVP 는 CLIP 의 품질에 ceiling 의존 | CLIP 이 못하는 task 는 제한됨 |
| **Target 시간적 안정성** | HOG, CLIP feature 는 video 에서 temporal consistency 부족 가능 | Temporal smoothing 필요할 수 있음 |
| **Downstream task generalization** | Semantic task 에는 우수하지만, 저수준 task (detail preservation) 는 MAE 우위 | Task 에 따라 target 선택 필요 |

## 📌 핵심 정리

$$\boxed{\text{MaskFeat}: \, \mathbf{h}_i \to W \mathbf{h}_i \approx \mathrm{HOG}(p_i)}$$

$$\boxed{\text{MVP}: \, \mathbf{h}_i \to W \mathbf{h}_i \approx \mathrm{CLIP}(p_i)}$$

| 항목 | Pixel (MAE) | HOG (MaskFeat) | CLIP (MVP) |
|------|-------------|----------------|-----------|
| **Target abstraction** | Low (0-255) | Mid (9-256 dim) | High (768 dim) |
| **Robustness** | Low (noise-sensitive) | High (gradient-based) | High (pretrained) |
| **Semantic alignment** | Low | Mid | Very high |
| **Target ceiling** | Image's max entropy | HOG 표현력 | CLIP accuracy |
| **Fine-tuning** | 83-85% (ImageNet) | 85-87% | 87-90% |
| **Downstream diversity** | Excellent (general) | Good (mid-level) | Excellent (semantic) |

## 🤔 생각해볼 문제

### 문제 1 (기초): HOG Feature 의 Invariance

HOG 는 왜 **global brightness shift** 에 불변이면서도
**local edge** 정보는 보존하는가?

Gradient 와 normalization 의 수학적 성질을 이용해 설명하시오.

<details>
<summary>해설</summary>

**Gradient 의 brightness-invariance**:
$$\nabla(I + c) = \nabla I$$

Gradient 는 intensity 의 **변화율** 만 보므로, constant offset 에 불변.

**Edge 정보 보존**:
Gradient 가 0 이 아닌 곳 = intensity 가 변하는 곳 = edge

따라서:
$$\mathrm{magnitude}(G(p)) \neq 0 \quad \Rightarrow \quad \text{edge present}$$

**Histogram normalization**:
$$\mathrm{HOG}_{\text{norm}} = \frac{H}{|H|_1}$$

개별 gradient magnitude 의 절댓값은 중요하지 않고,
**상대적 orientation distribution** 만 중요 → brightness invariance.

**결론**: Gradient (불변) + Histogram (정규화) = brightness-invariant edge detector

</details>

### 문제 2 (심화): Target Ceiling 과 Downstream Performance

MVP (CLIP target) 의 ceiling 은 "frozen CLIP ViT 가 도달한 성능" 인가,
아니면 "CLIP 의 semantic alignment" 인가?

두 관점이 다른 이유와, 각각의 현실적 함의를 논의하시오.

<details>
<summary>해설</summary>

**Ceiling 정의 1: CLIP accuracy**
```
MVP performance ≤ CLIP 's ImageNet accuracy
```
- CLIP 이 못하는 것 (예: fine-grained recognition) → MVP 도 못함
- 구체적: CLIP-ViT-L ≈ 75% ImageNet top-1
- 따라서 MVP fine-tuning ≤ 75% ceiling

**Ceiling 정의 2: Semantic alignment**
```
MVP performance ≤ (task 의 semantic content)
```
- Task 가 semantic 할수록 CLIP feature 이 useful
- Task 가 low-level (detail, texture) 에만 의존 → CLIP 의 이점 없음
- 예: edge detection task → CLIP 은 edges 에 focus 안 함

**현실적 함의**:

**Semantic task (ImageNet classification)**:
- MVP 우위 (CLIP 과 align)
- Ceiling: CLIP 의 semantic knowledge

**Low-level task (depth estimation, edge detection)**:
- MAE 우위 (pixel-level detail)
- MVP 는 CLIP 이 도움 안 되므로 overhead 만 발생

**결론**: MVP 는 "semantic task 의 transfer learning" 에 특화.
General-purpose pretraining 으로는 target-diversity (pixel + HOG + CLIP mixed) 더 낫음.

</details>

### 문제 3 (논문 비평): MaskFeat vs MVP vs MAE 의 trade-off

각 방법의 target 선택이 implicit 하게 가정하는
"downstream task distribution" 을 명확히 하시오.
현실의 vision application 에서 어느 것이 best general-purpose 인가?

<details>
<summary>해설</summary>

**MAE (pixel target)**:
- Assumes: low-level detail (edge, texture) 이 중요
- Best for: detection, segmentation, dense prediction
- Weakness: semantic task (classification, retrieval) 에서 sub-optimal

**MaskFeat (HOG target)**:
- Assumes: mid-level structure (edge/shape orientation) 이 중요
- Best for: balanced (detection + classification)
- Strength: robustness, generality

**MVP (CLIP target)**:
- Assumes: high-level semantic 이 최우선
- Best for: classification, retrieval, zero-shot
- Weakness: detail 관련 task, 낮은 resolution

**현실의 distribution**:

Recent vision application (2023 onwards):
- Foundation model 의 부상 → semantic alignment 중시
- Multi-task learning (detection + segmentation + classification) → balanced target 필요
- Large-scale pretraining → target quality 의존도 증가

**권장**:
- **General foundation model**: MaskFeat (mid-level, balanced)
- **Semantic-focused**: MVP (high-level, CLIP align)
- **Dense prediction**: MAE (pixel-level detail)
- **State-of-art**: Multi-objective (MAE loss + HOG loss + CLIP loss 동시)

</details>

---

<div align="center">

[◀ 이전](./04-simmim.md) | [📚 README](../README.md) | [다음 ▶](../ch6-multimodal/01-clip.md)

</div>
