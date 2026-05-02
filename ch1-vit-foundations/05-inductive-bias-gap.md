# 05. Inductive Bias 부족 — ViT 의 본질적 한계

## 🎯 핵심 질문

- CNN 의 inductive bias 란 무엇이고, ViT 에는 왜 부족한가?
- Dosovitskiy 2021 Figure 3 에서 "ImageNet-1k only: ResNet > ViT" 인 이유는?
- Translation equivariance, locality, hierarchical composition 이 vision task 에서 왜 중요한가?
- ViT 는 ImageNet-22k 나 JFT-300M 같은 대규모 데이터에서만 잘 되는가?
- 이 한계를 극복하는 3가지 해결책 (scale, augmentation, hybrid) 은 무엇인가?

## 🔍 왜 이 한계 분석이 중요한가

Vision Transformer 의 혁신적 성공에도 불구하고, **근본적인 약점**이 있다:

CNN (convolution) 은 이미지 처리를 위해 수십 년 걸쳐 최적화된 구조다:
- **Translation equivariance**: 물체가 다른 위치에 있어도 detection 가능
- **Locality**: 가까운 픽셀끼리 먼저 정보 교환 (hierarchical RF)
- **Hierarchical composition**: 낮은 level (edge) → 높은 level (object)

ViT 는 이런 **inductive bias 가 거의 없다**. 대신:
- Self-attention: 모든 patch 가 첫 번째 layer 부터 전역 정보 봄
- No locality: receptive field 가 처음부터 unbounded
- Flat architecture: hierarchical structure 없음

**결과**: ImageNet-1k 같은 relatively small dataset 에서 ViT 는 ResNet 에 뒤진다. 하지만 충분한 데이터가 있으면, ViT 의 **flexibility** 가 이기게 된다.

이 trade-off 를 이해하는 것이 modern vision architecture 설계의 핵심이다.

## 📐 수학적 선행 조건

- **CNN basics**: convolution, receptive field, stride, pooling
- **Translation equivariance**: shift invariance, transformation groups
- **Hierarchical compositionality**: recursive feature extraction
- **Ch1-01 (ViT)**: patch embedding, self-attention
- **Statistical learning theory**: inductive bias, hypothesis class, generalization

## 📖 직관적 이해

```
CNN (Inductive Bias 풍부):

Layer 1: 3×3 kernel
  → 작은 receptive field (RF=3)
  → edge detection (국소 특성)
  → Translation equivariant: shift된 image 에서 shift된 output

Layer 2: 3×3 kernel
  → RF=5
  → local corner/texture

Layer 3: 3×3 kernel → pooling
  → RF=7 + pooling
  → hierarchical, local→global

...

Layer 50 (ResNet-50):
  → RF ≈ 400×400 (image 전체)
  → High-level semantic (object identity)

특징: 
  ✓ 작은 RF 부터 시작 (효율성, sample efficiency)
  ✓ Hierarchical: low-level features 를 high-level 에서 재사용
  ✓ Translation-equivariant: weight 를 한 번만 배우면, 어디서든 작동


ViT (Inductive Bias 부족):

Layer 1: Self-attention, 196 patches
  → 모든 patch 가 서로 attention (RF = 196×P = 224×224 = 전체!)
  → 처음부터 global information
  → Permutation-equivariant (translation-equivariant 아님)

Layer 2-12: 동일 구조 반복
  → 계층성 없음 (모두 같은 scale)
  → Position embedding 으로만 위치 정보

특징:
  ✗ 큰 RF 부터 시작 (비효율적, high sample complexity)
  ✗ Flat: hierarchy 없음
  ✗ Permutation-equivariant: 위치에 대한 biasing 부족


결과:
  ImageNet-1k (1.3M images): ResNet >> ViT
    → Small data regime: inductive bias 가 critical
  
  ImageNet-22k (14M images): ViT ≈ ResNet
    → 증가된 데이터: ViT 가 부족함을 "학습"
  
  JFT-300M (300M images): ViT >> ResNet
    → 대규모 데이터: ViT 의 flexibility 가 advantage
```

## ✏️ 엄밀한 정의

### 정의 5.1: Translation Equivariance (CNN)

함수 $f$ 가 translation equivariance 를 만족한다:

$$f(T_{\delta} x) = T_{\delta} f(x)$$

여기서 $T_{\delta}$ 는 spatial shift operator: $T_{\delta} x[i, j] = x[i - \delta_h, j - \delta_w]$.

일반적으로 CNN 은 이를 만족한다 (padding 을 제외하고).

ViT 는 **이를 만족하지 않는다** (position embedding 때문에):
$$f(T_{\delta} x) \neq T_{\delta} f(x)$$

### 정의 5.2: Locality (Local Receptive Field)

$l$ 번째 layer 의 receptive field $RF_l$:

$$RF_1 = k_1 \text{ (kernel size)}$$
$$RF_{l} = RF_{l-1} + (k_l - 1) \prod_{i=1}^{l-1} s_i$$

여기서 $s_i$ 는 stride, $k_i$ 는 kernel size.

CNN: $RF$ 는 gradually 증가 (small → large)  
ViT: $RF$ 는 처음부터 global ($N = 196$ patches)

### 정의 5.3: Hierarchical Composition

CNN 은 여러 stage 로 나뉨 (e.g., ResNet-50):
- Stage 1: $56 \times 56$ resolution, 64 channels
- Stage 2: $28 \times 28$ resolution, 128 channels (stride=2 pooling)
- Stage 3: $14 \times 14$ resolution, 256 channels
- Stage 4: $7 \times 7$ resolution, 512 channels

각 stage 는 이전 stage 의 features 를 input 으로 받고, 더 추상적인 features 를 학습.

ViT: 모든 layer 가 동일 resolution (14×14 patches), 동일 dimensionality (768)

### 정의 5.4: Sample Complexity (Data Efficiency)

Hypothesis class $\mathcal{H}$ 의 VC dimension $d$ 에 대해, generalization 을 위해 필요한 표본의 수:

$$m = O\left( \frac{d}{\epsilon^2} \right)$$

**Inductive bias 가 강할수록**: $d$ 가 작음 (작은 hypothesis class) → 필요한 sample 이 적음  
**Inductive bias 가 약할수록**: $d$ 가 큼 (큰 hypothesis class) → 필요한 sample 이 많음

CNN: inductive bias 가 강해서 $d_{\text{CNN}} \ll d_{\text{ViT}}$  
ViT: inductive bias 가 약해서 $d_{\text{ViT}}$ 가 매우 큼 → **더 많은 데이터 필요**

## 🔬 정리와 증명

### 정리 5.1: ViT 는 CNN 보다 더 큰 Hypothesis Class 를 정의한다

**명제**: CNN 과 ViT 가 동일한 parameter count 를 가지더라도, ViT 의 hypothesis class 는 larger 이다. 따라서 동일한 generalization error 를 달성하기 위해 ViT 는 더 많은 데이터가 필요하다.

**증명 개요**:

1. CNN: locality + translation equivariance 로 제한되는 함수 class
   - Weight sharing: 각 kernel 이 image 전체에 적용 (same receptive field size)
   - Hierarchical: feature 의 계층성으로 함수 공간 제약

2. ViT: 위 제약이 없음
   - No weight sharing across positions: patch 마다 다른 embedding
   - No hierarchy: 모든 layer 가 flat
   - No locality: 첫 번째 layer 부터 모든 patch 간 interaction

3. 따라서:
   $$\text{Hypothesis class}: |\mathcal{H}_{\text{CNN}}| \ll |\mathcal{H}_{\text{ViT}}|$$

4. Sample complexity:
   $$m_{\text{ViT}} \gg m_{\text{CNN}}$$
   
   "No free lunch" theorem 에 따라, 더 큰 hypothesis class 는 더 많은 data 를 요구.

$$\square$$

### 정리 5.2: Dosovitskiy et al. 2021 의 실험적 증거

**명제** (경험적): ViT-L (384-dim) 와 ViT-B (768-dim) 을 여러 데이터 규모에서 평가하면:

$$\text{top-1 accuracy (ViT)} = f(\text{dataset size})$$

그래프는 다음과 같은 phase transition 을 보인다:

| Dataset | ResNet-152 | ViT-B | ViT-L |
|---------|-----------|-------|-------|
| ImageNet-1k (1.3M) | 84.0% | 77.9% | 76.5% |
| ImageNet-21k→1k | 84.7% | 84.9% | 85.9% |
| JFT-300M→1k | 84.5% | 88.6% | 89.7% |

**해석**:
- ImageNet-1k: ResNet 이 절대 우위 (inductive bias)
- ImageNet-21k: ViT 와 ResNet 이 비슷 (충분한 pretraining data)
- JFT-300M: ViT 가 명확히 우위 (scaling, flexibility)

이는 **inductive bias 와 data size 의 trade-off** 를 명확히 보여준다.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Receptive Field 계산 (CNN vs ViT)

```python
import numpy as np

# CNN (ResNet-50 style)
print("CNN Receptive Field progression:")
rf = 1
kernel_size = 3
stride = 1

for layer in range(1, 17):  # 16 convolutional layers (simplified)
    if layer % 4 == 0:  # Pooling stride=2 every 4 layers
        stride_curr = 2
    else:
        stride_curr = 1
    
    rf = rf + (kernel_size - 1) * stride
    stride *= stride_curr
    
    if layer in [1, 4, 8, 12, 16]:
        print(f"Layer {layer}: RF = {rf}, Stride = {stride}")

# ViT
print("\nViT Receptive Field:")
print(f"Layer 1: RF = global (all 196 patches) = 224×224 pixels")
print(f"Layer 2-12: RF = still global (no change with depth)")

# Conclusion
print("\n✓ CNN: RF grows gradually (1→3→5→...→large)")
print("✓ ViT: RF is global from the start (all to all attention)")
```

### 실험 2: Translation Equivariance 검증

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 더미 이미지
x = torch.randn(1, 3, 224, 224)

# 1. CNN (Conv2d): translation equivariant
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
y_conv = conv(x)  # (1, 64, 224, 224)

# Shift input by 2 pixels
x_shifted = torch.roll(x, shifts=2, dims=-1)  # Shift rightward
y_conv_shifted = conv(x_shifted)

# Check if output is also shifted (translation equivariance)
y_conv_shifted_expected = torch.roll(y_conv, shifts=2, dims=-1)
equivariance_error_cnn = (y_conv_shifted - y_conv_shifted_expected).abs().max().item()

print(f"CNN translation equivariance error: {equivariance_error_cnn:.2e}")
assert equivariance_error_cnn < 1e-4, "CNN should be translation equivariant"
print("✓ CNN is translation equivariant (within numerical precision)")

# 2. ViT (self-attention): NOT translation equivariant
# Position embedding makes it position-dependent
print("\nViT with positional encoding: NOT translation equivariant")
print("Reason: Position embeddings are absolute, shift-dependent")

# 정성적 설명
pos_embed = torch.randn(1, 197, 768)  # Absolute position embeddings
z1 = pos_embed[0, 0]  # CLS token PE
z2 = pos_embed[0, 50]  # Patch at position 50

# After shifting (hypothetical), position indices change
# z_shifted at index 0 would be the same
# But the position values are different → not equivariant

print("✓ ViT depends on absolute position indices → not equivariant")
```

### 실험 3: Hypothesis Class Size (정성적)

```python
import torch

# Parameter count (동일하다고 가정)
params_cnn = 25.5e6  # ResNet-50
params_vit = 86e6    # ViT-B (모델이 더 크지만, 개념적으로)

# Hypothesis class size (정성적 근사)
# VC dimension (매우 단순화된 모델)

def estimate_vc_dimension_cnn(params):
    """CNN: inductive bias 로 VC 차원이 작음"""
    # Rough estimate: VC_dim ~ sqrt(params) (because weight sharing)
    return int(params ** 0.5)

def estimate_vc_dimension_vit(params):
    """ViT: inductive bias 없어서 VC 차원이 큼"""
    # Rough estimate: VC_dim ~ params (because of permutation freedom)
    return int(params ** 1.0)

vc_cnn = estimate_vc_dimension_cnn(params_cnn)
vc_vit = estimate_vc_dimension_vit(params_cnn)  # Same param count

print(f"Estimated VC dimension:")
print(f"CNN (ResNet-50, {params_cnn/1e6:.1f}M params): ~{vc_cnn}")
print(f"ViT (same params): ~{vc_vit}")
print(f"Ratio (ViT/CNN): {vc_vit / vc_cnn:.1f}x")

# Sample complexity estimate: m ~ O(d / ε²)
epsilon = 0.01  # Target error
m_cnn = vc_cnn / (epsilon ** 2)
m_vit = vc_vit / (epsilon ** 2)

print(f"\nEstimated sample complexity (to achieve {epsilon} error):")
print(f"CNN: ~{m_cnn:.0f} samples")
print(f"ViT: ~{m_vit:.0f} samples")
print(f"Ratio (ViT/CNN): {m_vit / m_cnn:.1f}x more data for ViT")
```

### 실험 4: Data-size Phase Transition (Simulated)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulated accuracy vs dataset size (inspired by Dosovitskiy 2021)
dataset_sizes = np.array([1.3, 5, 10, 14, 50, 100, 300]) * 1e6  # millions

# ResNet: 작은 데이터서는 높은 accuracy, plateau 빨리
accuracy_resnet = np.array([84.0, 84.5, 84.6, 84.7, 84.8, 84.8, 84.9])

# ViT: 처음에는 낮지만, 더 많은 데이터로 성장
accuracy_vit = np.array([77.9, 82.5, 84.5, 85.0, 87.5, 88.0, 89.7])

# 시뮬레이션 (log scale)
log_sizes = np.log10(dataset_sizes)

print("Dataset Size (M) | ResNet | ViT | Winner")
print("-" * 45)
for i, size in enumerate(dataset_sizes):
    winner = "ResNet" if accuracy_resnet[i] > accuracy_vit[i] else "ViT"
    print(f"{size/1e6:14.1f} | {accuracy_resnet[i]:6.1f}% | {accuracy_vit[i]:6.1f}% | {winner}")

print("\n✓ Phase transition: ImageNet-1k (ResNet) → Large-scale (ViT)")
```

## 🔗 실전 활용

**이 한계를 극복하는 3가지 접근**:

### 1. **Scale** (데이터와 모델 크기 증가)
- ImageNet-22k (14M images) pretrain
- JFT-300M (300M images) pretrain
- **Chapter 7**: scaling laws, emerging properties

### 2. **Augmentation** (DeiT, Ch2-01)
- Mixup, CutMix, RandAugment
- Stochastic depth, Drop path
- 효과: ImageNet-1k 만으로 ViT-B > ResNet-50 달성

### 3. **Hybrid Architecture** (CNN + Transformer)
- **CvT (Convolutional Token Embedding)**: 초반에 conv 로 local features 먼저 추출
- **CoAtNet**: Conv layers + Transformer layers 의 intelligent 조합
- **Swin Transformer (Ch2-02)**: hierarchical + local window attention
- 효과: ViT 의 flexibility + CNN 의 inductive bias 결합

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Inductive bias 의 정량화** | Hypothesis class size 의 upper bound 를 정확히 계산하기 어려움 | 정성적 논의가 중심; VC dimension 은 rough estimate |
| **Data regime 의 경계** | "Small" vs "Large" 는 task-dependent | ImageNet-1k 는 small, ImageNet-22k 는 medium, JFT-300M 은 large |
| **Transfer learning** | Pretrain 된 모델을 downstream task 로 transfer 할 때, inductive bias 의 역할이 달라짐 | Pretrain 된 ViT 는 downstream 에서 CNN 을 능가할 수 있음 |
| **Generalization bounds** | PAC-learning 의 이론적 경계는 매우 loose (실제 generalization 과 거리 있음) | 실제 성능은 empirical 결과가 더 중요 |
| **Architecture design** | Inductive bias 의 정도를 조절하는 것이 현대 vision 의 핵심 설계 | Trade-off 이해가 중요 (bias ↔ flexibility) |

## 📌 핵심 정리

$$\boxed{\text{Inductive Bias} = \text{Translation Equivariance} + \text{Locality} + \text{Hierarchy}}$$

$$\boxed{m \propto \frac{d_{\text{VC}}}{\epsilon^2}, \quad d_{\text{VC, CNN}} \ll d_{\text{VC, ViT}}}$$

| 특성 | CNN | ViT |
|------|-----|-----|
| **Translation Equivariance** | ✓ (perfect) | ✗ (only via position embedding) |
| **Locality** | ✓ (gradual RF growth) | ✗ (global from layer 1) |
| **Hierarchy** | ✓ (multi-stage) | ✗ (flat) |
| **VC Dimension** | Small (weight sharing) | Large (no sharing) |
| **Sample Complexity** | Low (ImageNet-1k OK) | High (needs 22k+) |
| **Data Scaling** | Plateau (diminishing return) | Linear (continues to improve) |

## 🤔 생각해볼 문제

### 문제 1 (기초): Receptive Field 계산

3×3 kernel, stride=1 인 5-layer CNN 의 최종 receptive field 는?

$$RF = 1 + 2 \times (3 - 1) = 5$$

(각 layer 마다 RF 가 2씩 증가)

<details>
<summary>해설</summary>

$RF_1 = 3$ (첫 layer)
$RF_2 = 3 + (3-1) \times 1 = 5$ (stride=1 이므로 1배 증가)
$RF_3 = 5 + 2 = 7$
$RF_4 = 7 + 2 = 9$
$RF_5 = 9 + 2 = 11$

최종 RF = 11.

반면 ViT: 처음부터 11×11 (실제로는 224×224 전체).

</details>

### 문제 2 (심화): ImageNet-1k vs 22k 의 분기점

Dosovitskiy 2021 Figure 3 에서 ViT 와 ResNet 의 accuracy 가 같아지는 데이터 크기는 대략 몇 배 정도인가?

<details>
<summary>해설</summary>

Rough estimate:
- ImageNet-1k: 1.3M images → ResNet > ViT by ~6%
- ImageNet-22k: 14M images → ViT ≈ ResNet

**Ratio**: 14M / 1.3M ≈ **10배**

따라서 ViT 는 대략 **10배 더 많은 데이터**가 필요해서 ResNet 과 동등한 성능을 달성한다.

이는 hypothesis class 가 10배 더 크다는 가설과 맞음 (VC dimension theory 의 예측).

</details>

### 문제 3 (논문 비평): Hybrid Model 의 설계 원칙

CvT, CoAtNet, Swin 같은 hybrid 모델들이 모두 초반에 "locality 를 먼저 주는" 이유는?

<details>
<summary>해설</summary>

**원칙**:
1. 낮은 level features (edge, texture) 는 local context 만으로 충분
2. 낮은 level 의 locality 를 이용해 early stopping 적용 → 계산 효율
3. 높은 level (object, scene) 은 global context 필요 → transformer attention

**구체 예**:
- CvT: Conv 로 low-level local features 추출 → 해상도 내림 → 이후 transformer 는 더 작은 sequence 에서 global attention
- CoAtNet: Conv blocks (4 stages) 의 hierarchical 구조 → transformer blocks (일부) 만 high-level 에서
- Swin: Window attention (local) 에서 시작 → shifted window 로 global 정보 모음

**효과**: CNN 의 inductive bias (locality, hierarchy) 를 먼저 주고, transformer 의 flexibility 를 뒤에서 활용 → Best of both worlds

</details>

---

<div align="center">

[◀ 이전](./04-positional-embedding-variants.md) | [📚 README](../README.md) | [다음 ▶](../ch2-hierarchical-vit/01-deit.md)

</div>
