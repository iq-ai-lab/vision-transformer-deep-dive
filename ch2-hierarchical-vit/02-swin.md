# 02. Swin Transformer — Shifted Window Attention

## 🎯 핵심 질문

- Vanilla ViT 의 $O(n^2)$ attention 복잡도를 어떻게 $O(n \cdot w^2)$ 로 줄일 수 있는가?
- Window attention 이 local receptive field 를 강제하므로, global information flow 는 어떻게 보장하는가?
- Shifted window 의 cyclic shift + attention mask 는 어떻게 작동하는가?
- Patch merging 을 통한 4-stage hierarchical structure 는 detection · segmentation 에 왜 중요한가?

## 🔍 왜 이 window 가 Hierarchical ViT 의 핵심인가

ViT 는 patch embedding 을 하나의 sequence 로 취급하고, 모든 patch 간의 self-attention 을 계산한다. 
이는 long-range dependency 를 직접 모델링하는 강점이지만, 복잡도 $O(n^2)$ 이라는 치명적 약점을 가진다.

ImageNet-1k (224×224 이미지, 16×16 patch) 기준으로 $n = 196$ patch 이므로 
$O(196^2) = O(38416)$ 번의 attention 연산이 필요하다.

**Liu et al. (2021) Swin Transformer 의 통찰**: 
- CNN 의 강점: **locality** (인접한 픽셀끼리만 convolution) → $O(n)$ 복잡도
- ViT 의 강점: **long-range dependency** (모든 token 과의 상호작용) → $O(n^2)$ 복잡도

DeiT 는 augmentation 과 distillation 으로 ViT 의 데이터 효율성을 개선했다. 
Swin 은 다른 접근: **window attention** 으로 복잡도를 선형화하면서도, 
**shifted window** 로 global receptive field 를 회복한다.

결과: Dense prediction (detection, segmentation) 에서 CNN backbone (ResNet, EfficientNet) 을 능가. 
**Hierarchical ViT** 의 시작.

## 📐 수학적 선행 조건

- **ViT 기초** (Ch1-01 ~ 04): Self-attention, multi-head, position embedding
- **DeiT** (Ch2-01): Augmentation, distillation 의 inductive bias 주입
- **Receptive field** (CNN Deep Dive): Local structure, hierarchical features (FPN 스타일)
- **Attention complexity**: $O(n^2)$ vs $O(n)$ 의 trade-off
- **Cyclic shift & masking**: 2D 배열 manipulation, index arithmetic
- **Patch merging**: Channel concatenation, linear layer 를 통한 dimension reduction

## 📖 직관적 이해

```
ViT 의 Global Attention vs Swin 의 Window Attention:

┌───────────────────────────────────┐
│ ViT: Global Self-Attention        │
│                                   │
│ [Patch]     [Patch]     [Patch]   │
│    │          ╱│╲          │      │
│    └─────────┼─┼─────────┘        │
│    (모든 patch 쌍이 attend)       │
│    Complexity: O(n²) = O(196²)    │
│                                   │
│ 장점: Long-range 정보 직접 포착   │
│ 단점: 매우 느림                   │
└───────────────────────────────────┘

┌───────────────────────────────────┐
│ Swin: Local Window Attention      │
│                                   │
│ Window 1        Window 2           │
│ ┌────────┐   ┌────────┐           │
│ │[P][P]  │   │[P][P]  │           │
│ │[P][P]  │   │[P][P]  │           │
│ └────────┘   └────────┘           │
│   (각 window 내에만 attend)       │
│   Complexity: O(n·w²) = O(196·49) │
│                                   │
│ 장점: 선형 복잡도 + 지역 구조 포착│
│ 단점: Window 사이 정보 연결 필요   │
└───────────────────────────────────┘

┌───────────────────────────────────┐
│ Swin: Shifted Window Mechanism     │
│                                   │
│ Layer L (Window):          Layer L+1 (Shifted):
│ ┌──┬──┬──┐            ┌──┬──┬──┐  │
│ │  │  │  │            │  │  │  │  │
│ ├──┼──┼──┤            ├──┼──┼──┤  │
│ │ [Window] │ ──→ shift ──→ │[New]│  │
│ ├──┼──┼──┤            ├──┼──┼──┤  │
│ │  │  │  │            │  │  │  │  │
│ └──┴──┴──┘            └──┴──┴──┘  │
│                                   │
│ Shifted window 사이에는 새로운    │
│ connection 이 만들어짐            │
│ → Hierarchical receptive field    │
└───────────────────────────────────┘
```

**비유**: 
- ViT 는 "한 사람이 전체 회의실을 한눈에 보려고" (O(n^2) 노력)
- Swin 은 "작은 테이블들로 나누되, 테이블 간 사람을 옮겨가며" (O(n·w^2) 효율성 + global connectivity)

## ✏️ 엄밀한 정의

### 정의 2.5: Window Partition

이미지의 patch embedding 을 크기 $w \times w$ 의 non-overlapping windows 로 분할한다.

전체 patch 배열: $X \in \mathbb{R}^{H' \times W' \times D}$ (여기서 $H' = H/P, W' = W/P$ 는 patch 수)

Window partition 함수:
$$\mathcal{W}(X) = \{W_1, W_2, \ldots, W_M\}, \quad M = \frac{H'}{w} \cdot \frac{W'}{w}$$

각 window 의 크기: $w \times w \times D$ (즉, $n_w = w^2$ 개의 token 을 담음).

### 정의 2.6: Window Shifted Attention (WSA)

$l$ 번째 layer 에서:

**짝수 layer** ($l$ 이 짝수): Regular window partition
$$Z_l^{\text{WSA}} = \mathrm{WindowAttn}(\mathcal{W}(X_{l-1}))$$

**홀수 layer** ($l$ 이 홀수): Cyclic shifted + window partition
$$Z_l^{\text{WSA}} = \mathrm{WindowAttn}(\mathcal{W}(\mathrm{CyclicShift}(X_{l-1})))$$

Cyclic shift: $X' = \text{roll}(X, (\lfloor w/2 \rfloor, \lfloor w/2 \rfloor))$ (공간적 이동, spatial dimension 에서)

**Attention mask**: Shifted window 내 패딩 영역과 다른 semantic region 사이의 attention 을 mask 처리.

### 정의 2.7: Local Window Self-Attention

크기 $w \times w \times D$ 인 single window 내에서의 self-attention:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left( \frac{QK^\top}{\sqrt{d}} + B \right) V$$

여기서:
- $Q, K, V \in \mathbb{R}^{n_w \times d}$, $n_w = w^2$
- $B \in \mathbb{R}^{n_w \times n_w}$ 는 **relative position bias** (learnable, 정의 2.8)

Complexity: $O(n_w^2 \cdot D) = O(w^4 \cdot D)$ per window.

전체: $\sum_{m=1}^{M} O(w^4 \cdot D) = O(M \cdot w^4 \cdot D) = O((H' \cdot W') \cdot w^2 \cdot D) = O(n \cdot w^2)$ 
(여기서 $n = H' \cdot W'$ 는 총 patch 수).

### 정의 2.8: Relative Position Bias

절대 position embedding 대신, relative position 에 기반한 bias matrix:

$$B_{ij} = B_{\text{rel}}(\Delta_i - \Delta_j)$$

여기서 $\Delta_i, \Delta_j \in \{-(w-1), \ldots, w-1\}^2$ 는 상대 좌표.

2D bias: $B_{\text{rel}} \in \mathbb{R}^{(2w-1) \times (2w-1) \times n_h}$ (n_h 는 attention head 수).

이는 translation invariance 를 강화한다.

### 정의 2.9: Patch Merging

연속된 4개 patch 를 concatenate 하고, linear layer 로 channel 을 줄임.

입력: $X \in \mathbb{R}^{H' \times W' \times D}$

Reshape & concatenate:
$$X_{\text{merge}} = \mathrm{Concat}(X[::2, ::2, :], X[1::2, ::2, :], X[::2, 1::2, :], X[1::2, 1::2, :])$$

결과: $X_{\text{merge}} \in \mathbb{R}^{(H'/2) \times (W'/2) \times 4D}$

Linear projection:
$$X_{\text{out}} = W_{\text{linear}}(X_{\text{merge}}), \quad W_{\text{linear}} \in \mathbb{R}^{4D \times 2D}$$

최종: $X_{\text{out}} \in \mathbb{R}^{(H'/2) \times (W'/2) \times 2D}$

결과적으로 spatial resolution 은 $1/2$, channel 은 $2\times$ 증가.

## 🔬 정리와 증명

### 정리 2.3: Window Attention 의 선형 복잡도

**명제**: Window size $w$ 를 고정했을 때, 총 patch 수 $n$ 에 대해 
window attention 의 복잡도는 $O(n \cdot w^2)$ 이다 (선형).

**증명**:

전체 patch: $n = H' \times W'$ (2D grid)

Window partition: $M = \frac{H'}{w} \times \frac{W'}{w} = \frac{n}{w^2}$ 개의 window.

각 window 내 attention: $n_w = w^2$ 개의 token.
Single window 의 attention 복잡도: $O(n_w^2 \cdot d_{\text{head}}) = O(w^4 \cdot d_{\text{head}})$

전체 complexity:
$$\sum_{m=1}^{M} O(w^4 \cdot d_{\text{head}}) = \frac{n}{w^2} \cdot O(w^4 \cdot d_{\text{head}}) = O(n \cdot w^2 \cdot d_{\text{head}})$$

$w$ 가 상수 (보통 7 or 8) 이면, $O(n \cdot w^2) = O(n)$ (선형).

대조:
- Vanilla ViT: $O(n^2)$
- Swin: $O(n \cdot w^2) = O(n)$ (w 고정)

따라서 Swin 이 선형 복잡도. $\square$

### 정리 2.4: Shifted Window 의 Global Receptive Field

**명제**: 두 종류의 window (regular + shifted) 를 교대로 사용하면, 
$L$ 개 layer 를 거친 후 receptive field 가 $\Theta(L \cdot w)$ 로 증가한다.

**증명의 직관**:

Regular window (layer 1):
- Receptive field 는 window size $w \times w$ 로 제한됨.

Shifted window (layer 2):
- Shift 에 의해, 인접한 regular window 들이 새로운 shifted window 내에서 만남.
- 예: 오른쪽 아래 window 의 patch 가 왼쪽 위 window 의 patch 와 
  shifted window 에서 만남 → connection 생성.

재귀적:
- Layer 1: RF = $w$
- Layer 2: RF = $w + w = 2w$ (shifted 로 인해 인접 window 연결)
- Layer 3: RF = $2w + w = 3w$ (다시 한번 shifted)
- ...
- Layer $L$: RF $\approx L \cdot w$

따라서 $O(L \cdot w)$ 로 receptive field 가 확장. 

$L = 4$ stages (global receptive field 설정의 경우) 이면, RF $\approx 4 \cdot w \approx 28-32$ 
(full $224 \times 224$ image 를 대부분 커버).

$\square$

### 정리 2.5: Patch Merging 의 정보 보존

**명제**: Patch merging 을 통해 spatial resolution 을 $1/4$ (4 patches merge) 로 줄이면서도, 
channel 을 $2\times$ 늘리면, 정보 손실이 최소화된다.

**증명의 직관**:

Patch merging 전: $n = H' \times W'$ patches, 각 dimension $d$.
총 정보: $n \cdot d$ (bit units 기준)

Patch merging 후: $(H'/2) \times (W'/2)$ patches, 각 dimension $2d$.
총 정보: $(n/4) \cdot 2d = n \cdot d/2$

일견 정보 손실처럼 보이지만, 실제로는:
1. **Spatial 정보의 집약**: 4개 patch 의 위치 정보를 channel dimension 으로 인코딩 (4D → 2D spatial, 하지만 4x 더 집약)
2. **High-level feature 추출**: Early stage 의 low-level detail 은 late stage 에서 불필요; 대신 semantic structure 중요
3. **Hierarchical structure**: FPN (Feature Pyramid Network) 의 철학과 동일 — 각 stage 가 다른 semantic level 을 담당

따라서 정보 손실이 아니라, **hierarchical abstraction**. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Window Partition 과 Reversed Partition

```python
import torch
import torch.nn as nn

def window_partition(x, window_size):
    """
    x: (B, H, W, C)
    window_size: int
    return: windows (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, 
                  window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.reshape(-1, window_size, window_size, C)
    return windows

def window_unpartition(windows, window_size, hw_shape):
    """
    windows: (B*num_windows, window_size, window_size, C)
    hw_shape: (H, W)
    return: x (B, H, W, C)
    """
    H, W = hw_shape
    B = int(windows.shape[0] / (H // window_size) / (W // window_size))
    x = windows.reshape(B, H // window_size, W // window_size, 
                        window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.reshape(B, H, W, -1)
    return x

torch.manual_seed(42)
B, H, W, C = 2, 224, 224, 96
window_size = 7

# Create random feature map
x = torch.randn(B, H, W, C)

# Partition
windows = window_partition(x, window_size)
print(f"Original shape: {x.shape}")
print(f"Windows shape: {windows.shape}")
print(f"Number of windows: {windows.shape[0]}")

# Unpartition (reverse)
x_reconstructed = window_unpartition(windows, window_size, (H, W))
print(f"Reconstructed shape: {x_reconstructed.shape}")

# Check reconstruction error
error = torch.abs(x - x_reconstructed).max().item()
print(f"Max reconstruction error: {error:.2e}")
```

### 실험 2: Cyclic Shift 와 Attention Mask

```python
import torch
import numpy as np

def cyclic_shift(x, shift_size):
    """
    x: (B, H, W, C)
    shift_size: int (shift amount)
    return: shifted x
    """
    x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    return x

def generate_mask(window_size, shift_size, H, W):
    """
    Generate attention mask for shifted window attention.
    Window 내에서 다른 영역의 token 을 attend 하지 않도록 mask.
    """
    # Create a region index map
    img_mask = torch.zeros((1, H, W, 1))
    
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    
    # Partition mask
    mask_windows = torch.zeros((H // window_size, W // window_size, 
                               window_size, window_size))
    cnt_idx = 0
    for i in range(H // window_size):
        for j in range(W // window_size):
            mask_window = img_mask[0, i*window_size:(i+1)*window_size,
                                   j*window_size:(j+1)*window_size, 0]
            mask_windows[i, j] = mask_window
    
    # Reshape to (num_windows, window_size, window_size)
    mask_windows = mask_windows.reshape(-1, window_size, window_size)
    
    # Generate attention mask
    attn_mask = (mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2)).float()
    attn_mask = attn_mask * (-100.0)  # Large negative value for masked positions
    
    return attn_mask

torch.manual_seed(42)
B, H, W, C = 1, 56, 56, 96
window_size = 7
shift_size = window_size // 2

x = torch.randn(B, H, W, C)

# Cyclic shift
x_shifted = cyclic_shift(x, shift_size)
print(f"Original shape: {x.shape}")
print(f"Shifted shape: {x_shifted.shape}")

# Difference (first element)
diff = (x_shifted[0, 0, 0, :5] - x[0, shift_size, shift_size, :5]).abs().max().item()
print(f"Shift effect (first 5 channels): {diff:.4f}")

# Generate mask
num_windows = (H // window_size) ** 2
attn_mask = generate_mask(window_size, shift_size, H, W)
print(f"Attention mask shape: {attn_mask.shape}")
print(f"Num windows: {num_windows}, window tokens: {window_size**2}")
```

### 실험 3: Patch Merging 의 Spatial Downsampling

```python
import torch
import torch.nn as nn

class PatchMerging(nn.Module):
    """
    Patch merging layer.
    Merge 4개 neighboring patches into 1.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x):
        """
        x: (B, H, W, C)
        return: (B, H//2, W//2, 2C)
        """
        B, H, W, C = x.shape
        
        # Reshape: gather 4 patches
        x0 = x[:, 0::2, 0::2, :]    # Top-left
        x1 = x[:, 1::2, 0::2, :]    # Top-right
        x2 = x[:, 0::2, 1::2, :]    # Bottom-left
        x3 = x[:, 1::2, 1::2, :]    # Bottom-right
        
        # Concatenate along channel
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.reshape(B, H//2, W//2, 4*C)
        
        # Norm & reduce
        x = self.norm(x)
        x = self.reduction(x)
        
        return x

torch.manual_seed(42)
B, H, W, C = 2, 56, 56, 96

x = torch.randn(B, H, W, C)
merger = PatchMerging(C)
x_merged = merger(x)

print(f"Before merging: {x.shape}")
print(f"After merging: {x_merged.shape}")
print(f"Spatial reduction: {H}x{W} → {H//2}x{W//2}")
print(f"Channel change: {C} → {2*C}")

# Verify spatial information compression
total_before = H * W * C
total_after = (H//2) * (W//2) * (2*C)
print(f"Total elements before: {total_before}")
print(f"Total elements after: {total_after}")
print(f"Compression ratio: {total_before / total_after:.2f}x")
```

### 실험 4: Window Attention vs Global Attention 복잡도 비교

```python
import torch
import torch.nn.functional as F

def compute_attention_complexity(n, w=None):
    """
    n: number of patches (H*W)
    w: window size (for Swin); if None, compute global (ViT)
    return: (complexity, memory)
    """
    if w is None:  # Global ViT
        return n**2, n**2
    else:  # Swin window
        num_windows = n // (w**2)
        per_window = w**4
        total = num_windows * per_window
        return total, total

# ImageNet-1k: 224x224, patch=16
H, W, P = 224, 224, 16
n = (H // P) * (W // P)  # 196 patches

print(f"ImageNet-1k (224x224, patch_size=16): {n} patches")
print()

# Global ViT
ops_vit, mem_vit = compute_attention_complexity(n)
print(f"ViT (Global Attention):")
print(f"  Complexity: {ops_vit:,} ops ≈ {ops_vit / 1e3:.1f}k")
print(f"  Memory: O(n²)")
print()

# Swin with different window sizes
for w in [4, 7, 8]:
    ops_swin, mem_swin = compute_attention_complexity(n, w=w)
    speedup = ops_vit / ops_swin
    print(f"Swin (window={w}x{w}):")
    print(f"  Complexity: {ops_swin:,} ops ≈ {ops_swin / 1e3:.1f}k")
    print(f"  Speedup: {speedup:.1f}x")
    print()
```

## 🔗 실전 활용

**timm 에서의 Swin 구현**:
```python
from timm.models import swin_base_patch4_window7_224
model = swin_base_patch4_window7_224(pretrained=True)
```

**COCO Detection (Faster R-CNN backbone)**:
```python
from detectron2.modeling import build_backbone
backbone = build_backbone(cfg)  # Swin-B backbone
```

**Semantic Segmentation (Mask2Former)**:
```python
from mmdet.models import build_backbone
backbone = build_backbone(
    dict(type='SwinTransformer', embed_dim=128, ...)
)
```

**Dense prediction 장점**:
- Hierarchical features: 4-stage pyramid (4×→8×→16×→32×)
- Efficient computation: O(n·w²) 로 real-time inference 가능
- Strong local inductive bias: patch merging 으로 locality 강화

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Window size 고정** | Shifted window 의 효율성은 w 를 고정했을 때 | w 를 크게 하면 window 간 정보 흐름 악화 |
| **2D 구조 가정** | Window partition 과 shift 가 2D grid 기반 | Video 나 3D data 에는 별도 구현 필요 |
| **Relative position bias** | Learned bias 가 generalization 을 보장하지 않음 | 다른 resolution 으로 transfer 시 interpolation 필요 |
| **Patch merging 정보손** | 4개 patch 를 2D로 merge 할 때 경계 정보 손실 | Dense prediction (edge) 에서는 skip-connection 으로 보상 |
| **Computational overhead** | Cyclic shift + masking 의 구현 복잡도 | CUDA kernel 최적화 필수 (timm 구현 참고) |

## 📌 핵심 정리

$$\boxed{O(n) \text{ Complexity: } \mathrm{WindowAttn}(w=7) = O(n \cdot 49) \text{ vs } \mathrm{GlobalAttn} = O(n^2)}$$

| 개념 | 식 | 역할 |
|------|-----|------|
| **Window partition** | $\mathcal{W}(X) = \{W_1, \ldots, W_M\}$ | Spatial 분할로 복잡도 선형화 |
| **Shifted window** | $\mathrm{CyclicShift}(X, \lfloor w/2 \rfloor)$ | Window 간 connection, global RF |
| **Local attention** | $\mathrm{Attn}(Q, K, V) + B_{\text{rel}}$ | Relative position bias 로 translation invariance |
| **Patch merging** | Concat(4 patches) → Linear(4D → 2D) | Hierarchical downsampling |
| **Hierarchical stages** | 4×→8×→16×→32× | FPN 스타일 multi-scale features |

**결론**: Swin Transformer 는 window attention 과 shifted window 의 조합으로, 
ViT 의 $O(n^2)$ 복잡도를 $O(n \cdot w^2)$ (w 고정이면 $O(n)$) 으로 줄이면서도, 
global receptive field 를 유지한다. Hierarchical structure 로 dense prediction 에 적합.

## 🤔 생각해볼 문제

### 문제 1 (기초): Window Size 의 선택

Swin Transformer 에서 window size $w = 7$ 을 사용하는 이유는 무엇일까? 
$w = 4, 8, 14$ 를 비교하면 어떤 trade-off 가 생기는가?

<details>
<summary>해설</summary>

**$w = 4$ (작은 window)**:
- Complexity: 매우 낮음 ($O(n \cdot 16)$)
- 단점: Window 간 정보 흐름이 느림 (많은 layer 필요)
- 장점: 지역 구조 강하게 포착 (CNN 스타일)

**$w = 7$ (표준)**:
- Complexity: 적당함 ($O(n \cdot 49)$)
- 장점: ImageNet 224x224 에서 적절한 local/global balance
- 이유: 7x7 = 49 tokens per window ≈ 한 "semantic region" 크기

**$w = 14$ (큰 window)**:
- Complexity: 높음 ($O(n \cdot 196)$)
- 장점: Window 간 정보 흐름이 빠름 (fewer layers 로 global RF 달성)
- 단점: Global attention 에 가까워짐 → 효율성 감소

**$w = 8$ (2의 거듭제곱)**:
- Complexity: 적당함 ($O(n \cdot 64)$)
- 장점: 하드웨어 최적화 용이 (tensor blocking)
- 단점: 7 보다 약간 낮은 정확도

결론: $w = 7$ 은 경험적으로 ImageNet 규모에서 최적. 다른 resolution 에서는 조정 필요 (예: 384x384 이면 14 추천).

</details>

### 문제 2 (심화): Shifted Window 의 수학적 필요성

Swin Transformer 에서 shifted window 를 사용하지 않고, regular window 만 사용하면 어떻게 될까?
왜 shifted window 가 필수인가?

<details>
<summary>해설</summary>

Regular window 만 사용 시 (shifted 없이):
```
Layer 1:        Layer 2:        Layer 3:
┌─┬─┐          ┌─┬─┐          ┌─┬─┐
├─┼─┤          ├─┼─┤          ├─┼─┤
└─┴─┘          └─┴─┘          └─┴─┘
(같은 partition)
```

문제:
1. **Window 경계의 "벽"**: 각 window 는 독립적 → 인접 window 간 정보 교환 불가
2. **Receptive field 정체**: Layer 를 거쳐도 RF 가 증가하지 않음 (각 layer 마다 RF = w)
3. **Global connectivity 부재**: 깊은 layer 에서도 먼 곳의 정보에 접근 불가

Shifted window 사용 시:
```
Layer 1 (Regular):   Layer 2 (Shifted):
┌─┬─┐               ┌─┬─┐
├─┼─┤               ├─┼─┤
└─┴─┘               └─┴─┘
(다른 partition)
↓
인접 windows 가 새로운 shifted window 내에서 만남
→ Gradient flow 개선 + RF 증가
```

Shifted window 의 필요성:
- **Gradient flow**: Layer 를 거치면서 모든 token 이 상호작용 가능
- **Receptive field**: $O(L \cdot w)$ 로 증가 (L = layer 수)
- **Efficiency**: Regular window 만으로는 global connection 을 위해 매우 많은 layer 필요

따라서 shifted window 는 depth-efficiency 측면에서 필수적이다 (정리 2.4).

</details>

### 문제 3 (논문 비평): Swin vs ViT+DeiT 의 Inductive Bias

DeiT 는 augmentation + distillation 으로 ViT 의 inductive bias 부족을 보상했다.
Swin 은 architecture level 에서 locality 를 강제한다.

두 방식의 장단점을 논의하고, 어느 경우에 어느 것이 더 나을까?

<details>
<summary>해설</summary>

**DeiT (Augmentation + Distillation 관점)**:

장점:
- Flexibility: 같은 architecture 로 다양한 task 에 적응 가능
- Scalability: 매우 큰 데이터 (billions of images) 에서 좋은 성능
- Emergent properties: 충분한 데이터가 있으면 CNN 보다 나은 generalization

단점:
- Data 의존성: ImageNet-1k 만으로는 CNN 에 미치지 못함
- Computational cost: Pre-training 에 많은 자원 필요
- Dense prediction: Detection, segmentation 에서 pyramidal feature 구조가 없음 (U-Net 스타일 decoder 필요)

**Swin (Architectural Inductive Bias)**:

장점:
- Data efficient: 더 적은 데이터로도 좋은 결과 (CNN 수준)
- Hierarchical: Multi-scale features 가 자연스럽게 생성 → dense prediction 에 최적
- Computational efficiency: Local window attention 으로 실시간 inference 가능

단점:
- Architecture 의 rigidity: 2D grid 기반 (video 나 3D 는 별도 설계 필요)
- Scaling 한계: Extremely large-scale pre-training (LLaVA 같은 multimodal) 에서 ViT 가 더 나음

**언제 어느 것?**:

1. **Dense prediction (detection, segmentation)**: Swin 추천
   - Hierarchical structure 가 필수
   - Local inductive bias 가 도움

2. **Classification with large-scale data**: DeiT 스타일 ViT 추천
   - Enough data to overcome inductive bias 부족
   - Scaling 이 용이

3. **Multimodal (vision-language, video)**: ViT 추천
   - Sequential 구조의 flexibility
   - Self-attention 이 다양한 modality 를 유연하게 처리

4. **Resource-constrained (mobile, edge)**: Swin 추천
   - O(n) complexity 로 계산 효율 우수

결론: DeiT vs Swin 은 "flexibility vs efficiency" 의 trade-off. 
최근 추세는 hybrid (e.g., CoAtNet, CvT) 로 둘 다의 장점을 취하는 것.

</details>

---

<div align="center">

[◀ 이전](./01-deit.md) | [📚 README](../README.md) | [다음 ▶](./03-pvt.md)

</div>
