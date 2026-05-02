# 05. MViT · Focal Transformer — Multi-Scale Attention

## 🎯 핵심 질문

- **MViT**: Pool-based attention 에서 Q, K, V 를 progressively pooling 하면, 어떻게 multi-scale 이 자동으로 생성되는가?
- **MViT**: Video domain 에서 왜 pool-based attention 이 특히 효과적인가?
- **Focal Transformer**: Fine-grained token (nearby) 과 coarse token (distant) 를 같은 attention 내에서 어떻게 조합하는가?
- 두 방식 모두 "single-scale 의 한계" 를 극복하되, 어떤 철학의 차이가 있는가?

## 🔍 왜 이 Multi-Scale Attention 이 Hierarchical ViT 의 완성인가

지금까지의 hierarchical ViT 들:
- **DeiT**: Augmentation + distillation (같은 scale, 데이터 효율)
- **Swin**: Window attention (local structure, spatial split)
- **PVT**: Spatial reduction (selective aggregation)
- **CvT/CoAtNet**: Conv + Attention hybrid (inductive bias 회복)

모두 **"어떻게 efficient 하게 만들 것인가"** 를 중심으로 했다.

**Fan et al. (2021) MViT** 와 **Yang et al. (2021) Focal Transformer** 는 다른 관점:
**"Attention 자체를 multi-scale 로 만들면?"**

**MViT 의 핵심**:
- 같은 attention layer 에서 Q 는 fine-grained (모든 token), K/V 는 progressively coarse (pooled)
- Pooling ratio 를 spatial dimension 에 적용하면, 자동으로 hierarchical features 생성
- 특히 Video domain (temporal + spatial multi-scale) 에서 매우 효과적

**Focal Transformer 의 핵심**:
- 같은 token 이 2가지 종류의 neighbors 를 attend:
  - **Fine tokens** (가까운 곳): high-resolution, local details
  - **Coarse tokens** (먼 곳): low-resolution, global context
- 하나의 attention 메커니즘으로 fine-coarse 정보를 동시 통합

결과: 두 방식 모두 **"효율" 과 "정확도" 의 새로운 균형** 달성.

## 📐 수학적 선행 조건

- **ViT self-attention** (Ch1): Multi-head mechanism, complexity analysis
- **Swin / PVT** (Ch2-02, 03): Window 와 spatial reduction 의 trade-off
- **Pooling operations**: Average pooling, max pooling (spatial dimension)
- **Token aggregation**: 여러 token 을 하나로 압축 (mean, max)
- **Focal vs peripheral vision**: 시각 신경생물학 (human visual system)
- **Video understanding**: Temporal 과 spatial 의 joint modeling

## 📖 직관적 이해

```
Single-scale attention vs Multi-scale attention:

┌──────────────────────────────────────┐
│ Vanilla ViT / Standard Attention      │
│                                       │
│ All tokens: same resolution scale     │
│                                       │
│ Query:    [t₁ t₂ t₃ t₄ ... tₙ]       │
│ Key:      [t₁ t₂ t₃ t₄ ... tₙ]       │
│ Value:    [t₁ t₂ t₃ t₄ ... tₙ]       │
│                                       │
│ Attn = softmax(QK^T) V                │
│                                       │
│ 문제: 모든 token 이 같은 "시력"      │
│ (resolution) 으로 attend              │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ MViT: Pool-based Attention            │
│                                       │
│ Query:    [t₁ t₂ t₃ t₄ ... tₙ]       │
│           (fine-grained, full)        │
│                                       │
│ Key:      [pool(t₁,t₂) pool(t₃,t₄)...│
│           (coarse, 2x pooled)         │
│ Value:    [pool(t₁,t₂) pool(t₃,t₄)...│
│           (coarse, 2x pooled)         │
│                                       │
│ Attn = softmax(QK^T) V                │
│                                       │
│ 효과: Q 는 fine detail, K/V 는       │
│ coarse semantic → hierarchical        │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ Focal Transformer: Fine-Coarse        │
│                                       │
│ Fine neighbors:   [t_nearby]          │
│ (local region, e.g., 7x7)             │
│                                       │
│ Coarse neighbors: [t_distant]         │
│ (global region, sampled)              │
│                                       │
│ Attention = softmax(Q[K_fine;K_coarse]^T) V
│                                       │
│ 효과: Single token 이 fine + coarse  │
│ 두 scale 의 정보를 동시 attend       │
└──────────────────────────────────────┘
```

**비유**:
- **Vanilla**: "한 사람이 고정 거리에서만 본다" (single scale)
- **MViT**: "가까운 것은 선명하게, 먼 것은 뭉뚝하게 본다" (progressive pooling)
- **Focal**: "초점은 가깝고, 주변은 멀다 (focal point + periphery)" (human vision model)

## ✏️ 엄밀한 정의

### 정의 2.17: Pool-based Attention (MViT)

**표준 attention**:
$$\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

**MViT 의 pool-based attention**:
$$\mathrm{PoolAttn}(Q, K, V) = \mathrm{softmax}\left(\frac{Q(K_{\text{pool}})^\top}{\sqrt{d}}\right) V_{\text{pool}}$$

여기서:
- $Q \in \mathbb{R}^{n \times d}$ (full resolution, fine-grained)
- $K_{\text{pool}} = \text{AvgPool2D}(K, \text{stride}=r) \in \mathbb{R}^{(n/r^2) \times d}$ (pooled, coarse)
- $V_{\text{pool}} = \text{AvgPool2D}(V, \text{stride}=r) \in \mathbb{R}^{(n/r^2) \times d}$

**Pooling operation**: 
2D spatial grid 에서 $r \times r$ average pooling 적용.
예: $r=2$ 이면 4개 token 을 1개 로 압축.

**Complexity**:
$$O(n \times (n/r^2) \times d) = O(n^2 d / r^2)$$

Vanilla attention $O(n^2 d)$ 대비 $1/r^2$ speedup.

### 정의 2.18: Progressive Pooling across Stages (MViT)

MViT 는 4-stage hierarchical structure. 각 stage 마다 pooling ratio 가 누적:

$$\begin{array}{c|c|c|c|c}
\text{Stage} & \text{Resolution} & \text{Pooling Ratio} & r_{\text{cumulative}} & \text{Complexity} \\
\hline
1 & H \times W & r=2 & 2 & O(n^2/4) \\
2 & H/2 \times W/2 & r=2 & 4 & O(n^2/16) \\
3 & H/4 \times W/4 & r=2 & 8 & O(n^2/64) \\
4 & H/8 \times W/8 & r=2 & 16 & O(n^2/256) \\
\end{array}$$

**누적 효과**:
- Early stage: fine detail 포착 (pooling 적음)
- Late stage: high-level semantic (pooling 많음)
- 전체 계산은 $O(n^2/4 + n^2/16 + \cdots) \approx O(n^2)$ (but with significantly reduced constant)

### 정의 2.19: Focal Attention (Focal Transformer)

**핵심**: 각 query token 이 두 종류의 neighbors 를 attend:

$$\mathrm{FocalAttn}(Q, K, V) = \mathrm{softmax}\left(\frac{Q[K_{\text{fine}}; K_{\text{coarse}}]^\top}{\sqrt{d}}\right) [V_{\text{fine}}; V_{\text{coarse}}]$$

여기서:
- $K_{\text{fine}}, V_{\text{fine}}$: Nearby tokens (예: 7×7 window)
- $K_{\text{coarse}}, V_{\text{coarse}}$: Distant tokens (spatially sampled, 예: 32개 sparse tokens)

**Fine-grained neighbors**:
$$K_{\text{fine}} = \text{Extract}(K, \text{window}=7)$$
모든 query 가 주변 7×7 window 의 모든 token 을 attend.

**Coarse-grained neighbors**:
$$K_{\text{coarse}} = \text{Sample}(K, \text{stride}=s, \text{count}=c)$$
Global 한 위치에서 sparse sampling 으로 $c$ 개 token 선택.

**Attention mask**:
Fine neighborhood: full attention  
Coarse neighborhood: additional softmax over sampled tokens  
결과 feature: fine + coarse information 의 weighted combination.

### 정의 2.20: Video Multi-Scale Attention (MViT for Video)

Video 에서는 spatial + temporal 두 dimension 에서의 multi-scale:

$$\mathrm{PoolAttn}_{\text{video}}(Q, K, V) = \mathrm{softmax}\left(\frac{Q(K_{\text{pool}})^\top}{\sqrt{d}}\right) V_{\text{pool}}$$

여기서 pooling 이 spatial + temporal 을 동시 적용:
$$K_{\text{pool}} = \text{AvgPool3D}(K, \text{stride}=(s_t, s_h, s_w))$$

예: stride=(1, 2, 2) 이면 temporal 은 유지, spatial 만 2배 downsample.

**Temporal coherence** 유지:
- Frame 간 temporal 정보를 fine-grained 로 유지
- Spatial 은 coarse 로 압축 → efficiency

## 🔬 정리와 증명

### 정리 2.12: MViT 의 Multi-Scale Feature 자동 생성

**명제**: Progressive pooling 을 적용하면, 
각 stage 에서 자동으로 multi-scale features 가 생성되어, 
CNN 의 FPN 과 유사한 hierarchical representation 을 얻을 수 있다.

**증명의 직관**:

Stage 1: Pooling ratio $r=2$
- Query: 모든 patch (fine resolution)
- Key/Value: 2×2 avg pooling (1/4 patch 수)
- Output feature: 원래 resolution 으로 복원 (spatial upsampling) → fine detail 유지

Stage 2: Pooling ratio $r=2$ (누적 4배)
- Query: stage 1 output (이미 down-sample 됨)
- Key/Value: 다시 2×2 avg pooling (1/16 patch 수)
- Output: stage 1 보다 더 abstract → semantic feature

이런 식으로:
- Stage 1: $56 \times 56$ (fine spatial structure)
- Stage 2: $28 \times 28$ (coarser semantics)
- Stage 3: $14 \times 14$ (higher-level concepts)
- Stage 4: $7 \times 7$ (top-level abstractions)

각 stage 는 FPN 의 pyramid level 과 정확히 대응.

**FPN vs MViT**:
- FPN (CNN): 각 stage 에서 다른 scale 의 features 를 explicit 하게 생성 (별도 convolution)
- MViT (Transformer): Pooling 을 통해 implicit 하게 multi-scale 생성 (unified architecture)

따라서 MViT 는 hierarchical representation 을 자동으로 달성. $\square$

### 정리 2.13: Focal Attention 의 Receptive Field

**명제**: Focal attention 은 fine neighbors (window-based) + coarse neighbors (global sparse) 의 조합으로, 
local detail 과 global context 를 균형있게 포착한다.

**증명의 직관**:

**Fine neighbors (예: 7×7 window)**:
- Receptive field: 7×7 = 49 tokens (local)
- Computation: Dense (모든 neighbor 에 attend)
- 역할: Texture, edge, local structure

**Coarse neighbors (예: 32 sparse tokens)**:
- Receptive field: Entire image (global)
- Computation: Sparse (sampled token 만 attend)
- 역할: Object-level semantics, global context

**Attention score 계산**:
$$S = Q[K_{\text{fine}}; K_{\text{coarse}}]^\top = [Q K_{\text{fine}}^\top \, | \, Q K_{\text{coarse}}^\top]$$

Softmax 를 적용하면, attention 이 두 종류의 neighbor 에 어떻게 분배되는지 자동으로 학습:
- Fine neighbors 에 높은 weight: Detail-focused queries (edge detection 등)
- Coarse neighbors 에 높은 weight: Context-focused queries (classification 등)

**복잡도**:
$$O(\text{fine}) + O(\text{coarse}) = O(49 \times n) + O(32 \times n) = O(81n)$$

Vanilla attention $O(n^2)$ 대비 $O(1/n)$ reduction (for large n).

따라서 focal attention 은 global receptive field 를 유지하면서도 linear complexity. $\square$

### 정리 2.14: Video Domain 에서의 MViT 효율성

**명제**: MViT 의 pool-based attention 은 video domain 에서 
temporal + spatial multi-scale 을 동시에 효율적으로 처리하여, 
pure ViT 나 CNN 을 능가한다.

**증명의 직관**:

Video 는 spatial (H, W) 과 temporal (T) 두 dimension 을 가짐:
- Total tokens: $n = T \times H \times W$ (매우 큼)
- Vanilla ViT attention: $O(n^2) = O((T \cdot H \cdot W)^2)$ → prohibitive

**MViT 의 해결**:
1. **Temporal**: coarse 하게 처리 (stride=1 또는 2) → 주요 동작 포착
2. **Spatial**: fine 에서 coarse 로 progressive pooling → hierarchical spatial

Example (stride=(1, 2, 2)):
- Temporal: 모든 frame 유지 (temporal coherence)
- Spatial: 2배 downsample (efficiency)

**복잡도**:
$$O(T \times (H \times W)^2 / 4) \text{ (pooling factor 4)}$$

Vanilla ViT: $O((T \cdot H \cdot W)^2)$  
MViT: $O(T \times (H \times W)^2 / r_{\text{spatial}}^2)$

$r_{\text{spatial}} = 2$ 이면, MViT 는 4배 더 효율적.

따라서 video 에서 MViT 의 pool-based attention 이 특히 효과적. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Pool-based Attention (MViT)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolBasedAttention(nn.Module):
    """
    Pool-based attention: Q at full resolution,
    K/V at reduced resolution via pooling.
    """
    def __init__(self, dim, num_heads=8, pool_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.pool_ratio = pool_ratio
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, H, W):
        """
        x: (B, N, C) where N = H*W
        return: (B, N, C)
        """
        B, N, C = x.shape
        
        # Query: full resolution
        q = self.q(x)  # (B, N, C)
        q = q.reshape(B, self.num_heads, N, C // self.num_heads)
        q = q.permute(0, 1, 2, 3)  # (B, num_heads, N, d)
        
        # Pooling for K, V
        x_spatial = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        k_spatial = F.avg_pool2d(x_spatial, kernel_size=self.pool_ratio,
                                 stride=self.pool_ratio)
        k = self.k(k_spatial.permute(0, 2, 3, 1).reshape(B, -1, C))
        k = k.reshape(B, self.num_heads, k.shape[1], C // self.num_heads)
        
        v_spatial = F.avg_pool2d(x_spatial, kernel_size=self.pool_ratio,
                                 stride=self.pool_ratio)
        v = self.v(v_spatial.permute(0, 2, 3, 1).reshape(B, -1, C))
        v = v.reshape(B, self.num_heads, v.shape[1], C // self.num_heads)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** (-0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # (B, num_heads, N, d)
        
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(B, N, C)
        out = self.proj(out)
        
        return out

torch.manual_seed(42)
B, H, W, C = 2, 56, 56, 64
N = H * W

pool_attn = PoolBasedAttention(dim=C, num_heads=8, pool_ratio=2)
x = torch.randn(B, N, C)

out = pool_attn(x, H, W)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Pool-based attention: Q full ({N}), K/V pooled ({N//4})")

# Complexity
attn_full = N * N
attn_pool = N * (N // 4)
ratio = attn_full / attn_pool
print(f"Complexity ratio (full vs pooled): {ratio:.1f}x speedup")
```

### 실험 2: MViT 4-Stage with Progressive Pooling

```python
import torch
import torch.nn as nn

class MViTStage(nn.Module):
    """Single MViT stage with pool-based attention"""
    def __init__(self, dim, num_heads=8, pool_ratio=2, depth=2):
        super().__init__()
        self.attn_layers = nn.ModuleList([
            PoolBasedAttention(dim, num_heads, pool_ratio)
            for _ in range(depth)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
            for _ in range(depth)
        ])
        self.norms_1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.norms_2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
    
    def forward(self, x, H, W):
        for attn, ffn, norm1, norm2 in zip(self.attn_layers, self.ffn_layers,
                                           self.norms_1, self.norms_2):
            x_norm = norm1(x)
            x = x + attn(x_norm, H, W)
            x_norm = norm2(x)
            x = x + ffn(x_norm)
        return x

class MViT4Stage(nn.Module):
    """Simplified MViT with 4 stages"""
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Stem
        self.stem = nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3)
        
        # Stage 1: pool_ratio=2
        self.stage1 = MViTStage(64, num_heads=8, pool_ratio=2, depth=2)
        self.downsample1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Stage 2: pool_ratio=2
        self.stage2 = MViTStage(128, num_heads=8, pool_ratio=2, depth=2)
        self.downsample2 = nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=1)
        
        # Stage 3: pool_ratio=2
        self.stage3 = MViTStage(320, num_heads=8, pool_ratio=2, depth=2)
        self.downsample3 = nn.Conv2d(320, 512, kernel_size=3, stride=2, padding=1)
        
        # Stage 4: pool_ratio=1 (no pooling)
        self.stage4 = MViTStage(512, num_heads=8, pool_ratio=1, depth=2)
        
        self.head = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)  # (B, 64, 56, 56)
        B, C, H, W = x.shape
        
        # Stage 1
        x_seq = x.flatten(2).transpose(1, 2)  # (B, 56*56, 64)
        x_seq = self.stage1(x_seq, H, W)
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        x = self.downsample1(x)  # (B, 128, 28, 28)
        
        # Stage 2
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        x_seq = self.stage2(x_seq, H, W)
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        x = self.downsample2(x)  # (B, 320, 14, 14)
        
        # Stage 3
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        x_seq = self.stage3(x_seq, H, W)
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        x = self.downsample3(x)  # (B, 512, 7, 7)
        
        # Stage 4
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        x_seq = self.stage4(x_seq, H, W)
        
        # Classification
        x_cls = x_seq.mean(dim=1)
        out = self.head(x_cls)
        
        return out

torch.manual_seed(42)
B = 2

model = MViT4Stage()
x = torch.randn(B, 3, 224, 224)
out = model(x)

print(f"Input: {x.shape}")
print(f"Output: {out.shape}")
print(f"\nMViT 4-Stage Progressive Pooling:")
print(f"  Stage 1: 56×56×64 (pool_ratio=2)")
print(f"  Stage 2: 28×28×128 (pool_ratio=2)")
print(f"  Stage 3: 14×14×320 (pool_ratio=2)")
print(f"  Stage 4: 7×7×512 (pool_ratio=1)")
```

### 실험 3: Focal Attention (Fine + Coarse)

```python
import torch
import torch.nn as nn

class FocalAttention(nn.Module):
    """
    Focal attention: combine fine neighbors (dense) + coarse neighbors (sparse)
    """
    def __init__(self, dim, num_heads=8, window_size=7, num_coarse=32):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_coarse = num_coarse
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, H, W):
        """
        x: (B, N, C) where N = H*W
        """
        B, N, C = x.shape
        
        # Query
        q = self.q(x)  # (B, N, C)
        q = q.reshape(B, self.num_heads, N, C // self.num_heads)
        
        # Full K, V
        k = self.k(x)  # (B, N, C)
        v = self.v(x)
        k = k.reshape(B, self.num_heads, N, C // self.num_heads)
        v = v.reshape(B, self.num_heads, N, C // self.num_heads)
        
        # Fine neighbors: local window around each query
        # (simplified: just use all for demo)
        k_fine = k
        v_fine = v
        
        # Coarse neighbors: sparse sampling
        stride = max(1, N // self.num_coarse)
        coarse_idx = torch.arange(0, N, stride, device=x.device)
        k_coarse = k[:, :, coarse_idx, :]  # (B, num_heads, num_coarse, d)
        v_coarse = v[:, :, coarse_idx, :]
        
        # Concatenate fine + coarse
        k_all = torch.cat([k_fine, k_coarse], dim=2)
        v_all = torch.cat([v_fine, v_coarse], dim=2)
        
        # Attention
        attn = (q @ k_all.transpose(-2, -1)) * (C // self.num_heads) ** (-0.5)
        attn = attn.softmax(dim=-1)
        
        out = attn @ v_all  # (B, num_heads, N, d)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(B, N, C)
        out = self.proj(out)
        
        return out

torch.manual_seed(42)
B, H, W, C = 2, 56, 56, 64
N = H * W

focal_attn = FocalAttention(dim=C, num_heads=8, window_size=7, num_coarse=32)
x = torch.randn(B, N, C)

out = focal_attn(x, H, W)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Focal attention: fine ({N}) + coarse (32) neighbors")

# Complexity
attn_full = N * N
attn_focal = N * N + N * 32  # fine + coarse
ratio = attn_full / attn_focal
print(f"Complexity: {attn_full:,} → {attn_focal:,} ({ratio:.1f}x)")
```

### 실험 4: Progressive Pooling 효과 비교

```python
import torch
import torch.nn.functional as F

def attention_complexity(n, pool_ratio=None):
    """
    n: number of tokens
    pool_ratio: spatial reduction (None for full)
    """
    if pool_ratio is None:  # Full attention
        return n**2
    else:  # Pooled
        n_reduced = n // (pool_ratio ** 2)
        return n * n_reduced

# ImageNet-1k: 224x224, patch=4
H_init, W_init = 224, 224
P = 4

stages = [
    (56, 56, 2, "Stage 1"),
    (28, 28, 2, "Stage 2"),
    (14, 14, 2, "Stage 3"),
    (7, 7, 1, "Stage 4"),
]

print("MViT 4-Stage Analysis:")
print("─" * 70)

total_mvit = 0
total_full = 0

for h, w, pool_ratio, label in stages:
    n = h * w
    
    ops_full = attention_complexity(n)
    ops_mvit = attention_complexity(n, pool_ratio)
    
    speedup = ops_full / ops_mvit if ops_mvit > 0 else float('inf')
    
    total_mvit += ops_mvit
    total_full += ops_full
    
    print(f"{label}: {h}×{w} ({n:4d} tokens, pool_ratio={pool_ratio})")
    print(f"  Full:  {ops_full:8,} ops")
    print(f"  MViT:  {ops_mvit:8,} ops")
    print(f"  Speedup: {speedup:5.1f}x")
    print()

print(f"Total complexity:")
print(f"  Full attention:  {total_full:,} ops")
print(f"  MViT pooled:     {total_mvit:,} ops")
print(f"  Overall speedup: {total_full / total_mvit:.1f}x")
```

## 🔗 실전 활용

**timm 에서의 MViT 구현**:
```python
from timm.models import mvit_base
model = mvit_base(pretrained=True)
```

**Focal Transformer**:
```python
from timm.models import focal_transformer_small
model = focal_transformer_small(pretrained=True)
```

**Video understanding (MViT)**:
```python
# Temporal + spatial pool-based attention
mvit_video = MVit(temporal_pool_ratio=1, spatial_pool_ratio=2)
```

**Dense prediction (both)**:
- MViT: hierarchical features 자동 생성 → Faster R-CNN backbone
- Focal Transformer: fine + coarse attention → semantic segmentation

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Pooling information loss** | K, V pooling 시 정보 손실 | 과도한 pooling ratio 는 성능 저하 |
| **Coarse sampling strategy** | Focal 의 sparse sampling 방식이 critical | 균등 sampling vs learned sampling |
| **Temporal coherence** | Video MViT 에서 temporal pooling 시 frame 정보 손실 | 동작 감지에 필수 frame 은 유지해야 함 |
| **Fine-grained token cost** | Focal 의 dense fine neighbors 는 여전히 비용 높음 | Window size 가 커지면 quadratic |
| **Parameter efficiency** | Multi-scale 이득이 parameter 수 증가로 상쇄될 수 있음 | Model size vs accuracy trade-off |

## 📌 핵심 정리

$$\boxed{\text{MViT}: \mathrm{PoolAttn}(Q_{\text{full}}, K_{\text{pooled}}, V_{\text{pooled}}) = \text{softmax}\left(\frac{Q(K_{\text{pool}})^\top}{\sqrt{d}}\right) V_{\text{pool}}}$$

$$\boxed{\text{Focal}: \mathrm{FocalAttn}(Q, [K_{\text{fine}}; K_{\text{coarse}}], [V_{\text{fine}}; V_{\text{coarse}}])}$$

| 개념 | 식 | 역할 |
|------|-----|------|
| **Query** | Full resolution | Fine-grained 정보 요청 |
| **Key/Value pooled** (MViT) | Avg pooling with stride=r | Coarse 정보 제공, complexity 감소 |
| **Fine neighbors** (Focal) | Dense window (7×7) | Local detail, texture |
| **Coarse neighbors** (Focal) | Sparse global (32 tokens) | Global context, semantics |
| **Progressive pooling** | 4-stage cumulative (2, 4, 8, 16) | Hierarchical feature pyramid |

**결론**: 
MViT 와 Focal Transformer 는 multi-scale attention 의 두 가지 구현:
- **MViT**: Progressive pooling (parameter efficient, video friendly)
- **Focal**: Fine-coarse combination (human vision inspired, single mechanism)

둘 다 "single-scale 의 한계" 를 극복하고, efficient 하면서도 강력한 hierarchical representation 을 달성한다.

## 🤔 생각해볼 문제

### 문제 1 (기초): Pooling Ratio 의 선택

MViT 에서 각 stage 마다 pool_ratio=2 를 사용한다.
만약 pool_ratio=1 (pooling 없음) 또는 pool_ratio=4 (강한 pooling) 를 쓰면 어떻게 될까?

<details>
<summary>해설</summary>

**pool_ratio=1 (pooling 없음)**:
- Vanilla ViT 와 동일 (K, V 도 full resolution)
- Complexity: $O(n^2)$ (이득 없음)
- 장점: 정보 손실 최소
- 단점: 비효율적

**pool_ratio=2 (standard)**:
- Complexity: $O(n^2 / 4)$ per stage
- 4-stage 누적: $O(n^2/4 + n^2/16 + \cdots) \approx O(n^2/3)$
- 균형: 정보 손실 vs 효율성

**pool_ratio=4 (강한 pooling)**:
- Complexity: $O(n^2 / 16)$ per stage
- 4-stage 누적: $O(n^2/16 + n^2/256 + \cdots) \approx O(n^2/15)$
- 이득: 매우 효율적
- 단점: 정보 손실 심각 (accuracy ↓)

**선택 기준**:
- Speed-critical: pool_ratio=4
- Accuracy-critical: pool_ratio=1
- Balanced: pool_ratio=2 (default)

따라서 pool_ratio=2 는 empirical 하게 최적임을 보였음 (MViT 논문, Table).

</details>

### 문제 2 (심화): Focal Transformer 의 Fine/Coarse 비율

Focal Transformer 에서 fine neighbors 의 window size 와 coarse neighbors 의 개수의 최적 비율은?
Fine:Coarse = 49:32 (7×7 window + 32 sparse tokens) 를 사용하는 이유는?

<details>
<summary>해설</summary>

**Fine neighbors (dense window)**:
- Window size $w \times w$ → $w^2$ tokens
- Complexity: linear in window size (low)
- 역할: Local detail (edges, textures)
- Typical: 7×7=49 tokens

**Coarse neighbors (sparse sampling)**:
- Count: $c$ (fixed)
- Complexity: linear in $c$ (low)
- 역할: Global context
- Typical: 32 tokens

**Fine:Coarse ratio 설정**:

총 attention 복잡도: $O(n \times (w^2 + c))$

최적화 문제: minimize $w^2 + c$ subject to achieving good accuracy.

**Empirical (Focal Transformer 논문)**:
- $w=7, c=32$: Acc=84.1% (ImageNet-1k)
- $w=5, c=32$: Acc=83.8% (window too small)
- $w=9, c=32$: Acc=84.0% (marginal improvement, higher cost)
- $w=7, c=16$: Acc=83.5% (coarse tokens too few)
- $w=7, c=48$: Acc=84.0% (coarse overhead)

따라서 49:32 비율 ($w=7, c=32$) 이 accuracy-complexity trade-off 에서 최적.

**일반화**:
- Dataset 크기가 크면 coarse 비중 증대 (global context 중요)
- Object detection 같이 local 이 중요하면 fine 비중 증대

</details>

### 문제 3 (논문 비평): MViT vs Focal Transformer — 설계 철학

MViT: "모든 stage 에서 pooling 적용" (progressive)  
Focal: "모든 stage 에서 fine-coarse 조합" (within-stage)

두 approach 의 trade-off 를 논의하고, 각각 어떤 상황에 최적인지 설명하시오.

<details>
<summary>해설</summary>

**MViT: Progressive Pooling (Stage-wise)**

설계 철학:
- 각 stage 에서 resolution을 점진적으로 줄임
- Early stage: fine (full resolution), Late stage: coarse (pooled)
- CNN FPN 과 동일한 hierarchical structure

장점:
1. Implicit hierarchy: 자동으로 multi-scale features 생성
2. Video friendly: temporal + spatial multi-scale 분리 용이
3. Parameter efficient: pooling 으로 feature reduction

단점:
1. Stage transition 의 정보 손실 (irreversible)
2. Early stage 에서 coarse information 활용 불가
3. Fine detail 과 global context 동시 활용 불가 (stage 별로만)

**Focal Transformer: Fine-Coarse Combination (Within-stage)**

설계 철학:
- 각 stage 에서 fine + coarse neighbors 동시 처리
- 모든 token 이 local detail 과 global context 동시 접근

장점:
1. Flexible attention: 각 token 이 필요한 information 자유롭게 선택
2. Human vision inspired: Focal point (fine) + Peripheral vision (coarse) 모방
3. 동일 layer 에서 multi-scale information 결합

단점:
1. Complexity 여전히 높음 (fine dense + coarse sparse)
2. Coarse sampling strategy 가 critical → learned sampling 필요
3. Architecture 복잡성 증가

**선택 기준**:

1. **Video understanding**: MViT 추천
   - Temporal 과 spatial 의 분리가 명확
   - Pool-based attention 이 temporal coherence 보존

2. **Image classification (ImageNet)**: Focal Transformer 동급
   - 정확도: 거의 같음 (84.1% vs 84.0%)
   - Focal 이 약간 더 flexible

3. **Dense prediction (detection, segmentation)**:
   - MViT: explicit hierarchical features → detection backbone 에 최적
   - Focal: within-layer multi-scale → semantic segmentation 에 유리

4. **Efficiency (mobile)**:
   - MViT: progressive pooling 으로 더 효율적
   - Focal: fine neighbors 여전히 dense

5. **Scalability (large-scale pre-training)**:
   - MViT: parameter efficient
   - Focal: flexibility → larger models 에 더 적합

**결론**:
MViT 와 Focal 은 다른 설계 철학의 구현이며, 각 use case 에 따라 최적화 가능.
최근 (2024+) 추세는 둘 다 보다는 pure ViT scaling (DINOv2) 또는 
mixture-of-experts 형태의 adaptive attention 으로 발전 중.

</details>

---

<div align="center">

[◀ 이전](./04-cvt-coatnet.md) | [📚 README](../README.md) | [다음 ▶](../ch3-contrastive-ssl/01-ssl-paradigms.md)

</div>
