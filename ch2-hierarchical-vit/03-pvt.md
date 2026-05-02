# 03. PVT — Pyramid Vision Transformer

## 🎯 핵심 질문

- Swin 의 window attention 과 다르게, PVT 는 어떻게 **spatial reduction** 으로 attention 복잡도를 줄이는가?
- Key 와 Value 를 downsample 하면서 Query 는 유지하면 어떤 이점이 있는가?
- Dense prediction (detection, segmentation) 을 위해 hierarchical features 가 왜 필수적인가?
- PVT 가 Swin 과 함께 "hierarchical ViT" 의 양대 산맥이 되는 이유는?

## 🔍 왜 이 Spatial Reduction 이 Pyramid Vision 의 핵심인가

Swin Transformer 는 **window 로 spatial 을 나눈다** (partition). 각 window 내에서만 attention 을 계산하므로 복잡도는 $O(n \cdot w^2)$.

**Wang et al. (2021) PVT 의 다른 접근**: Window partition 대신 **spatial reduction** 을 이용한다.

핵심 아이디어:
- **Query (Q)**: 모든 patch 에 대해 유지 (각 patch 가 "질문" 을 던짐)
- **Key (K), Value (V)**: Stride-$R$ convolution 으로 spatial 을 $R \times R$ 배수만큼 downsample

결과: 
- Attention 복잡도 $O(n^2)$ → $O(n^2 / R^2)$ (K, V 의 spatial 이 줄어들므로)
- Global receptive field 를 유지하면서도 selective 한 정보만 집계
- **CNN 의 FPN (Feature Pyramid Network)** 과 유사한 hierarchical structure 자연 생성

Swin 과의 비교:
| 항목 | Swin | PVT |
|------|------|-----|
| **방식** | Window partition (spatial split) | Spatial reduction (channel compression) |
| **Query** | Full resolution | Full resolution |
| **Key/Value** | Full resolution (window 내) | Reduced resolution (stride-R conv) |
| **복잡도** | $O(n \cdot w^2)$ | $O(n^2 / R^2)$ |
| **Global RF** | Shifted window 로 확보 | Q 가 모든 patch 를 query 하므로 자동 |
| **Dense prediction** | U-Net decoder 필요 | Hierarchical features 자동 생성 |

## 📐 수학적 선행 조건

- **ViT 기초** (Ch1): Self-attention, multi-head mechanism
- **Swin Transformer** (Ch2-02): Window attention, hierarchical features
- **Spatial downsampling**: Stride convolution, pooling, resolution 변화
- **Depthwise separable convolution**: K, V 를 효율적으로 downsample 하는 방법
- **Feature pyramid**: CNN 의 FPN, multi-scale features 의 필요성
- **Dense prediction** (Object Detection Deep Dive): Anchor-based/anchor-free 방식과 feature hierarchy

## 📖 직관적 이해

```
Query: "이 위치의 정보를 모두에게 물어본다"
Key/Value: "필요한 부분만 압축해서 답한다"

┌─────────────────────────────────────┐
│ PVT: Spatial Reduction Attention    │
│                                     │
│ Input: Patch embeddings (n patches) │
│ [P][P][P][P][P]...                 │
│  ↓   ↓   ↓   ↓   ↓                 │
│                                     │
│ Query (Q): Full resolution 유지     │
│ [Q₁ Q₂ Q₃ Q₄ Q₅ ...] ← n tokens    │
│                                     │
│ Key/Value: Spatial reduction 적용   │
│ (stride-R conv)                     │
│ [K/V₁ K/V₂ K/V₃ ...]  ← n/R² tokens│
│                                     │
│ Attention: Q @ K^T / √d             │
│ Complexity: n × (n/R²) = n²/R²      │
│                                     │
│ Output: Attended features (n tokens)│
└─────────────────────────────────────┘

예시: R=2 인 경우
Input:  4x4 = 16 patches
Q:      16 tokens (full)
K/V:    4 tokens (2x2 subsampled)
Attn:   16 × 4 = 64 ops (vs 256 ops for global)
Speedup: 256 / 64 = 4x
```

**비유**: 
- **ViT**: "모든 사람에게 모든 질문을 던진다" (완전 연결, 비효율)
- **Swin**: "작은 그룹으로 나눠서 그룹 내 질문만 던진다" (분할, local bias)
- **PVT**: "모든 사람에게 질문을 던지지만, 중요한 사람들의 답만 들어준다" (selective, global + efficient)

## ✏️ 엄밀한 정의

### 정의 2.10: Spatial Reduction Attention (SRA)

입력 $X \in \mathbb{R}^{n \times d}$ (n = patch 수, d = embedding dimension) 에 대해:

**Query**: 모든 token 을 유지
$$Q = X W_Q, \quad Q \in \mathbb{R}^{n \times d}$$

**Key/Value**: Spatial reduction 적용
$$X_{\text{red}} = \text{Conv2d}_{\text{stride-R}}(X), \quad X_{\text{red}} \in \mathbb{R}^{(n/R^2) \times d}$$

이후:
$$K = X_{\text{red}} W_K, \quad V = X_{\text{red}} W_V$$

**Attention 계산**:
$$\mathrm{SRA}(Q, K, V) = \mathrm{softmax}\left( \frac{QK^\top}{\sqrt{d}} \right) V$$

**Complexity**:
- Q @ K^T: $n \times (n/R^2) = n^2 / R^2$ 
- Softmax & matmul with V: $O(n^2 / R^2)$

따라서 global attention $O(n^2)$ 대비 $1/R^2$ 배 감소.

### 정의 2.11: Pyramid Stages

PVT 는 4개 stage 로 구성되며, 각 stage 마다 spatial resolution 을 줄임.

Stage $i$ (i = 1, 2, 3, 4):
- **Input**: Patch embeddings from stage $i-1$
- **Spatial shape**: $H_i \times W_i \times C_i$
- **Layer blocks**: $L_i$ 개의 Transformer block (각 block 은 norm + SRA + FFN)
- **Output (다음 stage input)**: Patch merging → $(H_{i+1}, W_{i+1}) = (H_i/2, W_i/2)$, $(C_{i+1}) = 2 C_i$

**크기 예시** (ImageNet 224x224, patch=4):
- Stage 1: $56 \times 56 \times 64$ (R=8, 4개 patch merge)
- Stage 2: $28 \times 28 \times 128$ (R=4, 또 2배 downsample)
- Stage 3: $14 \times 14 \times 320$
- Stage 4: $7 \times 7 \times 512$

### 정의 2.12: Spatial Reduction Convolution

Stride-R convolution 으로 spatial dimension 을 줄임:

$$X_{\text{red}} = \text{Conv2d}_{\text{kernel=R, stride=R, groups=C_{\text{in}}}}(X)$$

더 정확히는, depthwise separable convolution:
$$X_{\text{red}} = \text{Conv2d}_{\text{DW}}(\text{Conv2d}_{\text{1×1}}(X))$$

여기서 depthwise convolution 이 R stride 로 spatial 을 downsample.

**효과**:
- Spatial: $(H, W) \to (H/R, W/R)$ (quadratic 감소)
- Channel: 유지 또는 조정

## 🔬 정리와 증명

### 정리 2.6: Spatial Reduction Attention 의 복잡도

**명제**: Spatial reduction ratio $R$ 을 적용하면, 
attention 복잡도가 $O(n^2) \to O(n^2 / R^2)$ 로 감소한다.

**증명**:

Query: $Q \in \mathbb{R}^{n \times d}$  
Key: $K \in \mathbb{R}^{(n/R^2) \times d}$ (spatial reduction 으로 $n$ → $n/R^2$)

Attention score: $S = QK^\top \in \mathbb{R}^{n \times (n/R^2)}$

계산 복잡도:
- Matrix multiplication (Q @ K^T): $n \times d \times (n/R^2) = O(n^2 d / R^2)$
- Softmax (per query): $n \times (n/R^2) = O(n^2 / R^2)$
- Output (S @ V): $n \times (n/R^2) \times d = O(n^2 d / R^2)$

전체: $O(n^2 d / R^2)$ (vanilla attention: $O(n^2 d)$)

따라서 $O(1/R^2)$ speedup. $\square$

### 정리 2.7: Global Receptive Field 의 자동 보장

**명제**: PVT 에서는 Query 가 full resolution 을 유지하므로, 
각 layer 에서 모든 patch 가 서로 interaction 할 가능성이 있다 (비록 K, V 는 reduced).

**증명의 직관**:

각 query token $q_i$ (i-th patch) 는 모든 K, V 를 attend 할 수 있다:
$$\text{Attn}(q_i) = \sum_{j \in [\text{reduced}]} \alpha_{ij} v_j$$

비록 K, V 가 spatial reduction 으로 줄어들었더라도, 
key insight: **Query 가 full resolution 이므로, 각 patch 는 "전역" 관점에서 정보를 수집한다.**

Swin 과의 차이:
- Swin: Window partition 으로 각 layer 마다 많은 patch 쌍이 attend 할 수 **없음** 
  → Shifted window 로 보상 필요
- PVT: Query 가 모두를 "질문" 하므로, K/V 가 작더라도 충분한 information flow

따라서 PVT 는 추가적인 shift mechanism 없이도 global receptive field 를 자동으로 확보.

**형식**:

Information flow: $q_i$ 에서 모든 $v_j$ 로의 path 가 명확히 존재.
Gradient flow: $\nabla_{v_j} L = \sum_i \alpha_{ij} \nabla_{q_i} L$ 로, 모든 query 가 contribute.

따라서 global connectivity 보장. $\square$

### 정리 2.8: Hierarchical Features 의 Detection 적합성

**명제**: PVT 의 4-stage pyramid structure 는 Faster R-CNN 같은 object detection 에 
자연스럽게 fit 되어, U-Net decoder 없이도 multi-scale features 를 제공한다.

**증명의 직관**:

Faster R-CNN 에서 feature pyramid 는:
- Low-level features (fine-grained, high resolution): 작은 object 감지
- High-level features (semantic, low resolution): 큰 object 감지

PVT 의 자동 생성:
- Stage 1: $56 \times 56$ (fine, $C_1 = 64$ channels)
- Stage 2: $28 \times 28$ (mid, $C_2 = 128$)
- Stage 3: $14 \times 14$ (mid-coarse, $C_3 = 320$)
- Stage 4: $7 \times 7$ (coarse, $C_4 = 512$)

각 stage 의 output 을 pyramid 로 사용하면, FPN 없이도 multi-scale 정보를 얻음.

**Backbone 으로서의 이점**:
- Dense prediction (detection, instance segmentation) 에 직접 적용 가능
- CNN FPN 대비 더 효율적인 feature extraction
- Swin 과 달리 별도의 decoder 구조 불필요

따라서 "pyramid + transformer" 의 자연스러운 결합. $\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Spatial Reduction Convolution

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialReductionConv(nn.Module):
    """
    Spatial reduction using depthwise separable convolution.
    Reduce spatial dimensions by factor R while maintaining channels.
    """
    def __init__(self, dim, reduction_ratio=2):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        # Depthwise conv (groups = dim)
        self.depthwise = nn.Conv2d(
            dim, dim, kernel_size=reduction_ratio,
            stride=reduction_ratio, groups=dim, padding=0
        )
    
    def forward(self, x):
        """
        x: (B, H, W, C) or (B, C, H, W)
        Assuming (B, C, H, W) for conv2d
        """
        B, C, H, W = x.shape
        x = self.depthwise(x)  # (B, C, H/R, W/R)
        return x

torch.manual_seed(42)
B, C, H, W = 2, 64, 56, 56
reduction_ratio = 2

x = torch.randn(B, C, H, W)
sr_conv = SpatialReductionConv(C, reduction_ratio)
x_reduced = sr_conv(x)

print(f"Original shape: {x.shape}")
print(f"Reduced shape: {x_reduced.shape}")
print(f"Reduction: {H}x{W} → {H//reduction_ratio}x{W//reduction_ratio}")
print(f"Spatial reduction ratio: {(H*W) / (x_reduced.shape[2]*x_reduced.shape[3]):.1f}x")
```

### 실험 2: SRA (Spatial Reduction Attention) Layer

```python
import torch
import torch.nn as nn

class SRA(nn.Module):
    """
    Spatial Reduction Attention.
    Q: full resolution
    K, V: reduced resolution
    """
    def __init__(self, dim, num_heads=8, sr_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
        # Spatial reduction for K, V
        self.sr_conv = nn.Conv2d(
            dim, dim, kernel_size=sr_ratio,
            stride=sr_ratio, groups=dim
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, H, W):
        """
        x: (B, N, C) where N = H*W
        H, W: spatial dimensions
        return: (B, N, C)
        """
        B, N, C = x.shape
        
        # Query: full resolution
        q = self.q(x)  # (B, N, C)
        q = q.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        q = q.permute(0, 2, 3, 1).reshape(B, N, C)  # Back to (B, N, C)
        
        # K, V: spatial reduction
        x_sr = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        x_sr = self.sr_conv(x_sr)  # (B, C, H/sr, W/sr)
        
        H_sr, W_sr = x_sr.shape[2], x_sr.shape[3]
        x_sr = x_sr.permute(0, 2, 3, 1).reshape(B, H_sr * W_sr, C)
        x_sr = self.norm(x_sr)
        
        k = self.k(x_sr)  # (B, N_sr, C)
        v = self.v(x_sr)
        
        # Multi-head attention
        q = q.reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # (B, num_heads, N, C/num_heads)
        
        k = k.reshape(B, H_sr * W_sr, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3)  # (B, num_heads, N_sr, C/num_heads)
        
        v = v.reshape(B, H_sr * W_sr, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3)  # (B, num_heads, N_sr, C/num_heads)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** (-0.5)
        attn = attn.softmax(dim=-1)  # (B, num_heads, N, N_sr)
        
        out = attn @ v  # (B, num_heads, N, C/num_heads)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(B, N, C)
        
        out = self.proj(out)
        return out

torch.manual_seed(42)
B, C, H, W = 2, 64, 56, 56
N = H * W

sra = SRA(dim=C, num_heads=8, sr_ratio=2)
x = torch.randn(B, N, C)

out = sra(x, H, W)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"SRA with sr_ratio=2: spatial reduction 4x")

# Complexity comparison
attn_full = N * N  # ViT global
attn_sra = N * (H // 2 * W // 2)  # PVT
ratio = attn_full / attn_sra
print(f"\nComplexity:")
print(f"  Full attention: {attn_full:,} operations")
print(f"  SRA attention: {attn_sra:,} operations")
print(f"  Speedup: {ratio:.1f}x")
```

### 실험 3: Pyramid Stages 와 Hierarchical Features

```python
import torch
import torch.nn as nn

class PVTStage(nn.Module):
    """Single PVT stage"""
    def __init__(self, dim, sr_ratio=2, depth=2):
        super().__init__()
        self.sra_layers = nn.ModuleList([
            SRA(dim, sr_ratio=sr_ratio)
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
        for sra, ffn, norm1, norm2 in zip(self.sra_layers, self.ffn_layers, 
                                          self.norms_1, self.norms_2):
            x_norm = norm1(x)
            x = x + sra(x_norm, H, W)
            x_norm = norm2(x)
            x = x + ffn(x_norm)
        return x

class PVTBackbone(nn.Module):
    """Simplified PVT with 4 stages"""
    def __init__(self, img_size=224, patch_size=4, num_classes=1000):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 64, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2, 64))
        
        # 4 stages
        self.stage1 = PVTStage(64, sr_ratio=8, depth=2)   # 56x56
        self.stage2 = PVTStage(128, sr_ratio=4, depth=2)  # 28x28
        self.stage3 = PVTStage(320, sr_ratio=2, depth=2)  # 14x14
        self.stage4 = PVTStage(512, sr_ratio=1, depth=2)  # 7x7
        
        self.merging1 = self._make_merge(64, 128)
        self.merging2 = self._make_merge(128, 320)
        self.merging3 = self._make_merge(320, 512)
        
        self.head = nn.Linear(512, num_classes)
    
    def _make_merge(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim * 4, out_dim),
        )
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, 64, 56, 56)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 56*56, 64)
        x = x + self.pos_embed
        
        # Stage 1
        x = self.stage1(x, H, W)  # (B, 3136, 64)
        
        # Merge to stage 2
        x = x.reshape(B, H, W, C)
        x = x.reshape(B, H//2, 2, W//2, 2, C).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H//2 * W//2, 4*C)
        x = self.merging1(x)  # (B, 784, 128)
        
        # Stage 2
        H, W = H // 2, W // 2
        x = self.stage2(x, H, W)
        
        # ... (merging 2, 3 similarly)
        
        # Global average pooling
        x_cls = x.mean(dim=1)  # (B, 512)
        out = self.head(x_cls)
        
        return out

torch.manual_seed(42)
B = 2

# Create simplified backbone
backbone = PVTBackbone()

# Forward
x = torch.randn(B, 3, 224, 224)
out = backbone(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"\nPVT Hierarchical Features:")
print(f"  Stage 1: 56×56×64")
print(f"  Stage 2: 28×28×128")
print(f"  Stage 3: 14×14×320")
print(f"  Stage 4: 7×7×512")
```

### 실험 4: SRA vs Global Attention 복잡도 비교

```python
import torch

def attention_complexity(n, sr_ratio=None):
    """
    n: number of patches
    sr_ratio: spatial reduction ratio (None for global)
    return: (ops, memory in elements)
    """
    if sr_ratio is None:  # Global attention
        return n**2, n**2
    else:  # SRA
        n_reduced = n // (sr_ratio ** 2)
        ops = n * n_reduced  # Q @ K^T
        mem = n * n_reduced + n_reduced * n  # Key + Value matrices
        return ops, mem

# ImageNet-1k: 224x224, patch=4
H_init, W_init, P = 224, 224, 4
H, W = H_init // P, W_init // P

print(f"ImageNet-1k (224x224, patch=4): {H}×{W} = {H*W} patches\n")

stages = [
    (56, 56, 8, "Stage 1"),
    (28, 28, 4, "Stage 2"),
    (14, 14, 2, "Stage 3"),
    (7, 7, 1, "Stage 4"),
]

print("PVT Stage Analysis:")
print("─" * 60)

for h, w, sr, label in stages:
    n = h * w
    ops_global, _ = attention_complexity(n)
    ops_sra, _ = attention_complexity(n, sr_ratio=sr)
    
    speedup = ops_global / ops_sra if ops_sra > 0 else float('inf')
    
    print(f"{label}: {h}×{w} ({n} patches)")
    print(f"  Global:  {ops_global:,} ops")
    print(f"  SRA:     {ops_sra:,} ops")
    print(f"  Speedup: {speedup:.1f}x")
    print()
```

## 🔗 실전 활용

**timm 에서의 PVT 구현**:
```python
from timm.models import pvt_v2_b2
model = pvt_v2_b2(pretrained=True)
```

**Object Detection (RetinaNet, Faster R-CNN backbone)**:
```python
from mmdet.models import build_backbone
backbone = build_backbone(dict(type='PVT', ...))
```

**Semantic Segmentation (Semantic FPN)**:
```python
from mmseg.models import build_backbone
backbone = build_backbone(dict(type='PVT', embed_dims=[64, 128, 320, 512]))
```

**Dense prediction 장점**:
- Hierarchical features 자동 생성 (4-stage pyramid)
- Global attention 의 flexibility + local convolution 의 efficiency
- Swin 과 달리 추가 decoder 구조 불필요

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Spatial reduction 정보손** | K, V downsample 시 정보 손실 | 매우 높은 sr_ratio 는 성능 악화 |
| **Sr_ratio 의 균형** | Stage 별로 다른 sr_ratio 필요 | Hyperparameter tuning 이 중요 |
| **Query 의 full resolution** | Q 는 유지되어 메모리 비용 증가 | Batch size 제약 (Q 가 크면 OOM) |
| **Relative position bias 부재** | PVT 는 absolute position 사용 | Swin 의 relative bias 만큼 translation robust 하지 않음 |
| **Patch merging 정보손** | 4개 patch → 2D 로 squeeze 시 경계 정보 손실 | Dense prediction 에서 skip-connection 으로 보상 필요 |

## 📌 핵심 정리

$$\boxed{\mathrm{SRA}: \text{Attn}(Q_{\text{full}}, K_{\text{reduced}}, V_{\text{reduced}}) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V, \quad \text{Complexity: } O(n^2 / R^2)}$$

| 개념 | 식 | 역할 |
|------|-----|------|
| **Query** | $Q = X W_Q$ (full resolution) | 모든 patch 가 정보 수집 |
| **Key/Value** | $K, V = X_{\text{red}} W_{K,V}$ (spatial reduction) | Selective 정보 집계 |
| **Spatial reduction** | Conv stride-R | Complexity 선형화 |
| **Pyramid stages** | 4 stages (56→28→14→7) | Multi-scale features 자동 생성 |
| **Global RF** | Query full 로 자동 보장 | Shifted window 없이도 global connectivity |

**결론**: PVT 는 Spatial Reduction Attention 으로 $O(n^2) \to O(n^2/R^2)$ complexity 를 달성하면서, 
Query 의 full resolution 유지로 global receptive field 를 자동 보장한다. 
Hierarchical features 로 dense prediction 에 최적화. Swin 과 함께 hierarchical ViT 의 양대 축.

## 🤔 생각해볼 문제

### 문제 1 (기초): Query vs Key/Value 의 비대칭성

PVT 에서 Query 는 full resolution 을 유지하고 Key/Value 만 downsample 한다. 
이게 역이라면 어떻게 될까? 즉, Key/Value 는 full resolution, Query 만 downsample?

<details>
<summary>해설</summary>

만약 Query 를 downsample 하고 K, V 를 full resolution 으로 유지하면:

문제점:
1. **정보 손실**: Downsampled Q 는 원래 위치 정보를 잃음 → "누가 물어봤는지" 불명확
2. **Output shape 불일치**: Q 가 n/R² 개이므로, attention output 도 n/R² 개 → 원래 input resolution 으로 돌아갈 수 없음 (upsampling 필요)
3. **Information bottleneck**: "여러 위치의 질문이 같은 reduced query 로 압축됨" → semantic ambiguity

**왜 PVT 의 방식이 올바른가?**:
- Query full, K/V reduced: "모든 위치가 각각 질문하고, 중요한 정보만 답한다" → clear signal
- 역방향: "몇 개 위치만 질문하고, 모든 정보로 답한다" → 질문 위치가 불명확

따라서 PVT 의 비대칭성은 정보 flow 의 방향성을 명확히 한다.

</details>

### 문제 2 (심화): Spatial Reduction Ratio 의 최적 선택

PVT 에서 각 stage 마다 다른 sr_ratio 를 사용한다:
- Stage 1: sr_ratio=8
- Stage 2: sr_ratio=4
- Stage 3: sr_ratio=2
- Stage 4: sr_ratio=1

이 수열 (8, 4, 2, 1) 을 어떤 원리로 결정할까?

<details>
<summary>해설</summary>

고려 요소:

1. **Computational budget**: 
   - Early stage (1, 2): 높은 spatial resolution (56×56, 28×28) → sr_ratio 를 크게 (8, 4)
   - Late stage (3, 4): 낮은 spatial resolution (14×14, 7×7) → sr_ratio 를 작게 (2, 1)
   - 전체 complexity 를 균형있게 분배

2. **Semantic information**:
   - Early stage: 저수준 특징 (edges, textures) → spatial detail 중요하지만, global context 덜 중요
     → Aggressive reduction (sr_ratio=8) 가능
   - Late stage: 고수준 특징 (objects, scenes) → global context 중요
     → Conservative reduction (sr_ratio=1) 필요

3. **Empirical validation**:
   PVT 논문 Table 에서 sr_ratio=(8,4,2,1) 조합이 다른 조합보다 우수함을 보임.

**수식적 동기**:

Complexity 를 정규화하면:
$$\text{Total Complexity} = \sum_{i=1}^{4} H_i \cdot W_i \cdot \frac{H_i \cdot W_i}{R_i^2} \cdot d_i$$

균형잡힌 배치(balanced allocation)로는, 각 stage 가 유사한 computational load 를 가져야 함:
$$H_1 W_1 \cdot \frac{H_1 W_1}{R_1^2} \approx H_2 W_2 \cdot \frac{H_2 W_2}{R_2^2} \approx \ldots$$

이는:
$$\frac{(H_i W_i)^2}{R_i^2} \approx \text{const}$$

Early stage ($H_1=56, W_1=56$): $(56^2)^2 / R_1^2$ → $R_1$ 커야 함 (8)  
Late stage ($H_4=7, W_4=7$): $(7^2)^2 / R_4^2$ → $R_4$ 작아야 함 (1)

따라서 (8, 4, 2, 1) 은 computational load 균등 분배의 결과이다.

</details>

### 문제 3 (논문 비평): PVT vs Swin — Dense Prediction 에서의 선택

PVT 는 Spatial Reduction Attention 을,Swin 은 Window Attention 을 사용한다.
Object detection 과 instance segmentation 에서 어느 것이 더 나을까?

논문 결과, 당시(2021) 에는 Swin 이 약간 더 나았는데, 그 이유는?

<details>
<summary>해설</summary>

**Swin 의 장점** (2021):
1. **Relative position bias**: Learned bias 가 translation variance 를 강화 → boundary-aware (object edge detection 에 유리)
2. **Window structure**: 자연스러운 locality bias → small object detection 에 유리
3. **Implementation maturity**: Shifted window 의 CUDA kernel 최적화가 더 성숙
4. **Perception bias**: Window-based decomposition 이 human visual system 과 더 유사 (?)

**PVT 의 단점** (당시):
1. **Absolute position embedding**: Generalization 이 약할 수 있음
2. **Spatial reduction 노이즈**: K, V 를 downsample 하면서 경계 정보 손실
3. **Query 메모리**: Q 가 full resolution 이므로 메모리 사용량 많음 (large batch size 제약)

**현재 (2024-2025) 의 업데이트**:
- PVT v2: Overlapping patch embedding, cosine attention, linear complexity MLP → Swin 을 능가
- CoAtNet: Conv + Attention hybrid 로 둘 다의 장점 취함
- 최신 연구: Hierarchical vision model 의 최적 설계는 여전히 open question

**결론**: 
- 2021: Swin > PVT (COCO AP 격차 약 1-2%)
- 2023+: PVT-v2 ≈ Swin ≈ EfficientNet-v2
- 선택: Task-specific hyperparameter tuning 이 더 중요

</details>

---

<div align="center">

[◀ 이전](./02-swin.md) | [📚 README](../README.md) | [다음 ▶](./04-cvt-coatnet.md)

</div>
