# 04. CvT · CoAtNet — Conv–Attention Hybrid

## 🎯 핵심 질문

- **CvT**: Patch embedding 을 overlapping convolution 으로 하면, ViT 의 locality inductive bias 를 회복할 수 있는가?
- **CvT**: QKV projection 을 depthwise convolution 으로 하면, translation equivariance 를 강화할 수 있는가?
- **CoAtNet**: CNN (MBConv) 과 Transformer 의 4-stage hybrid 구조는 어느 배치가 최적인가?
- 같은 FLOPs 에서 pure CNN · pure ViT 를 모두 능가하는 hybrid 의 원리는?

## 🔍 왜 이 Hybrid 가 Architectural Evolution 의 핵심인가

DeiT, Swin, PVT 까지는 모두 "Transformer 를 어떻게 더 효율적으로 만들 것인가" 라는 질문에 답했다.

**Wu et al. (2021) CvT** 와 **Dai et al. (2021) CoAtNet** 은 다른 질문을 던진다:
"Transformer 와 CNN 의 inductive bias 를 결합하면 어떻게 될까?"

**CvT 의 핵심**:
- Patch embedding 을 **overlapping convolution** (stride < kernel) 로 → locality 회복
- QKV projection 을 **depthwise convolution** 으로 → translation equivariance 강화
- 결과: ViT-B 와 같은 parameter 에서 2-3% 높은 정확도

**CoAtNet 의 핵심**:
- 4-stage hybrid: 초반 2 stages 는 **MBConv** (CNN), 후반 2 stages 는 **Transformer**
- 직관: 저수준 특징 (edges, textures) 는 CNN 으로 효율적 추출, 고수준 의미론 (objects, scenes) 은 Transformer 로 장거리 관계 포착
- 결과: 같은 FLOPs 에서 ResNet, EfficientNet, ViT 모두 능가

## 📐 수학적 선행 조건

- **Convolution** (CNN Deep Dive): Kernel size, stride, padding, groups
- **Depthwise separable convolution**: Groups=C 인 convolution (channel-wise)
- **MobileNetV2 / EfficientNet**: Inverted residual, squeeze-excitation
- **ViT self-attention** (Ch1): Multi-head mechanism, complexity analysis
- **Inductive bias**: Translation equivariance, locality, weight sharing
- **FLOPs**: Computational complexity 측정 (memory-efficient 와는 다름)

## 📖 직관적 이해

```
CNN vs ViT 의 Inductive Bias Trade-off:

┌──────────────────────────────┐
│ Pure CNN (ResNet, EfficientNet)
│ ✓ Translation equivariance     │
│ ✓ Weight sharing (kernel)      │
│ ✓ Locality (convolution)       │
│ ✗ Long-range dependency ↓      │
│ ✗ Global receptive field (깊어야 함)
└──────────────────────────────┘

┌──────────────────────────────┐
│ Pure ViT                       │
│ ✓ Long-range dependency        │
│ ✓ Global receptive field (1 layer)
│ ✗ Translation equivariance ↑   │
│ ✗ Locality (global attention)  │
│ ✗ More data needed             │
└──────────────────────────────┘

┌──────────────────────────────┐
│ CvT: Conv in Patch Embed + QKV
│ Overlapping conv patch embed   │
│ → Locality + position-awareness│
│                                │
│ Depthwise conv for QKV         │
│ → Translation equivariance     │
│                                │
│ Still global self-attention    │
│ → Long-range dependency        │
└──────────────────────────────┘

┌──────────────────────────────┐
│ CoAtNet: Stage-wise Hybrid     │
│                                │
│ Stage 1: MBConv               │
│ ├─ Low-level features         │
│ ├─ Spatial structure 포착      │
│ └─ Inductive bias strong      │
│                                │
│ Stage 2: MBConv               │
│ ├─ Mid-level features         │
│ └─ Receptive field expand    │
│                                │
│ Stage 3: Transformer          │
│ ├─ High-level semantics       │
│ ├─ Long-range relation        │
│ └─ Global context             │
│                                │
│ Stage 4: Transformer          │
│ ├─ Abstract representation    │
│ └─ Task-specific refinement    │
└──────────────────────────────┘
```

**비유**:
- **Pure CNN**: "지역 주민들 (pixel) 끼리만 대화" (local receptive field)
- **Pure ViT**: "모든 사람이 동시에 전체 이야기" (global attention, data-hungry)
- **CvT**: "지역 구조는 유지하면서 전체를 본다" (conv patch embed + global attention)
- **CoAtNet**: "처음엔 지역 얘기 (MBConv), 나중엔 전체를 본다" (hierarchical hybrid)

## ✏️ 엄밀한 정의

### 정의 2.13: CvT — Overlapping Convolution Patch Embedding

Vanilla ViT 의 patch embedding:
$$z_0 = X_{\text{patch}} W_E, \quad X_{\text{patch}} \in \mathbb{R}^{P^2 C \times D}$$

여기서 $P$ 는 patch size, stride = P (non-overlapping).

**CvT 의 overlapping patch embedding**:
$$z_0 = \text{Conv2d}(X, C, D, \text{kernel=}K, \text{stride=}S)$$

여기서 $K > S$ (overlapping). 보통 $K=7, S=4$.

**효과**:
- Overlapping → neighboring patches 간 redundancy 감소 + locality 강화
- Stride < kernel → position awareness 증가 (pixel 의 절대 위치 정보 유지)
- Vanilla ViT (stride=patch size) 보다 translation equivariance 우수

### 정의 2.14: CvT — Depthwise Conv for QKV Projection

Vanilla ViT 의 QKV projection:
$$[Q, K, V] = \text{Linear}(X) \in \mathbb{R}^{n \times 3d}$$

**CvT 의 depthwise conv QKV**:

1. Reshape to spatial: $X \in \mathbb{R}^{B \times n \times d} \to X' \in \mathbb{R}^{B \times C \times H \times W}$
2. Depthwise convolution (groups=d):
$$[Q', K', V'] = \text{DepthwiseConv2d}(X', d, 3d, \text{kernel=3, padding=1})$$
3. Reshape back: $[Q, K, V] \in \mathbb{R}^{B \times n \times d}$ (각각)

**효과**:
- Channel 별 convolution (depthwise) → translation equivariance 보존
- Local spatial information 을 QKV 에 직접 encoding
- Global self-attention 의 receptive field 는 유지하면서 locality 추가

### 정의 2.15: CoAtNet — 4-Stage Hybrid Architecture

**4-stage structure**:

$$\begin{array}{c|c|c|c}
\text{Stage} & \text{Blocks} & \text{Spatial Size} & \text{Operator} \\
\hline
1 & 2 \times \text{MBConv} & H/4 \times W/4 & \text{CNN} \\
2 & 2 \times \text{MBConv} & H/8 \times W/8 & \text{CNN} \\
3 & 6 \times \text{Transformer} & H/8 \times W/8 & \text{Attention} \\
4 & 2 \times \text{Transformer} & H/8 \times W/8 & \text{Attention} \\
\end{array}$$

**MBConv** (Mobile Inverted Bottleneck, EfficientNet):
$$\text{MBConv}(X) = X + \text{PWConv}(\text{ReLU}(\text{DWConv}(\text{PWConv}(X))))$$

여기서 PWConv = pointwise (1×1) convolution, DWConv = depthwise.

**Transformer block**: Standard multi-head self-attention + FFN.

**핵심**: Spatial resolution 이 Stage 2 부터 동일 (H/8 × W/8) 이므로, 
MBConv 와 Transformer 가 같은 resolution 에서 정보를 처리.

### 정의 2.16: Hybrid Loss 와 Gradient Flow

CoAtNet 전체 loss 는 classification loss 만 사용하되, 
각 stage 의 feature 에 다른 기여도를 갖도록 설계 가능:

$$L = L_{\text{task}} + \sum_{i=1}^{4} \lambda_i \cdot L_{\text{aux}_i}$$

하지만 standard CoAtNet 은 $\lambda_i = 0$ (단일 classifier).

**Gradient flow**:
- Early CNN stages: Local receptive field + spatial structure → low-level gradient 명확
- Late Transformer stages: Long-range information → high-level semantic gradient

이 두 gradient 가 hybrid 에서 어떻게 상호작용하는가는 아직도 미해결 문제.

## 🔬 정리와 증명

### 정리 2.9: CvT 의 Translation Equivariance 강화

**명제**: Depthwise conv for QKV projection 을 사용하면, 
vanilla ViT 의 position embedding 에 의존하지 않아도 translation equivariance 를 부분적으로 회복할 수 있다.

**증명의 직관**:

Vanilla ViT:
- Q, K, V 는 linear projection → translation variance (position embedding 에만 의존)
- Input spatial shift → Q, K, V 는 변하지 않음 (position embedding 이 변함)

CvT (depthwise conv):
- Q, K, V 를 spatial convolution 으로 계산 → translation equivariance 부분적 유지
- Input spatial shift $\Delta x$ → Q, K, V 도 같은 방향 shift
- Attention score: $QK^\top$ 도 shift-invariant (spatial position 에 기반한 relative distance 유지)

**형식**:

Input shift: $X \to X_{\text{shift}}(x) = X(x + \Delta x)$

Vanilla ViT: $Q_{\text{shift}} = X_{\text{shift}} W_Q = X(x+\Delta x) W_Q \neq Q(x+\Delta x)$

CvT (conv): $Q_{\text{shift}} = \text{Conv}(X_{\text{shift}}) = \text{Conv}(X(x+\Delta x)) = Q(x+\Delta x)$

따라서 depthwise conv QKV 로 translation equivariance 복원. $\square$

### 정리 2.10: CoAtNet 의 Optimal Hybrid Ratio

**명제**: 4-stage 중 CNN 2-stage + Transformer 2-stage 의 분배가 
ImageNet-1k 에서 FLOPs-Accuracy trade-off 상 최적이다.

**증명의 직관**:

CNN 의 장점: 저수준 특징 추출에 효율적 (parameter 당 feature quality ↑)  
Transformer 의 장점: 고수준 의미론 포착에 효율적 (global context ↑)

**Optimal allocation** (FLOPs constraint):

1. **Stage 1-2 (CNN)**:
   - Resolution: High (H/4, H/8)
   - Task: Edge detection, texture → CNN 의 locality bias 최적
   - FLOPs: CNN 이 Transformer 보다 효율적 (O(k²) vs O(n²))

2. **Stage 3-4 (Transformer)**:
   - Resolution: H/8 (down-sampled 후 fixed)
   - Task: Semantic understanding, object detection → Transformer 의 global receptive field 최적
   - FLOPs: Resolution down-sample 되어 O(n²) 이 관리 가능

**수식**:

CNN complexity per pixel: $O(k^2)$ (kernel size $k$)  
Transformer complexity per patch: $O((H/8)^2)$ (fixed)

Early stage (high resolution): CNN 이 더 효율적  
Late stage (low resolution): Transformer 가 acceptable

따라서 2 CNN + 2 Transformer 가 balanced. $\square$

### 정리 2.11: CoAtNet 의 FLOPs Scaling

**명제**: CoAtNet 은 같은 FLOPs 에서 ResNet, EfficientNet, ViT 모두보다 높은 정확도를 달성한다.

**증명의 직관**:

**ResNet 의 한계**:
- Deep architecture (50, 101, 152 layers) 필요
- Global receptive field 를 얻기 위해 많은 layer 필요 (O(depth) ∝ RF)
- Large kernel 사용 → FLOPs 증가

**EfficientNet 의 한계**:
- Scaling laws 에 기반 (width, depth, resolution)
- Compound scaling 으로 효율적이지만, 여전히 CNN 의 제약 (locality → global RF 로의 transition 이 느림)

**ViT 의 한계**:
- Global attention 은 강력하지만, low-level details 추출 비효율 (position embedding 에만 의존)
- Small data regime 에서 데이터 부족 (데이터 증강, distillation 필요)

**CoAtNet 의 우위**:
1. CNN stage: low-level → FLOPs 효율 높음
2. Transformer stage: high-level → data-efficient (resolution 낮아서)
3. 조합: 각 stage 에서 최적 → 같은 FLOPs 에서 높은 정확도

**Empirical** (CoAtNet 논문, ImageNet-1k):
- ResNet-152: 11.6M params, FLOPs 60G, Acc 80.9%
- EfficientNet-B7: 66M params, FLOPs 37G, Acc 84.4%
- ViT-B: 86M params, FLOPs 55G, Acc 80.9%
- **CoAtNet-1**: 42M params, FLOPs 18G, Acc **83.5%** ← 가장 효율적

$\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Overlapping Conv Patch Embedding (CvT)

```python
import torch
import torch.nn as nn

class OverlappingPatchEmbed(nn.Module):
    """
    Overlapping convolution for patch embedding.
    Vanilla ViT: kernel = stride (non-overlapping)
    CvT: kernel > stride (overlapping)
    """
    def __init__(self, in_channels=3, out_channels=64, 
                 kernel_size=7, stride=4):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.proj = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=kernel_size, 
                             stride=stride, 
                             padding=padding)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        """
        x: (B, 3, H, W)
        return: (B, H', W', C) where H'=H/stride, W'=W/stride
        """
        x = self.proj(x)  # (B, C, H', W')
        x = x.permute(0, 2, 3, 1)  # (B, H', W', C)
        x = self.norm(x)
        return x

torch.manual_seed(42)
B, H, W = 2, 224, 224

# Vanilla ViT (non-overlapping)
vanilla_patch = nn.Conv2d(3, 64, kernel_size=16, stride=16)

# CvT (overlapping)
cvt_patch = OverlappingPatchEmbed(3, 64, kernel_size=7, stride=4)

x = torch.randn(B, 3, H, W)

x_vanilla = vanilla_patch(x)  # (B, 64, 14, 14)
x_cvt = cvt_patch(x)  # (B, 56, 56, 64)

print(f"Input shape: {x.shape}")
print(f"Vanilla ViT patch embed: {x_vanilla.shape} ({14*14} patches)")
print(f"CvT overlapping patch embed: {x_cvt.shape} ({56*56} patches)")

print(f"\nOverlapping benefit:")
print(f"  More patches (56x56 vs 14x14) → more spatial detail")
print(f"  Kernel size > stride → receptive field + locality")
```

### 실험 2: Depthwise Conv for QKV (CvT)

```python
import torch
import torch.nn as nn

class DepthwiseConvQKV(nn.Module):
    """
    Depthwise convolution for QKV projection.
    Preserves translation equivariance.
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        
        # Spatial reshape + depthwise conv
        self.qkv_dw = nn.Conv2d(dim, dim * 3, kernel_size=3, 
                               padding=1, groups=dim)
        self.proj = nn.Linear(dim * 3, dim)
    
    def forward(self, x, H, W):
        """
        x: (B, N, C) where N = H*W
        return: Q, K, V (B, N, C)
        """
        B, N, C = x.shape
        
        # Reshape to spatial
        x_spatial = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Depthwise conv (groups=C)
        qkv = self.qkv_dw(x_spatial)  # (B, 3C, H, W)
        
        # Reshape back
        qkv = qkv.permute(0, 2, 3, 1).reshape(B, N, 3*C)  # (B, N, 3C)
        
        # Split Q, K, V
        q, k, v = qkv.split(C, dim=-1)
        
        return q, k, v

torch.manual_seed(42)
B, H, W, C = 2, 56, 56, 64
N = H * W

qkv_layer = DepthwiseConvQKV(C, num_heads=8)
x = torch.randn(B, N, C)

q, k, v = qkv_layer(x, H, W)

print(f"Input shape: {x.shape}")
print(f"Q shape: {q.shape}")
print(f"K shape: {k.shape}")
print(f"V shape: {v.shape}")
print(f"\nDepthwise conv QKV:")
print(f"  Preserves spatial structure")
print(f"  Groups = C (channel-wise)")
print(f"  Translation equivariance maintained")
```

### 실험 3: MBConv Block (CoAtNet)

```python
import torch
import torch.nn as nn

class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck (EfficientNet style).
    Used in CoAtNet early stages.
    """
    def __init__(self, in_channels, out_channels, expand_ratio=6, 
                 kernel_size=3, stride=1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        
        # Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ) if expand_ratio != 1 else nn.Identity()
        
        # Depthwise
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     padding, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.use_residual = stride == 1 and in_channels == out_channels
    
    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        
        if self.use_residual:
            out = out + identity
        
        return out

torch.manual_seed(42)
B, C, H, W = 2, 64, 56, 56

mbconv = MBConv(in_channels=C, out_channels=C, expand_ratio=6, stride=1)
x = torch.randn(B, C, H, W)
out = mbconv(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"MBConv: {C}→{C*6}→{C} (inverted residual)")
print(f"FLOPs efficient due to depthwise convolution")
```

### 실험 4: CoAtNet 4-Stage Hybrid

```python
import torch
import torch.nn as nn

class CoAtNet4Stage(nn.Module):
    """
    Simplified CoAtNet with 4 stages: 2 CNN + 2 Transformer
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Stem: 1 conv layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1: MBConv (CNN)
        self.stage1 = nn.Sequential(
            MBConv(64, 64, expand_ratio=4, kernel_size=3),
            MBConv(64, 64, expand_ratio=4, kernel_size=3),
        )
        
        # Stage 2: MBConv (CNN)
        self.stage2 = nn.Sequential(
            MBConv(64, 128, expand_ratio=4, kernel_size=3, stride=2),
            MBConv(128, 128, expand_ratio=4, kernel_size=3),
        )
        
        # Downsample to H/8 (if needed)
        
        # Stage 3: Transformer blocks
        self.stage3 = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=128, nhead=8, 
                                      dim_feedforward=512, 
                                      batch_first=True),
            nn.TransformerEncoderLayer(d_model=128, nhead=8, 
                                      dim_feedforward=512, 
                                      batch_first=True),
        )
        
        # Stage 4: Transformer blocks
        self.stage4 = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=128, nhead=8, 
                                      dim_feedforward=512, 
                                      batch_first=True),
            nn.TransformerEncoderLayer(d_model=128, nhead=8, 
                                      dim_feedforward=512, 
                                      batch_first=True),
        )
        
        # Classification head
        self.head = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Stem + Stage 1-2 (CNN)
        x = self.stem(x)  # (B, 64, 112, 112)
        x = self.stage1(x)  # (B, 64, 112, 112)
        x = self.stage2(x)  # (B, 128, 56, 56)
        
        # Prepare for transformer (sequence format)
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Stage 3-4 (Transformer)
        x_seq = self.stage3(x_seq)
        x_seq = self.stage4(x_seq)
        
        # Global average pooling
        x_cls = x_seq.mean(dim=1)
        out = self.head(x_cls)
        
        return out

torch.manual_seed(42)
B = 2

model = CoAtNet4Stage(num_classes=1000)
x = torch.randn(B, 3, 224, 224)
out = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"\nCoAtNet Stages:")
print(f"  Stem + Stage 1-2: CNN (MBConv)")
print(f"  Stage 3-4: Transformer")
print(f"  Hybrid design: Low-level CNN + High-level Transformer")
```

## 🔗 실전 활용

**timm 에서의 CoAtNet 구현**:
```python
from timm.models import coatnet_0, coatnet_1
model = coatnet_1(pretrained=True)
```

**CvT 구현**:
```python
from timm.models import cvt_13
model = cvt_13(pretrained=True)
```

**응용**:
- **Lightweight models (mobile)**: CoAtNet-0 (25M params, 2.5G FLOPs)
- **Efficient backbone (detection)**: CoAtNet-1 (42M params, 18G FLOPs)
- **Large-scale models**: CoAtNet-3 (168M params, 230G FLOPs)

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Conv + Attn 의존성** | Hybrid 가 optimal 이라고 보장하지 않음 | Task-specific tuning 필요 |
| **Overlapping patch 의 위치** | CvT 의 overlapping 이 모든 layer 에 적용되지 않음 | 초기 layer 만 적용 |
| **MBConv 의 제약** | Depthwise conv 는 group convolution → accuracy loss | Channel 수가 충분해야 함 |
| **Hybrid gradient flow** | CNN 과 Transformer gradient 의 상호작용 미분석 | Explainability 부족 |
| **Transfer learning** | Pre-trained CNN backbone 의 weight initialization 이 중요 | Fine-tuning 시 주의 필요 |

## 📌 핵심 정리

$$\boxed{\text{CvT}: \text{OverlapConv}(K=7, S=4) + \text{DepthwiseConv}(QKV) \rightarrow \text{Translation Equivariance}}$$

$$\boxed{\text{CoAtNet}: [\text{MBConv}, \text{MBConv}, \text{Transformer}, \text{Transformer}] \rightarrow \text{Optimal Hybrid}}$$

| 개념 | 식 | 역할 |
|------|-----|------|
| **Overlapping patch** | Conv(K=7, stride=4, padding=3) | Locality + position-awareness |
| **Depthwise QKV** | DepthwiseConv(groups=C) | Translation equivariance |
| **MBConv** | PWConv→DWConv→PWConv | Low-level efficient feature extraction |
| **4-stage hybrid** | 2 MBConv + 2 Transformer | Optimal FLOPs-Accuracy trade-off |
| **Global average pooling** | Mean over spatial dim | Classification feature aggregation |

**결론**: CvT 와 CoAtNet 은 CNN 과 Transformer 의 inductive bias 를 결합하여, 
각 stage 에서 최적의 연산 효율성을 달성한다. 
CoAtNet 은 같은 FLOPs 에서 ResNet, EfficientNet, ViT 모두를 능가.

## 🤔 생각해볼 문제

### 문제 1 (기초): Overlapping vs Non-overlapping Patch

CvT 는 overlapping patch embedding (kernel > stride) 를 사용한다.
이것이 position embedding 의 역할과 어떤 관계가 있는가?

<details>
<summary>해설</summary>

**Non-overlapping (Vanilla ViT)**:
- Patch embedding: $z_0 = \text{Conv2d}(k=16, s=16)$
- Patch 간 gap 존재 → position embedding 이 필수 (어느 patch 인지 구분)
- Position embedding: Learnable 또는 sinusoidal 1D sequence

**Overlapping (CvT)**:
- Patch embedding: $\text{Conv2d}(k=7, s=4)$
- Neighboring patches 간 spatial overlap → 상대 위치 정보가 자연스럽게 encoded
- Position embedding: Still needed, but carry less information (redundancy 감소)

**비유**:
- Non-overlapping: 완전히 분리된 조각들 → 순서 정보가 필수 (위치 라벨)
- Overlapping: 겹치는 부분이 있는 조각들 → 순서 정보가 자동으로 있음 (상대 위치 infer 가능)

**결론**: Overlapping patch 는 position embedding 의 필요성을 줄이고, 
translation equivariance 를 자연스럽게 회복한다.

</details>

### 문제 2 (심화): CoAtNet 의 최적 Stage Transition

CoAtNet 의 4-stage 중 stage 2에서 stage 3으로 전환할 때 (MBConv → Transformer), 
이 boundary 에서 gradient flow 가 어떻게 유지되는가?

<details>
<summary>해설</summary>

Stage 2 (MBConv) 의 output: $(B, C, H/8, W/8)$ spatial tensor  
Stage 3 (Transformer) 의 input: $(B, (H/8)*(W/8), C)$ sequence

**Reshape operation**:
$$X_{\text{seq}} = \text{Reshape}(X_{\text{spatial}}), \quad \text{shape: } (B, N, C) \text{ where } N = (H/8)^2$$

**Gradient flow**:

Forward: Spatial → Reshape → Sequence → Transformer  
Backward: Transformer gradient → Reshape^T → Spatial gradient

Key: Reshape 는 **differentiable** & **invertible** → gradient 무손실 통과.

그러나 주의점:
1. **Spatial structure loss**: Transformer 는 sequence 로만 봄 → 2D spatial structure 정보 손실
2. **MBConv 의 inductive bias 손실**: Transformer stage 에서 depthwise conv 의 locality 이점 소실

실제로, 최적 성능을 위해서는:
- Stage 2 의 output channel 을 충분히 크게 (semantic feature 포장)
- Stage 3 의 attention head 를 충분히 크게 (global context 포착)

따라서 **CNN-Transformer 경계** 설계가 매우 중요.

</details>

### 문제 3 (논문 비평): CvT vs CoAtNet — Design Philosophy

CvT 는 "Transformer 에 CNN 의 특징을 주입" (conv patch embed + conv QKV)  
CoAtNet 은 "CNN 과 Transformer 를 stage 별로 분리" (4-stage hybrid)

어느 philosophy 가 더 나을까? 각각의 장단점을 논의하시오.

<details>
<summary>해설</summary>

**CvT Philosophy: "Transformer 에 CNN 주입"**

장점:
1. Transformer 의 global receptive field 유지
2. CNN 의 translation equivariance 추가 (depthwise conv QKV)
3. Unified architecture (모든 layer 가 같은 구조)

단점:
1. Depthwise conv 의 overhead (QKV 계산마다)
2. Overlapping patch 의 computational cost (더 많은 token)
3. Incremental improvement 만 달성 (ViT 대비 2-3%)

**CoAtNet Philosophy: "CNN 과 Transformer 분리"**

장점:
1. 각 stage 에서 최적 operator 사용 (CNN for low-level, Transformer for high-level)
2. 명확한 information hierarchy (semantics 의 점진적 추상화)
3. Larger gap improvement (ResNet/EfficientNet 대비 3-5%)

단점:
1. Architecture 복잡성 증대 (4개 다른 block type)
2. Stage transition 의 설계 자유도 → hyperparameter 튜닝 필요
3. CNN-Transformer 경계에서의 gradient flow 불명확

**선택 기준**:

1. **실시간 성능**: CvT 추천
   - Unified architecture → inference optimization 용이
   - Depthwise conv overhead 작음

2. **최고 정확도**: CoAtNet 추천
   - 각 stage 최적화 가능
   - FLOPs 대비 accuracy 최고

3. **Hardware efficiency**: CoAtNet 추천
   - CNN stage 는 GEMM-heavy (가속기 친화)
   - Transformer stage 는 고도로 최적화 가능 (NVIDIA Tensor Core)

4. **Transfer learning**: CvT 추천
   - Unified architecture 로 pre-trained weight 활용 용이

**결론**: 
2021-2022 시점에는 CoAtNet 이 SOTA 였으나, 
최근(2024+)에는 pure ViT scaling (DINOv2, LLaVA) 이 더 강력함을 보임.
따라서 "최적 hybrid" 는 시간과 context 에 따라 변함.

</details>

---

<div align="center">

[◀ 이전](./03-pvt.md) | [📚 README](../README.md) | [다음 ▶](./05-mvit-focal.md)

</div>
