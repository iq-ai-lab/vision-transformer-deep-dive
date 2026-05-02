# 01. ViT 의 수학적 구조 (Dosovitskiy 2021)

## 🎯 핵심 질문

- Vision Transformer 의 입력에서 출력까지 전체 수식 파이프라인은 무엇인가?
- Patch embedding 은 이미지를 어떻게 tokenize 하고, 왜 이 방식이 Transformer 에 적합한가?
- Pre-LN (Pre-Layer Normalization) transformer block 의 정확한 정의와 차이점은 무엇인가?
- ViT-B/16 의 hyperparameter (depth, heads, hidden dimension) 는 어떻게 선택되는가?
- Classification 을 위해 왜 CLS token 이 필요한가?

## 🔍 왜 이 수학이 Vision Transformer 의 핵심인가

Vision Transformer (ViT) 는 Transformer 를 이미지에 처음 적용한 방법이다. CNN 처럼 convolution 이나 inductive bias 에 의존하지 않고, 순수하게 **self-attention 만으로** 이미지를 처리한다.

Dosovitskiy et al. (2021) 의 핵심 통찰:
1. 이미지를 정사각형 patch 로 쪼개고 (tokenization)
2. 각 patch 를 선형 projection 으로 embedding 한 뒤 (patch embedding)
3. Position embedding 을 더해 순서 정보를 인코딩
4. Transformer encoder stack 을 통과시키고 (self-attention)
5. CLS token 의 최종 hidden state 로 classification

이 **간단한 설계**가 대규모 데이터 (ImageNet-22k, JFT-300M) 에서 ResNet 을 능가하는 이유를 수학적으로 이해하는 것이 vision deep learning 의 가장 기본이다.

## 📐 수학적 선행 조건

- **Linear Algebra**: 행렬 곱셈, reshape, 벡터 정규화, 정사각행렬의 고유값
- **Transformer fundamentals** (별도 repo): Multi-head self-attention, layer normalization, feed-forward network
- **CNN basics**: 2D convolution, receptive field, stride
- **Probability**: 확률 분포, cross-entropy loss, softmax
- **PyTorch basics**: tensor operations, nn.Module, forward pass

## 📖 직관적 이해

```
입력 이미지: H×W×C (예: 224×224×3)
    ↓
16×16 patch 로 쪼개기 (P=16)
    ↓
N = HW/P² = 196개 patch (각 16×16×3 = 768차원)
    ↓
각 patch 를 선형 변환: 768 → D=768 (embedding)
    ↓
[CLS; patch₁; patch₂; ...; patch₁₉₆] (197개 token)
    ↓
Positional encoding 더하기 (절대 위치)
    ↓
Transformer encoder stack (12 layers × 12 heads)
    ↓
CLS token 의 최종 output z^0_L
    ↓
Linear classification head (768 → 1000 classes)
    ↓
ImageNet label prediction
```

**핵심 아이디어**: CNN 의 local receptive field 와 달리, 첫 번째 layer 부터 **모든 patch 가 서로 attention 을 주고받을 수 있다**. 따라서 **global context** 를 빠르게 수집할 수 있다.

## ✏️ 엄밀한 정의

### 정의 1.1: 이미지의 Patch Tokenization

입력 이미지 $x \in \mathbb{R}^{H \times W \times C}$ 를 $(P \times P)$ 크기의 겹치지 않는 patch 로 분할:

$$x_p^i = x\left[ iP : (i+1)P, jP : (j+1)P, : \right] \in \mathbb{R}^{P \times P \times C}, \quad i, j \in \{0, 1, \ldots, N-1\}, \quad N = \frac{HW}{P^2}$$

각 patch 를 1차원 벡터로 flatten:

$$\tilde{x}_p^i = \text{flatten}(x_p^i) \in \mathbb{R}^{P^2 C}$$

### 정의 1.2: Patch Embedding

learnable projection matrix $E \in \mathbb{R}^{P^2 C \times D}$ 를 이용하여:

$$z_0^i = \tilde{x}_p^i E \quad \text{for } i = 1, 2, \ldots, N$$

여기서 $z_0^i \in \mathbb{R}^D$ 는 $i$ 번째 patch 의 embedding (보통 $D = 768$).

### 정의 1.3: CLS Token 과 Positional Encoding

학습 가능한 CLS token $x_{\text{class}} \in \mathbb{R}^D$ 를 도입:

$$z_0^0 = x_{\text{class}}$$

Position embedding $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ 를 더하기 (1-indexed):

$$z_0 = [z_0^0; z_0^1; \ldots; z_0^N] + E_{\text{pos}}$$

결과: $z_0 \in \mathbb{R}^{(N+1) \times D}$

### 정의 1.4: Pre-LN Transformer Block

표준 Transformer block 과 달리, layer normalization 을 **self-attention 과 MLP 이전**에 적용:

$$\tilde{z}_\ell = \text{MultiHeadAttn}(\text{LN}(z_{\ell-1})) + z_{\ell-1}$$

$$z_\ell = \text{MLP}(\text{LN}(\tilde{z}_\ell)) + \tilde{z}_\ell$$

여기서:
- $\text{MultiHeadAttn}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$
- $\text{head}_i = \text{softmax}\left( \frac{Q_i K_i^\top}{\sqrt{d_k}} \right) V_i$
- $\text{MLP}(x) = W_2 \sigma(W_1 x + b_1) + b_2$ (ReLU or GELU)

$h$ = attention head 개수 (보통 12), $d_k = D / h$.

### 정의 1.5: 최종 Classification

마지막 layer 의 CLS token representation $z_L^0 \in \mathbb{R}^D$ 를 classification head 로 통과:

$$\hat{y} = \text{softmax}(z_L^0 W_c + b_c), \quad W_c \in \mathbb{R}^{D \times K}, \, b_c \in \mathbb{R}^K$$

여기서 $K$ 는 클래스 개수 (ImageNet: $K = 1000$).

## 🔬 정리와 증명

### 정리 1.1: ViT 의 전체 Forward Pass 는 Affine + Attention + MLP 의 Composition 이다

ViT forward pass:

$$z_\ell = \text{TBlock}(z_{\ell-1}), \quad \ell = 1, 2, \ldots, L$$

$$\hat{y} = \text{softmax}(z_L^0 W_c)$$

는 다음과 같이 전개된다:

1. Patch embedding: $z_0 = \text{Flatten}(x) E + E_{\text{pos}}$
2. Transformer stack: 각 block 은 (Pre-LN) + (Multi-head Attn) + (residual) + (Pre-LN) + (MLP) + (residual)
3. Classification: final CLS token 으로 linear projection

이는 **순수 Affine transformation, Softmax, Layer Norm, 및 attention mechanism** 의 composition 이며, **inductive bias (convolution, locality) 가 없다**.

**증명 개요**: 각 component 가 미분 가능하고 parameter 를 갖는 neural network module 이므로, 전체 forward pass 는 이들의 composition. 특히, positional encoding 을 제외한 모든 operation 이 patch 의 **순서에 depend 하지 않으므로** (attention 은 permutation-equivariant), spatial structure 는 오직 $E_{\text{pos}}$ 로만 인코딩된다.

$$\square$$

### 정리 1.2: ViT-B/16 의 Parameter 개수

ViT-B/16 (Base, 16×16 patch):
- Image size: $H = W = 224$
- Patch size: $P = 16$ → $N = 196$
- Hidden dimension: $D = 768$
- Heads: $h = 12$ → $d_k = 64$
- Depth: $L = 12$
- MLP hidden: $4D = 3072$

**Parameter count**:

| 모듈 | 개수 |
|------|------|
| Patch embedding $E$ | $P^2 C \times D = 768 \times 768 \approx 0.6$M |
| Position embedding | $(N+1) \times D = 197 \times 768 \approx 0.15$M |
| Transformer block (1개) | Attn: $3D^2 + D^2 \approx 2.4$M, MLP: $2 \times D \times 4D \approx 4.7$M, LN: $2D \approx 1.5$K |
| Total (12 blocks) | $12 \times (2.4 + 4.7)$M $\approx 86$M |
| Classification head | $D \times K = 768 \times 1000 \approx 0.77$M |
| **Total** | $\approx 86$M parameters |

ResNet-50 은 $\approx 26$M, ResNet-152 는 $\approx 60$M 이므로, ViT-B 는 더 큼. 그러나 효율성이 높다 (이후 장).

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Patch Embedding 의 Forward Pass

```python
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

# 하이퍼파라미터
B, H, W, C = 2, 224, 224, 3
P, D = 16, 768

# 더미 이미지
x = torch.randn(B, C, H, W)

# Patch embedding layer
patch_embed = nn.Linear(P * P * C, D)

# 수동 patch tokenization
N = (H // P) * (W // P)
x_p = x.reshape(B, C, H // P, P, W // P, P)
x_p = x_p.permute(0, 2, 4, 1, 3, 5).contiguous()
x_p = x_p.reshape(B, N, P * P * C)  # (B, 196, 768)

z_0 = patch_embed(x_p)  # (B, 196, 768)
print(f"Patch embedding output shape: {z_0.shape}")
assert z_0.shape == (B, N, D), f"Expected {(B, N, D)}, got {z_0.shape}"
print("✓ Patch embedding forward pass verified")
```

### 실험 2: CLS Token 과 Position Encoding 추가

```python
import torch
import torch.nn as nn

B, N, D = 2, 196, 768

# CLS token (learnable)
cls_token = nn.Parameter(torch.randn(1, 1, D))

# Position embedding (learnable)
pos_embed = nn.Parameter(torch.randn(1, N + 1, D))

# Dummy patch embedding
z_p = torch.randn(B, N, D)

# CLS token 추가
cls_expand = cls_token.expand(B, -1, -1)  # (B, 1, D)
z_with_cls = torch.cat([cls_expand, z_p], dim=1)  # (B, 197, D)

# Position embedding 더하기
z_0 = z_with_cls + pos_embed  # (B, 197, D)

print(f"After CLS + pos_embed: {z_0.shape}")
assert z_0.shape == (B, N + 1, D)
print("✓ CLS token and position encoding verified")
```

### 실험 3: Pre-LN Transformer Block 의 재현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreLNTransformerBlock(nn.Module):
    def __init__(self, D, h=12, mlp_ratio=4.0):
        super().__init__()
        self.D = D
        self.h = h
        self.d_k = D // h
        
        # Multi-head attention
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)
        self.W_o = nn.Linear(D, D)
        
        # MLP
        hidden = int(D * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Linear(hidden, D)
        )
        
        # Layer norms (Pre-LN)
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
    
    def forward(self, z):
        # Pre-LN attention
        z_norm = self.ln1(z)
        Q = self.W_q(z_norm).reshape(-1, z.size(1), self.h, self.d_k).transpose(1, 2)
        K = self.W_k(z_norm).reshape(-1, z.size(1), self.h, self.d_k).transpose(1, 2)
        V = self.W_v(z_norm).reshape(-1, z.size(1), self.h, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        
        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(-1, z.size(1), self.D)
        attn_out = self.W_o(attn_out)
        
        # Residual
        z = z + attn_out
        
        # Pre-LN MLP
        z_norm = self.ln2(z)
        mlp_out = self.mlp(z_norm)
        z = z + mlp_out
        
        return z

# Test
B, N, D = 2, 197, 768
block = PreLNTransformerBlock(D, h=12, mlp_ratio=4.0)
z = torch.randn(B, N, D)
z_out = block(z)

print(f"Pre-LN block input shape: {z.shape}")
print(f"Pre-LN block output shape: {z_out.shape}")
assert z_out.shape == z.shape
print("✓ Pre-LN Transformer block verified")
```

### 실험 4: ViT-B 의 전체 Forward Pass (축소 버전)

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, H=224, W=224, C=3, P=16, D=768, L=12, h=12, K=1000):
        super().__init__()
        self.H, self.W, self.C = H, W, C
        self.P = P
        self.N = (H // P) * (W // P)
        self.D = D
        
        # Patch embedding
        self.patch_embed = nn.Linear(P * P * C, D)
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, D))
        self.pos_embed = nn.Parameter(torch.randn(1, self.N + 1, D))
        
        # Transformer blocks (simplified: just 2 for demo)
        self.blocks = nn.ModuleList([
            PreLNTransformerBlock(D, h=h) for _ in range(L)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, K)
    
    def forward(self, x):
        B = x.size(0)
        
        # Patch embedding
        x_p = x.reshape(B, self.C, self.H // self.P, self.P, self.W // self.P, self.P)
        x_p = x_p.permute(0, 2, 4, 1, 3, 5).contiguous()
        x_p = x_p.reshape(B, self.N, self.P * self.P * self.C)
        z = self.patch_embed(x_p)
        
        # CLS token + position embedding
        cls = self.cls_token.expand(B, -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            z = block(z)
        
        # Classification
        z = self.norm(z)
        logits = self.head(z[:, 0])  # CLS token only
        
        return logits

# Test (small image for speed)
vit = VisionTransformer(H=224, W=224, C=3, P=16, D=768, L=2, h=12, K=1000)
x = torch.randn(1, 3, 224, 224)
logits = vit(x)

print(f"Input shape: {x.shape}")
print(f"Logits shape: {logits.shape}")
assert logits.shape == (1, 1000)
print("✓ ViT forward pass complete")
```

## 🔗 실전 활용

**timm (PyTorch Image Models)**:
```python
from timm.models import vit_base_patch16_224
model = vit_base_patch16_224(pretrained=True)
# 자동으로 patch embedding, pos encoding, transformer, head 포함
```

**Hugging Face Transformers**:
```python
from transformers import ViTImageProcessor, ViTForImageClassification
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
```

**DeiT (Data-efficient Image Transformers, Ch2-01)**:
- ViT-B 의 기본 구조를 유지하면서, distillation + augmentation 으로 small-data 성능 향상

**Swin Transformer (Ch2-02)**:
- Hierarchical 구조 + shifted window attention 으로 지역성 복원
- 여전히 동일한 patch embedding 과 transformer block 사용

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Patch independence** | 각 patch 를 독립적으로 embedding 한다 (cross-patch 정보는 transformer 에서 처리) | 이웃 patch 와의 경계 정보 손실 가능; CNN receptive field 와 다름 |
| **Absolute position** | Position embedding 이 절대 위치를 인코딩 | 다른 spatial context 에 transfer 할 때 문제; Swin 의 상대 PE 가 해결책 |
| **Quadratic complexity** | Self-attention 이 $O(N^2)$ 복잡도 | High-res 이미지 ($N > 1000$) 에서 비효율; linear attention 으로 해결 가능 |
| **Small-data regime** | ImageNet-1k 만으로는 ResNet 에 뒤짐 | ImageNet-22k 나 JFT-300M 같은 대규모 데이터 필요; DeiT 의 augmentation 으로 부분 완화 |
| **No locality bias** | Convolution 같은 지역성 편향이 없음 | 초반 layer 에서도 전역 정보 처리; 일부는 장점 (global context), 일부는 단점 (sample efficiency) |

## 📌 핵심 정리

$$\boxed{z_0 = \text{Flatten}(x) E + E_{\text{pos}}, \quad z_\ell = \text{TBlock}(z_{\ell-1}), \quad \hat{y} = \text{softmax}(z_L^0 W_c)}$$

| 개념 | 식 | 역할 |
|------|-----|------|
| **Patch embedding** | $z_0^i = \tilde{x}_p^i E$ | Tokenization; CNN receptive field 와 다르게 global view |
| **CLS token** | $z_0^0 = x_{\text{class}}$ | Image representation; learnable global pooling |
| **Position embedding** | $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ | Spatial information; absolute index (1D learned) |
| **Pre-LN block** | LN → Attn → residual → LN → MLP → residual | Layer normalization 을 먼저 적용; training stability 향상 |
| **Parameter count** | $\approx 86$M (ViT-B) | ResNet-50 (26M) 보다 3배; 효율성은 더 좋음 |

## 🤔 생각해볼 문제

### 문제 1 (기초): ViT-B/16 의 Token 개수 계산

224×224 이미지를 16×16 patch 로 나눌 때, 몇 개의 patch token 이 생기는가?  
CLS token 을 포함하면 총 몇 개인가?

<details>
<summary>해설</summary>

Image size: 224×224  
Patch size: 16×16  
Patches per dimension: $224 / 16 = 14$  
Total patches: $14 \times 14 = 196$  
With CLS token: $196 + 1 = 197$ tokens  

따라서 transformer 에 입력되는 sequence length 는 197.

</details>

### 문제 2 (심화): Patch Embedding 과 Conv2d 의 관계

Patch embedding ($E \in \mathbb{R}^{768 \times 768}$) 은 실제로 $\mathrm{Conv2d}(C=3, D=768, \text{kernel}=16, \text{stride}=16)$ 과 어떤 관계가 있는가?

가중치 reshape 를 통해 두 연산이 bit-exact 로 같음을 보이시오.

<details>
<summary>해설</summary>

Conv2d 의 가중치는 $W_{\text{conv}} \in \mathbb{R}^{D \times C \times P \times P}$.  
출력: $y_{\text{conv}} = \text{Conv2d}(x, W_{\text{conv}})$ → shape: $(B, D, 14, 14)$  

Patch embedding:  
1. Flatten patches: $(B, 196, 768)$
2. Linear: $(B, 196, 768)$

**Equivalence**: $W_{\text{conv}} \in \mathbb{R}^{768 \times 3 \times 16 \times 16}$ 를 $\mathbb{R}^{768 \times 768}$ 로 reshape 하면, conv 출력을 flatten·transpose 한 것과 정확히 일치한다.

이 등가성이 Ch1-02 의 핵심 정리.

</details>

### 문제 3 (논문 비평): ViT vs ResNet 의 학습 곡선

Dosovitskiy 2021 의 Figure 3 에서:
- **ImageNet-1k only**: ResNet > ViT
- **ImageNet-22k pretrain**: ViT >> ResNet

왜 이런 격차가 발생하는가? ViT 의 어떤 성질 때문인가?

<details>
<summary>해설</summary>

**ViT 의 inductive bias 부족** (Ch1-05):
- CNN (ResNet) 은 translation equivariance, locality 같은 vision prior 를 갖는다.
- ViT 는 이런 prior 가 없고, position embedding 으로만 spatial info 를 인코딩한다.

**ImageNet-1k 만으로는**:
- ViT 가 이런 visual structure 를 학습하기에 충분한 데이터가 아니다.
- ResNet 의 CNN bias 가 유리하다.

**ImageNet-22k / JFT-300M 에서**:
- 충분한 데이터가 주어지면, ViT 는 position + scale 에서 더 효율적이다.
- Self-attention 의 global receptive field 가 이점이 된다.

**해결책** (Ch2-01, DeiT):
- Augmentation (Mixup, CutMix, RandAugment)
- Distillation (CNN teacher 에서)
- 이들을 통해 ImageNet-1k 만으로도 ViT-B > ResNet-50 달성

</details>

---

<div align="center">

[◀ 이전](../README.md) | [📚 README](../README.md) | [다음 ▶](./02-patch-conv-equivalence.md)

</div>
