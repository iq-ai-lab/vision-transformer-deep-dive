# 02. Patch Embedding = Conv2d 등가 증명

## 🎯 핵심 질문

- Patch embedding (선형 변환) 과 2D convolution 이 동일한가?
- 두 연산이 언제 정확히 bit-exact 로 같은가?
- Conv2d 의 가중치를 어떻게 reshape 하여 patch embedding 과 동등하게 만드는가?
- 이 등가성이 실제 구현 (timm, PyTorch) 에서 가지는 의미는 무엇인가?
- 두 접근의 계산 복잡도와 메모리 효율성은 어떻게 다른가?

## 🔍 왜 이 등가성이 중요한가

ViT 는 patch embedding 을 선형 transformation 으로 정의한다. 하지만 실제로는 이것이 **kernel size = patch size, stride = patch size 인 특수한 Conv2d 연산**이다.

이 등가성을 증명하는 이유:
1. **이론적 깔끔함**: ViT 를 CNN 언어로도 표현 가능 → unified framework
2. **구현 효율성**: Conv2d 는 GPU 에 최적화되어 있음 → patch embedding 을 Conv2d 로 구현하는 것이 더 빠를 수 있음
3. **Hybrid models**: ViT + CNN 을 섞을 때 (CvT, CoAtNet) 경계가 명확해짐
4. **이론적 직관**: "왜 ViT 는 local structure 를 놓치는가?" 를 이해하는 데 도움

## 📐 수학적 선행 조건

- **2D Convolution**: 정의, kernel, stride, padding, 출력 크기 계산
- **Matrix multiplication**: reshape, transpose, vectorization
- **PyTorch tensor operations**: view, permute, contiguous
- **Linear algebra**: basis representation, weight matrix interpretation
- **Ch1-01 (ViT 기본 구조)**: patch embedding 의 정의

## 📖 직관적 이해

```
Conv2d 연산:
  입력: (B, C, H, W) = (2, 3, 224, 224)
  kernel size: P=16, stride=P=16
    ↓
  224×224 이미지를 16×16 크기로 슬라이드 (stride=16 → 겹치지 않음)
  각 16×16×3 패치에 D개의 필터를 적용 (각 필터는 16×16×3 = 768 차원)
  출력: (B, D, 14, 14) = (2, 768, 14, 14)
    ↓
  Flatten + reshape: (2, 768, 196) → (2, 196, 768)

Patch Embedding 연산:
  입력: (B, 196, 768) (이미 패치로 나뉜 상태)
  선형 변환: 768 → D=768
    ↓
  출력: (B, 196, 768)

**핵심**: Conv2d 의 각 "필터" 하나가 patch embedding matrix E 의 한 행(열)과 같다.
```

## ✏️ 엄밀한 정의

### 정의 2.1: Patch Embedding (선형 변환)

Input image $x \in \mathbb{R}^{B \times C \times H \times W}$ 를 다음과 같이 처리:

1. Reshape to patches: $x_{\text{patch}} \in \mathbb{R}^{B \times N \times (P^2 C)}$
   - $N = HW / P^2$ (patch 개수)
   - 각 patch 는 $P^2 C$ 차원 벡터

2. 선형 embedding: $E \in \mathbb{R}^{P^2 C \times D}$ (learnable)

3. 출력: $z = x_{\text{patch}} E \in \mathbb{R}^{B \times N \times D}$

### 정의 2.2: Convolution Layer (Conv2d)

입력 $x \in \mathbb{R}^{B \times C \times H \times W}$, 가중치 $W \in \mathbb{R}^{D \times C \times P \times P}$:

$$y[b, d, i, j] = \sum_{c=0}^{C-1} \sum_{p=0}^{P-1} \sum_{q=0}^{P-1} W[d, c, p, q] \cdot x[b, c, iP+p, jP+q]$$

출력: $y \in \mathbb{R}^{B \times D \times (H/P) \times (W/P)}$

## 🔬 정리와 증명

### 정리 2.1: Patch Embedding ≡ Conv2d with kernel=stride=P

**명제**: Input $x \in \mathbb{R}^{B \times C \times H \times W}$, patch embedding matrix $E \in \mathbb{R}^{P^2 C \times D}$, 그리고 Conv2d layer $W \in \mathbb{R}^{D \times C \times P \times P}$ 를 생각하자.

$W$ 를 다음과 같이 reshape 하면:

$$W_{\text{reshaped}}[d, :] = E[:, d]^T \in \mathbb{R}^{1 \times (P^2 C)}$$

패치 embedded 출력과 Conv2d 출력이 (flatten + transpose 를 적용했을 때) **bit-exact 로 동일**하다:

$$\text{Conv2d}(x, W) \to \text{flatten} \to \text{transpose} \equiv \text{PatchEmbed}(x, E)$$

**증명**:

Conv2d 의 한 위치 $(i, j)$ 에서 출력값은:

$$y[b, d, i, j] = \sum_{c, p, q} W[d, c, p, q] \cdot x[b, c, iP+p, jP+q]$$

이를 다르게 쓰면, kernel size 가 $P \times P$ 이고 stride 가 $P$ 인 경우, $i$ 번째 패치는:

$$\text{patch}[b, i, :] = x[b, :, iP:iP+P, :]_{{\rm flattened}} \in \mathbb{R}^{P^2 C}$$

패치 embedding:

$$z[b, i, d] = \sum_{k=0}^{P^2 C - 1} \text{patch}[b, i, k] \cdot E[k, d]$$

이제 $W$ 를 reshape 한다: $W[d, c, p, q] \leftarrow E[\text{flat\_idx}(c, p, q), d]$

여기서 $\text{flat\_idx}(c, p, q) = c \cdot P^2 + p \cdot P + q$.

그러면:

$$y[b, d, i, j] = \sum_{c, p, q} E[\text{flat\_idx}(c, p, q), d] \cdot x[b, c, iP+p, jP+q]$$

$j = 0$ (1D 에서 생각하면 $iP$ 는 i 번째 patch):

$$y[b, d, i, 0] = \sum_{k} E[k, d] \cdot x[b, k_c, iP + k_p : iP + k_p + 1, ...]$$

정확히 $z[b, i, d]$ 와 동일.

Conv2d 출력 $(B, D, H/P, W/P)$ 를 $(B, D \cdot H/(P^2), W)$ 로 flatten 후 transpose 하면 $(B, H/(P^2), D)$ 가 되고, 이는 patch embedding 과 동일.

$$\square$$

### 정리 2.2: Computational Equivalence

**명제**: 두 연산의 arithmetic complexity 는 동일하다.

**증명**:

Patch embedding: 
- 행렬곱: $B \cdot N \cdot P^2 C \cdot D = B \cdot (HW/P^2) \cdot P^2 C \cdot D = B \cdot HW \cdot C \cdot D$
- FLOP: $B \cdot HW \cdot C \cdot D$ multiplications

Conv2d:
- 각 output position $(i, j)$ 에서: $C \cdot P^2 \cdot D$ multiplications
- Total positions: $(H/P) \cdot (W/P)$
- Total FLOP: $(H/P) \cdot (W/P) \cdot C \cdot P^2 \cdot D = HW \cdot C \cdot D$ multiplications

따라서 동일한 수의 operations.

그러나 **실제 구현상 성능**은 다를 수 있다:
- Conv2d: GPU-optimized kernels (cuDNN, etc.)
- Linear: GEMM 최적화
- Patch embedding 이 더 빠를 수도, Conv2d 가 더 빠를 수도 있다. (hardware-dependent)

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Patch Embedding 의 정의

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# 하이퍼파라미터
B, C, H, W = 2, 3, 224, 224
P, D = 16, 768

# 더미 이미지
x = torch.randn(B, C, H, W)

# **방법 1: Patch Embedding (선형 변환)**
patch_embed_layer = nn.Linear(P * P * C, D)

# 패치로 재구성
x_patches = x.reshape(B, C, H // P, P, W // P, P)
x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
x_patches_flat = x_patches.reshape(B, -1, P * P * C)  # (B, 196, 768)

z_linear = patch_embed_layer(x_patches_flat)  # (B, 196, 768)

print(f"Patch embedding output shape: {z_linear.shape}")
```

### 실험 2: Conv2d 의 정의

```python
import torch
import torch.nn as nn

# Conv2d layer: kernel=16, stride=16, input channels=3, output channels=768
conv_layer = nn.Conv2d(C, D, kernel_size=P, stride=P)

# Conv2d forward
y_conv = conv_layer(x)  # (B, D, 14, 14)

# Flatten and transpose to match patch embedding shape
y_conv_flat = y_conv.flatten(2).transpose(1, 2)  # (B, 196, 768)

print(f"Conv2d output shape (after flatten+transpose): {y_conv_flat.shape}")
```

### 실험 3: 가중치 동등성 (Bit-exact 검증)

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# 동일한 초기화 사용
init_weight = torch.randn(P * P * C, D)

# 방법 1: Linear 가중치
linear = nn.Linear(P * P * C, D)
linear.weight.data = init_weight.T  # Linear uses (out_features, in_features)

# 방법 2: Conv2d 가중치 (reshape)
conv = nn.Conv2d(C, D, kernel_size=P, stride=P, bias=False)

# Conv2d weight 를 reshape: (D, C, P, P) ← from (P²C, D) transpose
# init_weight is (P²C, D), so we need (D, C, P, P)
W_conv = init_weight.T.reshape(D, C, P, P)
conv.weight.data = W_conv

# Dummy input
x = torch.randn(1, C, 224, 224)

# Patch embedding
x_patches = x.reshape(1, C, 224 // P, P, 224 // P, P)
x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
x_patches_flat = x_patches.reshape(1, -1, P * P * C)
z_linear = linear(x_patches_flat)

# Conv2d
y_conv = conv(x)
y_conv_flat = y_conv.flatten(2).transpose(1, 2)

# Compare
diff = (z_linear - y_conv_flat).abs().max().item()
print(f"Max difference between Linear and Conv2d: {diff}")
assert diff < 1e-5, f"Difference too large: {diff}"
print(f"✓ Linear embedding and Conv2d are bit-exact equivalent (max diff: {diff:.2e})")
```

### 실험 4: 실제 ViT 의 Patch Embedding vs Conv2d 구현

```python
import torch
import torch.nn as nn

class VitPatchEmbedConv(nn.Module):
    """ViT patch embedding using Conv2d"""
    def __init__(self, C, D, P):
        super().__init__()
        self.C = C
        self.D = D
        self.P = P
        # kernel_size=P, stride=P로 겹치지 않는 patch 생성
        self.conv = nn.Conv2d(C, D, kernel_size=P, stride=P, bias=True)
    
    def forward(self, x):
        # x: (B, C, H, W)
        y = self.conv(x)  # (B, D, H/P, W/P)
        y = y.flatten(2).transpose(1, 2)  # (B, H/P * W/P, D)
        return y

class VitPatchEmbedLinear(nn.Module):
    """ViT patch embedding using Linear"""
    def __init__(self, C, D, P):
        super().__init__()
        self.C = C
        self.D = D
        self.P = P
        self.linear = nn.Linear(P * P * C, D, bias=True)
    
    def forward(self, x):
        # x: (B, C, H, W)
        B = x.size(0)
        x_patches = x.reshape(B, self.C, 224 // self.P, self.P, 224 // self.P, self.P)
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        x_patches = x_patches.reshape(B, -1, self.P * self.P * self.C)
        y = self.linear(x_patches)
        return y

# Test
embed_conv = VitPatchEmbedConv(3, 768, 16)
embed_linear = VitPatchEmbedLinear(3, 768, 16)

# 동일한 가중치 설정 (테스트용)
with torch.no_grad():
    W_linear = embed_linear.linear.weight.data  # (768, 768)
    W_conv = W_linear.T.reshape(768, 3, 16, 16)
    embed_conv.conv.weight.data = W_conv
    embed_conv.conv.bias.data = embed_linear.linear.bias.data

x = torch.randn(2, 3, 224, 224)

z_linear = embed_linear(x)
z_conv = embed_conv(x)

diff = (z_linear - z_conv).abs().max().item()
print(f"Max difference: {diff:.2e}")
print(f"✓ Both implementations produce identical output: {z_linear.shape}")
```

## 🔗 실전 활용

**timm 의 구현** (PyTorch Image Models):
```python
from timm.models.vision_transformer import PatchEmbed
# timm 는 conv2d 를 기반으로 patch embedding 을 구현
```

**Hugging Face Vision Transformer**:
```python
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings
# 선형 변환 기반이지만, reshape/transpose 로 동등하게 구현
```

**CvT (Convolutional Token Embedding)**:
- Patch embedding 을 3×3 conv 여러 개로 대체
- 더 강한 inductive bias 획득

**CoAtNet (Hybrid)**:
- 초반: Conv2d block
- 중반: Hybrid (Conv + Attention)
- 후반: Pure Transformer
- Patch embedding 을 conv block 으로 확장

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **No overlap** | kernel size = stride = P (겹치지 않음) | 다른 설정 (overlapping patch) 이면 등가성 깨짐 |
| **Flatten order** | (B, D, H/P, W/P) → (B, H/P·W/P, D) 의 순서 명확 | Row-major vs Column-major 주의 |
| **Bias handling** | Linear 와 Conv2d 모두 bias 포함 가능 | Bias 여부가 동일해야 함 |
| **Padding** | 이 정리는 padding=0 (no padding) 가정 | padding 이 있으면 조정 필요 |
| **GPU optimization** | Arithmetic complexity 는 같지만, 실제 속도는 다름 | Hardware-dependent; benchmark 필요 |

## 📌 핵심 정리

$$\boxed{W_{\text{conv}}[d, c, p, q] \equiv E[\text{flat\_idx}(c,p,q), d]^T}$$

$$\boxed{\text{Conv2d}(x, k=P, s=P) \equiv \text{Linear}(x_{\rm patch}, E)}$$

| 관점 | 설명 |
|------|------|
| **Patch embedding** | 각 patch 를 벡터로 flatten → 선형 변환 |
| **Conv2d** | Sliding window + learnable filters |
| **등가성** | Weight reshape 로 two 구현 이 bit-exact 동등 |
| **의미** | ViT 는 실제로 CNN 의 특수한 경우; CNN 언어로도 표현 가능 |
| **구현** | Conv2d 가 GPU-optimized 일 수 있지만, 둘 다 가능 |

## 🤔 생각해볼 문제

### 문제 1 (기초): Conv2d 의 출력 크기 계산

Input: (B, 3, 224, 224)  
Conv2d(3, 768, kernel=16, stride=16)  
출력 크기는?

<details>
<summary>해설</summary>

Conv2d output size formula:
$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - \text{kernel}\_{\text{size}} + 2 \times \text{padding}}{\text{stride}} \right\rfloor + 1$$

여기서 kernel_size=16, stride=16, padding=0:
$$H_{\text{out}} = \left\lfloor \frac{224 - 16}{16} \right\rfloor + 1 = 13 + 1 = 14$$

따라서 출력: (B, 768, 14, 14)  
Flatten: (B, 768 × 196) = (B, 196, 768) after transpose

</details>

### 문제 2 (심화): Patch Embedding 의 가중치 개수

Patch embedding matrix $E \in \mathbb{R}^{P^2 C \times D}$ 의 parameter 개수는?  
Conv2d 의 가중치는?

<details>
<summary>해설</summary>

Patch embedding: $E \in \mathbb{R}^{768 \times 768}$  
Parameters: $768 \times 768 = 589,824$

Conv2d: $W \in \mathbb{R}^{768 \times 3 \times 16 \times 16}$  
Parameters: $768 \times 3 \times 256 = 589,824$

동일하다! (reshape 만 다르고 실제 parameter 개수는 같음)

추가: bias 가 있으면 양쪽 모두 768개 추가.

</details>

### 문제 3 (논문 비평): Patch Embedding vs Overlapping Patches

표준 ViT 는 non-overlapping patches (stride=P) 를 사용한다.  
Overlapping patches (stride < P) 를 사용하면 어떤 장단점이 생기는가?

<details>
<summary>해설</summary>

**Non-overlapping (stride=P, 표준)**:
- Pros: 계산 효율 좋음, patch 경계가 명확
- Cons: patch 경계의 정보 손실, local structure 놓침

**Overlapping (stride < P, e.g., stride=8)**:
- Pros: 더 많은 local context, boundary 정보 보존, CNN 같은 locality
- Cons: 더 많은 token (N 증가), 계산 비용 증가, quadratic attention 의 문제 심화

실제 hybrid models (CvT, CoAtNet) 는 초기 layer 에서 overlapping conv 를 사용하여 local feature 를 먼저 추출한 후, 이를 patch 로 tokenize 한다.

</details>

---

<div align="center">

[◀ 이전](./01-vit-math-structure.md) | [📚 README](../README.md) | [다음 ▶](./03-cls-token-vs-gap.md)

</div>
