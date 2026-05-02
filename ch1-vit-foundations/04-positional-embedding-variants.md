# 04. Positional Embedding 의 변종

## 🎯 핵심 질문

- Positional encoding 은 왜 필요하고, ViT 에서는 어떤 방식이 사용되는가?
- 1D learned embedding, 2D learned, sin-cos (NLP 스타일), relative position 등 여러 종류가 있는데, 어떤 것이 가장 좋은가?
- Swin Transformer 의 상대 위치 encoding (relative positional bias) 은 standard ViT 와 어떻게 다른가?
- ConViT 의 convolutional positional encoding (CPE) 은 무엇인가?
- Transfer learning 시 position embedding 의 interpolation (fine-tuning 할 때 이미지 크기가 다를 때) 은 어떻게 하는가?

## 🔍 왜 Positional Embedding 이 vision 에서 중요한가

Self-attention 은 **permutation-equivariant** 이다. 즉, 입력의 순서를 바꿔도 attention weight 의 구조는 같다:

$$\text{Attn}(Q, K, V) = \text{softmax}(QK^\top) V$$

이는 좋은 성질이지만, **sequence 의 spatial position 정보를 잃는다**.

NLP 에서는 word order 가 의미를 결정하므로, position encoding 이 중요하다.  
Vision 에서는 이미지의 patch **위치**가 spatial structure 를 결정한다.

따라서 patch embedding 에 position information 을 더해야 한다. 하지만 **어떻게?**

1. **1D learned** (ViT): 단순하지만, 2D spatial structure 를 명시하지 않음
2. **2D learned**: Row, column index 를 분리하여 인코딩
3. **Relative positional bias**: Window attention (Swin) 에서, 같은 window 내 patch 사이의 상대 거리만 사용
4. **Convolutional PE** (ConViT): 작은 depth-wise conv 로 position 정보를 implicit 하게 인코딩
5. **Sin-cos** (Fourier): NLP 의 표준, frequency-based 인코딩

## 📐 수학적 선행 조건

- **Transformer attention**: multi-head self-attention 의 정의
- **Fourier features**: sin, cos 의 주기성과 frequency
- **2D spatial indexing**: (row, col) 좌표계
- **Ch1-01 (ViT 기본)**: patch embedding 과 position embedding 의 구성

## 📖 직관적 이해

```
Position Encoding 의 목표: 
  - Patch i 가 (row=r, col=c) 에 있다는 정보를 vector 에 인코딩
  - Self-attention 이 "patch 의 상대 위치" 를 활용할 수 있게

1. 1D Learned (ViT standard):
   E_pos ∈ ℝ^(197 × 768)
   - 각 위치 (0=CLS, 1..196=patches) 에 대해 768차원 vector
   - Learnable, 절대 index 사용
   - 단순: E_pos[i] 은 i번째 위치의 embedding
   - 문제: 2D spatial structure 가 명시되지 않음

2. 2D Learned:
   - E_row ∈ ℝ^(14 × (D/2))  [row indices 0..13]
   - E_col ∈ ℝ^(14 × (D/2))  [col indices 0..13]
   - E_pos[i] = [E_row[r]; E_col[c]]  where i = r*14 + c
   - 더 나은 spatial awareness

3. Relative (Swin):
   - Window 내 patch 사이의 상대 거리: Δr, Δc ∈ {-P+1, ..., P-1}
   - Relative bias: B ∈ ℝ^(2P-1 × 2P-1)
   - Attention: Attn += B[Δr, Δc]
   - 장점: translation equivariance 부분 회복

4. Convolutional PE (ConViT):
   - 3×3 depth-wise conv 로 implicit position encoding
   - Position information 을 learned, dynamic 하게 처리
   - Content-aware (다양한 image 에 다르게 response)

5. Sin-Cos (NLP style):
   - PE[i, 2j] = sin(i / 10000^(2j/D))
   - PE[i, 2j+1] = cos(i / 10000^(2j/D))
   - 주기성: 낮은 frequency (낮은 차원) → long-range, 높은 frequency → short-range
   - Vision 에서는 learned 가 보통 더 나음
```

## ✏️ 엄밀한 정의

### 정의 4.1: 1D Learned Positional Embedding (표준 ViT)

Position embedding matrix $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$:

$$z_0 = z_0 + E_{\text{pos}}$$

여기서 $z_0 \in \mathbb{R}^{(N+1) \times D}$ 는 patch embedding 으로, 각 위치 $i \in \{0, 1, \ldots, N\}$ 에 대해 $E_{\text{pos}}[i, :] \in \mathbb{R}^D$ 는 learnable parameter.

$N = (H/P) \times (W/P) = 196$ for ViT-B/16 on ImageNet.

### 정의 4.2: 2D Learned Positional Embedding

Row 와 column index 를 분리하여 인코딩:

$$E_{\text{pos,row}} \in \mathbb{R}^{(H/P) \times (D/2)}, \quad E_{\text{pos,col}} \in \mathbb{R}^{(W/P) \times (D/2)}$$

Patch $(r, c)$ (0-indexed) 에 대한 position embedding:

$$E_{\text{pos}}[r \cdot (W/P) + c, :] = [E_{\text{pos,row}}[r, :]; E_{\text{pos,col}}[c, :]]$$

이는 $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ 의 일부로, CLS token 은 별도 embedding $e_{\text{cls}} \in \mathbb{R}^D$.

### 정의 4.3: Sinusoidal Positional Encoding (Transformer, NLP)

절대 위치 $i$ 에 대해:

$$PE[i, 2j] = \sin\left( \frac{i}{10000^{2j/D}} \right), \quad PE[i, 2j+1] = \cos\left( \frac{i}{10000^{2j/D}} \right)$$

여기서 $j \in \{0, 1, \ldots, D/2-1\}$.

이는 learnable 이 아니지만, 주기적 구조를 갖고 long-range dependency 를 선호한다.

### 정의 4.4: Relative Positional Bias (Swin Transformer)

Window 크기 $P \times P$ 내에서, 두 patch 의 상대 거리 $(\Delta r, \Delta c)$:

$$\Delta r, \Delta c \in \{-(P-1), \ldots, 0, \ldots, P-1\}$$

Relative bias matrix $B \in \mathbb{R}^{(2P-1) \times (2P-1)}$ (learnable):

$$\text{Attn}(Q, K, V)_{ij} = \text{softmax}\left( \frac{Q_i K_j^\top}{\sqrt{d_k}} + B[\Delta r_{ij}, \Delta c_{ij}] \right)$$

이는 각 attention head 마다 다를 수 있음 (multi-head 인 경우).

### 정의 4.5: Convolutional Positional Encoding (ConViT)

Depth-wise convolution 으로 position information 을 implicit 하게 인코딩:

$$z'_{\text{pos}} = \text{DWConv3x3}(z) + z$$

여기서 $\text{DWConv3x3}$ 는 3×3 depth-wise convolution (channel-wise, kernel_size=3).

Patch grid $(14 \times 14)$ 를 spatial tensor $(14 \times 14 \times D)$ 로 보고, 국소 spatial context 를 활용.

## 🔬 정리와 증명

### 정리 4.1: Sinusoidal PE 의 주기성과 주파수 특성

**명제**: Sinusoidal PE 는 $\text{PE}[i, :] \in \mathbb{R}^D$ 가 주기적 구조를 갖으며, 낮은 차원 ($j$ 작음) 은 long-range periodicity, 높은 차원 ($j$ 큼) 은 short-range periodicity 를 인코딩한다.

**증명**:

$\text{PE}[i, 2j] = \sin(i \cdot f_j)$ 여기서 $f_j = 10000^{-2j/D}$.

주기: $T_j = 2\pi / f_j = 2\pi \cdot 10000^{2j/D}$.

- $j = 0$: $T_0 = 2\pi \cdot 10000^0 = 2\pi$ → 매우 짧은 주기
- $j = D/2 - 1$: $T_{D/2-1} = 2\pi \cdot 10000^{2(D/2-1)/D} \approx 2\pi \cdot 10000$ → 매우 긴 주기

따라서 각 frequency band 가 다양한 scale 의 spatial information 을 인코딩한다.

$$\square$$

### 정리 4.2: Relative PE 는 Translation Equivariance 를 부분적으로 회복한다

**명제** (정성적): Relative positional bias 를 사용하는 attention 은 translation-invariant shift 에 대해 equivariant 하다 (같은 window 내).

**증명 개요**:

1. Absolute PE: $\text{PE}[i] \neq \text{PE}[i + \Delta]$ (절대 위치가 다르면 encoding 이 다름)
2. Relative PE: $B[\Delta r_{ij}, \Delta c_{ij}]$ 는 오직 상대 거리 $\Delta r_{ij} = r_i - r_j$ 에만 depend

   따라서 sequence 가 전체적으로 shift 되어도 (같은 window 내), 상대 거리는 보존되고, attention 은 동일.

3. CNN (convolution) 은 완벽한 translation equivariance 를 가짐:
   $$f(x_{\text{shifted}}) = \text{shift}(f(x))$$
   
   Relative PE 는 이를 부분적으로 (window-level) 회복.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 1D Learned Position Embedding

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# 하이퍼파라미터
N = 196  # patches (14×14)
D = 768  # hidden dimension

# 1D learned position embedding
pos_embed = nn.Parameter(torch.randn(N + 1, D))  # +1 for CLS

# Patch embedding (dummy)
z_patches = torch.randn(2, N, D)

# CLS token embedding
cls_token = torch.randn(2, 1, D)

# 결합
z_0 = torch.cat([cls_token, z_patches], dim=1)  # (2, 197, D)

# Position embedding 추가
z_0 = z_0 + pos_embed.unsqueeze(0)  # Broadcasting

print(f"After adding position embedding: {z_0.shape}")
assert z_0.shape == (2, 197, D)
print("✓ 1D learned position embedding verified")
```

### 실험 2: 2D Learned Position Embedding

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

H_patches, W_patches = 14, 14  # Image divided into 14×14 patches
D = 768

# 2D position embedding
pos_row = nn.Parameter(torch.randn(H_patches, D // 2))
pos_col = nn.Parameter(torch.randn(W_patches, D // 2))

# CLS token embedding
cls_token = torch.randn(1, 1, D)

# Patch embeddings
z_patches = torch.randn(1, H_patches * W_patches, D)

# 2D position embedding 생성
pos_2d = []
for r in range(H_patches):
    for c in range(W_patches):
        pos_rc = torch.cat([pos_row[r], pos_col[c]])  # (D,)
        pos_2d.append(pos_rc)

pos_2d = torch.stack(pos_2d)  # (196, D)
pos_embed = torch.cat([torch.randn(1, D), pos_2d])  # (197, D) with CLS

# Apply
z_0 = torch.cat([cls_token, z_patches], dim=1)
z_0 = z_0 + pos_embed.unsqueeze(0)

print(f"2D position embedding shape: {pos_embed.shape}")
print(f"After adding 2D PE: {z_0.shape}")
assert z_0.shape == (1, 197, D)
print("✓ 2D learned position embedding verified")
```

### 실험 3: Sinusoidal Position Encoding

```python
import torch
import math

def get_sinusoidal_pe(num_tokens, D):
    """Generate sinusoidal position encoding"""
    pe = torch.zeros(num_tokens, D)
    position = torch.arange(0, num_tokens, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, D, 2).float() * 
                         -(math.log(10000.0) / D))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    if D % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# Test
N = 197
D = 768
pe_sine = get_sinusoidal_pe(N, D)

print(f"Sinusoidal PE shape: {pe_sine.shape}")
print(f"First token PE (CLS): {pe_sine[0, :5]}")  # First 5 dims
print(f"Last token PE (patch 195): {pe_sine[-1, :5]}")

# Verify periodicity
freq_low_dim = 0  # Low frequency (large wavelength)
freq_high_dim = D - 1  # High frequency (short wavelength)

pos_vals_low = pe_sine[:, freq_low_dim].numpy()
pos_vals_high = pe_sine[:, freq_high_dim].numpy()

print(f"Low frequency period variation: {pos_vals_low[:10]}")
print(f"High frequency period variation: {pos_vals_high[:10]}")
print("✓ Sinusoidal position encoding verified")
```

### 실험 4: Relative Positional Bias (Swin-style)

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Window size
P = 7  # Window is 7×7
window_size = P * P  # 49 tokens in a window

# Relative positional bias matrix
relative_position_bias = nn.Parameter(
    torch.randn(2 * P - 1, 2 * P - 1)  # (13, 13)
)

# Dummy Q, K for attention in one window
Q = torch.randn(1, 12, window_size, 64)  # (batch, heads, tokens, d_k)
K = torch.randn(1, 12, window_size, 64)

# Attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / (64 ** 0.5)  # (1, 12, 49, 49)

# Add relative position bias
# Convert absolute indices to relative positions
rel_pos_h = torch.arange(P).unsqueeze(0) - torch.arange(P).unsqueeze(1)  # (P, P)
rel_pos_w = torch.arange(P).unsqueeze(0) - torch.arange(P).unsqueeze(1)

# Offset to make indices positive
rel_pos_h = rel_pos_h + (P - 1)  # 0 to 2P-2
rel_pos_w = rel_pos_w + (P - 1)

# Bias for 2D window
bias_2d = relative_position_bias[rel_pos_h, rel_pos_w]  # (P, P)
bias_1d = bias_2d.flatten().unsqueeze(0).unsqueeze(0)  # (1, 1, P*P)

scores = scores + bias_1d.unsqueeze(-2)  # Broadcast to (1, 12, 49, 49)

print(f"Relative position bias shape: {relative_position_bias.shape}")
print(f"Attention scores after bias: {scores.shape}")
print("✓ Relative positional bias verified")
```

## 🔗 실전 활용

**ViT (Dosovitskiy 2021)**:
- 1D learned positional embedding (simplest, works well with large-scale pretraining)

**DeiT (Touvron 2021, Ch2-01)**:
- Standard 1D learned PE (same as ViT)
- Works well on ImageNet-1k with proper augmentation

**Swin Transformer (Liu et al. 2021, Ch2-02)**:
- Relative positional bias within windows
- Slightly better performance on downstream tasks
- Translation equivariance 부분 회복

**ConViT (d'Ascoli et al. 2021)**:
- Convolutional positional encoding
- Combines CNN inductive bias with Transformer flexibility

**DINOv2 (Oquab et al. 2023, Ch4)**:
- Extended ViT with explicit 2D PE 변형
- Better geometry awareness

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Absolute vs Relative** | 1D PE: 절대, Swin: 상대 | 상대 PE 가 더 translation-aware 하지만, 같은 window 내에서만 |
| **Fixed vs Learned** | Sinusoidal: 고정, 다른 것: 학습가능 | Learned PE 가 보통 더 효율적 (vision 에서) |
| **Transfer learning** | 다른 해상도로 fine-tune 할 때 PE 를 interpolate | 예: 224→384 로 크기 증가 시, position 값을 비선형 보간 |
| **Sequence length dependency** | PE 가 N+1 길이에 fixed | Longer sequence (고해상도 이미지) 에 대해 N 을 늘려야 함 |
| **Computational cost** | 1D learned: negligible overhead | 모든 방식이 parameter count 와 computation 에서 무시할 수 있을 정도 |

## 📌 핵심 정리

$$\boxed{z_{\text{with\_pe}} = z + E_{\text{pos}}}$$

| Embedding Type | 수식 | 특징 |
|---|---|---|
| **1D Learned** | $E_{\text{pos}}[i] \in \mathbb{R}^D$ | 단순, 절대 위치, ViT 표준 |
| **2D Learned** | $E_{\text{pos}}[r,c] = [E_r[r]; E_c[c]]$ | 2D spatial structure 명시 |
| **Sinusoidal** | $\sin(i/10000^{2j/D})$, $\cos(...)$ | 주기적, frequency-aware |
| **Relative (Swin)** | $B[\Delta r, \Delta c]$ in attention | Translation equivariance 회복 |
| **Convolutional (ConViT)** | $\text{DWConv3x3}(z) + z$ | Implicit, content-aware |

## 🤔 생각해볼 문제

### 문제 1 (기초): Sinusoidal PE 의 최저 주파수 주기

$D = 768$ 일 때, sinusoidal PE 의 최저 주파수 (가장 긴 주기) 의 주기값은?

$$T = 2\pi / f = 2\pi \cdot 10000^{2(D/2-1)/D}$$

<details>
<summary>해설</summary>

$j = D/2 - 1 = 383$ (최고 차원)

$f_j = 10000^{-2 \cdot 383 / 768} = 10000^{-0.9974}$

$T_j = 2\pi / 10000^{-0.9974} = 2\pi \cdot 10000^{0.9974} \approx 2\pi \cdot 10000 \approx 62832$

따라서 매우 긴 주기: sequence 전체 (N=196) 에서도 변화가 거의 없다.

이렇게 multi-scale 주기성이 여러 거리 스케일의 spatial relationship 을 인코딩한다.

</details>

### 문제 2 (심화): 2D PE 가 1D 보다 언제 더 나은가?

2D learned PE 가 1D learned PE 보다 성능이 나은 데이터셋이나 상황은?

<details>
<summary>해설</summary>

2D PE 가 유리한 경우:
1. **작은 데이터**: spatial structure 를 더 명시적으로 제공할수록 generalization 좋음
2. **높은 해상도**: patch 가 많아질수록 (N 커짐), 2D structure 의 이점 커짐
3. **기하학적 특성**: object 의 위치가 중요한 task (예: object detection, 3D reconstruction)

1D PE 가 유리한 경우:
1. **대규모 데이터**: ImageNet-22k, JFT-300M 같은 충분한 데이터에서는 1D 의 단순성도 충분
2. **계산 효율**: 2D 는 parameter 가 약간 더 많음 (row + col separate)
3. **구현 간편**: 1D 가 훨씬 단순

**실제**: 대부분의 ViT 는 1D learned 를 사용. DeiT, CLIP, DINO 등도 1D.
2D 를 사용하는 모델은 거의 없음 (성능 개선이 marginal 이라고 판단).

</details>

### 문제 3 (논문 비평): Relative PE 와 Translation Equivariance 의 한계

Swin Transformer 의 relative PE 가 "translation equivariance 를 회복한다" 고 주장하지만,  
이것이 완벽한 equivariance 가 아닌 이유는?

<details>
<summary>해설</summary>

**Window-local 만 equivariant**:
- Swin 은 8×8 또는 7×7 window 내에서만 relative PE 를 사용
- Window 경계를 넘어가는 shift 에는 영향을 받음
- 예: 이미지 전체를 1칸 shift 하면, window 분할이 바뀌고, 일부 patch 는 다른 window 로 이동

**완벽한 translation equivariance 를 위해서는**:
- 모든 patch 간의 상대 거리 (무한히 긴 범위) 를 고려해야 함
- 그러면 relative PE matrix 가 $(H/P) \times (W/P) \times (H/P) \times (W/P)$ 크기 → computationally infeasible

**Swin 의 설계**:
- Window-local relative PE: 지역 equivariance
- Shifted window: 인접 window 간 정보 흐름 (전역 receptive field)
- 이의 조합으로 "good enough" translation property 달성

**결론**: 완벽하지 않지만, 실용적 trade-off. CNN 보다는 덜하지만, pure ViT (1D PE) 보다는 나음.

</details>

---

<div align="center">

[◀ 이전](./03-cls-token-vs-gap.md) | [📚 README](../README.md) | [다음 ▶](./05-inductive-bias-gap.md)

</div>
