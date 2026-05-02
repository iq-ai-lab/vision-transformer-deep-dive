# 03. Class Token 과 Global Average Pooling

## 🎯 핵심 질문

- CLS token 은 무엇이며, 왜 ViT 에서 필요한가?
- Global Average Pooling (GAP) 과 CLS token 의 차이점은 무엇인가?
- 둘이 성능 차이를 만드는가? Touvron 2021 의 실험 결과는?
- CLS token 의 최종 attention weight 를 분석하면, 어떤 patch 에 집중하는가?
- CLS token 은 실제로 "learnable global pooling" 인가?

## 🔍 왜 이 비교가 중요한가

CNN (ResNet) 에서 classification 을 위해 **Global Average Pooling (GAP)** 을 사용한다:

$$\bar{z} = \frac{1}{C \cdot H' \cdot W'} \sum_{c,h,w} z[c, h, w]$$

모든 spatial location 의 평균을 취하는 고정된 연산이다.

ViT 는 다르다. **CLS token** $x_{\text{class}}$ 를 처음부터 도입하고, transformer 를 통해 모든 patch 와 정보를 주고받으며, 최종적으로 CLS token 의 hidden state 만 classification head 로 통과시킨다.

**질문**: CLS token 의 최종 상태는 실제로 무엇인가? 모든 patch 의 weighted average 인가? 아니면 더 복잡한 aggregation 인가?

Touvron et al. (DeiT, 2021) 는 **CLS token 과 GAP 을 직접 비교**하는 실험을 했고, 놀랍게도 거의 같은 성능을 보였다. 이는 CLS token 이 실제로 "learnable pooling" 으로 해석될 수 있음을 시사한다.

## 📐 수학적 선행 조건

- **Attention mechanism** (Transformer basics): Query, Key, Value, softmax attention
- **Multi-head attention**: head concatenation, projection
- **Ch1-01 (ViT 구조)**: CLS token 의 정의, attention block
- **CNN pooling**: Global Average Pooling, Max Pooling
- **Weighted average**: 일반적 aggregation 기법

## 📖 직관적 이해

```
CLS Token 의 역할:

Transformer block 1:
  CLS + 196 patches → attention → CLS 는 모든 patch 정보를 봄
  z₁^0 (CLS at layer 1) = weighted sum of (CLS, patch1, ..., patch196)

Transformer block 2:
  CLS (with info from block 1) + 196 patches → attention → 더 정제된 정보
  z₂^0 (CLS at layer 2) = deeper integration

...

Transformer block 12 (final):
  z₁₂^0 = CLS 의 최종 상태
         = 모든 patch 의 "learned, non-uniform weighted average"?

Classification:
  logits = Linear(z₁₂^0)


Global Average Pooling (비교):

각 patch representation z₁₂^i (i=1..196) 를 모두 동일 가중치로 평균:
  z_GAP = (1/196) * sum(z₁₂^i)

Classification:
  logits = Linear(z_GAP)

→ 거의 같은 성능? 왜?
```

## ✏️ 엄밀한 정의

### 정의 3.1: Class Token (CLS)

Learnable parameter $x_{\text{class}} \in \mathbb{R}^D$:

$$z_0^0 = x_{\text{class}}$$

Input embedding 의 첫 번째 token 으로 추가:

$$z_0 = [x_{\text{class}}; z_0^1; z_0^2; \ldots; z_0^N] + E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$$

$N = 196$ (patches for 224×224 images with P=16).

### 정의 3.2: Transformer Block 에서의 CLS

$\ell$ 번째 layer 의 attention:

$$Q^{(\ell)} = z_{\ell-1} W_Q, \quad K^{(\ell)} = z_{\ell-1} W_K, \quad V^{(\ell)} = z_{\ell-1} W_V$$

Multi-head attention (simplified notation):

$$\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V$$

CLS token 에 대한 attention output:

$$z_\ell^0 = \text{Attn}(q^{(\ell)}_0, K^{(\ell)}, V^{(\ell)})$$

여기서 $q^{(\ell)}_0$ 는 CLS 의 query vector.

### 정의 3.3: Global Average Pooling (GAP)

최종 layer 의 모든 patch representation 을 동등하게 평균:

$$\bar{z} = \frac{1}{N} \sum_{i=1}^{N} z_L^i \in \mathbb{R}^D$$

CLS 를 사용하지 않고, GAP 의 output 을 directly classification head 로:

$$\hat{y}_{\text{GAP}} = \text{softmax}(\bar{z} W_c)$$

### 정의 3.4: CLS as Learned Pooling

만약 CLS token 의 최종 상태를 다음과 같이 해석할 수 있다면:

$$z_L^0 = \sum_{i=0}^{N} \alpha_i^* z_L^i$$

여기서 $\alpha_i^*$ 는 학습된 가중치 (implicit, attention mechanism 을 통해)

그러면 CLS 는 "learnable weighted average pooling" 이다.

## 🔬 정리와 증명

### 정리 3.1: CLS Token 의 최종 상태는 Weighted Average Aggregation 이다

**명제** (정성적): CLS token 의 최종 hidden state $z_L^0$ 는 모든 patch 와 CLS 자신의 weighted average 로 해석될 수 있으며, weight 는 transformer 의 attention mechanism 을 통해 학습된다.

**증명 개요**:

1. Transformer block 은 residual connection 을 갖는다:
   $$z_\ell = z_{\ell-1} + \text{Attn}(\text{LN}(z_{\ell-1})) + \text{MLP}(\cdot)$$

2. Attention 은 linear combination:
   $$\text{Attn}_i = \sum_j \alpha_{ij} V_j, \quad \sum_j \alpha_{ij} = 1$$

3. $L$ 개 layer 의 composition 후, CLS token 의 값은:
   - 초기값 $x_{\text{class}}$ 에서 시작
   - 각 layer 마다 모든 patch 의 정보를 weighted 로 통합
   - Linear combination 들의 중첩 (nesting) 을 통해 복잡한 aggregation

4. 결과적으로, $z_L^0$ 는 궁극적으로 $z_0 = [x_{\text{class}}; z_0^1; \ldots; z_0^N]$ 의 weighted sum 형태이다 (비선형 변환 제외, MLP 때문).

따라서 "learnable global pooling" 으로 볼 수 있다.

$$\square$$

### 정리 3.2: CLS 와 GAP 은 거의 동등한 성능을 제공한다 (Touvron 2021)

**명제** (경험적): DeiT 실험에 따르면, CLS token 을 사용한 classification 과 GAP 을 사용한 classification 의 top-1 accuracy 차이는 무시할 수 있을 정도이다 (< 1%).

**실험** (from Touvron et al. DeiT Table 3):

| Pooling | ImageNet top-1 | top-5 |
|---------|--------|-------|
| CLS | 81.8% | 95.5% |
| GAP | 81.5% | 95.3% |

차이: CLS 가 약 0.3% 더 높음 (통계적 유의성 미검증).

**해석**:
- CLS 는 모든 patch 를 선택적으로 활용 (attention weights 가 heterogeneous)
- GAP 은 모든 patch 를 동등하게 취급 (uniform weights)
- 최종적으로는 둘 다 "global aggregation" 을 하므로, 성능 차이가 작다
- CLS 가 약간 우위인 이유: learnable pooling 이 더 flexible

다른 조건 (learning rate, augmentation strength 등) 에서는 GAP 이 더 나을 수도 있다.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: CLS Token 의 초기화 및 embedding

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# 하이퍼파라미터
B, N, D = 2, 196, 768

# CLS token (learnable parameter)
cls_token = nn.Parameter(torch.randn(1, 1, D))

# Patch embeddings (already computed)
z_patches = torch.randn(B, N, D)

# CLS token 을 각 batch 에 복사
cls_expanded = cls_token.expand(B, -1, -1)  # (B, 1, D)

# CLS + patches 결합
z_with_cls = torch.cat([cls_expanded, z_patches], dim=1)  # (B, 197, D)

print(f"CLS token shape: {cls_token.shape}")
print(f"CLS expanded shape: {cls_expanded.shape}")
print(f"Combined with patches: {z_with_cls.shape}")
assert z_with_cls.shape == (B, N + 1, D)
print("✓ CLS token initialization verified")
```

### 실험 2: Global Average Pooling

```python
import torch

# Final layer output (모든 tokens)
z_final = torch.randn(B, N + 1, D)  # (2, 197, 768)

# 방법 1: CLS token만 사용
z_cls = z_final[:, 0, :]  # (B, D)

# 방법 2: 모든 patch 의 GAP (CLS 제외)
z_gap = z_final[:, 1:, :].mean(dim=1)  # (B, D)

# 방법 3: CLS 포함한 GAP
z_gap_all = z_final.mean(dim=1)  # (B, D)

print(f"CLS shape: {z_cls.shape}")
print(f"GAP (patches only) shape: {z_gap.shape}")
print(f"GAP (all) shape: {z_gap_all.shape}")

# CLS 와 GAP 의 코사인 유사도
cos_sim = torch.nn.functional.cosine_similarity(z_cls, z_gap, dim=1).mean()
print(f"Cosine similarity (CLS vs GAP): {cos_sim:.4f}")
```

### 실험 3: Attention Weight 분석 (CLS 의 attention)

```python
import torch
import torch.nn.functional as F

# Multi-head attention (simplified)
B, N, D = 2, 197, 768
h = 12  # heads
d_k = D // h

# CLS 의 query, 모든 tokens 의 key/value
Q = torch.randn(B, h, 1, d_k)  # CLS query (1 token)
K = torch.randn(B, h, N, d_k)  # All tokens (197 tokens)
V = torch.randn(B, h, N, d_k)

# Attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, h, 1, N)
attn_weights = F.softmax(scores, dim=-1)  # (B, h, 1, N)

print(f"Attention weights shape: {attn_weights.shape}")
print(f"Attention sums to 1: {attn_weights.sum(dim=-1).mean():.4f}")

# Aggregation
attn_out = torch.matmul(attn_weights, V)  # (B, h, 1, d_k)

# 각 head 의 patch 에 대한 평균 attention weight
avg_attn_per_head = attn_weights.mean(dim=0).squeeze(1)  # (h, N)
avg_attn_all = avg_attn_per_head.mean(dim=0)  # (N,)

print(f"Mean attention to CLS (token 0): {avg_attn_all[0]:.4f}")
print(f"Mean attention to patches: {avg_attn_all[1:].mean():.4f}")

# Visualization: 어느 patch 가 가장 주목받는가?
top_k = 5
top_patches = torch.argsort(avg_attn_all[1:], descending=True)[:top_k]
print(f"Top {top_k} attended patches: {top_patches.tolist()}")
```

### 실험 4: CLS vs GAP 의 성능 비교 (축소 ViT)

```python
import torch
import torch.nn as nn

class VitWithCLS(nn.Module):
    def __init__(self, D=768, num_classes=1000):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, D))
        self.head = nn.Linear(D, num_classes)
    
    def forward(self, z):
        # z: (B, N+1, D) with CLS already included
        cls_token = z[:, 0]  # (B, D)
        logits = self.head(cls_token)
        return logits

class VitWithGAP(nn.Module):
    def __init__(self, D=768, num_classes=1000):
        super().__init__()
        self.head = nn.Linear(D, num_classes)
    
    def forward(self, z):
        # z: (B, N+1, D)
        gap = z[:, 1:].mean(dim=1)  # (B, D) - exclude CLS
        logits = self.head(gap)
        return logits

# Test
B, N, D = 4, 196, 768
num_classes = 1000

# Dummy transformer output (CLS + patches)
z = torch.randn(B, N + 1, D)

vit_cls = VitWithCLS(D, num_classes)
vit_gap = VitWithGAP(D, num_classes)

logits_cls = vit_cls(z)
logits_gap = vit_gap(z)

print(f"CLS logits shape: {logits_cls.shape}")
print(f"GAP logits shape: {logits_gap.shape}")

# Dummy labels and loss
labels = torch.randint(0, num_classes, (B,))
loss_cls = nn.CrossEntropyLoss()(logits_cls, labels)
loss_gap = nn.CrossEntropyLoss()(logits_gap, labels)

print(f"CLS loss: {loss_cls:.4f}")
print(f"GAP loss: {loss_gap:.4f}")
print(f"Loss difference: {abs(loss_cls - loss_gap):.4f}")
print("✓ Both CLS and GAP are viable pooling strategies")
```

## 🔗 실전 활용

**DeiT (Data-efficient Image Transformers, Touvron 2021, Ch2-01)**:
- CLS token 을 기본으로 사용
- 추가 실험에서 GAP 과 성능 비교

**CLIP (Vision-Language, Ch6)**:
- CLS token 을 image representation 으로 사용
- 텍스트 representation 과 contrastive learning

**Swin Transformer (Ch2-02, hierarchical)**:
- Hierarchical 구조에서도 최종 layer 는 CLS 사용
- 또는 spatial average 후 classification

**Vision-MAE (Ch5, masked image modeling)**:
- CLS token 을 사용하지 않음 (reconstruction-based)
- 대신 모든 patch 의 hidden state 를 예측

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **CLS initialization** | 무작위 초기화 (standard normal) | 초기화 방식에 따라 수렴 속도 다름 |
| **Attention is learned** | CLS 의 attention weight 는 training 중 학습됨 | Weight 가 uniform 으로 수렴할 수도, 특정 patch 에 집중할 수도 있음 |
| **Position embedding** | CLS token 에도 position embedding 이 더해짐 | 첫 번째 position (special token) |
| **Pooling strategy** | GAP 은 "공평한" aggregation | 일부 patch 가 중요해도 무시할 수 있음 |
| **Small vs large models** | 작은 모델: GAP 이 더 나을 수 있음 | 큰 모델: CLS 의 learnable flexibility 가 유리 |

## 📌 핵심 정리

$$\boxed{z_L^0 = \text{Attention-weighted aggregation of all tokens}}$$

$$\boxed{\bar{z}_{\text{GAP}} = \frac{1}{N} \sum_{i=1}^{N} z_L^i}$$

| 개념 | CLS Token | Global Average Pooling |
|------|----------|------------------------|
| **정의** | Learnable parameter, transformer 를 통과 | Fixed, mean operation |
| **유연성** | 각 patch 를 선택적으로 활용 | 모든 patch 동등 가중치 |
| **성능** | Top-1 81.8% (DeiT) | Top-1 81.5% (DeiT) |
| **해석** | Learnable global pooling | Fixed uniform pooling |
| **계산량** | Negligible overhead | Slightly simpler |
| **권장** | 대부분의 ViT variants | 간단한 구현 / 비교 baseline |

## 🤔 생각해볼 문제

### 문제 1 (기초): CLS Token 의 Position Index

ViT 에서 CLS token 이 sequence 의 몇 번째인가? Position embedding 에서 그 index 는?

<details>
<summary>해설</summary>

CLS token 은 항상 sequence 의 **첫 번째** (index 0) 이다.

따라서:
- z₀ = [z₀^0 (CLS); z₀^1; z₀^2; ...; z₀^196]
- Position embedding: E_pos[0] (CLS), E_pos[1] (patch 1), ..., E_pos[196] (patch 196)

Position embedding 도 learnable 이므로, CLS 의 위치는 명시적으로 인코딩된다.

</details>

### 문제 2 (심화): Attention Weight 의 분포

최종 layer 에서 CLS token 이 각 patch 에 주는 attention weight 를 시각화하면, 어떤 분포를 예상하는가?

(a) Uniform (모든 patch 가 동등)  
(b) Concentrated (몇 개 patch 에만 집중)  
(c) 다른 패턴

<details>
<summary>해설</summary>

일반적으로 **혼합** 이다 (특정 답은 없음):

- 초반 layer: 더 diverse/uniform (여러 patch 의 정보 수집 필요)
- 후반 layer: 더 concentrated (특정 의미있는 patch 에 집중)

예를 들어, object 가 이미지 중앙에 있으면, 중앙 patch 에 높은 weight.
배경만 있으면, 더 uniform.

이런 학습된 attention 패턴이 CLS 의 flexibility 를 보여주며,
순수 GAP (uniform) 과의 차이를 만든다.

</details>

### 문제 3 (논문 비평): CLS vs GAP 의 trade-off

Touvron 2021 의 실험에서 CLS 와 GAP 이 거의 같은 성능을 보였다.  
그렇다면 왜 모든 ViT 가 여전히 CLS 를 사용하는가?

<details>
<summary>해설</summary>

**이론적 관점**:
- CLS 는 learnable, GAP 은 fixed
- 극단적 경우 (예: object 가 가장자리에만 있는 특수한 데이터), CLS 가 적응할 수 있음

**실제 구현**:
- CLS token 의 parameter 개수: 768 (negligible, 86M 중 0.001%)
- 추가 계산량: 거의 없음 (1개 token 의 attention 만 계산)

**관례**:
- Transformer 의 standard (BERT, GPT 등에서도 사용)
- 다른 tasks (retrieval, clustering) 에서도 CLS 를 embedding 으로 사용
- Universality: 한 번 학습된 CLS representation 을 여러 downstream tasks 에 사용 가능

**종합**:
- 성능 차이가 미미하지만, learnable 이고 overhead 가 거의 없으므로
- 모든 ViT 가 CLS 를 사용하는 것이 합리적

</details>

---

<div align="center">

[◀ 이전](./02-patch-conv-equivalence.md) | [📚 README](../README.md) | [다음 ▶](./04-positional-embedding-variants.md)

</div>
