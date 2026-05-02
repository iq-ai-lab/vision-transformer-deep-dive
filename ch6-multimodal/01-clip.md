# 01. CLIP — Contrastive Language-Image Pretraining

## 🎯 핵심 질문

- Vision Transformer 와 Transformer 텍스트 인코더를 대칭적으로 학습하려면 어떤 손실함수가 필요한가?
- 왜 symmetric contrastive loss 를 쓰는가? 그리고 온도 파라미터 $\tau$ 의 역할은?
- 400M 이미지-텍스트 쌍으로 학습하면 zero-shot transfer 가 정말 가능한가?
- CLIP 이 Vision Transformer 를 재정의한 이유는?

## 🔍 왜 이 손실함수가 CLIP 인가

Vision-Language 연결의 핵심은 **대칭성** 이다. 이미지 인코더도 텍스트 인코더도 같은 embedding space 에 놓아야 한다.

Contrastive 학습은 positive pair (이미지-텍스트) 는 가깝게, negative pair 는 멀게 한다.  
하지만 **양쪽 인코더를 동시에 학습** 하려면, 양방향 contrastive loss 가 필요하다.
— image-to-text matching + text-to-image matching 을 동시에.

CLIP 은 이를 $\frac{1}{2}(L_{i \to t} + L_{t \to i})$ 로 풀었고, 이 단순함이 web-scale 학습을 가능하게 했다.

## 📐 수학적 선행 조건

- **Softmax cross-entropy**: $-\log \frac{e^{z_{\text{pos}}}}{\sum_j e^{z_j}}$
- **InfoNCE (Oord 2018)**: Mutual Information 의 하한; $L = -\log \frac{e^{z_{\text{pos}}/\tau}}{\sum_j e^{z_j/\tau}}$
- **Batch-wise negatives**: minibatch 내 다른 샘플들이 hard negative 역할
- **Temperature scaling**: softmax 의 sharpness 제어
- **Symmetry in learning**: 양방향 gradient flow

## 📖 직관적 이해

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          CLIP 의 이미지-텍스트 정렬

  이미지 인코더 (ViT)           텍스트 인코더 (Transformer)
       ↓                               ↓
   [img embedding]              [text embedding]
     shape: (B, D)               shape: (B, D)
       ↓ normalize                     ↓ normalize
     I ∈ ℝ^{B×D}               T ∈ ℝ^{B×D}
       ↓                               ↓
  ┌─────────────────────────────────────┐
  │  Similarity Matrix S = I @ T^T      │
  │  S_{ij} = cos_sim(img_i, text_j)   │
  │  shape: (B, B)                      │
  │  대각선 원소가 positive pair         │
  └─────────────────────────────────────┘
       ↓
  Image-to-Text Loss:
  각 이미지 i 마다, S_i 에서 대각선 i 만 positive
  L_{i→t} = cross-entropy(S / τ, target=[0,1,...,0])

  Text-to-Image Loss:
  각 텍스트 j 마다, S^T_j 에서 대각선 j 만 positive
  L_{t→i} = cross-entropy(S^T / τ, target=[0,1,...,0])

  Final Loss:
  L = 0.5 * (L_{i→t} + L_{t→i})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**비유**: 이미지와 텍스트가 두 개의 프로젝터로 떨어져 있고, 
"같은 대상" 을 가리키는 것끼리만 겹치도록 정렬하는 것.
온도 $\tau$ 는 두 프로젝터의 "초점" 을 맞추는 초점거리.

## ✏️ 엄밀한 정의

### 정의 6.1: CLIP 이미지-텍스트 쌍

batch 크기 $B$ 에 대해, paired dataset 에서:
- 이미지: $\mathbf{I} = (I_1, I_2, \ldots, I_B) \in \mathbb{R}^{B \times H \times W \times 3}$
- 텍스트: $\mathbf{T} = (T_1, T_2, \ldots, T_B)$ (각각 tokenize 됨)

대각선 쌍 $(I_i, T_i)$ 이 positive, $(I_i, T_j)$ ($i \neq j$) 이 negative.

### 정의 6.2: Normalized Embeddings

이미지 인코더 $f_I$ (ViT), 텍스트 인코더 $f_T$ (Transformer) 에 대해:
$$\mathbf{z}_I^{(i)} = \frac{f_I(I_i)}{\|f_I(I_i)\|_2}, \quad \mathbf{z}_T^{(j)} = \frac{f_T(T_j)}{\|f_T(T_j)\|_2}$$

모두 L2-normalized (단위 벡터). Embedding dimension: $D$ (보통 512).

### 정의 6.3: Similarity Matrix

$$S_{ij} = \tau \cdot (\mathbf{z}_I^{(i)})^T \mathbf{z}_T^{(j)}$$

여기서 $\tau$ 는 학습 가능한 온도 파라미터 (초기값 $\log(1/0.07) \approx 2.603$, 보통 $\tau \leq 100$ 으로 clamp).

### 정의 6.4: Image-to-Text Loss

$$L_{i \to t} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{S_{ii}}}{\sum_{j=1}^{B} e^{S_{ij}}}$$

CrossEntropyLoss 관점: 이미지 $i$ 를 query 로, 텍스트 $\{1, \ldots, B\}$ 를 candidate 로 보면,
정답은 index $i$.

### 정의 6.5: Text-to-Image Loss (대칭)

$$L_{t \to i} = -\frac{1}{B} \sum_{j=1}^{B} \log \frac{e^{S_{jj}}}{\sum_{i=1}^{B} e^{S_{ij}}}$$

Text query 에 image candidate.

### 정의 6.6: 대칭 InfoNCE 손실

$$L_{\text{CLIP}} = \frac{1}{2}(L_{i \to t} + L_{t \to i})$$

## 🔬 정리와 증명

### 정리 6.1: 대칭 손실과 정보량 하한

**명제**: CLIP 의 대칭 손실 $L_{\text{CLIP}} = \frac{1}{2}(L_{i \to t} + L_{t \to i})$ 는
두 개의 InfoNCE 손실의 합과 동치이며, 각각 상호 정보량의 variational lower bound 를 이룬다.

**증명**:

(1) Image-to-Text 손실을 정보론 관점에서 본다.  
Oord et al. (2018, InfoNCE) 에 따르면, positive 와 negatives 의 classification 손실은:
$$L_{i \to t} = -\mathbb{E}_{(I,T)} \left[ \log \frac{e^{f_I(I)^T f_T(T) / \tau}}{e^{f_I(I)^T f_T(T) / \tau} + \sum_{j \neq i} e^{f_I(I)^T f_T(T_j) / \tau}} \right]$$

이를 다시 쓰면:
$$L_{i \to t} = -\log \frac{1}{1 + \sum_{j \neq i} e^{(f_I(I)^T f_T(T_j) - f_I(I)^T f_T(T)) / \tau}}$$

Batch size $B$ 에서:
$$L_{i \to t} = \mathbb{E} \left[ -\log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}} \right]$$

마찬가지로 $L_{t \to i}$ 도 정의.

(2) 대칭성:  
Gradient 흐름을 생각해보면,
$$\frac{\partial L_{i \to t}}{\partial S_{ij}} = \frac{1}{B} (p_{ij} - \delta_{ij})$$
where $p_{ij} = \frac{e^{S_{ij}/\tau}}{\sum_k e^{S_{ik}/\tau}}$ (softmax 확률).

대칭 손실 $\frac{1}{2}(L_{i \to t} + L_{t \to i})$ 는 gradient 가 양쪽 인코더에 대칭적으로 작용한다는 뜻.

(3) InfoNCE lower bound:  
Poole et al. (2019) 에서: 정렬된 positive pair 와 batch negatives 에 대한 contrastive loss 는
$$I(I; T) \geq L_{\text{InfoNCE}} = \log B - L$$
의 mutual information lower bound 를 갖는다. 따라서 $L_{\text{CLIP}}$ 을 minimize 하는 것은 
두 인코더 embedding 의 상호정보를 maximize 하는 것과 동치.

$$\square$$

### 정리 6.2: 온도 파라미터의 역할

**명제**: 온도 $\tau$ 는 softmax 의 sharpness 를 제어하며, 
너무 작으면 overconfidence (gradient 소실), 너무 크면 underconfidence (낮은 discriminability) 를 야기한다.

**증명**:

Softmax 의 entropy:
$$\mathcal{H}(\tau) = -\sum_{j=1}^B p_j(\tau) \log p_j(\tau), \quad p_j(\tau) = \frac{e^{S_{ij} / \tau}}{\sum_k e^{S_{ik} / \tau}}$$

$\tau \to 0^+$ 일 때: $\mathcal{H}(\tau) \to 0$ (one-hot), gradient 가 sparse 해짐.  
$\tau \to \infty$ 일 때: $\mathcal{H}(\tau) \to \log B$ (uniform), 정보 손실.

Optimal $\tau$ 는 gradient 의 signal-to-noise ratio 를 최대화.  
CLIP 논문에서는 $\tau$ 를 학습 가능하게 하되 log scale 로 초기화 ($\log(1/0.07)$) 하고,
$\tau \in (1/100, 100)$ 로 clamp 해서 numerical stability 를 보장.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 간단한 CLIP 손실 구현

```python
import torch
import torch.nn.functional as F

def clip_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    image_embeddings: (B, D) normalized
    text_embeddings: (B, D) normalized
    temperature: float
    Returns: scalar loss
    """
    # Similarity matrix: (B, B)
    logits = image_embeddings @ text_embeddings.T / temperature
    
    # Image-to-text loss
    labels = torch.arange(len(image_embeddings), device=image_embeddings.device)
    loss_i2t = F.cross_entropy(logits, labels)
    
    # Text-to-image loss (transpose)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    # Symmetric
    loss = (loss_i2t + loss_t2i) / 2.0
    return loss

# Toy data
B, D = 4, 16
torch.manual_seed(42)
img_emb = F.normalize(torch.randn(B, D), dim=1)
txt_emb = F.normalize(torch.randn(B, D), dim=1)

loss = clip_loss(img_emb, txt_emb, temperature=0.07)
print(f"Batch loss: {loss.item():.4f}")

# Expected: ~log(B) = log(4) ≈ 1.386 (random embeddings, no alignment)
```

**예상 결과**: B=4 일 때 random embeddings 로 약 1.386 (모든 클래스가 등확률).

### 실험 2: 온도 파라미터의 영향

```python
import matplotlib.pyplot as plt

temperatures = [0.01, 0.07, 0.2, 0.5, 1.0]
losses = []

for tau in temperatures:
    l = clip_loss(img_emb, txt_emb, temperature=tau)
    losses.append(l.item())
    print(f"τ={tau:.3f}: L={l.item():.4f}")

# 온도가 높아질수록 loss 는 log(B) 에 수렴
```

### 실험 3: Alignment 학습 단계별

```python
import torch.optim as optim

# 약간 aligned 된 embeddings (sin/cos function)
img_emb = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.0, -1.0]
], dtype=torch.float32)
img_emb = F.normalize(img_emb, dim=1)

# Text embeddings: image 와 약간 다름
txt_emb = img_emb + 0.3 * torch.randn_like(img_emb)
txt_emb = F.normalize(txt_emb, dim=1)

# Trainable parameters
img_emb.requires_grad = True
txt_emb.requires_grad = True

optimizer = optim.SGD([img_emb, txt_emb], lr=0.1)

print("Step | Loss")
for step in range(11):
    optimizer.zero_grad()
    loss = clip_loss(img_emb, txt_emb, temperature=0.07)
    loss.backward()
    optimizer.step()
    
    if step % 2 == 0:
        print(f"{step:4d} | {loss.item():.4f}")

# Output (example):
# Step | Loss
#    0 | 1.2345
#    2 | 1.1234
#    4 | 1.0567
#    ...
```

### 실험 4: 대칭성 검증

```python
# Gradient 가 양쪽 인코더에 균형잡혀 있는지 확인

img_emb = F.normalize(torch.randn(B, D), dim=1, requires_grad=True)
txt_emb = F.normalize(torch.randn(B, D), dim=1, requires_grad=True)

loss = clip_loss(img_emb, txt_emb, temperature=0.07)
loss.backward()

grad_img_norm = torch.norm(img_emb.grad).item()
grad_txt_norm = torch.norm(txt_emb.grad).item()

print(f"Gradient norm (image): {grad_img_norm:.6f}")
print(f"Gradient norm (text):  {grad_txt_norm:.6f}")
print(f"Ratio: {grad_img_norm / grad_txt_norm:.4f}")  # ≈ 1.0 (대칭)
```

## 🔗 실전 활용

**CLIP (Radford et al. 2021)**:
- ViT-L/14 또는 ViT-L/32 이미지 인코더
- 12-layer Transformer 텍스트 인코더
- 400M image-text 쌍 (web scrape)
- 학습률 $5 \times 10^{-4}$, batch size 32K (분산 학습)
- 32 epochs, 약 2-3주 A100 cluster

**open_clip (Cherti et al. 2023)**:
- CLIP 을 재현 + 개선
- 다양한 backbone (ViT-g/14, convnext, etc.)
- 공개 weights: ViT-L/14 on LAION-400M, LAION-2B

**실무 활용**:
1. Zero-shot classification (Ch.6-02)
2. Vision-language retrieval
3. Semantic search (product discovery)
4. Vision grounding (bounding box 생성)

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **배치 크기 충분** | Negatives 가 minibatch 에서 충분히 커야 함 | B < 100 이면 training instability |
| **Contrastive learning 가정** | Negative pairs 가 충분히 negative 함 | Web data 는 noise 많음 (관련 없는 이미지-텍스트) |
| **L2 정규화** | Embedding 을 단위벡터로 normalize | Cosine similarity 만 사용 가능; inner product 불가 |
| **온도 범위** | $\tau$ 를 합리적 범위로 clamp | $\tau > 100$ 이면 underflow, $\tau < 0.01$ 이면 overflow |
| **분포 균형** | Positive pair 와 negative pair 의 개수가 대칭 | Imbalanced batch 는 bias 야기 |

## 📌 핵심 정리

$$\boxed{L_{\text{CLIP}} = \frac{1}{2}(L_{i \to t} + L_{t \to i}) = \frac{1}{2}\left( -\log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}} - \log \frac{e^{S_{ii}/\tau}}{\sum_i e^{S_{ij}/\tau}} \right)}$$

| 개념 | 정의 | 역할 |
|------|------|------|
| **Similarity Matrix** | $S_{ij} = \tau \cdot \mathbf{z}_I^{(i)T} \mathbf{z}_T^{(j)}$ | 이미지-텍스트 matching score |
| **Image-to-Text Loss** | $L_{i \to t} = -\sum_i \log p(T_i \mid I_i)$ | 이미지→텍스트 대응 학습 |
| **Text-to-Image Loss** | $L_{t \to i} = -\sum_j \log p(I_j \mid T_j)$ | 텍스트→이미지 대응 학습 |
| **Temperature** | $\tau$ (학습 가능, $\in [1/100, 100]$) | Softmax sharpness 조절; 안정성 |
| **Batch Negatives** | Minibatch 의 다른 쌍들 | Hard negative 제공; $N$ 커질수록 informative |
| **L2 Normalization** | $\mathbf{z} = f(x) / \|f(x)\|_2$ | Cosine similarity 유지; scale invariance |

## 🤔 생각해볼 문제

### 문제 1 (기초): 대칭성의 필요성

CLIP 이 비대칭 손실 $L_{i \to t}$ 만 쓰면 안 되는 이유는?  
양쪽 인코더가 균등하게 학습되지 않을 것 같은데, 증명하시오.

<details>
<summary>해설</summary>

비대칭 손실 $L_{i \to t} = -\log p(T \mid I)$ 만 쓰면:
$$\frac{\partial L_{i \to t}}{\partial \mathbf{z}_I} = \frac{1}{\tau}(\mathbf{z}_T - \bar{\mathbf{z}}_T^{\text{softmax}})$$
$$\frac{\partial L_{i \to t}}{\partial \mathbf{z}_T} = \frac{1}{\tau}(\mathbf{z}_I^{\text{aligned}} - \bar{\mathbf{z}}_I^{\text{softmax}})$$

이미지 인코더는 "이미지를 텍스트에 가깝게" 밀려고 하지만,
텍스트 인코더는 "전체 배치에 대한" gradient 를 받는다.  
특히 text-to-image 정보는 학습되지 않아서, 이미지는 텍스트 embedding space 에 완전히 정렬되지 않음.

대칭 손실을 쓰면 양쪽 다 "상호 정보"를 maximize 하므로 균등 학습.

</details>

### 문제 2 (심화): 온도 파라미터의 학습 곡선

초기 $\tau = \log(1/0.07) \approx 2.603$ 에서 시작할 때,  
학습 중 $\tau$ 가 어떻게 변하는가? 그리고 왜 upper bound 100을 두는가?

<details>
<summary>해설</summary>

CLIP 논문 (Radford et al. 2021):
- 온도는 학습 가능한 매개변수: $\log \tau$ 를 최적화
- 초기값: $\log \tau_0 = \log(1/0.07) \approx 2.603$
- 학습 중: 자동으로 조정됨. 보통 training 진행되면서 $\tau$ 는 약간 감소 (sharper softmax)

Upper bound 100 의 이유:
- $\tau > 100$ 이면 $S_{ij} / \tau$ 의 range 가 너무 작아져서, softmax 가 거의 uniform 에 수렴
- Numerical underflow 위험 (모든 logits 이 거의 0)
- Gradient 가 소실됨

Lower bound $1/100$ 의 이유:
- $\tau < 0.01$ 이면 softmax 가 one-hot 에 가까워짐 (extreme confidence)
- Gradient 가 sparse 해져서 learning 이 불안정
- Mode collapse 위험

Practical: 보통 log scale 로 constraint 를 두고 $\log \tau \in [\log(1/100), \log 100]$ 로 clamp.

</details>

### 문제 3 (논문 이해): Batch Size 와 Hard Negatives

CLIP 에서 배치 크기가 32K 로 엄청 큰 이유는?  
이론적으로 배치 크기와 손실/성능의 관계를 설명하시오.

<details>
<summary>해설</summary>

Minibatch contrastive learning 에서:
- Negatives 의 개수: $B - 1$
- InfoNCE lower bound: $I(I; T) \geq \log B - L$
- 즉, $B$ 가 크면 더 많은 hard negatives 가 자동으로 샘플됨

CLIP 에서 B=32K 를 쓰는 이유:
1. **Hard negative 충분성**: 32000개 후보 중에 semantic 으로 가까운 것들이 있을 확률이 높음
2. **정보량**: $\log(32000) \approx 10.4$ bits 의 정보 제약
3. **대규모 데이터**: 400M 쌍을 학습하려면 충분히 큰 배치로 각 pair 의 가치를 최대화
4. **분산 학습**: 32K 는 여러 GPU 에 나뉨 (모든 replica 의 embeddings 를 모아야 함)

Smaller batch (e.g., B=256) 에서는:
- Negative 수가 적음 → informative 도 적음
- Training 이 불안정 (큰 variance)
- Zero-shot transfer 성능이 떨어짐

이는 contrastive learning 에서 배치 크기가 매우 중요한 이유.

</details>

---

<div align="center">

[◀ 이전](../ch5-masked-image-modeling/05-maskfeat-mvp.md) | [📚 README](../README.md) | [다음 ▶](./02-clip-zero-shot.md)

</div>
