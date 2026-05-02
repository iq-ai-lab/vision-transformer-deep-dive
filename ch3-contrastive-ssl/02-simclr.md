# 02. SimCLR — Simple Contrastive Framework (Chen 2020)

## 🎯 핵심 질문

- SimCLR pipeline (augmentation → encoder → projection → InfoNCE) 의 각 단계가 왜 필수적인가?
- Augmentation strength (color + crop) 가 contrastive learning 의 성능에 미치는 영향은?
- Projection head $g$ 와 encoder $f$ 의 역할 분담 — why use $f$ for downstream but $g$ for pretraining?
- Large batch size (8K) 가 정말 필요한가 — negative count 의 수학적 의존성은?
- SimCLR 의 gradient flow 형태는 무엇인가? Information bottleneck 관점에서 어떻게 해석되는가?

## 🔍 왜 SimCLR 인가

2020년 Chen et al. 이 발표한 SimCLR 은 contrastive learning 의 가장 단순하면서도 효과적인 framework 이다.

핵심은:
1. **단순함**: 추가 structure (memory bank, momentum encoder) 없음.
2. **효과성**: Large batch + strong augmentation 으로 SOTA 달성.
3. **해석 가능**: 수학적으로 명확한 InfoNCE loss, instance discrimination.

SimCLR 의 성공이 contrastive learning 이 NLP (BERT) 만큼 vision 에서도 가능함을 증명했다.

## 📐 수학적 선행 조건

- **InfoNCE Loss**: Poole (2019) — noise contrastive estimation
- **Mutual Information**: $I(X; Y) = H(X) - H(X|Y)$, lower bound (Chapter 3.3 에서 상세)
- **Cosine Similarity**: $\cos(a, b) = \frac{a^\top b}{\|a\| \|b\|}$ 의 미분
- **Softmax & Cross-entropy**: temperature scaling, log-sum-exp stability
- **Data Augmentation**: Random crop, color jitter, Gaussian blur (torchvision.transforms)

## 📖 직관적 이해

```
┌─────────────────────────────────────────────────────┐
│ SimCLR Full Pipeline (End-to-end)                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Input: x (image)                                    │
│   ↓                                                 │
│ Two random augmentations:                           │
│   x̃₁ = Aug(x)  and  x̃₂ = Aug(x)                   │
│   (same image, different noise samples)             │
│   ↓                                                 │
│ Encode both through f (shared encoder):             │
│   h₁ = f(x̃₁)  and  h₂ = f(x̃₂)  ∈ ℝᴰ              │
│   ↓                                                 │
│ Project through MLP g:                              │
│   z₁ = g(h₁)  and  z₂ = g(h₂)  ∈ ℝᴰ'               │
│   ↓                                                 │
│ Compute InfoNCE loss:                               │
│   L = -log(exp(sim(z₁,z₂)/τ) / Σ exp(...))         │
│   ↓                                                 │
│ Downstream: use h (not z) with linear probe         │
│                                                     │
└─────────────────────────────────────────────────────┘

Key insight: z for contrastive, h for downstream
(projection head improves contrastive learning but
 hurts downstream — explain why in theory)

┌─────────────────────────────────────────────────────┐
│ Why Projection Head Hurts Downstream (Appendix)    │
├─────────────────────────────────────────────────────┤
│                                                     │
│ g(h) = W₂ σ(W₁ h + b₁) + b₂  (2-layer MLP)         │
│                                                     │
│ During pretraining: z = g(h) optimized for         │
│ contrastive task. But this forces g to "waste"     │
│ dimensions on discriminating views, not semantics. │
│                                                     │
│ During downstream: if we use z instead of h,       │
│ we lose semantic info that g didn't need.          │
│                                                     │
│ Solution: throw away g, use h for linear probe     │
│ (or fine-tune with h).                             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**비유**: 하나의 사진을 여러 각도에서 촬영 (augmentation) 해서, 
같은 물체임을 판별 (InfoNCE) 하도록 neural network 를 훈련한다.
훈련 후, 네트워크의 "눈" (h) 는 물체의 semantic feature 를 알지만,
"판별기 뇌" (g) 는 단지 두 사진이 같은지만 판별할 수 있도록 특화되었다.
따라서 downstream task 에는 "눈" (h) 을 사용한다.

## ✏️ 엄밀한 정의

### 정의 3.4: Data Augmentation in SimCLR

Image $x$ 에 대한 stochastic augmentation $\mathcal{T}$ 는 다음 composition:
$$\tilde{x} = \mathcal{T}(x) = T_{\text{color}} \circ T_{\text{crop}} \circ T_{\text{resize}} (x)$$

여기서:
- $T_{\text{resize}}$ : random size (224×224)
- $T_{\text{crop}}$ : random crop to 224×224 (50~100% of image)
- $T_{\text{color}}$ : random color jitter (brightness, contrast, saturation, hue)
- Additional: Gaussian blur (50% probability), random grayscale (10%)

**중요**: 같은 $x$ 로부터 **독립적으로** $\tilde{x}_i, \tilde{x}_j \sim \mathcal{T}(x)$ 를 샘플링.

### 정의 3.5: SimCLR Encoder & Projection

**Encoder** $f: \mathbb{R}^{3 \times 224 \times 224} \to \mathbb{R}^D$
- ResNet50 또는 ViT backbone
- Output: $h = f(x) \in \mathbb{R}^D$ (hidden representation)
- Usually $D = 2048$ (ResNet) or $768$ (ViT)

**Projection head** $g: \mathbb{R}^D \to \mathbb{R}^{D'}$
$$g(h) = W_2 \sigma(W_1 h + b_1) + b_2$$

여기서:
- $W_1 \in \mathbb{R}^{D_{\text{mid}} \times D}$, $W_2 \in \mathbb{R}^{D' \times D_{\text{mid}}}$
- $\sigma = \text{ReLU}$
- Usually $D_{\text{mid}} = D$ (no bottleneck), $D' = 128$
- $\ell_2$ normalization: $z = g(h) / \|g(h)\|_2$

### 정의 3.6: InfoNCE Loss (SimCLR)

Batch $\{x_k\}_{k=1}^B$ 에서 augmentation pair 생성:
$$\{\tilde{x}_{k,i}, \tilde{x}_{k,j}\}_{k=1}^{B}$$

For each sample, two augmented views pass through encoder + projection:
$$z_{k,i} = g(f(\tilde{x}_{k,i})), \quad z_{k,j} = g(f(\tilde{x}_{k,j}))$$

**InfoNCE loss** for the pair $(z_{k,i}, z_{k,j})$:
$$\ell_{k,i} = -\log \frac{\exp(\mathrm{sim}(z_{k,i}, z_{k,j}) / \tau)}{\sum_{m=1}^{2B} \mathbb{1}_{[m \neq k]} \exp(\mathrm{sim}(z_{k,i}, z_{k,m}) / \tau)}$$

여기서:
- $\mathrm{sim}(a, b) = \cos(a, b) = a^\top b / (\|a\| \|b\|)$ (cosine similarity)
- $\tau$ : temperature parameter (typically 0.07)
- Denominator 의 $2B$ : 현재 batch 의 모든 view ($B$ images × 2 augmentations)
- $\mathbb{1}_{[m \neq k]}$ : self-contrastive pair 제외

**Total loss**:
$$L = \frac{1}{2B} \sum_{k=1}^{B} (\ell_{k,i} + \ell_{k,j})$$

(symmetric loss — 두 방향 모두 계산)

## 🔬 정리와 증명

### 정리 3.3: Gradient Form of SimCLR (InfoNCE)

정의 3.6 의 loss $\ell_k$ 에 대해:

$$\nabla_{z_{k,i}} \ell_k = \frac{1}{\tau} \left[ \frac{\exp(\mathrm{sim}(z_{k,i}, z_{k,j})/\tau)}{\sum_m \exp(\mathrm{sim}(z_{k,i}, z_{k,m})/\tau)} (z_{k,i} - z_{k,j}) + \text{(correction from denominator)} \right]$$

**직관적 해석**:
1. Positive pair term: $z_{k,i}$ 와 $z_{k,j}$ 를 끌어당김 (negative gradient 방향).
2. Negative pair term: 모든 negative pair 로부터 밀어냄 (positive gradient 방향).
3. The softmax weight: 어려운 negative (sim 이 높은 것) 에만 집중.

**증명** (요약):

Cross-entropy loss $L = -\log p_{k,j}$ where
$$p_{k,j} = \frac{\exp(s_{k,i,j}/\tau)}{\sum_m \exp(s_{k,i,m}/\tau)}$$

Let $q = $ softmax denominator. Then:
$$\nabla_{s} L = -\frac{1}{q} \nabla_s \exp(s/\tau) + \ldots = -\mathbb{1}_{[j]} + p_j$$

Gradient of cosine sim $\mathrm{sim}(a,b)$ w.r.t. $a$:
$$\nabla_a \mathrm{sim}(a,b) = \frac{1}{\|a\|} \left( b - \mathrm{sim}(a,b) a \right)$$

Combining: complex but interpretable gradient flow.

$$\square$$

### 정리 3.4: Necessity of Projection Head

**Claim**: Projection head $g$ 가 없으면 (즉, $z = h$ 직접 사용) contrastive learning 성능이 떨어진다.

**증명 (경험적)**:

1. **정보 이론적 관점** (Wang et al. 2020):
   - Encoder $f$ 는 augmentation-invariant 을 최대화.
   - Projection $g$ 는 information bottleneck 을 도입 → high-level semantic 만 남김.
   - Without $g$: $h$ 에 augmentation detail (color shift, crop position 등) 가 섞여 있음.

2. **실증적** (SimCLR paper, Table 3):
   
   | Configuration | Linear Probe (ImageNet top-1) |
   |---|---|
   | Without $g$ (use $h$ directly) | 69.3% |
   | With $g$ (use $z = g(h)$) | 71.3% |
   | **But downstream (fine-tune)**: |  |
   | Use $z$ from pretrained | 76.5% |
   | Use $h$ from pretrained | 77.2% |

   → $g$ 는 contrastive task 에는 도움이지만, downstream 에는 $h$ 가 더 낫다!

$$\square$$

### 정리 3.5: Batch Size & Negative Count Dependency

정리 3.2 (Poole 2019) 의 mutual information lower bound:
$$I(z_i; z_j) \geq \log N - L_{\text{InfoNCE}}$$

여기서 $N = $ number of negatives $= 2(B-1)$ (SimCLR).

**Corollary**: Loss 가 고정되어 있으면, $N$ 이 크면 $I$ 의 하한도 커진다. 
즉, **더 많은 negative = 더 높은 MI lower bound**.

**실제 영향** (Chen et al. 2020, Figure 3):

$$\text{Top-1 Linear Probe Accuracy} \approx 60\% @ B=64, 65\% @ B=256, 69\% @ B=4096, 71\% @ B=4096+$$

→ Batch size 가 $2^x$ 로 증가하면, performance 도 logarithmically 증가.

**문제**: 큰 batch size 는 GPU memory 를 많이 차지. Solution: gradient accumulation, distributed training (이 때문에 MoCo 가 나중에 queue 로 해결).

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: SimCLR Loss 구현 (Full)

```python
import torch
import torch.nn.functional as F

def simclr_loss(z_i, z_j, temperature=0.07):
    """
    SimCLR loss for a batch of positive pairs.
    z_i, z_j: (B, D) tensors, already l2-normalized
    """
    B = z_i.shape[0]
    
    # Concatenate to handle both (z_i, z_j) and (z_j, z_i) in one shot
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    
    # Compute similarity matrix (2B, 2B)
    sim_matrix = torch.mm(z, z.t()) / temperature  # (2B, 2B)
    
    # Positive pairs: diagonal (i, i+B) and (i+B, i)
    # i-th element paired with (i+B)-th and vice versa
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])  # [B, B+1, ..., 2B-1, 0, 1, ..., B-1]
    
    # Cross-entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# Toy test
B, D = 32, 128
z_i = torch.randn(B, D)
z_j = torch.randn(B, D)

# L2 normalize
z_i = F.normalize(z_i, dim=1)
z_j = F.normalize(z_j, dim=1)

loss = simclr_loss(z_i, z_j, temperature=0.07)
print(f"SimCLR loss (random init): {loss:.4f}")
# Expected: ~log(2B) ≈ log(64) ≈ 4.16
```

### 실험 2: Augmentation Importance (Ablation)

```python
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch.optim as optim

def simclr_transform(image_size=32):
    """SimCLR augmentation pipeline."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1*image_size)*2+1, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

# CIFAR-100 dataset (small for quick test)
dataset = CIFAR100(root='./data', train=True, download=True, 
                   transform=simclr_transform(32))
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

# Simple ResNet18 encoder
from torchvision.models import resnet18
encoder = resnet18(pretrained=False)
encoder.fc = torch.nn.Identity()  # Remove final FC

# Projection head
proj = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 128)
)

optimizer = optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=1e-3)

# 1-epoch training (toy)
encoder.train()
proj.train()
losses = []

for batch_idx, (x, _) in enumerate(loader):
    # Two augmented views from same batch
    x_aug1 = simclr_transform(32)(x)
    x_aug2 = simclr_transform(32)(x)
    
    # Forward pass
    h1 = encoder(x_aug1)
    h2 = encoder(x_aug2)
    z1 = F.normalize(proj(h1), dim=1)
    z2 = F.normalize(proj(h2), dim=1)
    
    # Loss
    loss = simclr_loss(z1, z2, temperature=0.07)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if batch_idx % 50 == 0:
        print(f"Batch {batch_idx}: loss = {loss:.4f}")

print(f"Final average loss: {sum(losses[-10:]) / 10:.4f}")
```

**기댓값**: 처음에 ~4.0, 50 steps 후 ~2.5

### 실험 3: Projection Head Ablation

```python
def train_simclr_ablation():
    """Compare with/without projection head."""
    
    # Model configurations
    configs = {
        'no_proj': False,  # Use h directly
        'with_proj': True  # Use z = g(h)
    }
    
    results = {}
    
    for name, use_proj in configs.items():
        encoder = resnet18(pretrained=False)
        encoder.fc = torch.nn.Identity()
        
        if use_proj:
            proj = torch.nn.Sequential(
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 128)
            )
            params = list(encoder.parameters()) + list(proj.parameters())
        else:
            proj = None
            params = encoder.parameters()
        
        optimizer = optim.Adam(params, lr=1e-3)
        
        final_losses = []
        for epoch in range(3):  # 3 epochs
            for batch_idx, (x, _) in enumerate(loader):
                x_aug1 = simclr_transform(32)(x)
                x_aug2 = simclr_transform(32)(x)
                
                h1 = encoder(x_aug1)
                h2 = encoder(x_aug2)
                
                if use_proj:
                    z1 = F.normalize(proj(h1), dim=1)
                    z2 = F.normalize(proj(h2), dim=1)
                else:
                    z1 = F.normalize(h1, dim=1)
                    z2 = F.normalize(h2, dim=1)
                
                loss = simclr_loss(z1, z2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                final_losses.append(loss.item())
        
        results[name] = sum(final_losses[-20:]) / 20
    
    print("Final average loss (last 20 steps):")
    for name, loss in results.items():
        print(f"  {name}: {loss:.4f}")

# Expected: with_proj < no_proj (projection helps)
```

### 실험 4: Temperature Effect

```python
def temperature_sensitivity():
    """Evaluate different temperature values."""
    
    temperatures = [0.01, 0.07, 0.1, 0.5, 1.0]
    results = {tau: [] for tau in temperatures}
    
    for tau in temperatures:
        encoder = resnet18(pretrained=False)
        encoder.fc = torch.nn.Identity()
        proj = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128)
        )
        
        optimizer = optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=1e-3)
        
        for batch_idx, (x, _) in enumerate(loader):
            x_aug1 = simclr_transform(32)(x)
            x_aug2 = simclr_transform(32)(x)
            
            h1 = encoder(x_aug1)
            h2 = encoder(x_aug2)
            z1 = F.normalize(proj(h1), dim=1)
            z2 = F.normalize(proj(h2), dim=1)
            
            loss = simclr_loss(z1, z2, temperature=tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            results[tau].append(loss.item())
            
            if batch_idx >= 50:  # Just 50 steps
                break
    
    print("Final loss by temperature:")
    for tau in temperatures:
        final_loss = sum(results[tau][-10:]) / 10
        print(f"  τ={tau}: {final_loss:.4f}")

# Expected: τ=0.07 is sweet spot (lowest final loss)
```

## 🔗 실전 활용

**Linear Probe Evaluation**:
- Freeze encoder weights, train only linear classifier on top of $h$ with labeled data.
- Standard in SSL papers (ImageNet-1k top-1 accuracy).
- SimCLR: ~71% (ResNet50), ~76% (ViT-B/32).

**Fine-tuning**:
- Unfreeze encoder + projection, fine-tune on downstream task.
- Use $h$ (not $z$), throw away $g$.
- Often achieves higher accuracy than linear probe.

**Computational Optimization**:
- **Gradient accumulation**: effective batch size without GPU memory explosion.
- **Distributed training**: all-reduce operations for gradient averaging.
- **Mixed precision**: reduce memory footprint.

## ⚖️ 가정과 한계

| 항목 | 설명 | 한계 |
|------|------|------|
| **Augmentation 품질** | Color jitter + crop 이 instance discrimination 충분 | 특정 domain (의료영상, 적외선) 에서는 부적절 가능 |
| **Large batch** | 정보이론상 더 많은 negative 가 필요 | GPU memory limit (DDP/gradient accumulation 필수) |
| **Gaussian blur** | 고주파 detail 제거 (contrastive 에 도움) | Dense task 에는 detail 손실 가능 |
| **Projection head** | Information bottleneck (contrastive 최적화) | Downstream 에는 $h$ 를 사용해야 함 |
| **Symmetric loss** | 두 방향 모두 계산 (더 안정적) | 계산량 2배 |

## 📌 핵심 정리

$$\boxed{\text{SimCLR}: \quad L = -\log \frac{\exp(\cos(z_i, z_j)/\tau)}{\sum_k \exp(\cos(z_i, z_k)/\tau)}}$$

$$\boxed{z = g(f(\tilde{x})), \quad \text{use } h = f(\tilde{x}) \text{ for downstream}}$$

| 구성 | 목적 | 핵심 파라미터 |
|------|------|------------|
| **Data augmentation** | Instance discrimination 신호 생성 | crop ratio, color strength |
| **Encoder $f$** | Augmentation-invariant 학습 | ResNet50 / ViT backbone |
| **Projection $g$** | Information bottleneck (semantic 추출) | 2-layer MLP, output dim=128 |
| **InfoNCE loss** | Contrastive objective | temperature $\tau = 0.07$ |

## 🤔 생각해볼 문제

### 문제 1 (기초): Temperature 의 역할

정의 3.6 의 loss 에서 temperature $\tau = 0.07$ 이라고 하자.
- $\tau$ 를 10배 크게 (0.7) 하면 loss 는 어떻게 변하는가?
- $\tau$ 를 10배 작게 (0.007) 하면?
- 이것이 alignment-uniformity trade-off 와 어떤 관계가 있는가?

<details>
<summary>해설</summary>

**$\tau$ 크게 (0.7)**:
- Softmax 가 flatten → 모든 negative 가 비슷한 확률
- Loss 가 덜 sharp → 천천히 수렴 (hard negative 무시)
- Uniformity 우선: 모든 negative 를 동등하게 밀어냄

**$\tau$ 작게 (0.007)**:
- Softmax 가 매우 sharp → 높은 similarity 인 negative 에 집중
- Loss 가 매우 sharp → 빠른 수렴 (hard negative 에 큰 gradient)
- Alignment 우선: positive pair 에만 집중

**Alignment-Uniformity (Wang et al. 2020)**:
- Alignment $= -\mathbb{E}[\cos(z_i, z_j)]$ (positive pair 의 similarity)
- Uniformity $= \mathbb{E}[\log \| z_i - z_j \|]$ (negative pair 의 거리)
- Temperature $\tau$ 는 이 둘의 trade-off 를 제어

$\tau$ 작으면 uniformity 를 더 강조 (hard negative),
$\tau$ 크면 alignment 를 더 강조 (soft matching).

**최적값**: 0.07 (ImageNet-1k scale 에서 검증됨, 다른 dataset 에서는 0.1~0.2 가능)

</details>

### 문제 2 (심화): Projection Head 의 Information Bottleneck

Theorem 3.4 에서 projection head $g$ 가 contrastive learning 을 돕는다고 했다.
하지만 downstream task 에는 $g$ 를 떼고 $h$ 를 사용해야 한다고 했다.

이것을 information-theoretic 관점에서 설명하시오.
$I(h; y)$ vs $I(z; y)$ (여기서 $y$ = downstream label) 를 비교하시오.

<details>
<summary>해설</summary>

**Pretraining 단계**:
- Contrastive loss 는 $z = g(h)$ 를 augmentation-invariant 하도록 최적화.
- $g$ 는 augmentation detail 을 "squeeze out" 하는 information bottleneck 역할.
- Result: $I(z; z') > I(h; h')$ (다른 augmentation 의 유사성)

**Downstream task** (예: ImageNet classification):
- Downstream label $y$ 는 대부분 semantic (물체 종류, 위치 등).
- 하지만 some detail 도 중요: 질감, 색상 (특히 fine-grained task).
- $I(h; y) > I(z; y)$ (projection 으로 손실된 detail 이 있음)

**구체적 예**:
- $h$ = [semantic features, texture, color, location, ...]
- $z = g(h)$ = [semantic features] (g 가 texture, color 를 squeeze)
- Classification 이면 semantic 으로 충분하지만, segmentation 이면 texture 도 필요.
- 따라서 $h$ 가 더 general-purpose.

**한계**: 이는 경험적 관찰이고, 이론적으로 정확히 증명하려면 
downstream task 의 information spectrum 을 알아야 한다.

</details>

### 문제 3 (논문 비평): SimCLR vs MoCo — Batch Size Dependency

Chen et al. (SimCLR, 2020) 는 large batch (8K) 가 필수라고 했지만,
He et al. (MoCo, 2020) 는 queue 를 사용해 작은 batch 에서도 잘 작동한다고 했다.

정보이론적 관점에서, 왜 queue 가 large batch 를 대체할 수 있는지 설명하시오.
두 방식의 장단점은?

<details>
<summary>해설</summary>

**SimCLR (Large Batch)**:
- Negative count $N = 2(B-1)$
- 정리 3.5: $I \geq \log N$ (Poole bound)
- $B = 4096$ → $N = 8190$ → $\log N ≈ 8.9$

**MoCo (Queue)**:
- Maintain dictionary (queue) of $K = 65536$ keys from past batches.
- Current batch 의 query 와 queue 의 keys 를 비교.
- Negative count $N = K ≈ 65536$ → $\log N ≈ 11$
- Batch size 는 256 (SimCLR 의 32배 작음)

**관점**:
- Poole bound 는 N 에만 의존.
- Queue 가 N 을 크게 유지 → small batch 에서도 큰 MI lower bound.

**장단점**:

| 항목 | SimCLR | MoCo |
|------|--------|------|
| Batch size | 필수로 큼 (8K) | 작아도 됨 (256) |
| Memory | high | moderate (queue buffer) |
| Momentum encoder | 필요 없음 | 필수 ($\tau=0.999$ update) |
| 수렴 속도 | 빠름 | 느림 (EMA inertia) |
| Consistency | high (fresh negatives) | good (queue freshness) |

**결론**: Trade-off. SimCLR 은 parallel computing (많은 GPU) 에, 
MoCo 는 single machine (fewer GPU) 에 적합.

</details>

---

<div align="center">

[◀ 이전](./01-ssl-paradigms.md) | [📚 README](../README.md) | [다음 ▶](./03-infonce-mi-bound.md)

</div>
