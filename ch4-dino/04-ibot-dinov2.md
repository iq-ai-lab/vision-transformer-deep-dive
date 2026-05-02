# 04. iBOT 와 DINOv2 (Oquab 2024)

## 🎯 핵심 질문

- **iBOT** (image BERT) 는 DINO 에 어떤 새로운 mechanism 을 더했는가?
- **Masked image modeling** 과 self-distillation 을 joint 하면 어떤 이득이 있는가?
- **KoLeo 정규화** 는 왜 representation collapse 를 방지하는가?
- **DINOv2** 가 supervised ImageNet-21k 능력에 도달한 비결은?

## 🔍 왜 iBOT 에서 DINOv2 로 진화했는가

DINO (2021) 의 성공 이후, 두 가지 질문이 남았다:

1. **Unsupervised learning 의 한계**: Linear probe 77.1% vs supervised 81.8%
2. **Scalability**: 더 큰 모델, 더 큰 데이터셋으로 나아갈 때 무엇이 필요한가?

iBOT (2022) + DINOv2 (2024) 는 이를 해결한다:

- **iBOT**: DINO + masked image modeling (MIM)
  - Student 가 masked patches 의 teacher output 을 예측
  - Self-supervised signal 추가 → representation 개선
  
- **DINOv2**: iBOT + data curation + KoLeo + model scaling
  - 1.1B parameter ViT-g
  - 142M curated images (LVD)
  - **최종 결과**: Linear probe 86% (supervised 능가)

## 📐 수학적 선행 조건

- **DINO fundamentals** (Ch4-01, 02, 03)
- **Masked language modeling** (BERT, Ch5-01)
- **Regularization** : KL divergence, spectral methods
- **Clustering** : prototypes, codebook quantization
- **Data curation** : large-scale vision datasets

## 📖 직관적 이해

```
┌─────────────────────────────────────────────────────────────┐
│  DINO Evolution: DINO → iBOT → DINOv2                      │
│                                                             │
│  DINO (2021):                                              │
│  ├─ Multi-crop (global + local)                            │
│  ├─ EMA teacher                                            │
│  └─ Centering + sharpening                                 │
│     Result: 77.1% linear probe                             │
│                                                             │
│  iBOT (2022):                                              │
│  ├─ DINO backbone                                          │
│  ├─ + Masked image modeling                                │
│  │   (mask 30% of patches, predict teacher tokens)         │
│  ├─ + KoLeo regularizer                                    │
│  │   (maximize feature diversity)                          │
│  └─ Result: 83.2% linear probe                             │
│                                                             │
│  DINOv2 (2024):                                            │
│  ├─ iBOT core                                              │
│  ├─ + Model scaling (ViT-g 1.1B)                           │
│  ├─ + Data curation (LVD 142M images)                      │
│  ├─ + Cluster regularization (SwAV-style prototypes)       │
│  └─ Result: 86% linear probe (≈ supervised!)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

iBOT 의 핵심:
  
  Original image + Augmentation
        ↓
  Patch masking (30% of patches)
        ↓
  [Student Head]                [Teacher Head]
  - Visible patches            - All patches (unmasked)
  - Masked patches → predict   - EMA updated
        ↓                              ↓
  Loss = L_DINO + L_masked_prediction
        └─ Two signals working together
        
  + KoLeo:
    maximize ||z_i - z_j|| for all i ≠ j
    → prevent feature collapse
    → 새로운 collapse mode 방지

DINOv2 의 추가요소:

  Data Curation:
  - Web 에서 142M 고품질 이미지 선별
  - CLIP vision encoder 로 filtering
  - 다양한 도메인 (documents, scenes, objects, etc.)
  
  Cluster Regularization:
  - Prototypes: L2-normalized centers
  - 각 sample → nearest K prototypes 과 match
  - Smooths optimization → better generalization
```

## ✏️ 엄밀한 정의

### 정의 4.13: iBOT — Masked Image Modeling with Distillation

Input: Image $x$, multi-crop (2 global + 8 local).

**Step 1**: Random masking 적용 (30% of patches):
$$M \sim \text{Bernoulli}(0.3) \in \{0, 1\}^{HW}$$

**Step 2**: Patch-level output:
- Student: $\mathbf{z}_s^{(i)} = h_s(g(x_i))$, $i \in$ all crops
- Teacher: $\mathbf{z}_t^{(i)} = h_t(g(x_i))$, $i \in$ globals only

**Step 3**: Two objectives:

DINO loss (정의 4.4, 이전):
$$\mathcal{L}_{\text{DINO}} = -\sum_{x_t \in V_g} \sum_{x_s \in V, x_s \neq x_t} p_t(x_t)^\top \log p_s(x_s)$$

Masked patch prediction loss:
$$\mathcal{L}_{\text{masked}} = -\sum_{i: M_i=1} \mathbf{z}_t^{(i)} \cdot \mathbf{z}_s^{(i)} / (\|\mathbf{z}_t^{(i)}\| \|\mathbf{z}_s^{(i)}\|)$$

(Cosine similarity between student masked output and teacher)

**Total loss**:
$$\mathcal{L}_{\text{iBOT}} = \alpha \mathcal{L}_{\text{DINO}} + (1-\alpha) \mathcal{L}_{\text{masked}}$$

보통 $\alpha = 0.5$ (equal weighting).

### 정의 4.14: KoLeo Regularizer

**주장**: Output 의 diversity 를 maximize 하는 정규화.

$$\mathcal{R}_{\text{KoLeo}} = -\frac{1}{B^2} \sum_{i=1}^B \log \min_{j \neq i} \| \mathbf{z}_i - \mathbf{z}_j \|_2$$

여기서:
- $\mathbf{z}_i$: student output for sample $i$
- $\min_{j \neq i}$: nearest neighbor distance
- $\log$: takes logarithm

**해석**: 각 sample 이 nearest neighbor 와 최대한 멀어지도록 강제.

이를 모든 샘플에 대해 평균.

**효과**:
- One-hot / uniform collapse 방지
- Feature space 를 균등하게 활용

**Combined loss**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{iBOT}} + \lambda \mathcal{R}_{\text{KoLeo}}$$

보통 $\lambda = 0.005$.

### 정의 4.15: SwAV-style Cluster Prototypes (DINOv2 추가)

Prototypes: $\mathbf{w}_1, \ldots, \mathbf{w}_K \in \mathbb{R}^D$ (learnable, L2-normalized).

Output 에서:
$$s_k = \mathbf{z}^\top \mathbf{w}_k / \tau_c$$

Optimal transport (soft assignment):
$$q_k = \frac{\exp(s_k / \tau_{\text{ot}})}{\sum_k \exp(s_k / \tau_{\text{ot}})}$$

Loss (cross-entropy on assignment):
$$\mathcal{L}_{\text{cluster}} = -\sum_k q_k \log q_k'$$

여기서 $q_k'$ 는 "previous iteration 의 assignment" (consistency).

**효과**: Clustering loss 가 regularization 역할.

### 정의 4.16: Data Curation for DINOv2

**LVD (Large Vision Dataset)** = 142M curated images.

선별 기준:
1. 해상도 ≥ 256px
2. CLIP ViT-L 로 계산한 feature quality (높은 semantic 신호)
3. 다양성 보장 (도메인 균형)
4. 중복 제거 (near-duplicate removal)

결과: ImageNet-1k (1.2M) 의 ~100배 크기, 
하지만 훨씬 높은 질.

## 🔬 정리와 증명

### 정리 4.9: Masked Image Modeling 이 representation 을 개선하는 이유

**주장**: Masked patch prediction 은 explicit spatial reasoning 을 강제하여, 
DINO-only 보다 더 robust features 를 학습한다.

**증명 스케치**:

DINO 만 있으면:
- Multi-crop consistency loss
- Patch-level spatial structure 는 implicit

Masked patch prediction 추가:
- Student 가 masked patches 를 예측
- Teacher 가 전체 image context 를 본 출력을 target

=> Student 는 **surrounding context 로부터 masked patch 의 의미를 유추** 해야 함.

이는:
1. **Local-global alignment** (DINO)
2. **Context-based inpainting** (MIM)

두 가지 constraint 를 동시에 적용.

Formally, optimization landscape:
- DINO-only: critical point set $\mathcal{C}_{\text{DINO}}$
- iBOT: critical point set $\mathcal{C}_{\text{iBOT}} = \mathcal{C}_{\text{DINO}} \cap \mathcal{C}_{\text{MIM}}$

Intersection 은 더 작은 set 이므로, generalization 이 더 좋을 가능성.

Empirically (대리 이론):
$$\text{rank}(\nabla^2 \mathcal{L}_{\text{iBOT}}) < \text{rank}(\nabla^2 \mathcal{L}_{\text{DINO}})$$

=> 더 낮은 effective dimension → better transfer learning.

$$\square$$

### 정리 4.10: KoLeo 정규화의 효과

**주장**: KoLeo 는 nearest neighbor distance 를 최대화함으로써, 
feature space 의 "dead zones" 를 제거한다.

**증명**:

KoLeo loss:
$$\mathcal{R}_{\text{KoLeo}} = -\frac{1}{B} \sum_{i=1}^B \log \min_{j \neq i} d_{ij}$$

Gradient (w.r.t. $\mathbf{z}_i$):
$$\frac{\partial \mathcal{R}}{\partial \mathbf{z}_i} \propto -\frac{\nabla d_{i,j^*}}{\text{NN distance}_{i,j^*}}$$

여기서 $j^* = \arg\min_j d_{ij}$ (nearest neighbor).

이는:
- 만약 $d_{i,j^*}$ 가 작으면, gradient magnitude 가 크다.
- 강제로 $\mathbf{z}_i$ 를 $\mathbf{z}_{j^*}$ 에서 멀어지도록.

결과:
- Feature space 에서 과도하게 집중된 영역 제거
- Uniform coverage 유도

**Collapse prevention**:

정의 4.6 (one-dimensional collapse): $\mathbf{z}_i \approx \alpha_i \mathbf{v}$

이 경우, nearest neighbor distance:
$$\min_{j \neq i} \|\alpha_i \mathbf{v} - \alpha_j \mathbf{v}\| = |\alpha_i - \alpha_j| \|\mathbf{v}\|$$

KoLeo 가 이를 최대화하려고 하면, $|\alpha_i - \alpha_j|$ 를 spread 해야 하는데,
이는 결국 모든 $\alpha_i$ 가 선 위에 분산되도록 강제한다.

비효율적이므로, network 가 다른 방향으로 feature 를 spread 시키게 된다.

=> **Rank 증가, collapse 회피**.

$$\square$$

### 정리 4.11: Scaling Laws in DINOv2

**주장**: Model size 와 data size 를 동시에 증가시키면, 
선형 probe accuracy 는 지속적으로 향상된다 (saturation 없음).

**증명** (경험적 관찰):

| 모델 | Data | Linear Probe | 
|-----|------|-------------|
| ViT-S (22M) | ImageNet-1k | 77.1% |
| ViT-B (87M) | ImageNet-1k | 78.2% |
| ViT-L (307M) | ImageNet-1k | 79.0% |
| ViT-g (1.1B) | LVD-142M | 86.0% |

관찰:
- Model scaling 만: diminishing returns
- Data 추가 (but same distribution): plateau
- **Model + Data scaling jointly**: continued improvement

이는 다음을 시사:
$$\text{Accuracy} \propto \log(\text{model size}) + \log(\text{data size})$$

(Chinchilla scaling law 와 유사)

따라서:
- 단순히 "bigger model" 이 아니라, "well-curated large data" 가 critical.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Masked Image Modeling Loss 구현

```python
import torch
import torch.nn.functional as F

class MaskedPredictionLoss(torch.nn.Module):
    """iBOT 의 masked patch prediction loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, z_student, z_teacher, mask):
        """
        z_student: [B*N, D] - student outputs (all crops)
        z_teacher: [B*G, D] - teacher outputs (globals only)
        mask: [B*N, P] - binary mask (1 = masked, 0 = visible)
        """
        # Extract masked predictions
        z_s_masked = z_student[mask.bool()]  # [num_masked, D]
        z_t_masked = z_teacher[mask.bool()]  # [num_masked, D]
        
        # Normalize
        z_s_masked = F.normalize(z_s_masked, dim=-1)
        z_t_masked = F.normalize(z_t_masked, dim=-1)
        
        # Cosine similarity (positive = should be high)
        sim = (z_s_masked * z_t_masked).sum(dim=-1)
        
        # Loss: maximize cosine similarity
        loss = (1 - sim).mean()
        
        return loss

# Test
B = 32
N = 10  # crops
D = 256
P = 196  # patches

z_s = torch.randn(B * N, D)
z_t = torch.randn(B * 2, D)  # Only 2 globals
mask = torch.bernoulli(torch.full((B * N, P), 0.3))  # 30% masked

# Replicate z_t for all crops (simplified)
z_t_expanded = torch.cat([z_t] * (N // 2), dim=0)

loss_fn = MaskedPredictionLoss()
loss = loss_fn(z_s, z_t_expanded, mask)
print(f"Masked prediction loss: {loss.item():.4f}")
```

### 실험 2: KoLeo Regularizer 구현

```python
import torch

class KoLeoRegularizer(torch.nn.Module):
    """KoLeo regularization: maximize nearest neighbor distance"""
    def __init__(self):
        super().__init__()
    
    def forward(self, z):
        """
        z: [B, D] - batch of features
        returns: KoLeo loss (lower is better for minimization)
        """
        # Compute pairwise distances
        # z: [B, D]
        # dist: [B, B]
        dist_sq = torch.cdist(z, z, p=2) ** 2
        
        # Set diagonal to infinity (don't compare with self)
        dist_sq.fill_diagonal_(float('inf'))
        
        # Find minimum distance for each sample
        min_dist_sq, _ = torch.min(dist_sq, dim=1)
        min_dist = torch.sqrt(min_dist_sq + 1e-8)
        
        # KoLeo: -log(min_dist)
        # Negative log encourages large min_dist
        koleo = -torch.log(min_dist + 1e-8).mean()
        
        return koleo

# Test
z = torch.randn(64, 256)
z = F.normalize(z, dim=-1)

koleo_fn = KoLeoRegularizer()
koleo_loss = koleo_fn(z)

print(f"KoLeo loss: {koleo_loss.item():.4f}")
print(f"(Lower is better; encourages feature diversity)")
```

### 실험 3: Cluster Prototype Loss (DINOv2 스타일)

```python
import torch
import torch.nn.functional as F

class ClusterPrototypeLoss(torch.nn.Module):
    """SwAV-style cluster regularization"""
    def __init__(self, n_prototypes=256, feat_dim=256):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.register_buffer('prototypes', torch.randn(n_prototypes, feat_dim))
        self.prototypes.data = F.normalize(self.prototypes, dim=-1)
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
    
    def forward(self, z, tau_c=0.1, tau_ot=0.07):
        """
        z: [B, D] - batch features (normalized)
        tau_c: temperature for softmax
        tau_ot: temperature for optimal transport
        """
        # Similarity to prototypes
        sim = torch.mm(z, self.prototypes.t()) / tau_c  # [B, K]
        
        # Soft assignment
        q = F.softmax(sim / tau_ot, dim=1)  # [B, K]
        
        # Consistency loss: encourage consistent assignments
        # (in practice, compare with previous iteration q_prev)
        # For simplicity, use self-consistency here
        
        p = F.softmax(sim, dim=1)  # [B, K]
        
        # Cross-entropy
        loss = -(p * F.log_softmax(sim, dim=1)).sum(dim=1).mean()
        
        return loss

# Test
z = torch.randn(64, 256)
z = F.normalize(z, dim=-1)

cluster_fn = ClusterPrototypeLoss(n_prototypes=256, feat_dim=256)
cluster_loss = cluster_fn(z)

print(f"Cluster loss: {cluster_loss.item():.4f}")
print(f"(Regularization for consistent clustering)")
```

### 실험 4: Multi-Objective Ablation

```python
import torch
import torch.nn.functional as F

def compute_linear_probe_accuracy(features, labels, epochs=10):
    """
    Simplified linear probe: train linear layer on frozen features
    """
    D = features.shape[1]
    n_classes = labels.max().item() + 1
    
    W = torch.randn(D, n_classes, requires_grad=True)
    b = torch.randn(n_classes, requires_grad=True)
    
    optimizer = torch.optim.SGD([W, b], lr=0.01)
    
    for _ in range(epochs):
        logits = features @ W + b
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Evaluate
    with torch.no_grad():
        logits = features @ W + b
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
    
    return acc

# Simulate different feature quality
n_samples = 1000
feat_dim = 256
n_classes = 10

labels = torch.randint(0, n_classes, (n_samples,))

# Scenario 1: DINO only
print("=" * 50)
print("Ablation Study: DINO vs iBOT vs DINOv2")
print("=" * 50)

feats_dino = torch.randn(n_samples, feat_dim) * 0.5
for c in range(n_classes):
    mask = labels == c
    feats_dino[mask] += torch.randn(feat_dim) * 1.5

acc_dino = compute_linear_probe_accuracy(feats_dino, labels)
print(f"DINO (multi-crop only): {acc_dino:.3f}")

# Scenario 2: iBOT (DINO + MIM + KoLeo)
# More structured features due to additional losses
feats_ibot = feats_dino.clone()
# Simulate better clustering
for c in range(n_classes):
    mask = labels == c
    # Tighter clusters
    feats_ibot[mask] = (feats_ibot[mask] + 2 * torch.randn(feat_dim)) / 1.5

acc_ibot = compute_linear_probe_accuracy(feats_ibot, labels)
print(f"iBOT (DINO + MIM + KoLeo): {acc_ibot:.3f}")

# Scenario 3: DINOv2 (iBOT + scaling + curation)
# Even better features: larger effective dimension
feats_dinov2 = torch.randn(n_samples, feat_dim) * 0.3
for c in range(n_classes):
    mask = labels == c
    # Very tight, well-separated clusters
    feats_dinov2[mask] += torch.randn(feat_dim) * 3.0

acc_dinov2 = compute_linear_probe_accuracy(feats_dinov2, labels)
print(f"DINOv2 (full setup): {acc_dinov2:.3f}")

print("\nConclusion: Each component improves feature quality")
```

## 🔗 실전 활용

### iBOT 훈련 파이프라인

```
1. Load DINO backbone (pretrained or random init)
2. Per-batch loop:
   a. Generate multi-crop (2 global + 8 local)
   b. Apply random masking (30% of patches)
   c. Forward student (all crops + masks)
   d. Forward teacher (globals only)
   e. Compute L_DINO (정의 4.4)
   f. Compute L_masked (정의 4.13)
   g. Compute R_KoLeo (정의 4.14)
   h. Total loss = α*L_DINO + (1-α)*L_masked + λ*R_KoLeo
   i. Backward & optimize
   j. Update teacher via EMA
3. After training: Linear probe / k-NN evaluation
```

### DINOv2 최적화 팁

| 하이퍼파라미터 | 권장값 | 이유 |
|-------------|-------|------|
| Masking ratio | 30% | BERT 관례, DINO 와 유사 |
| α (DINO vs MIM weight) | 0.5 | 균형 |
| λ (KoLeo weight) | 0.005 | Subtle regularization |
| Prototype count | 256-1024 | 충분한 diversity |
| Model size | ≥ ViT-B | Scaling law 의존 |
| Data size | ≥ 50M | Curated, high-quality |

### 성능 비교

| 모델 | 데이터 | Linear Probe | 논평 |
|-----|------|-----------|------|
| DINO ViT-B | ImageNet-1k | 78.2% | Baseline |
| iBOT ViT-B | ImageNet-1k | 83.2% | +5% from MIM |
| DINOv2 ViT-B | LVD-142M | 85.1% | +2% from curation |
| DINOv2 ViT-g | LVD-142M | 86.0% | +0.9% from scale |
| Supervised ViT-L | ImageNet-21k | 88.6% | Ceiling (다른 방법) |

**핵심**: DINOv2 ≈ supervised 성능, **unsupervised 로는 기록적**.

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Mask ratio** | 30% 가 "optimal" | Task 별로 다름; 예: 15% (documents) |
| **KoLeo 안정성** | Nearest neighbor distance 기반 | Batch size 작으면 noisy |
| **Prototype count** | K=256 가 충분 | 더 크면 computational cost ↑ |
| **Data curation** | LVD 는 proprietary | Open-source alternative (CC12M, LAION) 사용 가능 |
| **Scaling assumption** | Chinchilla scaling law 가정 | Edge case 에서 정확도 떨어짐 |
| **Compute requirement** | ViT-g 훈련: 대규모 GPU 클러스터 필요 | 2048 A100 GPU-days |

## 📌 핵심 정리

$$\boxed{\mathcal{L}_{\text{iBOT}} = \alpha \mathcal{L}_{\text{DINO}} + (1-\alpha) \mathcal{L}_{\text{masked}} + \lambda \mathcal{R}_{\text{KoLeo}}}$$

$$\boxed{\mathcal{R}_{\text{KoLeo}} = -\frac{1}{B} \sum_{i=1}^B \log \min_{j \neq i} \| \mathbf{z}_i - \mathbf{z}_j \|_2}$$

| 요소 | 역할 | 이득 |
|------|------|------|
| **L_DINO** | Multi-crop consistency | View-invariance |
| **L_masked** | Context-based MIM | Spatial reasoning |
| **R_KoLeo** | Diversity regularization | Collapse prevention |
| **Model scaling** | ViT-S → ViT-g | +9% accuracy |
| **Data curation** | Quality over quantity | +3% from LVD |

## 🤔 생각해볼 문제

### 문제 1 (기초): 왜 30% masking ratio 를 선택했는가?

BERT 는 15%, MAE 는 75% 인데, iBOT 는 30%?

<details>
<summary>해설</summary>

Trade-off:
- Ratio 너무 낮음 (< 10%): reconstruction 이 너무 쉬움 → 신호 약함
- Ratio 너무 높음 (> 80%): student 가 context 로부터 추론 힘듦 → 불가능한 task

iBOT 의 30% 는:
- DINO 의 self-supervised signal 과 compatible
- MIM 신호가 "가능하지만 도전적"
- Cross-validation 으로 optimal 

domain 별로 다름:
- Natural images: 30% optimal
- Document images: 15% (높은 redundancy)
- Scenes: 40% (더 복잡한 reasoning)

</details>

### 문제 2 (심화): KoLeo vs Batch Normalization

KoLeo 는 nearest neighbor distance 를 maximize 하는데,
Batch Norm 도 feature 를 normalize/spread 하지 않나?

<details>
<summary>해설</summary>

**Batch Norm:**
- Per-feature normalization (mean=0, std=1)
- **sample-wise** diversity 는 보장하지 않음
- 같은 값을 여러 samples 이 가질 수 있음

**KoLeo:**
- Per-sample regularity (nearest neighbor distance)
- Sample space 전체에서 spread
- Batch Norm 과 orthogonal 한 목표

결합:
- Batch Norm: feature dimension scale 조정
- KoLeo: sample space 에서 coverage

실제로 DINOv2 에서:
- Batch Norm 유지
- KoLeo 추가
- 둘 다 필요

</details>

### 문제 3 (논문 비평): LVD 데이터 캐레이션의 타당성

DINOv2 의 성공이 단순히 "더 많은 데이터" 때문 아닌가?
알고리즘 개선인가, 데이터 개선인가?

<details>
<summary>해설</summary>

**분리 분석** (DINOv2 논문):

| 설정 | Linear Probe |
|-----|------------|
| iBOT + ImageNet-1k | 83.2% |
| iBOT + LVD (uncurated) | 84.5% |
| iBOT + LVD (curated) | 85.2% |
| DINOv2 (iBOT + curation + cluster) | 86.0% |

결론:
- Curation 효과: +0.7%
- Clustering 추가: +0.8%
- 알고리즘 개선: +2.0% (total)
- 데이터 개선: +2.0% (LVD vs 1k)

=> **Data 와 Algorithm 이 동등하게 중요**.

더 나아가:
- Large-scale "random web" 데이터: saturates at ~80%
- Curated high-quality data: continues to improve

Practical implication:
- 단순히 "bigger" 가 아니라 "better curated" 중요.

</details>

---

<div align="center">

[◀ 이전](./03-emergent-properties.md) | [📚 README](../README.md) | [다음 ▶](../ch5-masked-image-modeling/01-beit.md)

</div>
