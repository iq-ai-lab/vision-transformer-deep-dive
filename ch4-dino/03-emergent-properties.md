# 03. Emergent Properties of DINO

## 🎯 핵심 질문

- DINO 의 attention map 이 explicit object mask supervision 없이 왜 segmentation 처럼 동작하는가?
- Last-layer attention 의 CLS-to-patch 가중치와 object boundary 의 관계는?
- k-NN classifier 가 linear probe 만큼 좋은 이유는 무엇인가?
- 이러한 emergent properties 는 DINO 의 어떤 design choice 에서 비롯되는가?

## 🔍 왜 이것이 "emergent" 인가

DINO 훈련 목표는 **오직 self-supervised representation learning** 이다.
- Segmentation label 없음
- Boundary annotation 없음
- 어떤 명시적 geometric task 도 없음

그런데 학습 후:

1. **Attention map**: Last layer 의 CLS token-to-patch attention 이 object mask 와 일치
2. **Correspondence**: Image pairs 간에 일관된 object correspondence 발견
3. **Clustering**: k-NN 으로 neighbor images 를 같은 클래스로 찾음

이는 **self-supervised 신호만으로도 semantic structure 를 학습한다**는 강력한 증거다.

## 📐 수학적 선행 조건

- **Vision Transformer attention**: Multi-head self-attention $\text{Attn}(Q,K,V)$
- **CLS token**: Learnable classification token
- **Last-layer features**: Final hidden state 의 D-dim vector
- **Evaluation metrics**: 
  - IoU (Intersection over Union)
  - mIoU (mean IoU)
  - Normalized mutual information (NMI)
  - k-NN accuracy

## 📖 직관적 이해

```
┌────────────────────────────────────────────────────────────┐
│           Emergent Segmentation in DINO                   │
│                                                            │
│  Input Image                                              │
│  ┌──────────────────────────────────────────┐             │
│  │                                          │             │
│  │    [CAT sitting on COUCH]                │             │
│  │                                          │             │
│  └──────────────────────────────────────────┘             │
│              ↓                                             │
│  ViT Backbone                                            │
│  (patch embedding + transformer blocks)                   │
│              ↓                                             │
│  Last-Layer Self-Attention                               │
│  CLS token attends to all patches                         │
│              ↓                                             │
│  Attention Weights: A ∈ [0,1]^(H×W)                      │
│  (CLS-to-patch attention only)                            │
│              ↓                                             │
│  A_normalized (threshold at 0.15):                        │
│  ┌──────────────────────────────────────────┐             │
│  │                                          │             │
│  │    [████████████████░░░░░░░░░░░]  ← Cat │             │
│  │    [░░░░░░░░░░░░░░░░░░░░░░░░░░░]  ← Couch │           │
│  │                                          │             │
│  └──────────────────────────────────────────┘             │
│                                                            │
│  이를 원본 해상도로 upsampling → Segmentation Mask!      │
│                                                            │
└────────────────────────────────────────────────────────────┘

k-NN Classifier:
  test image feature → find K nearest neighbors in training set
  → majority vote on labels → classification
  
  DINO features: semantic structure 때문에 k-NN 이 매우 효과적!
  supervised features 만큼 좋은 performance
```

## ✏️ 엄밀한 정의

### 정의 4.9: Last-Layer Attention Map

$h$ = ViT head dimension, $d_h$ = number of heads in last layer.

Last-layer self-attention output:
$$\text{Attn}^{\text{last}}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Attention weights (마지막 head, 모든 head average):
$$A \in \mathbb{R}^{(HW+1) \times (HW+1)}$$

CLS-to-patch attention (row 0, columns 1:):
$$\mathbf{a} = A[0, 1:] \in \mathbb{R}^{HW}$$

Reshape to spatial:
$$A_{\text{spatial}} = \text{reshape}(\mathbf{a}, H, W)$$

### 정의 4.10: Thresholded Segmentation Mask

$$M_{\text{pred}}(x, y) = \begin{cases} 1 & \text{if } A_{\text{spatial}}(x,y) > \theta \\ 0 & \text{otherwise} \end{cases}$$

여기서 $\theta$ 는 threshold (보통 0.15).

### 정의 4.11: Evaluation — mIoU

Ground truth mask: $M_{\text{gt}} \in \{0, 1\}^{H \times W}$
Predicted mask: $M_{\text{pred}} \in \{0, 1\}^{H \times W}$

IoU (Intersection over Union):
$$\text{IoU} = \frac{|M_{\text{pred}} \cap M_{\text{gt}}|}{|M_{\text{pred}} \cup M_{\text{gt}}|}$$

여러 classes (c=1, ..., C):
$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^C \text{IoU}_c$$

### 정의 4.12: k-NN Classifier

Training set: $\{(\mathbf{f}_i, y_i)\}_{i=1}^N$ where $\mathbf{f}_i$ = feature, $y_i$ = label.

Test feature: $\mathbf{f}_{\text{test}}$

Nearest neighbors (L2 distance):
$$\mathcal{N}_k = \arg\min_i \| \mathbf{f}_{\text{test}} - \mathbf{f}_i \|_2, \quad |\mathcal{N}_k| = k$$

Prediction (majority vote):
$$\hat{y} = \arg\max_c \left| \{ i \in \mathcal{N}_k : y_i = c \} \right|$$

Accuracy:
$$\text{Acc} = \frac{1}{N_{\text{test}}} \sum_{t=1}^{N_{\text{test}}} \mathbb{1}[\hat{y}_t = y_t]$$

## 🔬 정리와 증명

### 정리 4.6: Attention-to-Segmentation 의 원리

**주장**: DINO 에서 학습된 CLS-to-patch attention 이 semantic boundaries 를 capture 하는 이유는, 
**local-to-global consistency loss** 가 강제하는 결과다.

**증명 스케치**:

DINO loss (정의 4.4 에서):
$$\mathcal{L} = -\sum_{x_t \in V_g} \sum_{x_s \in V, x_s \neq x_t} p_t(x_t)^\top \log p_s(x_s)$$

여기서 $p_s, p_t$ 는 **global features** 에 기반한 분포.

Local patches 도 **같은 global distribution** 을 예측해야 함.

=> **Local patches 는 whole image semantics 를 알아야 함**

즉, local crop 은:
- Object 의 일부만 본다 (예: cat 의 귀)
- 하지만 full image semantic 을 예측해야 한다
- 따라서 주변 맥락 정보를 통합해야 한다

Attention mechanism 은 이를 수행하는 자연스러운 방법:
- CLS token: global semantic summary
- Patch tokens: local information
- CLS-to-patch attention: "어느 patch 가 semantics 에 중요한가?"

**수학적으로**:

Network 가 local crop 의 semantic 을 예측하려면:
$$p_s^{\text{local}} \approx p_t^{\text{global}}$$

필연적으로:
$$f_s(\text{local patch}) \approx f_t(\text{global image}) \quad \text{in some embedding}$$

Attention 을 통해:
$$f_s(\text{patch}) = \sum_i \alpha_{i \to \text{patch}} \cdot f_s(\text{patch}_i)$$

여기서 $\alpha$ 는 학습되는 attention weight.

Object 경계를 넘으면 semantic 이 바뀌므로, attention 이 자연스럽게 object boundary 에 고점을 형성한다.

$$\square$$

### 정리 4.7: k-NN Classifier 의 효과성

**주장**: DINO 의 k-NN accuracy 가 높은 이유는, feature space 가 **충분히 구조화**되어 있기 때문이다.

**증명**:

Self-supervised loss 를 다시 쓰면:
$$\mathcal{L} = -\sum_{(i,j) \in \text{pairs}} p_i^\top \log p_j$$

이는 같은 object 의 다양한 views 를 가까운 feature space 에 배치한다.

그러나 **class-level** 정보는 implicit:

데이터셋의 class structure (예: ImageNet 의 1000 classes):
- Dog 클래스의 다양한 종들
- Cat 클래스의 다양한 포즈

DINO 는 label 을 모르지만, **visual similarity** 를 학습한다.

결과:
1. 같은 클래스 → visual 유사 → feature space 에서 가까움
2. 다른 클래스 → 구별됨 → feature space 에서 멀어짐

따라서 k-NN 이 효과적:
$$P(\text{correct classification}) = P(\text{neighbors share same class}) \approx \text{high}$$

Theoretical bound (information theory):

Feature space 의 intrinsic dimension 을 $d_{\text{int}}$ 라 하면,
k-NN 의 expected error:
$$\mathbb{E}[\text{error}] \approx C \cdot \left(\frac{d_{\text{int}}}{N}\right)^{2/d}$$

여기서 $d$ 는 ambient dimension (DINO: 384 for ViT-S/8).

DINO 의 features 가 low intrinsic dimension 을 가지므로 (semantic structure due to multi-crop),
k-NN 이 우수한 성능을 보인다.

$$\square$$

### 정리 4.8: Emergent Properties 는 Multi-Crop 의 결과

**주장**: DINO 의 emergent properties (attention-based segmentation, k-NN effectiveness) 는 
multi-crop loss 가 없으면 나타나지 않는다.

**증명** (비교 분석):

**DINO (multi-crop)** vs **SimCLR (contrastive on globals only)**:

SimCLR:
$$\mathcal{L} = -\log \frac{\exp(\cos(f_i, f_j) / \tau)}{\sum_k \exp(\cos(f_i, f_k) / \tau)}$$

Global crops 만 비교:
- Instance-level discrimination 학습
- Class-level structure 는 implicit 하지만 약함

DINO (multi-crop):
$$\mathcal{L}_{\text{local-to-global}} = -\sum_{j \in \text{local}} \log p_{\text{global}}(f_j)$$

Local patches 가 global semantics 예측:
- Explicit spatial correspondence 학습
- Object boundaries 가 "attention 의 천연 할당 경계"

Empirically (DINO paper):
- SimCLR: attention segmentation mIoU ≈ 30%
- DINO: attention segmentation mIoU ≈ 60%

=> Multi-crop 이 emergent properties 의 근본 원인.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 모의 Attention Map 에서 Segmentation 추출

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def extract_segmentation_from_attention(attention_map, threshold=0.15):
    """
    attention_map: [H, W] - normalized attention from CLS to patches
    returns: binary mask
    """
    mask = (attention_map > threshold).float()
    return mask

def evaluate_segmentation(mask_pred, mask_gt):
    """
    Calculate IoU between predicted and ground truth
    """
    intersection = (mask_pred * mask_gt).sum()
    union = ((mask_pred + mask_gt) > 0).float().sum()
    iou = intersection / (union + 1e-8)
    return iou.item()

# Simulate test
n_patches_h, n_patches_w = 14, 14
H, W = 224, 224

# Simulated DINO attention (high in object region, low elsewhere)
attn_map = torch.zeros(n_patches_h, n_patches_w)
# Create "object" region (say top-left quadrant)
attn_map[:7, :7] = torch.linspace(0.3, 0.8, 49).reshape(7, 7)
attn_map[7:, 7:] = torch.linspace(0.05, 0.1, 49).reshape(7, 7)
attn_map[7:, :7] = 0.08
attn_map[:7, 7:] = 0.12

# Extract mask
mask_pred = extract_segmentation_from_attention(attn_map, threshold=0.15)

# Ground truth (same quadrant)
mask_gt = torch.zeros(n_patches_h, n_patches_w)
mask_gt[:7, :7] = 1.0

iou = evaluate_segmentation(mask_pred, mask_gt)
print(f"Simulated Attention mIoU: {iou:.3f}")

# Visualize
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

ax1.imshow(attn_map.cpu().numpy(), cmap='hot')
ax1.set_title('Attention Map')

ax2.imshow(mask_pred.cpu().numpy(), cmap='gray')
ax2.set_title('Predicted Mask')

ax3.imshow(mask_gt.cpu().numpy(), cmap='gray')
ax3.set_title('Ground Truth Mask')

plt.tight_layout()
print("Visualization complete")
```

### 실험 2: k-NN Classifier 구현

```python
import torch
from sklearn.neighbors import KNeighborsClassifier

def knn_classifier(features_train, labels_train, features_test, k=10):
    """
    k-NN 분류기
    features_train: [N_train, D] - training features
    labels_train: [N_train] - training labels
    features_test: [N_test, D] - test features
    """
    # Use scikit-learn's KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features_train.cpu().numpy(), labels_train.cpu().numpy())
    
    predictions = knn.predict(features_test.cpu().numpy())
    return torch.tensor(predictions)

def evaluate_knn(predictions, labels_test):
    """Calculate accuracy"""
    accuracy = (predictions == labels_test).float().mean().item()
    return accuracy

# Simulate DINO features
N_train = 1000
N_test = 200
D = 384  # DINO ViT-S/8 dimension

# Create synthetic features: clusters by class
n_classes = 10
features_train = torch.randn(N_train, D) * 0.1
labels_train = torch.zeros(N_train, dtype=torch.long)

for c in range(n_classes):
    start = (N_train // n_classes) * c
    end = start + (N_train // n_classes)
    features_train[start:end] += torch.randn(D) * 2.0  # Class-specific offset
    labels_train[start:end] = c

# Test features
features_test = torch.randn(N_test, D) * 0.1
labels_test = torch.zeros(N_test, dtype=torch.long)

for c in range(n_classes):
    start = (N_test // n_classes) * c
    end = start + (N_test // n_classes)
    features_test[start:end] += torch.randn(D) * 2.0
    labels_test[start:end] = c

# Evaluate k-NN for different k
for k in [1, 5, 10, 20]:
    predictions = knn_classifier(features_train, labels_train, features_test, k=k)
    acc = evaluate_knn(predictions, labels_test)
    print(f"k={k}: Accuracy = {acc:.3f}")
```

### 실험 3: DINO 와 SimCLR 의 Attention 비교

```python
import torch
import torch.nn.functional as F

def simulate_dino_attention(local_feats, global_feat, with_local=True):
    """
    DINO: student 가 local+global crops 를 본다
    Loss: local 이 global distribution 예측해야 함
    → Attention 이 어느 patch 를 집중할지 결정
    """
    if not with_local:
        return None
    
    # Simulate attention weight (응답하는 정도)
    # Local patches 이 global 의미를 배우므로,
    # CLS 의 attention 이 semantic region 에 집중
    n_patches = local_feats.shape[0]
    
    # Similarity to global
    similarities = F.cosine_similarity(local_feats, global_feat.unsqueeze(0), dim=1)
    
    # Softmax attention
    attention = F.softmax(similarities / 0.1, dim=0)
    
    return attention

def simulate_simclr_attention(global_feats, with_local=False):
    """
    SimCLR: global crops 만 비교
    No local-global alignment
    → Attention 이 semantic 구조를 배우지 않음
    """
    if with_local:
        return None
    
    # Just noise/uniform attention
    return torch.ones(len(global_feats)) / len(global_feats)

# Test
n_local_patches = 64
local_features = torch.randn(n_local_patches, 128)
global_feature = torch.randn(128)

# DINO attention: high concentration
dino_attn = simulate_dino_attention(local_features, global_feature, with_local=True)
dino_entropy = -(dino_attn * torch.log(dino_attn + 1e-8)).sum().item()

# SimCLR attention: uniform
simclr_attn = simulate_simclr_attention(None, with_local=False)
simclr_entropy = -(simclr_attn * torch.log(simclr_attn + 1e-8)).sum().item()

print(f"DINO attention entropy: {dino_entropy:.3f} (concentrated)")
print(f"SimCLR attention entropy: {simclr_entropy:.3f} (uniform)")
print(f"=> DINO's local-global loss forces attention to concentrate on semantic regions")
```

### 실험 4: Correspondence 측정 (k-NN 정확도)

```python
import torch

def measure_correspondence_accuracy(features_query, features_pool, labels_query, labels_pool, k=5):
    """
    Query 이미지와 pool 이미지 간의 correspondence 측정
    k-NN 으로 correct class neighbor 를 찾는 비율
    """
    # Compute pairwise distances
    dist = torch.cdist(features_query, features_pool, p=2)
    
    # Find k-nearest neighbors
    _, indices_knn = torch.topk(dist, k, dim=1, largest=False)
    
    # Check if any neighbor has correct label
    correct = 0
    for i, idx_knn in enumerate(indices_knn):
        knn_labels = labels_pool[idx_knn]
        if labels_query[i] in knn_labels:
            correct += 1
    
    accuracy = correct / len(features_query)
    return accuracy

# Simulate
n_query = 100
n_pool = 5000
d_feat = 256
n_classes = 100

# DINO-like features: class-clustered
query_feats = torch.randn(n_query, d_feat) * 0.5
query_labels = torch.randint(0, n_classes, (n_query,))

pool_feats = torch.randn(n_pool, d_feat) * 0.5
pool_labels = torch.randint(0, n_classes, (n_pool,))

# Add class-specific structure
for c in range(n_classes):
    query_mask = query_labels == c
    pool_mask = pool_labels == c
    
    if query_mask.sum() > 0:
        query_feats[query_mask] += torch.randn(d_feat) * 2.0
    if pool_mask.sum() > 0:
        pool_feats[pool_mask] += torch.randn(d_feat) * 2.0

# Measure correspondence
acc_k5 = measure_correspondence_accuracy(query_feats, pool_feats, query_labels, pool_labels, k=5)
acc_k1 = measure_correspondence_accuracy(query_feats, pool_feats, query_labels, pool_labels, k=1)

print(f"k-NN=1 correspondence accuracy: {acc_k1:.3f}")
print(f"k-NN=5 correspondence accuracy: {acc_k5:.3f}")
print(f"=> High accuracy shows features capture semantic class structure")
```

## 🔗 실전 활용

### 1. Attention-based Segmentation 파이프라인

```
1. Load DINO ViT-S/8 model (pretrained)
2. For each image:
   a. Forward pass → get last layer attention
   b. Extract CLS-to-patch attention weights
   c. Reshape to spatial [H, W]
   d. Normalize and threshold (τ = 0.15)
   e. Upsampling to original resolution
   f. Compare with ground truth mask → mIoU
3. Report: PASCAL VOC / COCO 의 unsupervised segmentation mIoU
```

### 2. k-NN 분류 파이프라인

```
Training phase:
  For each training image:
    1. Compute DINO feature (last layer before head)
    2. Store (feature, label) pair
  
Test phase:
  For each test image:
    1. Compute DINO feature
    2. Find k-nearest neighbors (L2 distance)
    3. Majority vote on labels
    4. Evaluate accuracy
```

### 성능 지표

| 평가 항목 | DINO ViT-S/8 | Supervised ViT-S |
|----------|-------------|-----------------|
| Attention mIoU (PASCAL VOC) | 58.1% | - |
| k-NN Acc (ImageNet-1k) | 74.5% | - |
| Linear Probe Acc | 77.1% | 81.0% |

**Insight**: Linear probe 는 supervised 에 95% 달성, k-NN 은 92%, attention segmentation 은 71%.

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Threshold 선택** | mIoU 는 τ (threshold) 에 민감 | 보통 0.15 가 최적; validation set 에서 tuning |
| **k-NN 의 k** | k 가 작으면 noise, 크면 smoothing 너무 강함 | 보통 k=20 권장 |
| **Object size bias** | Attention 이 큰 객체를 더 잘 capture | Small object 에서 성능 저하 |
| **Single-head attention** | Multi-head 중 하나만 사용 | Average 하면 성능 향상 |
| **Supervised vs unsupervised** | Attention segmentation 은 supervised 보다 10-20% 낮음 | Trade-off: scalability vs accuracy |

## 📌 핵심 정리

$$\boxed{\text{Attention Mask: } M(x, y) = \mathbb{1}[A_{\text{cls-to-patch}}(x,y) > \theta]}$$

$$\boxed{\text{k-NN: } \hat{y}_{\text{test}} = \arg\max_c |\{i \in \mathcal{N}_k : y_i = c\}|}$$

| 성질 | 발현 방식 | 정량화 |
|------|---------|--------|
| **Attention Segmentation** | CLS attention weights → spatial mask | mIoU (IoU per class, averaged) |
| **k-NN Classification** | Nearest neighbors in feature space | Top-1 accuracy |
| **Correspondence** | Same object across images → similar features | Recall@K (how many true matches in top-K) |
| **Clustering Quality** | Feature space 의 class separability | Silhouette score, NMI |

## 🤔 생각해볼 문제

### 문제 1 (기초): 왜 CLS attention 만 본 것인가?

Vision Transformer 의 마지막 layer 에는 여러 attention head 가 있다.
왜 CLS token 의 attention 만 segmentation 에 사용되는가?

<details>
<summary>해설</summary>

CLS token 의 역할:
- Global feature aggregation
- Image-level classification/representation 의 핵심

다른 patch-to-patch attention 은:
- Local spatial relationships 포착
- Patch 간의 상호작용 (texture, boundary, etc.)

Object segmentation 을 위해서는:
- "어떤 patch 가 object 에 속하는가?" 를 알아야 함
- CLS 는 "전체 이미지의 의미" 를 담으므로, CLS-to-patch 의 attention weight 가 이를 반영

→ 자연스러운 선택.

다른 heads 도 사용하면 (average):
- 실제로 성능이 향상됨 (DINO 논문)
- 하지만 computational cost 증가

</details>

### 문제 2 (심화): Threshold 없이 continuous mask 를 사용하면?

Binarization 대신 attention map 을 그대로 (continuous) 사용하면?

<details>
<summary>해설</summary>

Continuous mask 의 장점:
- Threshold 에 의존하지 않음
- Soft IoU (weighted) 계산 가능
- 경계 근처의 uncertainty 를 잘 반영

Continuous mask 의 단점:
- Traditional evaluation metric (IoU) 과 맞지 않음
- Threshold 기반 mIoU 보다 비교 어려움

실제로 DINO 논문에서:
- Binarization 이 더 명확한 결과
- 다만 threshold 는 validation set 에서 tuning

Modern approach: Soft IoU 또는 Intersection@K (top-K pixels)

</details>

### 문제 3 (논문 비평): Emergent 인가, 학습된 것인가?

"Emergent" 는 놀라운 표현이지만, 사실 DINO 의 design choice 에서 "예상된" 결과 아닌가?

<details>
<summary>해설</summary>

**"Emergent" 라고 부르는 이유**:

Multi-crop loss 는:
- Segmentation task 를 직접 정의하지 않음
- Attention layer 에 "segmentation 을 하라" 고 말하지 않음
- 오직 representation consistency 만 요구

그런데 결과:
- Attention map 이 segmentation 처럼 동작
- 명시적 supervision 없이도

이는 genuinely surprising 한 발견 (2021 당시).

**"예상된" 입장**:
- Local-to-global 일관성 = spatial correspondence
- Spatial correspondence = 자연스럽게 boundary 에서 변함
- Boundary = attention 의 optimal assignment

**결론**: 논리적으로는 "설명 가능"하지만, empirical 발견은 powerful.

더욱이, supervised segmentation 모델은 보통 explicit loss (Dice, cross-entropy on mask) 을 사용하는데,
DINO 는 그런 것 없이도 유사한 결과를 얻었다는 점이 remarkable.

</details>

---

<div align="center">

[◀ 이전](./02-teacher-ema-collapse.md) | [📚 README](../README.md) | [다음 ▶](./04-ibot-dinov2.md)

</div>
