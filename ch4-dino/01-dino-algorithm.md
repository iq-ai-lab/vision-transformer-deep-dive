# 01. DINO 알고리즘 (Caron 2021)

## 🎯 핵심 질문

- Self-supervised learning 에서 **teacher-student** 구조는 왜 필요한가?
- **Multi-crop 전략** (2개 global, 8개 local) 이 학습을 어떻게 개선하는가?
- DINO 의 **asymmetric loss** 는 모든 crop 조합을 동등하게 다루지 않는데, 왜 이렇게 설계했는가?
- Linear probe / k-NN classifier 에서 unsupervised 모델이 supervised 에 가까운 성능을 얻는 기초가 무엇인가?

## 🔍 왜 이 알고리즘이 self-supervised 혁명인가

DINO (DIstillation with NO labels) 는 2021년 처음 제시된 self-supervised vision 의 이정표다.

주요 혁신:
1. **Contrastive 없음**: InfoNCE 처럼 explicit 한 negative pair 가 필요 없다.
2. **Teacher-student 증류**: 매 step 마다 student weight 를 teacher 로 exponential moving average (EMA) 한다.
3. **Multi-crop 강제**: Local patch 들도 global image 의 의미를 배워야 한다는 inductive bias.
4. **Emergent properties**: 학습된 attention map 이 명시적 supervision 없이 object mask 를 형성한다.

결과: ImageNet linear probe 에서 77.1% (ViT-S/8), 기존 BYOL/SimSiam 능가.

## 📐 수학적 선행 조건

- **Self-supervised learning basics** (Ch3: contrastive, BYOL)
- **Exponential moving average (EMA)**: $\theta_t \leftarrow \lambda \theta_t + (1-\lambda) \theta_s$
- **Cross-entropy loss**: $L = -\sum_i p_i \log q_i$
- **Softmax temperature**: $p(x) = \exp(x/\tau) / \sum_j \exp(x_j/\tau)$
- **KL divergence**: $D_{\text{KL}}(p \| q) = \sum_i p_i \log(p_i / q_i)$

## 📖 직관적 이해

```
┌─────────────────────────────────────────────────────────────┐
│                DINO Pipeline                                │
│                                                             │
│  Image x  ──→ Multi-crop (2 global + 8 local)             │
│              ↙                              ↘               │
│         STUDENT                          TEACHER            │
│         (trained)                        (EMA copy)         │
│              │                              │               │
│         all crops                    only 2 global          │
│         (10 forward)                  (2 forward)           │
│              │                              │               │
│         head (MLP)                   head (MLP)             │
│              ↓                              ↓               │
│          logits_s                      logits_t             │
│        (normalized)                  (normalized)           │
│              │                              │               │
│              └──────────────────────────────┘               │
│                        Cross-entropy:                       │
│         L = -Σ_{x_t ∈ V_g} Σ_{x_s ∈ V} p_t(x_t)^T log p_s(x_s)
│                                                             │
│  여기서:                                                     │
│  - p_t = softmax(logits_t / τ_t), τ_t = 0.04 (sharp)     │
│  - p_s = softmax(logits_s / τ_s), τ_s = 0.1 (soft)        │
│  - logits 는 output normalization + centering              │
└─────────────────────────────────────────────────────────────┘

학습 목표:
  student 가 teacher 의 확률분포를 모든 view-pair 에서 모방
  → local crops 도 global crops 와 일관된 표현을 학습
```

**비유**: Student 는 exam 을 푸는 학생, teacher 는 정답지. 하지만 정답지는 계속 갱신된다 (EMA). 
모든 문제 (crop) 를 풀어야 하지만, 채점 (loss) 은 정답지의 두 "신뢰할 수 있는 버전" 만 본다.

## ✏️ 엄밀한 정의

### 정의 4.1: Multi-Crop 생성

Image $x$ 에서 10 개의 crop 을 만든다:

$$\mathcal{V} = \{x_1^g, x_2^g\} \cup \{x_1^{\ell}, \ldots, x_8^{\ell}\}$$

여기서:
- $x_i^g$: global crop (전체 이미지의 96% 이상), 2개
- $x_j^{\ell}$: local crop (전체 이미지의 50% 미만), 8개

각 crop 은 independent augmentation (random flip, rotation, color jitter, etc.) 을 거친다.

### 정의 4.2: Student & Teacher 네트워크

Student: $f_{\theta_s}(x) = h_s(g(x))$ 
- $g(x)$: ViT backbone (feature 추출)
- $h_s$: projection head (MLP, $d \to 65536$)

Teacher: $f_{\theta_t}(x) = h_t(g(x))$
- **가중치**: $\theta_t \leftarrow \lambda \theta_t + (1-\lambda) \theta_s$ (EMA, $\lambda \in [0.996, 1.0)$ cosine schedule)
- **Head**: 동일 구조이지만 서로 다른 parameters

### 정의 4.3: Normalization & Temperature Scaling

$\mathbb{C}$ = running centering vector (mean of teacher output)

Student logits (all crops):
$$\tilde{z}_s = \frac{h_s(g(x)) - \mathbb{C}}{\left\| h_s(g(x)) - \mathbb{C} \right\|} \quad (\text{normalized})$$

Teacher logits (only globals):
$$\tilde{z}_t = \frac{h_t(g(x)) - \mathbb{C}}{\left\| h_t(g(x)) - \mathbb{C} \right\|} \quad (\text{normalized})$$

Probability (with temperature):
$$p_s(x) = \mathrm{softmax}(\tilde{z}_s / \tau_s), \quad p_t(x) = \mathrm{softmax}(\tilde{z}_t / \tau_t)$$

- $\tau_s = 0.1$ (student, soft target)
- $\tau_t = 0.04$ (teacher, sharp target)

### 정의 4.4: DINO Loss

Global crops 만 teacher 역할:

$$\mathcal{L} = -\sum_{x_t \in V_g} \sum_{x_s \in V, x_s \neq x_t} p_t(x_t)^\top \log p_s(x_s)$$

여기서:
- Outer sum: teacher 가 보는 2개 global crops
- Inner sum: student 가 보는 모든 10개 crops (제외: 자신의 쌍 $x_t$)
- $p_t$: teacher 확률 (sharp)
- $p_s$: student 확률 (soft)

**총 loss**: (2 globals × 9 other crops) = 18 terms

## 🔬 정리와 증명

### 정리 4.1: Multi-Crop Consistency 의 최적화 효과

**주장**: Multi-crop loss 는 다음 두 가지를 동시에 달성한다:
1. **Global-level consistency**: 두 global crops 간의 semantic agreement
2. **Local-to-global alignment**: local patches 도 global representation 과 일관되어야 함

**증명 스케치**:

Loss 를 전개하면:
$$\mathcal{L} = -\sum_{i=1}^{2} \sum_{j=1, j \neq i}^{10} \log p_s(x_i^g)^\top p_t(x_j^g) - \sum_{i=1}^{2} \sum_{j=3}^{10} \log p_s(x_j^{\ell})^\top p_t(x_i^g)$$

첫 항: 두 global 간 consistency (standard contrastive like)
두 번째 항: local-to-global alignment (DINO 특유)

두 번째 항의 존재로 인해:
- Local patches 는 **full context** 를 배워야 한다.
- 단순한 texture 만으로는 부족; semantic content 필요.
- 결과: 더 robust, transferable features.

Gradient 관점:
$$\frac{\partial \mathcal{L}}{\partial \theta_s} = \frac{\partial}{\partial \theta_s} \left[ \text{global-global} + \text{local-global} \right]$$

Local-global term 이 student 를 "zoom out" 하도록 강제한다.

$$\square$$

### 정리 4.2: EMA Teacher 의 수렴성

**주장**: $\lambda = \lambda(t)$ 가 cosine schedule 로 $0.996 \to 1.0$ 이면, teacher weight 는 안정적 평형점으로 수렴한다.

**증명 스케치**:

EMA update:
$$\theta_t^{(n+1)} = \lambda(n) \theta_t^{(n)} + (1 - \lambda(n)) \theta_s^{(n)}$$

이를 unroll 하면:
$$\theta_t^{(n)} = \prod_{i=0}^{n-1} \lambda(i) \cdot \theta_t^{(0)} + \sum_{k=0}^{n-1} (1-\lambda(k)) \prod_{i=k+1}^{n-1} \lambda(i) \cdot \theta_s^{(k)}$$

$\lambda(n) \to 1$ 이면 $\prod_{i=k}^{n-1} \lambda(i) \to$ 지수적 감소.

따라서:
$$\theta_t^{(n)} \approx \int_{\text{recent}} (1-\lambda) e^{-\int \lambda} \theta_s(\tau) d\tau$$

즉, teacher 는 최근 student 의 **exponentially weighted average** 로 수렴하며, 급격한 진동을 피한다.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Multi-Crop DataLoader 구현

```python
import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class MultiCropAugmentation:
    """DINO 의 10-crop augmentation"""
    def __init__(self, global_scale=(0.4, 1.0), local_scale=(0.05, 0.4)):
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.n_globals = 2
        self.n_locals = 8
        
        # Base augmentation
        self.base_aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur((3, 3), sigma=(0.1, 2.0))
        ])
    
    def _random_crop(self, size, scale):
        """Random resized crop"""
        def transform(img):
            aug = T.RandomResizedCrop(
                size, scale=scale, 
                interpolation=T.InterpolationMode.BILINEAR
            )
            return aug(img)
        return transform
    
    def __call__(self, img):
        crops = []
        # 2 global crops
        for _ in range(self.n_globals):
            aug = T.Compose([
                self._random_crop(224, self.global_scale),
                self.base_aug,
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
            crops.append(aug(img))
        
        # 8 local crops
        for _ in range(self.n_locals):
            aug = T.Compose([
                self._random_crop(96, self.local_scale),
                self.base_aug,
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
            crops.append(aug(img))
        
        return torch.stack(crops)  # [10, 3, H, W]

# Test
dataset = CIFAR10(root='/tmp/data', train=True, download=True,
                   transform=MultiCropAugmentation())
loader = DataLoader(dataset, batch_size=2)

batch = next(iter(loader))
print(f"Batch shape: {batch.shape}")  # Expected: [2, 10, 3, H, W]
print(f"Multi-crop augmentation working: {batch.shape[1] == 10}")
```

### 실험 2: DINO Loss 구현

```python
import torch
import torch.nn.functional as F

class DINOLoss(torch.nn.Module):
    def __init__(self, tau_s=0.1, tau_t=0.04, n_globals=2):
        super().__init__()
        self.tau_s = tau_s
        self.tau_t = tau_t
        self.n_globals = n_globals
        self.center = torch.zeros(65536)
        
    def forward(self, student_out, teacher_out, center_momentum=0.99):
        """
        student_out: [B*10, D] - student outputs from all 10 crops
        teacher_out: [B*2, D]  - teacher outputs from 2 globals only
        """
        B = student_out.shape[0] // 10
        batch_size = B * self.n_globals
        
        # Reshape for multi-crop
        # student_out: [B, 10, D]
        student_out = student_out.reshape(B, 10, -1)
        teacher_out = teacher_out.reshape(B, self.n_globals, -1)
        
        # Normalize
        student_out = F.normalize(student_out, dim=-1)
        teacher_out = F.normalize(teacher_out, dim=-1)
        
        # Update center (momentum)
        center_new = teacher_out.mean(dim=(0, 1))
        self.center = self.center.to(center_new.device)
        self.center = center_momentum * self.center + (1 - center_momentum) * center_new
        
        # Apply centering & normalization
        student_out = student_out - self.center.unsqueeze(0).unsqueeze(0)
        student_out = F.normalize(student_out, dim=-1)
        
        teacher_out = teacher_out - self.center.unsqueeze(0).unsqueeze(0)
        teacher_out = F.normalize(teacher_out, dim=-1)
        
        # Apply temperature
        student_prob = F.softmax(student_out / self.tau_s, dim=-1)
        teacher_prob = F.softmax(teacher_out / self.tau_t, dim=-1)
        
        # Compute loss: for each global, compute loss against all other crops
        loss = 0.0
        for i in range(self.n_globals):
            # teacher view i
            p_t = teacher_prob[:, i]  # [B, D]
            
            # Loop over all crops
            for j in range(10):
                if j < self.n_globals and i == j:
                    continue  # Skip same-view pair
                
                p_s = student_prob[:, j]  # [B, D]
                
                # Cross-entropy: -sum(p_t * log(p_s))
                loss += -torch.sum(p_t * torch.log(p_s + 1e-8)) / (B * self.n_globals)
        
        return loss

# Test
student_out = torch.randn(32 * 10, 65536)  # 32 images, 10 crops each
teacher_out = torch.randn(32 * 2, 65536)   # 32 images, 2 global crops

dino_loss = DINOLoss(tau_s=0.1, tau_t=0.04)
loss_val = dino_loss(student_out, teacher_out)
print(f"DINO Loss: {loss_val.item():.4f}")
assert loss_val.item() > 0, "Loss should be positive"
```

### 실험 3: EMA Teacher 업데이트

```python
import torch
import math

class EMATeacher:
    def __init__(self, model, tau_base=0.996, tau_end=1.0, n_steps=100000):
        self.model = model
        self.tau_base = tau_base
        self.tau_end = tau_end
        self.n_steps = n_steps
        
        # Copy student to teacher
        self.teacher_model = self._copy_model(model)
    
    def _copy_model(self, model):
        import copy
        return copy.deepcopy(model)
    
    def _get_tau(self, step):
        """Cosine schedule from tau_base to tau_end"""
        return self.tau_end - (self.tau_end - self.tau_base) * (1 + math.cos(math.pi * step / self.n_steps)) / 2
    
    def update(self, step):
        tau = self._get_tau(step)
        
        for param_s, param_t in zip(self.model.parameters(), self.teacher_model.parameters()):
            param_t.data = tau * param_t.data + (1 - tau) * param_s.data.detach()
    
    def __call__(self, x):
        return self.teacher_model(x)

# Test: verify tau schedule
ema = EMATeacher(torch.nn.Linear(10, 10), n_steps=1000)
taus = [ema._get_tau(step) for step in [0, 250, 500, 750, 1000]]
print(f"Tau schedule: {[f'{t:.4f}' for t in taus]}")
# Expected: 0.9960, ~0.9980, ~0.9990, ~1.0000, 1.0000
assert taus[0] == 0.996 and taus[-1] == 1.0, "Schedule bounds incorrect"
```

### 실험 4: Attention Map 시각화 (emergent segmentation)

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def visualize_attention_map(attn_weights, image, threshold=0.1):
    """
    attn_weights: [H*W, H*W] - attention from CLS to all patches
    Reshape to spatial grid and threshold
    """
    n_patches = int(math.sqrt(attn_weights.shape[0]))
    attn_2d = attn_weights[:, 1:].reshape(n_patches, n_patches, -1)  # Exclude CLS
    
    # CLS to patch attention
    cls_attn = attn_weights[0, 1:]  # [H*W - 1]
    cls_attn_2d = cls_attn.reshape(n_patches, n_patches)
    
    # Threshold
    mask = (cls_attn_2d > threshold).float()
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=image.shape[-2:], mode='bilinear')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    ax1.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax1.set_title('Original Image')
    
    ax2.imshow(cls_attn_2d.cpu().numpy(), cmap='hot')
    ax2.set_title('CLS Attention')
    
    ax3.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    ax3.set_title(f'Mask (threshold={threshold})')
    
    plt.tight_layout()
    return fig

# Simulated test
image = torch.randn(3, 224, 224)
n_patches = 14  # 224 / 16 = 14
attn_weights = torch.randn(n_patches**2 + 1, n_patches**2 + 1)  # +1 for CLS
attn_weights = F.softmax(attn_weights, dim=-1)

fig = visualize_attention_map(attn_weights, image, threshold=0.1)
print("Attention visualization complete (not shown in script mode)")
```

## 🔗 실전 활용

### 학습 파이프라인

1. **Initialization**: Student = random, Teacher = copy of student
2. **Per-batch loop**:
   - Generate multi-crop (2 global + 8 local)
   - Forward student (all crops), teacher (globals only)
   - Compute loss (학생이 teacher 분포 모방)
   - Backward on student only
   - Update teacher weights via EMA
   - Update centering vector
3. **After training**: 
   - Discard heads
   - Use backbone features for linear probe or k-NN

### 성능 지표

| 모델 | ImageNet Linear Probe | k-NN 분류기 |
|------|----------------------|-----------|
| DINO ViT-S/8 | 77.1% | 74.5% |
| DINO ViT-B/8 | 78.2% | 75.3% |
| Supervised ViT-B | 81.8% | - |

**Insight**: DINO 는 supervised 의 ~95% 수준을 달성하며, explicit label 없이 semantic features 를 학습.

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Multi-crop 필요성** | Local-to-global alignment 을 위해 10개 crop 이 "최적" | 수는 하이퍼파라미터; 더 많거나 적으면 성능 변함 |
| **EMA schedule** | Cosine $\lambda$ 이 수렴성 보장 | Linear schedule 도 작동하지만 덜 안정적 |
| **Centering 안정성** | Momentum center 가 collapse 방지의 핵심 | Center 없으면 uniform distribution collapse (Ch4-02) |
| **Temperature gap** | $\tau_s > \tau_t$ (soft vs sharp) | 너무 크거나 작으면 불안정; 0.1 / 0.04 는 empirically optimal |
| **Large batch size** | 배치가 작으면 centering estimate 노이즈 높음 | 실전: batch size ≥ 128 권장 |
| **Data augmentation** | Strong augmentation (blur, color jitter) 중요 | Weak augmentation 만으로는 부족 |

## 📌 핵심 정리

$$\boxed{\mathcal{L} = -\sum_{x_t \in V_g} \sum_{x_s \in V, x_s \neq x_t} p_t(x_t)^\top \log p_s(x_s)}$$

$$\boxed{\text{Teacher: } \theta_t \leftarrow \lambda(t) \theta_t + (1-\lambda(t)) \theta_s, \quad \lambda(t) = \text{cosine}(0.996 \to 1.0)}$$

| 개념 | 수식 | 역할 |
|------|-----|------|
| **Multi-crop** | 2 globals, 8 locals per image | View-invariant representation 강제 |
| **EMA Teacher** | $\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$ | 안정적 target distribution 제공 |
| **Centering** | $\mathbb{C} = m \mathbb{C} + (1-m) \bar{z}_t$ | One-dimensional collapse 방지 |
| **Temperature** | $\tau_s = 0.1, \tau_t = 0.04$ | Student soft, teacher sharp (Sharpening) |
| **Loss** | Cross-entropy between probabilities | Student → teacher distribution alignment |

## 🤔 생각해볼 문제

### 문제 1 (기초): Multi-Crop Loss 계산

Image 1개, batch size 1일 때:
- Student output: 10개 crop (모두 처리)
- Teacher output: 2개 crop (globals 만)

Loss 를 계산하는 term 개수는? (자신의 쌍 제외)

<details>
<summary>해설</summary>

Teacher 가 보는 crop: 2개 (global 1, global 2)
Student 가 보는 crop: 10개 (global 1, 2 + local 1-8)

각 teacher crop 에 대해 student crop 중 자신을 제외한 9개와 pair 를 만든다.

Total: $2 \times 9 = 18$ loss terms.

이를 batch dimension 으로 확장하면, batch size B 일 때: $2 \times 9 \times B$ terms.

</details>

### 문제 2 (심화): EMA Schedule 의 효과

Cosine schedule: $\lambda(t) = 1 - 0.004 \cdot \frac{1 + \cos(\pi t / T)}{2}$

vs. 상수 $\lambda = 0.998$ 을 비교했을 때, 어느 것이 더 안정적인가? 왜?

<details>
<summary>해설</summary>

**Cosine schedule이 더 안정적**:

초기 ($t=0$): $\lambda(0) = 1 - 0.004 \cdot 1 = 0.996$ (aggressive update)
→ Student 가 아직 좋지 않을 때, teacher 가 더 자주 갱신

후기 ($t=T$): $\lambda(T) = 1 - 0.004 \cdot 0 = 1.0$ (no update)
→ Teacher 가 수렴했으므로, student 가 teacher 로 수렴

상수 $\lambda = 0.998$ 이면 처음부터 끝까지 같은 속도로 갱신되어, 
후기에 불필요한 jitter 가 생긴다.

**결론**: Curriculum of EMA momentum.

</details>

### 문제 3 (논문 비평): DINO vs SimSiam

DINO: EMA teacher + centering + multi-crop loss
SimSiam: Stop-gradient + predictor

DINO 가 centering 을 추가한 이유는? (Ch4-02 와 연계)

<details>
<summary>해설</summary>

SimSiam 은 stop-gradient 로 collapse 를 방지하지만, 
모든 batch 에서 일관된 gradient flow 를 보장하지 못한다.

DINO 는 cross-batch centering (momentum buffer) 로:
- **One-dimensional collapse** 차단: output 이 상수가 되는 것 방지
- **Uniform distribution collapse** 차단: output 이 균등분포가 되는 것 방지

EMA teacher 와 centering 은 **상호보완**:
- Teacher: 안정성 (smooth target)
- Centering: 다양성 (collapse 방지)

이를 joint 로 제거했을 때 성능이 저하된다.
(Details in Ch4-02 ablation)

</details>

---

<div align="center">

[◀ 이전](../ch3-contrastive-ssl/05-byol-simsiam.md) | [📚 README](../README.md) | [다음 ▶](./02-teacher-ema-collapse.md)

</div>
