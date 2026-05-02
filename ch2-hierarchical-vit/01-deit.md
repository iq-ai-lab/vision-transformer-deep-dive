# 01. DeiT — Data-Efficient ViT

## 🎯 핵심 질문

- ViT 가 CNN 보다 데이터를 더 많이 필요한 근본적 이유는 무엇인가?
- Distillation token $x_{\text{dist}}$ 이 CNN teacher 의 inductive bias 를 어떻게 하드 라벨로 주입하는가?
- Hard distillation 이 soft distillation 보다 왜 효과적인가?
- 강한 augmentation (Mixup, CutMix, RandAugment, Erasing, Stochastic Depth) 이 ViT 의 데이터 효율성을 어떻게 개선하는가?

## 🔍 왜 이 distillation 이 Vision Transformer 의 핵심인가

ViT (Dosovitskiy 2021) 는 patch 단위의 global attention 으로 long-range dependency 를 직접 모델링하는 강점을 얻었다. 그러나 대가는 명확했다: translation equivariance · locality 등 CNN 이 갖춘 inductive bias 를 완전히 상실했다.

결과적으로 ImageNet-1k 같은 100만 장 규모 데이터셋으로는 ViT 가 CNN 에 비해 2-3% 정도 낮은 정확도를 보였다. 

**Touvron et al. (2021) DeiT 의 통찰**: Inductive bias 의 부족은 **더 많은 데이터** 를 요구하지만, 동시에 **teacher CNN 의 지식을 직접 주입** 하거나 **강한 data augmentation** 으로 보상할 수 있다는 가설.

DeiT 는 이를 두 가지 방법으로 증명했다:
1. **Distillation token**: 별도의 token $x_{\text{dist}}$ 가 CNN teacher 의 hard label 을 모방 → inductive bias 를 간접적으로 주입
2. **Strong augmentation recipe**: Mixup, CutMix, RandAugment, Random Erasing, Stochastic Depth 의 조합 → 유효 데이터 다양성 증대

결과: ImageNet-1k 만으로 ResNet-50 · EfficientNet-B5 를 능가.

## 📐 수학적 선행 조건

- **ViT 기초** (Ch1-01 ~ 04): Patch embedding, multi-head self-attention, position embedding, CLS token
- **Inductive bias 개념** (Ch1-05): Translation equivariance, locality, CNN 의 receptive field
- **Cross-entropy loss & label smoothing**: $L_{\text{CE}}(z, y)$, label smoothing 의 regularization 효과
- **KL divergence**: Soft target 분포와의 거리, temperature scaling
- **Mixup / CutMix** (Regularization Deep Dive): Convex combination 또는 spatial mixing 으로 증대된 training set
- **Stochastic Depth** (Regularization Deep Dive): Residual connection 을 random 하게 skip 하는 regularization

## 📖 직관적 이해

```
Vision Transformer 의 데이터 효율성 문제:

┌─────────────────────────────────────────────────┐
│ ViT (Vanilla)                                   │
│                                                 │
│ [Patch]─┐                                       │
│ [Patch]─┼─→ Self-Attention ─→ [CLS]            │
│ [Patch]─┘    (모든 patch 간 관계)               │
│              그러나 local structure 정보 없음   │
│                                                 │
│ CNN teacher 의 inductive bias                   │
│ (locality, translation equivariance) 부재      │
│ → 데이터 효율성 ↓                               │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ DeiT 해결 방안 1: Distillation Token            │
│                                                 │
│ [Patch]─┐                                       │
│ [Patch]─┼─→ Self-Attention ─→ [CLS] ──→ Task  │
│ [Patch]─┤                        (label: y)     │
│ [DIST]──┘                     [DIST] ──→ CNN   │
│                               (label: y_teacher)│
│                                                 │
│ → CNN 의 dark knowledge 을 직접 주입            │
│ → Hard distillation (label match) 사용         │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ DeiT 해결 방안 2: Strong Augmentation           │
│                                                 │
│ Original Image 1장                              │
│  ↓                                              │
│ Mixup (2개 이미지 blending)                     │
│ + CutMix (patch 잘라 붙이기)                    │
│ + RandAugment (색감, 회전 등)                  │
│ + Random Erasing (영역 지우기)                 │
│ + Stochastic Depth (skip 일부 layer)           │
│  ↓                                              │
│ 유효 1장 = 여러 transform 의 조합               │
│ → 데이터 다양성 증대                            │
│ → Inductive bias 부족 보상                      │
└─────────────────────────────────────────────────┘
```

**비유**: CNN 은 "지도책을 구역 단위로 세세히 읽는" 방식이고, ViT 는 "지도책 전체를 한번에 이해하려는" 방식. ViT 는 더 강한 집중력이 필요하므로, distillation 은 CNN teacher 가 "구역별 설명을 해주는" 것이고, augmentation 은 "같은 지도를 여러 각도로 봐서 이해를 깊게 하는" 것.

## ✏️ 엄밀한 정의

### 정의 2.1: Distillation Token 을 포함한 ViT 출력

Vanilla ViT 의 경우:
$$z_L^{\text{cls}} = \mathrm{Attn}(x_0, x_1, \ldots, x_N)_{\text{cls token}}$$

DeiT 에서는 별도의 learnable token $x_{\text{dist}}$ 을 추가하여:
$$z_L^{\text{cls}}, z_L^{\text{dist}} = \mathrm{Attn}(x_0, x_1, \ldots, x_N, x_{\text{dist}})$$

여기서 $x_0$ 는 CLS token, $x_1 \ldots x_N$ 은 patch embedding, $x_{\text{dist}}$ 는 새로 추가된 distillation token. 모두 position embedding 을 더한 후 Transformer layer 에 입력.

### 정의 2.2: Hard Distillation Loss

Teacher model (CNN): $y_{\text{teacher}} = \arg\max_c f_{\text{teacher}}(x)$  
Student model (ViT): $z_L^{\text{cls}}, z_L^{\text{dist}}$  
Ground truth: $y$

Loss 함수:
$$L_{\text{DeiT}} = \frac{1}{2} L_{\text{CE}}(z_L^{\text{cls}}, y) + \frac{1}{2} L_{\text{CE}}(z_L^{\text{dist}}, y_{\text{teacher}})$$

여기서 $L_{\text{CE}}(z, \tilde{y}) = -\log \frac{\exp(z_{\tilde{y}})}{\sum_c \exp(z_c)}$ 는 cross-entropy loss.

**핵심**: CLS token 은 ground truth $y$ 를 학습하고, DIST token 은 teacher 의 hard label $y_{\text{teacher}}$ 를 학습한다. 두 신호가 같은 weight 로 합쳐진다.

### 정의 2.3: Soft Distillation Loss (비교 대상)

온도-스케일링된 분포 사용:
$$L_{\text{soft}} = D_{\mathrm{KL}}\left( \sigma(f_{\text{teacher}}(x) / \tau) \, \big\| \, \sigma(z_L / \tau) \right)$$

여기서 $\sigma$ 는 softmax, $\tau$ 는 temperature (보통 3-20).

**차이**: Hard 는 label 만 본다 (one-hot). Soft 는 전체 확률분포를 본다 (teacher 의 모든 class 신뢰도 정보 포함).

### 정의 2.4: Augmentation Pipeline (DeiT 의 강화 버전)

학습 시 각 이미지 $x$ 에 대해 순서대로 적용:

1. **Random Crop & Resize**: $224 \times 224$ 로 resize
2. **Mixup** (확률 $p_m = 0.8$): 
   $$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j, \quad \lambda \sim \mathrm{Beta}(\alpha, \alpha), \, \alpha=1$$
3. **CutMix** (확률 $p_c = 0.5$): 
   $$\tilde{x} = x_i \odot M + x_j \odot (1-M)$$
   여기서 $M \in \{0,1\}^{H \times W}$ 는 random rectangular mask.
4. **RandAugment** (N=2, magnitude=9): 
   색감 변환, 회전, 전단 변환 등 2개를 random 하게 선택 후 적용.
5. **Random Erasing** (확률 $p_e = 0.2$): 
   영역을 random 한 색으로 칠하기.
6. **Stochastic Depth** (확률 $p_d = 0.1$ 정도): 
   Residual block skip (layer level).

## 🔬 정리와 증명

### 정리 2.1: Hard Distillation 의 효율성

**명제**: Hard distillation ($y_{\text{teacher}}$ 의 hard label) 이 soft distillation (확률분포) 보다 
데이터 효율성이 높을 수 있다.

**증명의 직관**:

Soft distillation 은 teacher 의 모든 class 에 대한 신뢰도를 추정해야 한다. 예를 들어 "개" 이미지에 대해 teacher 가 개: 0.95, 늑대: 0.04, 고양이: 0.01 같은 분포를 제공한다면, student 는 이 분포를 따라야 한다.

반면 hard distillation 은 teacher 의 hard label (개) 만 본다. 따라서:
- Teacher 의 예측이 강할수록 (높은 confidence) hard label 도 명확하다.
- Teacher 가 헷갈리는 샘플 (confidence 낮음) 은 soft label 이 "흐릿하지만" hard label 은 여전히 명확하다.

ImageNet-1k 같은 작은 데이터셋에서는 hard label 의 **margin** (명확성) 이 중요하다:
$$\text{Margin} = \log \frac{\exp(z_{\text{true}})}{\sum_c \exp(z_c)} - \log \frac{\exp(z_{\text{2nd}}))}{\sum_c \exp(z_c)}$$

Hard distillation 은 이 margin 을 직접 최대화한다. Soft distillation 은 분포를 맞추려고만 한다.

**증명 (형식)**:

Teacher model $f_T$ 가 주는 label 을 $\hat{y} = \arg\max_c f_T^c$ 라 하자.

Hard loss: $L_H = L_{\text{CE}}(z, \hat{y})$  
Soft loss: $L_S = D_{\mathrm{KL}}(\sigma(f_T / \tau) \| \sigma(z / \tau))$

데이터 크기 $N$ 이 충분히 작을 때 (ImageNet-1k 스케일), hard label 이 주는 **신호의 명확성** (signal-to-noise ratio) 이 soft label 을 이용한 gradient 보다 크다:

$$\mathbb{E}_{x \sim D} [\|\nabla_z L_H\|_2] > \mathbb{E}_{x \sim D} [\|\nabla_z L_S\|_2]$$

이는 teacher 의 신뢰도가 높을 때 성립한다 ($f_T(\hat{y}) > 0.9$ 정도).

따라서 small-data regime 에서는 hard distillation 이 더 효율적. $\square$

### 정리 2.2: Augmentation 의 Effective Batch Size 증대

**명제**: Mixup + CutMix + RandAugment 의 조합은 사실상 training set 크기를 $\approx 3\times$ 증대시킨다.

**증명의 직관**:

Original ImageNet-1k: 약 130만 장.

- Mixup: 매번 다른 2 이미지를 mix → label 분포도 변함 → 각 iteration 마다 "새로운" 샘플 생성
- CutMix: patch 단위 섞기 → spatial 관점에서 새로운 조합
- RandAugment: 색감·회전 등 → 각 이미지당 여러 버전

Empirically, ViT-B + 이 augmentation recipe 로 학습하면:
- epoch 당 loss decay 속도가 3배 더 빠르다
- 같은 epoch 수에서 2-3% 정확도 향상
- 이는 "유효 샘플" 을 3배로 늘린 것과 동등하다.

형식적으로, 각 augmentation $A_i$ 의 확률 $p_i$ 와 다양성 지수 $D_i$ (변환의 범위) 를 고려하면:
$$N_{\text{eff}} \approx N \cdot \prod_i (1 + p_i \cdot D_i)$$

DeiT 의 augmentation pipeline 은:
$$N_{\text{eff}} \approx N \cdot (1 + 0.8 \cdot 2) \cdot (1 + 0.5 \cdot 2) \cdot (1 + 0.2 \cdot 1.5) \cdots \approx 2.5 - 3.0 \times N$$

$\square$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Distillation Token 의 동작 원리

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT_with_DistToken(nn.Module):
    """ViT with distillation token"""
    def __init__(self, hidden_dim=768, num_classes=1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dist_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=12, dim_feedforward=3072,
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x, dist_tokens], dim=1)
        x = self.transformer(x)
        cls_output = x[:, 0]
        dist_output = x[:, -1]
        logits_cls = self.head(cls_output)
        logits_dist = self.head(dist_output)
        return logits_cls, logits_dist

torch.manual_seed(42)
B, N, D, C = 2, 196, 768, 1000
model = ViT_with_DistToken(hidden_dim=D, num_classes=C)
x = torch.randn(B, N, D)
logits_cls, logits_dist = model(x)

y_true = torch.tensor([5, 10])
y_teacher = torch.tensor([5, 11])
loss_cls = F.cross_entropy(logits_cls, y_true)
loss_dist = F.cross_entropy(logits_dist, y_teacher)
loss_deit = 0.5 * loss_cls + 0.5 * loss_dist

print(f"Loss (CLS): {loss_cls.item():.4f}")
print(f"Loss (DIST): {loss_dist.item():.4f}")
print(f"Loss (DeiT): {loss_deit.item():.4f}")
```

### 실험 2: Hard vs Soft Distillation 비교

```python
import torch
import torch.nn.functional as F

def hard_distillation_loss(logits_student, labels_teacher, 
                           labels_true, weight=0.5):
    loss_teacher = F.cross_entropy(logits_student, labels_teacher)
    loss_true = F.cross_entropy(logits_student, labels_true)
    return weight * loss_true + (1 - weight) * loss_teacher

def soft_distillation_loss(logits_student, logits_teacher, 
                           labels_true, tau=4.0, weight=0.5):
    p_teacher = F.softmax(logits_teacher / tau, dim=-1)
    log_p_student = F.log_softmax(logits_student / tau, dim=-1)
    loss_teacher_kl = F.kl_div(log_p_student, p_teacher, 
                               reduction='batchmean')
    loss_true = F.cross_entropy(logits_student, labels_true)
    return weight * loss_true + (1 - weight) * loss_teacher_kl

torch.manual_seed(42)
B, C = 32, 1000
logits_student = torch.randn(B, C)
logits_teacher = torch.randn(B, C)
labels_true = torch.randint(0, C, (B,))
labels_teacher = torch.randint(0, C, (B,))

loss_hard = hard_distillation_loss(logits_student, labels_teacher, 
                                   labels_true)
loss_soft = soft_distillation_loss(logits_student, logits_teacher, 
                                   labels_true, tau=4.0)

print(f"Hard distillation loss: {loss_hard.item():.4f}")
print(f"Soft distillation loss: {loss_soft.item():.4f}")

with torch.no_grad():
    acc_hard = (logits_student.argmax(dim=-1) == labels_true).float().mean()
    print(f"Accuracy (hard): {acc_hard.item():.4f}")

temps = [1.0, 2.0, 4.0, 8.0]
print("\nTemperature effect:")
for tau in temps:
    loss = soft_distillation_loss(logits_student, logits_teacher, 
                                  labels_true, tau=tau)
    print(f"  tau={tau}: loss={loss.item():.4f}")
```

### 실험 3: Mixup 의 Effective Data Augmentation

```python
import torch
import torch.nn.functional as F
import numpy as np

def mixup(x, y, alpha=1.0):
    B = x.size(0)
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(B)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_mixed = lam * F.one_hot(y, num_classes=1000).float() + \
              (1 - lam) * F.one_hot(y[idx], num_classes=1000).float()
    return x_mixed, y_mixed

torch.manual_seed(42)
B, C, H, W = 16, 3, 224, 224
x = torch.randn(B, C, H, W)
y = torch.randint(0, 1000, (B,))

x_mixed, y_mixed = mixup(x, y, alpha=1.0)

print(f"Original x shape: {x.shape}")
print(f"Mixup x shape: {x_mixed.shape}")
print(f"Original y shape: {y.shape}")
print(f"Mixup y (soft labels) shape: {y_mixed.shape}")

y_mixed_entropy = -(y_mixed * torch.log(y_mixed + 1e-10)).sum(dim=-1).mean()
print(f"\nEntropy of mixed labels: {y_mixed_entropy.item():.4f}")
print(f"  (higher = more class mixing)")
```

### 실험 4: Stochastic Depth 의 Layer Skip 효과

```python
import torch
import torch.nn as nn

class StochasticDepth(nn.Module):
    """Stochastic Depth (DropPath)"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = [x.size(0)] + [1] * (x.ndim - 1)
        mask = torch.bernoulli(torch.ones(shape) * keep_prob).to(x.device)
        x_dropped = x * mask / keep_prob
        return x_dropped

torch.manual_seed(42)
B, D, N = 2, 768, 196

class Block(nn.Module):
    def __init__(self, drop_path=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(D)
        self.attn = nn.Linear(D, D)
        self.drop_path = StochasticDepth(drop_path)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        return x

model = nn.Sequential(
    Block(drop_path=0.05),
    Block(drop_path=0.10),
    Block(drop_path=0.15),
)

x = torch.randn(B, N, D)
model.train()
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print("Stochastic Depth: diverse paths via random layer skip")
```

## 🔗 실전 활용

**timm 에서의 DeiT 구현**:
```python
from timm.models import deit_base_patch16_224
model = deit_base_patch16_224(pretrained=True)
```

**주요 하이퍼파라미터**:
- `distillation_type='hard'` (기본값)
- `distillation_loss_weight=0.5`
- `augmentation='strong'` (Mixup + CutMix + RandAugment + Erasing)
- `stochastic_depth_prob=0.1`

**응용**:
- **Small ViT**: 모바일, edge device 학습 가능
- **Fine-tuning**: ImageNet pretrain + downstream task
- **Knowledge distillation**: ViT-L teacher → ViT-B student

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Teacher 의존성** | CNN teacher 를 요구함 | Teacher 정확도가 낮으면 제한됨 |
| **Hard label 안정성** | Teacher 가 confident 하다고 가정 | Noisy teacher 에선 성능 저하 |
| **Augmentation 일반성** | ImageNet 에 최적화됨 | 다른 도메인에서는 재조정 필요 |
| **Computational cost** | 2개 forward + teacher 평가 | Training 시간 1.5배 증가 |
| **Distillation 지속성** | Pre-training 단계만 사용 | Fine-tuning 시 제거하는 것이 표준 |

## 📌 핵심 정리

$$\boxed{L_{\text{DeiT}} = \tfrac{1}{2} L_{\text{CE}}(z^{\text{cls}}, y) + \tfrac{1}{2} L_{\text{CE}}(z^{\text{dist}}, y_{\text{teacher}})}$$

| 개념 | 식 | 역할 |
|------|-----|------|
| **CLS token** | $z^{\text{cls}} = \mathrm{Attn}(\ldots)_0$ | Ground truth 학습 |
| **DIST token** | $z^{\text{dist}} = \mathrm{Attn}(\ldots)_{-1}$ | Teacher inductive bias 주입 |
| **Hard distillation** | $L_{\text{CE}}(z, y_{\text{teacher}})$ | Label 직접 모방 |
| **Augmentation** | Mixup + CutMix + RandAugment + Erasing | Effective dataset size 증대 |
| **Stochastic Depth** | Random layer skip | Regularization + Ensemble 효과 |

**결론**: ViT 의 인덕티브 바이어스 부족은 **강한 teacher 신호** 와 **강한 데이터 증강** 으로 완전히 보상 가능하다. ImageNet-1k 만으로 CNN 능가. 이는 "inductive bias 는 학습 가능하다" 는 중요한 통찰을 제공한다.

## 🤔 생각해볼 문제

### 문제 1 (기초): Distillation Token 의 필요성

ViT 에 distillation token 을 추가하지 않고, CLS token 만으로 student 와 teacher 를 동시 학습하면 어떻게 될까?

<details>
<summary>해설</summary>

CLS token 만으로 동시 학습 시:
$$L = 0.5 \cdot L_{\text{CE}}(z^{\text{cls}}, y) + 0.5 \cdot L_{\text{CE}}(z^{\text{cls}}, y_{\text{teacher}})$$

문제점:
1. **Conflicting gradients**: $y \neq y_{\text{teacher}}$ 일 때, gradient 가 반대 방향일 수 있음
2. **Token capacity 부족**: 단일 token 에서 두 신호를 동시 처리 → information bottleneck
3. **정확도 하락**: 1-2% 하락 (실험적 결과)

Distillation token 별도 추가 시:
- CLS 와 DIST 가 independent 한 경로 진화
- Gradient conflict 감소
- Token capacity 증대

따라서 distillation token 은 gradient routing 관점에서 필수적이다.

</details>

### 문제 2 (심화): Hard vs Soft Distillation 의 Information-Theoretic 해석

Hard label $y_{\text{teacher}}$ vs soft distribution $p_{\text{teacher}}$ 
— 어느 것이 더 많은 정보를 담는가? Mutual Information 관점에서 설명하시오.

<details>
<summary>해설</summary>

Soft distribution 이 더 많은 정보를 가진 것처럼 보이지만, 
finite sample regime 에서는 다르다:

1. **Teacher confidence 의 노이즈**:
   ImageNet 에서 "개" 이미지가 "늑대" 로 약간 헷갈린다.
   Teacher 의 soft label: (개: 0.92, 늑대: 0.06, ...)
   이 0.06 은 semantic similarity 일까, 아니면 teacher 오류일까?

2. **Hard label 의 확실성**:
   Hard label (개) 은 teacher 의 argmax 이므로, 
   "teacher 가 가장 confident 한 class" 만 추출.
   이는 noise 에 강하다.

3. **Signal-to-Noise Ratio**:
   Teacher 의 신뢰도가 높으면 (confident), 
   hard label 의 MI 가 soft label 의 MI 보다 더 높다.

4. **Empirical validation** (DeiT Table 3):
   - Hard distillation: 84.7% (ViT-B)
   - Soft distillation (τ=4): 84.1%
   - No distillation: 81.8%

결론: Small-data regime 에서 hard distillation 이 정보 효율성이 더 높다.

</details>

### 문제 3 (논문 비평): DeiT 의 Augmentation Recipe 일반화

DeiT 의 augmentation recipe (Mixup, CutMix, RandAugment, Erasing)
는 ImageNet-1k 에 최적화되어 있다. 
CIFAR-10, 의료 이미지, 위성 이미지에서도 같은 recipe 를 쓸 수 있을까?

<details>
<summary>해설</summary>

**CIFAR-10** (32×32 이미지):
- Mixup: ✓ 효과적
- CutMix: ✗ 이미지가 너무 작아서 의미 없음
- RandAugment: △ magnitude 를 줄여야 함
- 권장: Mixup + 약한 RandAugment

**의료 이미지** (X-ray, CT):
- Mixup: ✗ 물리적 의미 손상
- CutMix: ✗ 같은 이유
- RandAugment: △ 색감 ✓, 회전 △
- 권장: Mild rotation + Elastic deformation + Noise

**위성 이미지** (고해상도):
- Mixup: ✓ 효과적
- CutMix: ✓ 효과적
- RandAugment: ✓ 효과적
- 권장: Mixup + CutMix + 제한된 RandAugment

**결론**: 
DeiT 의 recipe 는 일반화가능한 **framework** 이지만, 
각 도메인의 특성에 맞춰 **parameter tuning** 이 필수적이다.

</details>

---

<div align="center">

[◀ 이전](../ch1-vit-foundations/05-inductive-bias-gap.md) | [📚 README](../README.md) | [다음 ▶](./02-swin.md)

</div>
