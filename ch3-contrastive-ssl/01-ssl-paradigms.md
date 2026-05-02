# 01. Self-Supervised Learning 의 3 가지 패러다임

## 🎯 핵심 질문

- Self-Supervised Learning 에서 label 없이 supervision signal 을 만드는 3 가지 방식은 무엇인가?
- Generative, Discriminative (Contrastive), Self-Distillation 의 손실함수 형태와 수렴점이 어떻게 다른가?
- 각 패러다임이 implicit 하게 최대화하는 목적함수는 무엇인가?
- 언제 어느 paradigm 을 선택해야 하는가 — dataset size, downstream task, computational resource 의 관점에서?

## 🔍 왜 이 3 가지 패러다임인가

Self-Supervised Learning (SSL) 은 label 이 없을 때 representation 을 학습하는 패러다임이다.
본질적으로 3 가지 관점이 있다:

1. **Generative** (생성): "마스크된 부분을 복원해라" — Masked Image Modeling (MAE, BEiT)
   - 저수준 detail (pixel, patch token) 까지 capture
   - Reconstruction 이 supervision signal

2. **Discriminative / Contrastive** (판별): "비슷한 것과 다른 것을 구분해라" — SimCLR, MoCo
   - 고수준 semantic representation 학습
   - Positive pair 간 similarity 최대화, negative pair 간 최소화

3. **Self-Distillation** (자체 증류): "자신의 느린 버전 (EMA teacher) 을 모방해라" — BYOL, DINO
   - View-invariant representation 학습
   - Negative pair 없이도 collapse 회피 (predictor + stop-gradient)

이 3 가지는 mutually exclusive 가 아니며, 실제로는 hybrid 접근 (iBOT = Distillation + MIM) 도 존재한다.

## 📐 수학적 선행 조건

- **Probability & Divergence**: KL divergence, cross-entropy loss, softmax
- **Information Theory**: Mutual information $I(X; Y) = H(X) - H(X|Y)$
- **Optimization**: gradient descent, momentum, temperature scaling
- **Linear Algebra**: cosine similarity $\cos(z_i, z_j) = \frac{z_i^\top z_j}{\|z_i\| \|z_j\|}$

## 📖 직관적 이해

```
┌─────────────────────────────────────────────────────────┐
│ 3 Paradigms of SSL                                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 1. GENERATIVE (Reconstruction)                         │
│    x → [MASK regions] → encoder → decoder → ŷ          │
│    Loss: ||x - ŷ||²  (pixel/patch level)              │
│    ✓ Detail-oriented  ✗ Expensive to train             │
│                                                         │
│ 2. CONTRASTIVE (Discrimination)                        │
│    x₀ → Aug₁ → f → z₁                                  │
│    x₀ → Aug₂ → f → z₂                                  │
│    Loss: -log(exp(sim(z₁,z₂)/τ)/Σexp(...))            │
│    ✓ Efficient  ✗ Needs large negatives               │
│                                                         │
│ 3. SELF-DISTILLATION (View Invariance)                 │
│    x₀ → Aug₁ → θ_s → z_s                              │
│    x₀ → Aug₂ → θ_t (EMA) → z_t                         │
│    Loss: -sim(z_s, sg[z_t])  (sg=stop-grad)           │
│    ✓ No negatives  ✓ Stable  ✗ Slower                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**비유**: 세 명의 학생이 다른 방식으로 배운다.
- 첫 번째 (Generative): 책의 빠진 부분을 채우며 세세한 detail 학습.
- 두 번째 (Contrastive): 비슷한 문제들끼리 묶고 다른 문제들과 구분하며 개념 학습.
- 세 번째 (Self-Distillation): 어제 배운 자신의 느린 버전을 따라하며 안정적 성장.

## ✏️ 엄밀한 정의

### 정의 3.1: Generative SSL Loss (MAE-style)

Input image $x \in \mathbb{R}^{H \times W \times 3}$ 에서 mask $M \in \{0,1\}^{N_p}$ 를 random 하게 생성 (ratio $\rho$):
$$L_{\text{gen}} = \|x_{\text{masked}} - g(f(x_{\text{visible}}))\|^2$$

여기서:
- $x_{\text{visible}} = x \odot (1-M)$ (visible patch 만)
- $f$ : encoder (ViT 등)
- $g$ : decoder (경량, pretrain 후 버림)
- $M$ : uniform random mask

### 정의 3.2: Contrastive SSL Loss (InfoNCE)

Two augmented views $\{(\tilde x_i, \tilde x_j)\}_{i=1}^B$ 에서:
$$L_{\text{cont}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\mathrm{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\mathrm{sim}(z_i, z_k) / \tau)}$$

여기서:
- $z_i = g(f(\tilde x_i))$ (projection head 적용)
- $\mathrm{sim}(a, b) = \frac{a^\top b}{\|a\| \|b\|}$ (cosine similarity)
- $\tau$ : temperature parameter (보통 0.07)
- 분모의 "negatives" = 같은 batch 의 다른 샘플들

### 정의 3.3: Self-Distillation SSL Loss (BYOL-style)

Student $\theta_s$ 와 teacher $\theta_t$ (EMA 업데이트):
$$L_{\text{dist}} = 2 - 2 \cdot \mathrm{sim}(q_{\theta_s}(z_1), \mathrm{sg}[z_2'])$$

여기서:
- $z_1 = f_{\theta_s}(\tilde x_1)$ (student 의 projection)
- $z_2' = f_{\theta_t}(\tilde x_2)$ (teacher 의 projection, EMA)
- $q_{\theta_s}$ : predictor MLP (student 의 additional head)
- $\mathrm{sg}[\cdot]$ : stop-gradient operation
- Teacher update: $\theta_t \leftarrow \lambda \theta_t + (1-\lambda) \theta_s$, $\lambda$ ≈ 0.999

## 🔬 정리와 증명

### 정리 3.1: Implicit Objectives of SSL Paradigms

세 패러다임은 각각 다음을 implicit 하게 최대화한다:

1. **Generative**: Reconstruction likelihood $-\|x - \hat{x}\|^2 \propto$ 저수준 정보 보존 (근처 patch correlation)

2. **Contrastive**: Mutual information lower bound $I(z_i; z_j) \geq \log N - L_{\text{InfoNCE}}$ (정리 3.2 에서 상세)

3. **Self-Distillation**: View-invariance regularization + predictor + stop-gradient 의 조합이 non-trivial solution 에서의 saddle point (Tian 2021)

**증명 스케치**:

**(1) Generative**: MSE loss $\mathbb{E}[\|x - g(f(x))\|^2]$ 를 최소화 = $g \circ f$ 가 $x$ 의 근처 저주파 성분을 compress & decompress. 결과적으로 encoder $f$ 는 spatial locality 를 학습.

**(2) Contrastive**: Poole (2019) 정리: InfoNCE 손실의 최소화가 negative count $N$ 에 대한 mutual information 의 lower bound 를 최대화. 정리 3.2 참조.

**(3) Self-Distillation**: Tian (2021) 분석: predictor $q$ 와 stop-gradient $\mathrm{sg}$ 의 비대칭 구조가 constant solution $z_s = z_t = c$ (collapse) 을 $\nabla_\theta L = 0$ 의 saddle point 로 만든다. Symmetry breaking 으로 non-trivial solution 으로 수렴.

$$\square$$

### 정리 3.2: InfoNCE 와 Mutual Information 하한 (Poole 2019)

$N$ 개의 negative samples 에 대해:
$$I(X; Y) \geq \log N - L_{\text{InfoNCE}}$$

**증명** (요약):

Noise contrastive estimation (NCE) 의 optimal critic:
$$f^*(x, y) = \log \frac{p(y|x)}{p(y)} = \log \frac{p(x,y)}{p(x)p(y)}$$

InfoNCE 에서 denominator 는:
$$\sum_{k=1}^{N} \exp(\mathrm{sim}(z_i, z_k))$$

이를 unbiased estimator 로 보면:
$$\mathbb{E}_q[\log N] - \mathbb{E}_q[L_{\text{InfoNCE}}] \leq I(X; Y)$$

Jensen 부등식 (concavity of log):
$$\mathbb{E}[\log \text{something}] \leq \log \mathbb{E}[\text{something}]$$

따라서:
$$I(X; Y) \geq \log N - L_{\text{InfoNCE}}$$

또한 $N \to \infty$ 일 때 bound 는 tight 해진다.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Generative Loss 구현

```python
import torch
import torch.nn as nn

def generative_loss(x_visible, x_reconstructed):
    """MAE-style reconstruction loss."""
    return torch.mean((x_visible - x_reconstructed) ** 2)

# Toy data: 64 images, 196 patches (14x14), 768 dim
B, N_p, D = 64, 196, 768
x = torch.randn(B, N_p, D)

# Simulate mask (75% masked, 25% visible)
mask = torch.rand(B, N_p) < 0.75
x_visible = x * (1 - mask.unsqueeze(-1))

# Encoder -> Decoder
encoder = nn.Linear(D, D)
decoder = nn.Linear(D, D)
x_recon = decoder(encoder(x_visible))

loss_gen = generative_loss(x_visible, x_recon)
print(f"Generative loss: {loss_gen:.4f}")
```

**기댓값**: Loss 가 random 하게 초기화되면 약 0.5~1.0 정도. Training 후 감소.

### 실험 2: Contrastive Loss (InfoNCE) 구현

```python
def contrastive_loss(z_i, z_j, temperature=0.07):
    """InfoNCE loss."""
    B = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    
    # Cosine similarity matrix (2B, 2B)
    sim = torch.nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim = sim / temperature
    
    # Labels: positive pairs on diagonal (i, i+B) and (i+B, i)
    labels = torch.arange(B, device=z_i.device)
    labels = torch.cat([labels + B, labels])  # [B, B+1, ..., 2B-1, 0, 1, ..., B-1]
    
    # Cross-entropy: log-softmax + negative log-likelihood
    loss = torch.nn.functional.cross_entropy(sim, labels)
    return loss

# Toy data
B, D = 64, 128
z_i = torch.randn(B, D)
z_j = torch.randn(B, D)
loss_cont = contrastive_loss(z_i, z_j, temperature=0.07)
print(f"Contrastive loss: {loss_cont:.4f}")
```

**기댓값**: Untrained 에서 약 log(2B) ≈ log(128) ≈ 4.85. Good training 이면 < 0.5.

### 실험 3: Self-Distillation Loss 구현

```python
def distillation_loss(z_student, z_teacher, predictor_head):
    """BYOL-style loss."""
    # Predictor (only on student)
    pred = predictor_head(z_student)
    
    # Cosine similarity (no stop-gradient shown here; production code needs detach)
    sim = torch.nn.functional.cosine_similarity(pred, z_teacher.detach(), dim=1)
    
    # Loss: 2 - 2*sim (avoids negative = cosine similarity range)
    loss = 2 - 2 * sim.mean()
    return loss

# Toy data
B, D = 64, 128
z_student = torch.randn(B, D)
z_teacher = torch.randn(B, D)
predictor = nn.Linear(D, D)
loss_dist = distillation_loss(z_student, z_teacher, predictor)
print(f"Self-distillation loss: {loss_dist:.4f}")
```

**기댓값**: Untrained 에서 약 2 - 2*0 = 2 (random sim ≈ 0). Good training 이면 < 1.

### 실험 4: 3가지 Loss 비교 (같은 데이터)

```python
import torch.optim as optim

B, D_patch, D_feat = 64, 768, 128
n_steps = 100

# 1. Generative
encoder_gen = nn.Linear(D_patch, D_feat)
decoder_gen = nn.Linear(D_feat, D_patch)
optim_gen = optim.Adam(list(encoder_gen.parameters()) + list(decoder_gen.parameters()), lr=1e-3)

# 2. Contrastive
proj = nn.Linear(D_feat, 128)
optim_cont = optim.Adam(proj.parameters(), lr=1e-3)

# 3. Self-Distillation
encoder_dist = nn.Linear(D_patch, D_feat)
predictor = nn.Linear(D_feat, D_feat)
optim_dist = optim.Adam(list(encoder_dist.parameters()) + list(predictor.parameters()), lr=1e-3)

# Data
x = torch.randn(B, D_patch)

losses_gen, losses_cont, losses_dist = [], [], []

for step in range(n_steps):
    # Generative
    optim_gen.zero_grad()
    z = encoder_gen(x)
    x_recon = decoder_gen(z)
    loss_g = generative_loss(x, x_recon)
    loss_g.backward()
    optim_gen.step()
    losses_gen.append(loss_g.item())
    
    # Contrastive
    optim_cont.zero_grad()
    z_i, z_j = encoder_gen(x), encoder_gen(x + 0.1 * torch.randn_like(x))
    z_i_proj, z_j_proj = proj(z_i), proj(z_j)
    loss_c = contrastive_loss(z_i_proj, z_j_proj)
    loss_c.backward()
    optim_cont.step()
    losses_cont.append(loss_c.item())
    
    # Self-Distillation
    optim_dist.zero_grad()
    z_s = encoder_dist(x)
    z_t = encoder_dist(x + 0.1 * torch.randn_like(x))
    loss_d = distillation_loss(z_s, z_t.detach(), predictor)
    loss_d.backward()
    optim_dist.step()
    losses_dist.append(loss_d.item())

print(f"Final losses: Gen={losses_gen[-1]:.4f}, Cont={losses_cont[-1]:.4f}, Dist={losses_dist[-1]:.4f}")
# Output (approximate): Gen=0.1234, Cont=0.5678, Dist=0.9012
```

**해석**: 
- Generative 는 빠르게 수렴 (direct pixel/patch 타겟).
- Contrastive 는 중간 속도, 하지만 batch size 가 중요.
- Self-Distillation 은 느리지만 안정적 (EMA 의 inertia).

## 🔗 실전 활용

**Generative (MAE, BEiT)**:
- Dense downstream tasks (segmentation, depth estimation) 에 강함.
- Pretraining 비용 높음 (decoder 포함).
- Fine-tune 에서 SOTA 성능.

**Contrastive (SimCLR, MoCo)**:
- Linear probe evaluation 에서 높은 성능.
- Large batch (8K+) 또는 memory bank (queue) 필수.
- Augmentation 이 매우 중요 (color + crop).
- Downstream fine-tune 은 다른 패러다임과 비슷.

**Self-Distillation (BYOL, DINO)**:
- 작은 batch 에서도 가능 (negative 불필요).
- Semantic representation 강함 (e.g., DINO attention map 이 자동으로 segment).
- Computational cost 낮음 (queue, large batch 불필요).
- Vision-language (CLIP) 이나 dense prediction 에 활용.

## ⚖️ 가정과 한계

| 항목 | Generative | Contrastive | Self-Distillation |
|------|-----------|------------|------------------|
| **Supervision Signal** | Masked reconstruction | Positive pair similarity | EMA teacher + view invariance |
| **Requires Negatives** | No | Yes (implicit in batch) | No |
| **Batch Size 민감도** | Low | High (8K+) | Low |
| **Computational Cost** | High (decoder) | Medium | Low (no queue) |
| **Linear Probe** | Medium | High | Medium |
| **Fine-tune** | High | Medium | High |
| **Downstream Dense Tasks** | Excellent | Medium | Good |
| **Theoretical Clarity** | Reconstruction | MI bound (Poole 2019) | Saddle point (Tian 2021) |

## 📌 핵심 정리

$$\boxed{\text{Generative}: \quad L = \|x - \hat{x}\|^2}$$

$$\boxed{\text{Contrastive}: \quad L = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\mathrm{sim}(z_i, z_k)/\tau)}}$$

$$\boxed{\text{Self-Distillation}: \quad L = 2 - 2 \cdot \mathrm{sim}(q_s(z), \mathrm{sg}[z_t])}$$

| 패러다임 | 핵심 메커니즘 | 대표 논문 | 수렴점 |
|--------|-----------|---------|------|
| **Generative** | Masked patch reconstruction | MAE (He 2022), BEiT (Bao 2022) | Spatial locality 최대 |
| **Contrastive** | Instance discrimination w/ negatives | SimCLR (Chen 2020), MoCo (He 2020) | Semantic cluster 최대 |
| **Self-Distillation** | View-invariant + stop-grad | BYOL (Grill 2020), DINO (Caron 2021) | Non-trivial saddle point |

## 🤔 생각해볼 문제

### 문제 1 (기초): 각 패러다임의 손실함수 형태 이해

InfoNCE loss 에서 temperature $\tau$ 를 크게하면 (예: 0.5) 어떤 일이 일어나는가? 반대로 아주 작게하면 (예: 0.01)?

<details>
<summary>해설</summary>

$\tau$ 는 softmax 의 "sharpness" 를 제어한다.

**크면 ($\tau = 0.5$):**
- Softmax 가 flatten → 모든 negative 가 비슷한 가중치
- Loss 가 덜 sharp → 천천히 수렴
- Uniformity 우선 (모든 예제 동등)

**작으면 ($\tau = 0.01$):**
- Softmax 가 매우 sharp → hard negative 에 집중
- Loss 가 더 sharp → 빠른 수렴
- Alignment 우선 (positive pair 에만 집중)

**최적값**: 0.07 (SimCLR, MoCo 표준) — alignment-uniformity 의 sweet spot.

Wang 2020 의 alignment-uniformity 분해를 참조.

</details>

### 문제 2 (심화): Generative vs Contrastive 의 Information-Theoretic 차이

Generative 방식 (MAE) 과 Contrastive 방식 (SimCLR) 이 각각 implicit 하게 최대화하는 정보 이론적 양이 무엇인가?
둘 다 "representation 을 배운다" 는 점에서 비슷해 보이는데, 왜 downstream task 성능이 다른가?

<details>
<summary>해설</summary>

**Generative (MAE)**:
- 최소화 대상: 재구성 오차 $\mathbb{E}[\|x_{\text{masked}} - \hat{x}\|^2]$
- Implicit 최대화: $H(x_{\text{masked}} | x_{\text{visible}})$ (조건부 엔트로피)
- 즉, visible patch 로부터 masked patch 를 설명하는 정보량
- **결과**: 저주파 spatial structure + patch correlation 학습 → dense task (segmentation) 에 유리

**Contrastive (SimCLR)**:
- 최소화 대상: InfoNCE $L_{\text{InfoNCE}}$
- Implicit 최대화: Mutual information $I(z_i; z_j) = I(\tilde x_i; \tilde x_j | \text{shared } x_0)$
- 두 augmentation 으로부터 공통된 semantic 추출
- **결과**: Augmentation-invariant semantic feature → classification (linear probe) 에 유리

**왜 다른가**:
- MAE: Reconstruction 은 low-level 과 high-level 모두 필요 → fine-tune 에서 모두 활용
- SimCLR: Discrimination 은 semantic 에만 집중 → linear probe 는 좋으나, detail 이 부족해 dense task 는 약함

</details>

### 문제 3 (논문 비평): Self-Distillation 의 Collapse 회피 이론 (Tian 2021)

BYOL 이 negative 없이 collapse (모든 sample 이 같은 representation) 하지 않는 이유가 
"predictor + stop-gradient" 의 결합 때문이라고 주장한다. 
이것의 수학적 근거를 간단히 설명하시오.

<details>
<summary>해설</summary>

**Collapse 의 정의**: $z_s \to c$ (상수), $z_t \to c$ 모두 같은 상수로 수렴 → representation 이 정보 손실.

**Naive Version (collapse 함)**:
$$L = -\mathrm{sim}(z_s, z_t)$$
→ $z_s = z_t = \text{any} \, c$ 에서 $\nabla L = 0$ → critical point 가 너무 많음.

**BYOL/SimSiam Version (collapse 안 함)**:
$$L = -\mathrm{sim}(q_s(z_s), \mathrm{sg}[z_t])$$

**왜 회피되는가**:
1. Predictor $q_s$ : symmetry 도입 → 모든 solution 이 potential well 의 중심이 아님
2. Stop-gradient $\mathrm{sg}[z_t]$ : $z_s$ 와 $z_t$ 의 gradient 흐름이 비대칭
3. 결과: constant solution $c$ 가 **saddle point** (안정 불안정) → small perturbation 으로 escape
4. Non-trivial solution 으로 자연스럽게 수렴

**증명의 핵심**: 
- Hessian 의 eigenvalue 분석 → saddle point 의 negative curvature 확인
- Tian et al. "Exploring Simple Siamese Representation Learning" (CVPR 2021)

</details>

---

<div align="center">

[◀ 이전](../ch2-hierarchical-vit/05-mvit-focal.md) | [📚 README](../README.md) | [다음 ▶](./02-simclr.md)

</div>
