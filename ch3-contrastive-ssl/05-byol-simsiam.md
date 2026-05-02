# 05. BYOL · SimSiam — Without Negatives

## 🎯 핵심 질문

- BYOL 과 SimSiam 이 negative pair 없이 collapse 하지 않는 이유는 정말 "수학적 이유" 인가, 아니면 경험적 관찰인가?
- Tian et al. (2021) 의 saddle point 분석 — predictor + stop-gradient 의 정보기하학적 역할은?
- BYOL 의 EMA target 과 SimSiam 의 stop-gradient 의 본질적 차이는?
- 왜 predictor $q$ 를 제거하거나 stop-gradient 를 없으면 collapse 하는가?
- Collapse mode (모든 배치가 같은 representation) vs dimensional collapse (한 차원으로만 표현) 의 차이는?

## 🔍 왜 Negative 없이도 작동하는가

2020~2021년 혁신: BYOL (Grill et al.) 과 SimSiam (Chen et al.) 은
**negative pair 없이** self-supervised learning 이 가능함을 보여줬다.

이전 패러다임:
- Contrastive learning (SimCLR, MoCo): "positive 와 negative 를 구분"
- Information theory bound (Poole 2019): "많은 negative 가 필요"

새로운 패러다임:
- Self-distillation (BYOL): Student 가 teacher (EMA) 를 모방
- Implicit momentum (SimSiam): Predictor + stop-gradient = implicit EMA

핵심: **Predictor + stop-gradient** 또는 **EMA teacher** 가 collapse 를 방지.
이는 단순히 empirical trick 이 아니라, **saddle point 분석** (Tian 2021) 으로 증명 가능.

## 📐 수학적 선행 조건

- **Gradient Flow**: Neural network 의 미분과 역전파
- **Symmetry Breaking**: 대칭 시스템에서 비대칭 해로의 분기
- **Hessian Analysis**: Critical point 의 stability (eigenvalue 부호)
- **Implicit Regularization**: Optimization 자체가 주는 regularization
- **Stop-Gradient Operation**: Computational graph 에서 gradient 차단

## 📖 직관적 이해

```
┌──────────────────────────────────────────────────────────┐
│ Why Negatives Aren't Needed: A Symmetry Argument         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ COLLAPSE Problem:                                        │
│   All samples → same constant representation c           │
│   Loss: L(c, c) = 0 (trivial solution)                   │
│   = All information lost                                 │
│                                                          │
│ Solution 1 (Contrastive): Negatives prevent collapse      │
│   "Different samples must map to different features"     │
│                                                          │
│ Solution 2 (Self-Distillation): Asymmetry prevents it    │
│   Student θ_s tries to match Teacher θ_t (EMA)          │
│   But θ_t moves slower → forces θ_s to "chase"          │
│   = Can't all be same constant                          │
│                                                          │
│ Solution 3 (Predictor + Stop-grad): Asymmetry            │
│   min_θ L = 2 - 2·sim(q(z_s), sg[z_t])                  │
│   q: learnable predictor MLP                            │
│   sg: stop-gradient (breaks gradient flow)              │
│   Asymmetry between z_s (optimized) & sg[z_t] (frozen) │
│   = Saddle point analysis: constant is unstable!        │
│                                                          │
└──────────────────────────────────────────────────────────┘

Analogy:
  Contrastive: "Everyone is different" (enforcement)
  Self-Distill: "Follow a moving target" (pursuit)
  Predictor+SG: "Asymmetric game" (equilibrium)
```

**비유**: 마술 거울의 게임.
- Contrastive: 각자 다른 옷을 입으라 (강제).
- BYOL: 어제 자신의 이미지를 따라가라 (추격).
- SimSiam: 거울을 통해 자신을 보되, 거울을 직접 만질 수 없다 (비대칭).

## ✏️ 엄밀한 정의

### 정의 3.13: BYOL Loss

**Pipeline**:
1. Input image $x$ 에서 두 augmentation: $\tilde{x}_1, \tilde{x}_2$
2. Student encoder + projection: $z_1 = f_s(\tilde{x}_1)$
3. Target encoder (EMA) + projection: $z_2' = f_t(\tilde{x}_2)$ (with $f_t$ not on gradient path)
4. Student 에만 predictor $q$ 추가: $\hat{z}_1 = q(z_1)$ (extra MLP)

**Loss** (asymmetric, one direction):
$$L_{\text{BYOL}} = 2 - 2 \cdot \mathrm{sim}(\hat{z}_1, \mathrm{sg}[z_2'])$$

where:
- $\mathrm{sim} = $ cosine similarity
- $\mathrm{sg}[z_2']$ = stop-gradient (detach from graph)
- Symmetric version: $L = L_{\text{forward}} + L_{\text{backward}}$ (both directions)

**Target encoder update** (no gradient):
$$f_t \leftarrow \lambda f_t + (1 - \lambda) f_s, \quad \lambda = 0.999$$

### 정의 3.14: SimSiam Loss

Simpler version (no EMA, only stop-gradient):

**Pipeline**:
1. Same augmentation as BYOL
2. One encoder $f$ (shared): $z_1 = f(\tilde{x}_1), z_2 = f(\tilde{x}_2)$
3. Predictor $q$ on both: $p_1 = q(z_1), p_2 = q(z_2)$

**Loss** (symmetric):
$$L_{\text{SimSiam}} = -\frac{1}{2} (\mathrm{sim}(p_1, \mathrm{sg}[z_2]) + \mathrm{sim}(p_2, \mathrm{sg}[z_1]))$$

**Key difference from BYOL**:
- No EMA target encoder (single $f$)
- Stop-gradient only on $z$, not on predictor
- Simpler implementation

### 정의 3.15: Stop-Gradient Operation

Computational graph 에서:
$$\mathrm{sg}[x] = \begin{cases}
x & \text{forward pass (value)} \\
0 & \text{backward pass (gradient)}
\end{cases}$$

**PyTorch 구현**:
```python
def stop_gradient(x):
    return x.detach()  # or: x.clone().detach()
```

**Effect**: Gradient 가 $\mathrm{sg}[x]$ 를 통해 역전파되지 않음.
즉, loss 에서 $\mathrm{sg}[x]$ 의 값은 변하지만, gradient 에는 영향을 주지 않음.

## 🔬 정리와 증명

### 정리 3.11: Tian et al. (2021) — Saddle Point Analysis of SimSiam

**핵심 주장**: SimSiam loss 의 constant solution $z_s = z_t = c$ (collapse) 은
**saddle point** 이므로, small perturbation 으로도 escape 가능.

**정리 (informal)**:

Consider simplified loss:
$$L(\theta) = -\mathrm{sim}(q(z_s(\theta)), z_t(\theta))$$

where $z_s, z_t$ are encoder outputs (both depend on $\theta$ for $z_s$, 
$z_t$ depends on $\theta$ for stop-grad but gradient $\approx 0$).

**Case 1: Without predictor $q$ (direct similarity)**
$$L_{\text{no pred}} = -\mathrm{sim}(z_s(\theta), \mathrm{sg}[z_s(\theta)])$$

이 경우, $z_s = c$ (상수) 가 critical point.
Hessian $H = \nabla^2 L|_{z_s=c}$ 분석:
- 모든 eigenvalue < 0 → 극소점 (minimum)
- = Stable collapse mode!

**Case 2: With predictor $q$ (asymmetric)**
$$L_{\text{pred}} = -\mathrm{sim}(q(z_s(\theta)), \mathrm{sg}[z_s(\theta)])$$

Predictor $q(z) = Wz + b$ (affine) 추가:
- Hessian 이 mixed signs (positive & negative eigenvalues) → saddle point
- = Unstable collapse!

**증명 스케치**:

Define potential function $\Phi(z) = -\mathrm{sim}(q(z), z)$:

Without $q$: $\Phi(z) = -\mathrm{sim}(z, z) = -1$ for all normalized $z$ → flat landscape

With $q$: Predictor 가 $q(z) \neq z$ 를 학습 → non-trivial $\Phi$ landscape → critical points 가 saddle

자세한 수학은 Tian et al. "Exploring Simple Siamese Representation Learning" (CVPR 2021) 참조.

$$\square$$

### 정리 3.12: EMA Target 의 안정성 (BYOL)

**주장**: EMA target encoder $f_t \leftarrow \lambda f_t + (1-\lambda) f_s$ 가
SimSiam 의 predictor + stop-gradient 와 유사한 역할.

**증명 (직관적)**:

BYOL loss:
$$L = 2 - 2 \cdot \mathrm{sim}(q(f_s(\tilde{x}_1)), f_t(\tilde{x}_2))$$

EMA 때문에 $f_t$ 는 천천히 변함.
$t$ 시점에서: $\|f_t - f_s\| \approx (1-\lambda) \Delta$ (lag proportional to learning speed).

이 "lag" 이 predictor $q$ 처럼 작동:
- $f_s$ 와 $f_t$ 의 차이 → implicit asymmetry
- Collapse 시도 시 $f_t$ 가 천천히 따라와서 방해

따라서:
$$\text{EMA (BYOL)} \approx \text{Stop-gradient + Predictor (SimSiam)} + \text{Implicit momentum}$$

$$\square$$

### 정리 3.13: Dimensional Collapse vs Full Collapse

**Dimensional collapse**:
- Encoder output 이 한 차원 (또는 낮은 차원) 에 collapse.
- 예: $z_i \approx (\alpha, 0, 0, \ldots, 0)$ (첫 차원만)
- Information loss 는 적지만, semantic diversity 감소.

**Full collapse**:
- 모든 샘플이 동일한 representation: $z_i = c$ for all $i$
- Complete information loss.

**Prevention**:
- Contrastive: negative pairs 로 강제 분산 (uniformity)
- BYOL/SimSiam: predictor + EMA 로 implicit diversity

**관찰**: BYOL/SimSiam 은 full collapse 는 방지하지만, 
dimensional collapse 가능성 있음 (empirically rare).

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: BYOL Loss 구현

```python
import torch
import torch.nn as nn

class BYOLEncoder(nn.Module):
    def __init__(self, input_dim=224*224*3, hidden_dim=512, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 128)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

class BYOLModel(nn.Module):
    def __init__(self, input_dim=100, tau=0.999):
        super().__init__()
        self.student = BYOLEncoder(input_dim, 256, 256)
        self.target = BYOLEncoder(input_dim, 256, 256)
        
        # Copy initial weights
        self.target.load_state_dict(self.student.state_dict())
        
        # Predictor only on student
        self.predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.tau = tau
    
    def forward(self, x1, x2):
        z_s = self.student(x1)
        p_s = self.predictor(z_s)
        
        with torch.no_grad():
            z_t = self.target(x2)
        
        # BYOL loss: 2 - 2*sim(p_s, sg[z_t])
        z_s_norm = torch.nn.functional.normalize(p_s, dim=1)
        z_t_norm = torch.nn.functional.normalize(z_t.detach(), dim=1)
        
        loss = 2 - 2 * torch.sum(z_s_norm * z_t_norm, dim=1).mean()
        return loss
    
    def update_target(self):
        """EMA update of target encoder."""
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.target.parameters()):
                param_t.data = self.tau * param_t.data + (1 - self.tau) * param_s.data

# Test
model = BYOLModel(input_dim=100, tau=0.999)
optimizer = torch.optim.Adam(
    list(model.student.parameters()) + list(model.predictor.parameters()),
    lr=1e-3
)

for step in range(50):
    x1 = torch.randn(32, 100)
    x2 = torch.randn(32, 100)
    
    loss = model.forward(x1, x2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.update_target()
    
    if step % 10 == 0:
        print(f"Step {step}: loss = {loss:.4f}")
```

**Expected output**: Loss decreases from ~2.0 to ~0.5

### 실험 2: SimSiam Loss 구현

```python
class SimSiamModel(nn.Module):
    def __init__(self, input_dim=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        p1 = self.predictor(z1)
        
        z2 = self.encoder(x2)
        p2 = self.predictor(z2)
        
        # Symmetric loss with stop-gradient
        z1_norm = torch.nn.functional.normalize(z1, dim=1)
        z2_norm = torch.nn.functional.normalize(z2, dim=1)
        p1_norm = torch.nn.functional.normalize(p1, dim=1)
        p2_norm = torch.nn.functional.normalize(p2, dim=1)
        
        loss = -(torch.sum(p1_norm * z2_norm.detach(), dim=1).mean() +
                torch.sum(p2_norm * z1_norm.detach(), dim=1).mean()) / 2
        
        return loss

# Test
model = SimSiamModel(input_dim=100)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for step in range(50):
    x1 = torch.randn(32, 100)
    x2 = torch.randn(32, 100)
    
    loss = model.forward(x1, x2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step}: loss = {loss:.4f}")
```

### 실험 3: Collapse Ablation (Removing Predictor)

```python
def train_without_predictor():
    """SimSiam without predictor — should collapse."""
    
    class SimpleSiam(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        
        def forward(self, x1, x2):
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            
            z1_norm = torch.nn.functional.normalize(z1, dim=1)
            z2_norm = torch.nn.functional.normalize(z2, dim=1)
            
            # No predictor, direct similarity
            loss = -(torch.sum(z1_norm * z2_norm.detach(), dim=1).mean())
            
            return loss
    
    model = SimpleSiam()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    
    representations = []
    
    for step in range(100):
        x = torch.randn(32, 100)
        
        loss = model.forward(x, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Monitor representation collapse
        with torch.no_grad():
            z = torch.nn.functional.normalize(model.encoder(x), dim=1)
            center = z.mean(dim=0)
            representations.append(center.norm().item())
    
    return representations

reprs = train_without_predictor()
print("Representation center norm (should → 1.0 for collapse):")
print(f"Initial: {reprs[0]:.4f}")
print(f"Final: {reprs[-1]:.4f}")
# Expected: Final → close to 1.0 (collapsed to constant vector)
```

### 실험 4: Predictor vs No Predictor (Comparison)

```python
def compare_with_without_predictor():
    """Train SimSiam with and without predictor."""
    
    results_with = []
    results_without = []
    
    for use_predictor in [True, False]:
        if use_predictor:
            model = SimSiamModel(input_dim=100)
        else:
            model = SimpleSiam()
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        
        collapse_measure = []
        
        for step in range(50):
            x1 = torch.randn(32, 100)
            x2 = torch.randn(32, 100)
            
            if use_predictor:
                loss = model.forward(x1, x2)
            else:
                loss = model.forward(x1, x2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Measure collapse: variance across batch
            with torch.no_grad():
                z = torch.nn.functional.normalize(model.encoder(x1), dim=1)
                # Std of features across batch (low = collapse)
                collapse = z.std(dim=0).mean().item()
                collapse_measure.append(collapse)
        
        if use_predictor:
            results_with = collapse_measure
        else:
            results_without = collapse_measure
    
    print("Feature std (measure of collapse avoidance):")
    print(f"With predictor:    final = {results_with[-1]:.4f}")
    print(f"Without predictor: final = {results_without[-1]:.4f}")
    # Expected: With > Without (predictor prevents collapse)
```

## 🔗 실전 활용

**Scalability**:
- Negative 불필요 → smaller batch OK (256 이상면 충분)
- Distributed training 간단 (queue, large batch 불필요)
- Single GPU 훈련 가능

**Advantages over contrastive**:
- Memory-efficient (negative 저장 불필요)
- Stable training (no batch normalization dependency)
- Competitive or superior downstream performance

**Disadvantages**:
- EMA target (BYOL) 또는 stop-gradient (SimSiam) 의 implicit mechanism
- 수학적 이해가 덜 명확 (Poole bound 같은 명시적 theory 없음)
- 일부 domain 에서는 contrastive 가 우위

## ⚖️ 가정과 한계

| 항목 | 설명 | 한계 |
|------|------|------|
| **Predictor 필수성** | Stop-gradient + predictor 의 비대칭성이 collapse 방지 | 너무 복잡한 predictor 는 역효과 가능 |
| **EMA momentum** | 느린 target 이 stability 제공 (BYOL) | 너무 빠른 EMA ($\lambda$ 작음) 는 collapse 위험 |
| **Stop-gradient** | Gradient 흐름 차단이 asymmetry 생성 | 완전히 frozen 되는 것은 아님 (forward 는 있음) |
| **Representation quality** | Collapse 회피 = 다양한 representations | Contrastive 만큼 semantic 할지는 task 의존 |

## 📌 핵심 정리

$$\boxed{L_{\text{BYOL}} = 2 - 2 \cdot \mathrm{sim}(q_\theta(z_1), \mathrm{sg}[z_2'])}$$

$$\boxed{L_{\text{SimSiam}} = -\frac{1}{2}(\mathrm{sim}(p_1, \mathrm{sg}[z_2]) + \mathrm{sim}(p_2, \mathrm{sg}[z_1]))}$$

$$\boxed{\text{Saddle Point}: \quad \text{constant solution is unstable with } q + \mathrm{sg}}$$

| 항목 | BYOL | SimSiam | Contrastive |
|------|------|---------|------------|
| **Negative 필요** | No | No | Yes (large batch) |
| **EMA target** | Yes ($\tau=0.999$) | No | No |
| **Predictor** | Yes | Yes | No |
| **Batch size** | 256 가능 | 256 가능 | 4096+ 권장 |
| **이론적 명확성** | 중간 (saddle point) | 중간 (saddle point) | 높음 (MI bound) |

## 🤔 생각해볼 문제

### 문제 1 (기초): Stop-Gradient 의 역할

SimSiam 에서 stop-gradient 를 제거하면 (즉, $z_2$ 대신 $z_2.detach()$ 제거):
```python
loss = -torch.sum(p1 * z2, dim=1).mean()  # 오류!
```

왜 collapse 가 발생하는가? Gradient flow 관점에서 설명하시오.

<details>
<summary>해설</summary>

**With stop-gradient** ($\mathrm{sg}[z_2]$):
- Loss $= -\mathrm{sim}(p_1, \mathrm{sg}[z_2])$
- Gradient: $\frac{\partial L}{\partial p_1} \neq 0$, but $\frac{\partial L}{\partial z_2} = 0$ (detached)
- $p_1$ 과 $z_2$ 가 독립적으로 최적화 → asymmetry 생성

**Without stop-gradient** ($z_2$ 직접 사용):
- Loss $= -\mathrm{sim}(p_1, z_2)$
- Gradient: both $p_1$ 과 $z_2$ 에 대해 nonzero
- $p_1 = z_2$ 로 converge 하면, loss $= -1$
- 그런데 encoder 는 $p_1$ 을 만들므로, 
  $\mathrm{encoder} \to z_1 \to p_1 = z_2$ 로 강제
- 따라서 $z_1 = z_2 \to $ all samples map to same vector $\to$ **collapse**

**수학적으로**: Gradient 의 순환 경로가 생겨, 
모든 layer 가 같은 collapse 방향으로 업데이트.

</details>

### 문제 2 (심화): EMA vs Predictor — Which is Better?

BYOL (EMA) vs SimSiam (predictor only) 의 representation quality 를
정보이론적 관점에서 비교하시오.

특히, EMA 는 "slow-moving target" 이고 predictor 는 "affine transformation" 인데,
둘이 같은 collapse-prevention mechanism 을 제공하는가?

<details>
<summary>해설</summary>

**BYOL (EMA target)**:
- Target $z_t$ 가 계속 변함 (lag ≈ 1000 steps)
- Student 는 moving target 을 항상 따라가려고 함
- Implicit information: "track a slow-moving model"
- Result: rich representation (temporal consistency)

**SimSiam (predictor)**:
- Predictor $q$ 는 학습 가능한 mapping
- $q(z_1)$ 이 $z_2$ 를 맞춰야 하므로, 
  $q$ 가 점점 identity 에 가까워짐
- Implicit information: "learn an invertible mapping"
- Result: balanced representation (symmetry + asymmetry balance)

**정보이론적 비교**:
- EMA: 매 step 마다 target 이 변해, diversity 유지
- Predictor: $q$ 가 learned mapping 이므로, 
  encoder 의 representation space 를 직접 "twist" 하는 효과

**Empirically**: 
- BYOL: linear probe 약간 우수 (EMA 의 temporal diversity)
- SimSiam: fine-tune 에서 비슷
- 큰 차이 없음 (둘 다 collapse 회피 성공)

**결론**: 서로 다른 mechanism 이지만, 최종 representation quality 는 비슷.

</details>

### 문제 3 (논문 비평): Tian 2021 의 Saddle Point 분석 한계

Tian et al. (2021) 은 SimpleS 의 collapse 회피를 saddle point 분석으로 설명했다.

하지만 이 분석은 "simplified" model (affine predictor, quadratic-ish landscape) 을 가정한다.
실제 neural network 는 매우 복잡한 landscape 를 가지고 있다.

실제 collapse 회피의 주요 기여자는:
1. Saddle point 기하학?
2. Implicit regularization (SGD 자체)?
3. Batch normalization 의 부작용?
4. 뭔가 다른 것?

토론하시오.

<details>
<summary>해설</summary>

**Tian 분석의 한계**:
- Simplified quadratic loss 가정
- Affine predictor (선형)
- 실제 네트워크는 고도 nonlinear

**실제 collapse 회피의 다양한 요인**:

1. **Saddle point geometry** (Tian 의 주장):
   - 맞음, 하지만 이것만으로는 충분하지 않을 수 있음
   - Landscape 가 복잡해서 이론적 분석 어려움

2. **Implicit regularization** (SGD):
   - SGD 가 특정 solution 을 "선호"
   - Batch size, learning rate 에 따라 implicit bias 존재
   - Theoretical: "implicit bias of SGD" (연구 중)

3. **Batch normalization**:
   - BN 이 없으면 collapse 더 쉬움
   - BN 이 hidden representation 을 normalize → diversity 유지
   - SimSiam 의 성공에 BN 역할 큼

4. **Actual empirical observation**:
   - 모든 요인이 함께 작동: 
     "predictor + stop-grad + SGD + BN + batch diversity"
   - 단일 인수(single factor) 로 설명 불가

**결론**:
- Saddle point 분석은 "one piece of the puzzle"
- Complete explanation 은 여러 이론 (optimization, statistics, geometry) 의 조합 필요
- Active research area

</details>

---

<div align="center">

[◀ 이전](./04-moco.md) | [📚 README](../README.md) | [다음 ▶](../ch4-dino/01-dino-algorithm.md)

</div>
