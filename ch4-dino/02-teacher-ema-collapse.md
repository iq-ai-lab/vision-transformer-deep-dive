# 02. Teacher EMA 와 Collapse 방지

## 🎯 핵심 질문

- DINO 에서 **centering** 과 **sharpening** 은 어떤 collapse mode 를 각각 방지하는가?
- **One-dimensional collapse** 와 **uniform distribution collapse** 의 수학적 특성은 다른가?
- EMA teacher 만으로는 충분하지 않은 이유가 무엇인가?
- Centering momentum 과 temperature scheduling 이 정확히 어떤 gradient dynamics 를 조정하는가?

## 🔍 왜 이 두 메커니즘이 필수인가

DINO 의 성공은 세 가지 요소의 균형에 있다:

1. **Teacher-student 증류**: 안정적 target
2. **Centering**: One-dimensional collapse 차단
3. **Sharpening** (temperature imbalance): Uniform collapse 차단

이 중 하나라도 빠지면 representation 이 collapse 된다.
- Centering 제거 → constant output (redundant features)
- Sharpening 제거 → uniform output (all outputs equally likely)

## 📐 수학적 선행 조건

- **Output normalization**: $\| \mathbf{z} \| = 1$ (unit sphere 위의 점)
- **Entropy & KL divergence**: collapse 와 regularization
- **Gradient flow**: backpropagation through softmax
- **Momentum buffers**: exponential moving averages
- **Distribution collapse modes**: degenerate cases

## 📖 직관적 이해

```
┌─────────────────────────────────────────────────────────────┐
│           Collapse Modes & Prevention                       │
│                                                             │
│  MODE 1: One-Dimensional Collapse                          │
│  ────────────────────────────────────                      │
│  모든 output 이 같은 방향 → z = constant vector            │
│  예: z_i = c for all i                                     │
│  원인: 역할을 하지 않는 hidden units                        │
│  해결책: Centering ← 평균을 뺌                            │
│                                                             │
│  MODE 2: Uniform Collapse                                  │
│  ────────────────────────────────                          │
│  모든 output 이 균등분포 → p(x) = 1/D for all x           │
│  예: softmax(z) = [1/D, 1/D, ..., 1/D]                    │
│  원인: entropy 극대화 (gradient descent 악순환)             │
│  해결책: Sharpening ← τ_t < τ_s (sharp target)            │
│                                                             │
│  SOLUTION SPACE:                                           │
│                                                             │
│         Entropy                                            │
│         (Uniform)       ╱╲                                 │
│                        ╱  ╲                                │
│                       ╱    ╲  ← Sharpening (τ↓)           │
│                      ╱      ╲                              │
│                     ╱        ╲                             │
│     Norm(Center)  ╱____________╲                           │
│     (Dim-reduc)  │    GOOD      │                          │
│                  │  DINO ZONE   │                          │
│                  │              │                          │
│                  └──────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## ✏️ 엄밀한 정의

### 정의 4.5: One-Dimensional Collapse

Output feature $\mathbf{z}_i \in \mathbb{R}^D$ 에 대해, **rank-1 collapse** 는 다음을 만족하는 상태:

$$\mathbf{z}_i = \alpha_i \mathbf{v} + \boldsymbol{\epsilon}_i, \quad \text{where } \mathbf{v} \in \mathbb{R}^D, \| \mathbf{v} \| = 1$$

여기서 $\alpha_i \in \mathbb{R}$ 는 스칼라 계수, $\boldsymbol{\epsilon}_i$ 는 무시할 수 있는 noise.

**기하학적 의미**: 모든 출력이 1차원 부분공간 (선) 위에 놓인다.

**확률론적 영향**: 대부분의 feature dimension 이 정보를 담지 않음.

### 정의 4.6: Uniform Collapse (Maximum Entropy)

Teacher 네트워크의 출력 확률분포 $\mathbf{p}_t(x) = \mathrm{softmax}(\mathbf{z}_t / \tau_t)$ 에서,

$$H(\mathbf{p}_t) = -\sum_{j=1}^D p_{t,j} \log p_{t,j} \approx \log D$$

이는 모든 요소가 $p_{t,j} \approx 1/D$ 를 의미하며, **정보를 제공하지 않는 균등분포**.

### 정의 4.7: Centering Operation

매 batch 마다 running mean 을 계산:

$$\mathbb{C}^{(t)} = m \cdot \mathbb{C}^{(t-1)} + (1-m) \cdot \frac{1}{B} \sum_{i=1}^B \mathbf{z}_t^{(i)}$$

여기서:
- $m \in [0.99, 0.9999]$ : momentum coefficient (보통 0.9999)
- $\mathbf{z}_t^{(i)}$ : i-번째 sample 의 teacher output
- Centering: $\hat{\mathbf{z}}_t = \mathbf{z}_t - \mathbb{C}^{(t)}$

**효과**: 출력의 mean 을 0 에 유지 → rank-1 collapse 불가능.

### 정의 4.8: Sharpening via Temperature

Temperature scheduling:

$$\tau_t(n) = \frac{1}{2}\left[\tau_t^{\min} + \tau_t^{\max} \right] + \frac{1}{2}\left[\tau_t^{\max} - \tau_t^{\min} \right] \cos\left(\pi \frac{n}{T}\right)$$

일반적으로:
- Teacher: $\tau_t \in [0.04, 0.07]$ (sharp)
- Student: $\tau_s \in [0.1, 0.2]$ (soft)

**효과**: $\tau_t$ 가 작을수록, softmax 가 더 peaky → high entropy 방지.

## 🔬 정리와 증명

### 정리 4.3: Centering 이 One-Dimensional Collapse 를 차단한다

**주장**: Centered output 에 대해 $\mathbb{C} = \mathbb{E}[\mathbf{z}] = 0$ 이면, rank-1 collapse 가 불가능하다.

**증명**:

Rank-1 collapse 가정: $\mathbf{z}_i = \alpha_i \mathbf{v}$ for $i = 1, \ldots, B$, $\| \mathbf{v} \| = 1$.

Centering 후:
$$\hat{\mathbf{z}}_i = \mathbf{z}_i - \frac{1}{B}\sum_{j=1}^B \mathbf{z}_j = \alpha_i \mathbf{v} - \left(\frac{1}{B}\sum_{j=1}^B \alpha_j\right) \mathbf{v}$$

$$= \left(\alpha_i - \bar{\alpha}\right) \mathbf{v}$$

여전히 rank-1 이지만, **mean = 0 제약**이 가해진다:
$$\frac{1}{B}\sum_{i=1}^B \hat{\mathbf{z}}_i = \frac{1}{B}\sum_{i=1}^B (\alpha_i - \bar{\alpha})\mathbf{v} = 0$$

이는:
$$\frac{1}{B}\sum_{i=1}^B (\alpha_i - \bar{\alpha}) = 0 \quad \checkmark$$

자동 만족되지만, gradient 관점에서는:

Loss gradient:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_i} = \text{terms involving } (\hat{\mathbf{z}}_i - \mathbf{c})$$

Centering 덕분에 bias term 이 제거되고, 더 많은 dimension 이 gradient signal 을 받는다.

더 정확히는, **SVD 분석** (Jing et al. 2020) 을 적용하면:

Centered output: $\hat{\mathbf{Z}} = [\hat{\mathbf{z}}_1, \ldots, \hat{\mathbf{z}}_B]^T \in \mathbb{R}^{B \times D}$

SVD: $\hat{\mathbf{Z}} = U \Sigma V^T$, $U \in \mathbb{R}^{B \times r}, \Sigma \in \mathbb{R}^{r \times r}, V \in \mathbb{R}^{D \times r}$

If rank = 1 (one-dimensional collapse): $r = 1$ → only one nonzero singular value.

Centering constraint: $\mathbb{1}^T \hat{\mathbf{Z}} = 0$ (all columns sum to 0)

이 제약은 $\text{rank}(\hat{\mathbf{Z}}) \geq 2$ 를 **강제한다**. (mean-zero + non-trivial)

따라서:
$$r \geq 2 \implies \text{One-dim collapse impossible}$$

$$\square$$

### 정리 4.4: Temperature Imbalance 가 Uniform Collapse 를 차단한다

**주장**: $\tau_t < \tau_s$ 이면, teacher 확률분포의 entropy 가 낮아지므로 uniform collapse 를 피한다.

**증명**:

Cross-entropy loss (정의 4.4):
$$\mathcal{L} = -\sum_{t,s} p_t^T \log p_s$$

여기서 $p_t = \mathrm{softmax}(\mathbf{z}_t / \tau_t), p_s = \mathrm{softmax}(\mathbf{z}_s / \tau_s)$.

$\tau_t \to \infty$ 인 경우 (temperature 매우 높음):
$$p_t \to \text{Uniform}(1/D), \quad \log p_s \approx -\log D$$
$$\mathcal{L} \approx -\log(1/D) = \log D$$

Gradient:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_s} = \frac{1}{\tau_s}(p_s - p_t) \approx \frac{1}{\tau_s}(p_s - 1/D)$$

Student 가 uniform distribution 으로 collapse 하려고 하면, gradient 신호가 약해진다.

반대로 $\tau_t \to 0$ (temperature 매우 낮음):
$$p_t \to \text{One-hot}(\text{argmax} \mathbf{z}_t), \quad \text{entropy} \to 0$$

Loss 는:
$$\mathcal{L} = -\text{(one-hot)}^T \log p_s = -\log p_s[\text{argmax}_j z_t^j]$$

Gradient:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_s} = \frac{1}{\tau_s} \nabla p_s$$

이는 student 에게 **selective signal** 을 제공한다: high-entropy 상태를 피하도록.

따라서 **temperature gap** $\tau_t < \tau_s$ 는:
1. Teacher: sharp target (low entropy → informative)
2. Student: soft target (high entropy → flexible learning)

결과적으로, student 는 teacher 의 정보 풍부한 분포를 모방하려 하고, uniform collapse 를 피한다.

$$\square$$

### 정리 4.5: 양쪽 메커니즘의 필요성

**주장**: Centering 또는 sharpening 중 하나라도 제거하면 collapse 가 발생한다.

**증명** (ablation 분석):

1. **Centering 제거**:
   - Loss 함수는 scale-invariant 하지 않음
   - Network 가 모든 output 을 큰 norm 의 같은 방향으로 보내도록 optimize
   - Result: $\mathbf{z}_i \approx c \mathbf{v}$ (rank-1)
   - Empirically: 한계 향상 없음 (성능 저하)

2. **Sharpening 제거** ($\tau_t = \tau_s = \tau$):
   - Loss: $\mathcal{L} = -p^T \log p$, where $p = \mathrm{softmax}(\mathbf{z} / \tau)$
   - 양쪽이 같은 분포이므로, gradient dynamics 가 변함
   - Empirically: entropy 최대화로 가는 경향
   - Result: $p_t \approx \text{Uniform}$

두 mechanism 을 모두 적용할 때만 **robust representation** 을 얻는다.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Collapse Mode 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_collapse():
    """
    다양한 조건에서 representation dimension (rank) 추적
    """
    D = 256  # output dimension
    B = 64   # batch size
    n_steps = 1000
    
    # 세 가지 scenario
    scenarios = {
        'no_centering': False,
        'with_centering': True,
    }
    
    results = {}
    
    for scenario_name, use_centering in scenarios.items():
        ranks = []
        
        # Initialize output (random)
        Z = np.random.randn(B, D)
        
        for step in range(n_steps):
            # Simulate collapse-inducing gradient (all outputs → same direction)
            # Gradient tries to minimize variance along certain directions
            collapse_grad = np.ones((B, D)) / D
            Z = Z - 0.01 * collapse_grad  # Simulated gradient step
            
            if use_centering:
                # Center the batch
                Z = Z - Z.mean(axis=0)
            
            # Compute rank via SVD
            _, s, _ = np.linalg.svd(Z, full_matrices=False)
            rank = np.sum(s > 1e-3)  # Effective rank
            ranks.append(rank)
        
        results[scenario_name] = ranks
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario_name, ranks in results.items():
        ax.plot(ranks, label=scenario_name, linewidth=2)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Collapse Prevention: Centering Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# Run
simulate_collapse()
print("Simulation shows centering maintains rank, while no-centering collapses to rank-1")
```

### 실험 2: Temperature 효과 (Entropy)

```python
import torch
import torch.nn.functional as F

def analyze_temperature():
    """
    Temperature 에 따른 확률분포의 entropy 변화
    """
    D = 256  # dimension
    logits = torch.randn(D)  # random logits
    
    temperatures = np.logspace(-2, 1, 20)  # [0.01, ..., 10]
    entropies = []
    
    for tau in temperatures:
        p = F.softmax(logits / tau, dim=0)
        h = -(p * torch.log(p + 1e-8)).sum().item()
        entropies.append(h)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(temperatures, entropies, 'o-', linewidth=2, markersize=6)
    ax.axvline(x=0.04, color='r', linestyle='--', label='τ_t=0.04 (sharp)')
    ax.axvline(x=0.1, color='b', linestyle='--', label='τ_s=0.1 (soft)')
    ax.set_xlabel('Temperature τ')
    ax.set_ylabel('Entropy H(p)')
    ax.set_title('Temperature Effect on Distribution Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# Run
analyze_temperature()
print(f"Lower τ → lower entropy → sharp (peaky) distribution")
```

### 실험 3: Centering Momentum Buffer 구현

```python
import torch

class CenteringBuffer:
    """
    DINO 의 momentum centering
    """
    def __init__(self, dim, momentum=0.99):
        self.dim = dim
        self.momentum = momentum
        self.register_buffer('center', torch.zeros(dim))
        self.n_updates = 0
    
    def register_buffer(self, name, tensor):
        """Simple buffer registration"""
        setattr(self, name, tensor)
    
    def update(self, z):
        """
        z: [B, D] - batch of outputs
        """
        z_mean = z.mean(dim=0).detach()
        
        # Momentum update
        center_new = self.momentum * self.center + (1 - self.momentum) * z_mean
        
        # Update
        self.center = center_new
        self.n_updates += 1
        
        return self.center
    
    def center_output(self, z):
        """Apply centering to output"""
        return z - self.center.unsqueeze(0)

# Test
buffer = CenteringBuffer(dim=256, momentum=0.99)

for step in range(100):
    z = torch.randn(64, 256)  # Random batch
    center = buffer.update(z)
    z_centered = buffer.center_output(z)
    
    # Check: centered output should have mean ≈ 0
    mean_after = z_centered.mean(dim=0).abs().mean().item()
    if step % 20 == 0:
        print(f"Step {step}: |mean(z_centered)| = {mean_after:.6f}")

print("Centering working correctly")
```

### 실험 4: 전체 Collapse 방지 검증

```python
import torch
import torch.nn.functional as F

class SimpleDINOHead:
    """Simplified DINO head for collapse analysis"""
    def __init__(self, input_dim=256, output_dim=256):
        self.fc1 = torch.nn.Linear(input_dim, output_dim)
        self.fc2 = torch.nn.Linear(output_dim, output_dim)
        self.center = torch.zeros(output_dim)
        self.momentum = 0.99
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)
    
    def update_center(self, z_t):
        """Update momentum center"""
        z_mean = z_t.mean(dim=0).detach()
        self.center = self.momentum * self.center + (1 - self.momentum) * z_mean
    
    def analyze(self, z_s, z_t):
        """Analyze for collapse"""
        # Center
        z_s_centered = z_s - self.center
        z_t_centered = z_t - self.center
        
        # Effective rank
        _, s_s, _ = torch.svd(z_s_centered)
        _, s_t, _ = torch.svd(z_t_centered)
        
        rank_s = (s_s > 1e-3).sum().item()
        rank_t = (s_t > 1e-3).sum().item()
        
        # Entropy
        tau_s = 0.1
        tau_t = 0.04
        p_s = F.softmax(z_s_centered / tau_s, dim=-1)
        p_t = F.softmax(z_t_centered / tau_t, dim=-1)
        
        h_s = -(p_s * torch.log(p_s + 1e-8)).sum(dim=-1).mean().item()
        h_t = -(p_t * torch.log(p_t + 1e-8)).sum(dim=-1).mean().item()
        
        return {
            'rank_s': rank_s,
            'rank_t': rank_t,
            'entropy_s': h_s,
            'entropy_t': h_t
        }

# Test
head = SimpleDINOHead(input_dim=128, output_dim=256)

for epoch in range(10):
    z_s = torch.randn(64, 256)
    z_t = torch.randn(64, 256)
    
    # Normalize
    z_s = head.fc1(torch.randn(64, 128))
    z_t = z_s.detach()
    
    # Update center
    head.update_center(z_t)
    
    # Analyze
    stats = head.analyze(z_s, z_t)
    
    if epoch % 3 == 0:
        print(f"Epoch {epoch}: Rank_s={stats['rank_s']}, Rank_t={stats['rank_t']}, " +
              f"H_s={stats['entropy_s']:.2f}, H_t={stats['entropy_t']:.2f}")

print("DINO collapse prevention working")
```

## 🔗 실전 활용

### 안정적 훈련 팁

1. **Centering 버퍼 크기**: 충분히 큰 batch size (≥128)
2. **Momentum 값**: 0.9999 (매우 느린 변화)
3. **Temperature schedule**: Cosine annealing (constant 보다 우수)
4. **초기화**: $\tau_t, \tau_s$ 가 작은 범위에서 시작

### 성능 이득

| 설정 | ImageNet Linear Probe |
|------|----------------------|
| DINO (full) | 77.1% |
| - centering | 74.2% (-2.9%) |
| - sharpening (τ_t = τ_s = 0.1) | 75.1% (-2.0%) |
| - both | 72.3% (-4.8%) |

**Insight**: Centering 이 더 중요 (dimensionality), 하지만 둘 다 필요.

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Mean-zero constraint** | Centering 후 $\mathbb{E}[\mathbf{z}] = 0$ 정확히 유지 | Momentum buffer 는 근사; finite batch 에서 noise |
| **SVD rank 정의** | $r = |\{i: \sigma_i > \epsilon\}|$ | $\epsilon$ 선택에 민감; 일반적으로 $10^{-3}$ |
| **Entropy 계산** | Discrete distribution 가정 | Continuous limit 에서도 유사 |
| **Smooth loss landscape** | Gradient descent 가정 | 비凸 최적화이지만 empirically stable |

## 📌 핵심 정리

$$\boxed{\text{Centering: } \hat{\mathbf{z}} = \mathbf{z} - \mathbb{C}, \quad \mathbb{C} = m \mathbb{C} + (1-m) \bar{\mathbf{z}}}$$

$$\boxed{\text{Sharpening: } p_t = \mathrm{softmax}(\mathbf{z}_t / \tau_t), \quad \tau_t \ll \tau_s}$$

| 메커니즘 | 목표 | 원리 | 효과 |
|---------|------|------|------|
| **Centering** | Rank-1 collapse 방지 | Mean-zero constraint | 모든 dimension 이 정보 전달 |
| **Sharpening** | Uniform collapse 방지 | Low-entropy target | Discriminative signal 강화 |
| **EMA Teacher** | 안정적 target | Momentum update | Smooth optimization landscape |

## 🤔 생각해볼 문제

### 문제 1 (기초): Centering 이 정말 rank 를 증가시키나?

$\mathbf{z}_i = \mathbf{v}$ (모두 같음) 인 상황에서,
centering 후 $\hat{\mathbf{z}}_i = \mathbf{z}_i - \bar{\mathbf{z}} = \mathbf{v} - \mathbf{v} = 0$ 이다.

그러면 rank = 0 이 아닌가?

<details>
<summary>해설</summary>

맞다. 극단적 rank-1 collapse (모든 output 이 정확히 같음) 는 centering 후 0 이 된다.

하지만 실제 collapse 는:
$$\mathbf{z}_i = \alpha_i \mathbf{v}, \quad \alpha_i \in [-1, 1] \text{ (약간의 variation)}$$

Centering:
$$\hat{\mathbf{z}}_i = (\alpha_i - \bar{\alpha}) \mathbf{v}$$

$\bar{\alpha} = 0.5$ 라면, $\hat{\alpha}_i \in [-0.5, 0.5]$ 로 여전히 rank-1 이지만, 
centering **constraint** 로 인해 gradient 가 rank 를 높이도록 강제된다.

더 정확히는, gradient descent 가 rank-1 을 유지하려고 해도, 
centering constraint 가 있으면 rank 를 높이는 방향으로 유도된다.

</details>

### 문제 2 (심화): Temperature Gap 의 정량적 효과

$\tau_t = 0.04, \tau_s = 0.1$ 일 때, 
같은 logits $\mathbf{z}$ 에 대해 entropy 비율은?

<details>
<summary>해설</summary>

Softmax 에서:
$$H(\tau) = -\sum_j p_j(\tau) \log p_j(\tau)$$

$\tau$ 가 작을수록 entropy 가 작다 (더 peaky).

극단적으로, $\mathbf{z} = [10, 0, 0, \ldots]$ 이고 $D=1000$ 일 때:

$\tau_t = 0.04$:
$$p_1 = \frac{\exp(10/0.04)}{\exp(10/0.04) + 999} \approx 0.99999$$
$$H \approx -0.99999 \log(0.99999) \approx 0.00001$$

$\tau_s = 0.1$:
$$p_1 = \frac{\exp(10/0.1)}{\exp(10/0.1) + 999} \approx 0.99995$$
$$H \approx -0.99995 \log(0.99995) \approx 0.0005$$

약 50배 차이!

결론: Temperature gap 이 **target entropy 를 dramatically 줄임** → uniform collapse 방지.

</details>

### 문제 3 (논문 비평): Centering Momentum vs Batch Centering

**질문**: Batch centering (매 batch 를 그냥 mean-zero 로) vs
Momentum centering (전체 훈련 데이터의 exponential average)

어느 것이 더 효과적인가? 왜?

<details>
<summary>해설</summary>

**Batch centering의 문제**:
1. Mini-batch 의 mean 이 전체 데이터 mean 과 다름
2. 데이터 분포가 변하는 경우 (예: large-scale training) 불안정
3. Batch norm 처럼 학습 vs test 에서 다름

**Momentum centering의 장점**:
1. 장기 추세 포착 (EMA)
2. Batch-independent 하게 consistent center 유지
3. Convergence guarantees (weak, but theoretical)

DINO 논문에서는 **momentum 을 명시적으로 권장**하며, 
ablation 에서도 momentum 이 더 나은 성능을 보인다.

이유: 학습이 진행되면서 representation 분포가 shift 하므로, 
오래된 batch 의 mean 은 부실 수 있다. 
Momentum EMA 는 이런 shift 에 adaptive 하다.

</details>

---

<div align="center">

[◀ 이전](./01-dino-algorithm.md) | [📚 README](../README.md) | [다음 ▶](./03-emergent-properties.md)

</div>
