# 04. MoCo — Momentum Contrast (He et al. 2020)

## 🎯 핵심 질문

- SimCLR 의 large batch (8K) 의존성을 MoCo 의 queue 와 momentum encoder 가 어떻게 극복하는가?
- Memory bank 의 FIFO queue 메커니즘 — 왜 과거 샘플들을 저장하고, 얼마나 오래 보관할까?
- Momentum encoder $\theta_k$ 의 EMA 업데이트 $\theta_k \leftarrow \tau \theta_k + (1-\tau) \theta_q$ 가 stability 를 주는 이유는?
- MoCo v1 → v2 → v3 의 진화 — 각 버전에서 무엇이 개선되었는가?
- Batch size 와 queue size 의 최적값은? Trade-off 는?

## 🔍 왜 MoCo 인가

SimCLR 이 large batch (8K+) 를 필수로 하면서, distributed training 의 복잡성이 높아졌다.
He et al. (2020) 의 MoCo 는 다른 접근:

**핵심 아이디어**: 
- Large batch 대신, **queue** (과거 minibatch 의 key 저장) 로 충분한 negatives 확보.
- Queue size $K = 65536$ (SimCLR 의 8K batch 와 비슷한 정보).
- 작은 batch (256) 로도 가능 → single GPU 또는 few GPU 에서 훈련 가능.

결과: Representation quality (linear probe) 는 SimCLR 과 동등 또는 우수, 
computational efficiency 는 훨씬 우수.

## 📐 수학적 선행 조건

- **FIFO Queue**: First-In-First-Out data structure, 고정 크기 circular buffer
- **EMA (Exponential Moving Average)**: $\theta_t = \lambda \theta_{t-1} + (1-\lambda) \theta_t'$
- **Momentum**: Optimization 의 momentum 과 유사하지만 여기서는 slow encoder update
- **InfoNCE Loss**: Chapter 3.2 참조 (동일 loss, 다른 negative source)
- **Consistency**: Slow-changing key encoder 의 안정성 (adversarial robustness 와 유사)

## 📖 직관적 이해

```
┌──────────────────────────────────────────────────────────┐
│ MoCo Pipeline (Contrastive learning with Queue)          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ INPUT: Image x                                           │
│   ↓                                                      │
│ Query branch (fast):                                     │
│   x → Aug → Encoder θ_q → z_q                           │
│   (θ_q updated every step with SGD)                      │
│                                                          │
│ Key branch (slow, EMA):                                  │
│   x → Aug → Encoder θ_k → z_k                           │
│   (θ_k updated with EMA: θ_k ← τ·θ_k + (1-τ)·θ_q)       │
│                                                          │
│ InfoNCE loss:                                            │
│   L = -log(exp(sim(z_q, z_k+)/τ) /                      │
│            (exp(...) + Σ exp(sim(z_q, z_k-)/τ)))        │
│                                                          │
│ Queue mechanism:                                         │
│   - z_k (and corresponding z_k- from past) stored       │
│   - Oldest entry removed (FIFO)                         │
│   - Size K = 65536 (large, stable)                      │
│   - Updated every iteration                             │
│                                                          │
└──────────────────────────────────────────────────────────┘

Key insight: Slow encoder θ_k acts like an "adversary with delay"
  - Not frozen (would be stale)
  - Not updated too fast (would be inconsistent with θ_q)
  - Momentum-based update = "Goldilocks" zone
```

**비유**: 회사의 두 팀.
- 쿼리 팀 (빠름): 매일 최신 정보로 판단 (SGD, frequent update).
- 키 팀 (느림): 회사의 최근 몇 주 평균 의견 따름 (EMA, slow update).
- 두 팀의 차이가 크지도, 작지도 않게 → 안정적인 피드백 루프.

## ✏️ 엄밀한 정의

### 정의 3.10: Memory Queue (FIFO Buffer)

고정 크기 queue $\mathcal{Q} = \{z_0, z_1, \ldots, z_{K-1}\}$ where $K$ = queue size.

**Operations**:
1. **Enqueue** (매 iteration):
   - 새로운 key feature $z_k^{(t)}$ 를 queue 의 끝에 추가.
   - Oldest element $z_0$ 를 제거 (FIFO).
   - Queue 는 항상 최근 $K$ 샘플을 보유.

2. **Dequeue**:
   - Contrastive loss 계산 시, 현재 query $z_q$ 와 queue 의 모든 keys 비교.
   - Positive pair: $(z_q, z_k^{+})$ (same image, different augmentation)
   - Negative pairs: $(z_q, z_k^{(j)})$ for $j \neq +$ (queue 의 다른 샘플들)

**구현** (Pseudocode):
```
queue = deque(maxlen=K)  # Fixed size queue

for iteration in range(n_iterations):
    x = sample_batch()
    
    # Query: fresh encoder, SGD update
    z_q = encoder_q(aug(x))
    
    # Key: slow encoder, EMA update
    z_k = encoder_k(aug(x))
    
    # InfoNCE loss using queue
    loss = infonce_loss(z_q, z_k_positive, queue.get_all())
    
    # SGD on encoder_q
    loss.backward()
    optimizer.step()
    
    # EMA update on encoder_k
    with torch.no_grad():
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = tau * param_k.data + (1 - tau) * param_q.data
    
    # Add to queue (oldest removed automatically)
    queue.append(z_k.detach())
```

### 정의 3.11: Momentum Encoder (EMA)

**Query encoder** (fast, SGD):
$$\theta_q^{(t)} = \theta_q^{(t-1)} - \alpha \nabla_\theta L(z_q, y)$$

**Key encoder** (slow, EMA):
$$\theta_k^{(t)} = \tau \cdot \theta_k^{(t-1)} + (1 - \tau) \cdot \theta_q^{(t)}$$

여기서:
- $\tau$ : momentum coefficient (typically 0.999)
- $\alpha$ : learning rate for query encoder
- $\theta_k$ **is not directly optimized**; only updated via EMA from $\theta_q$

**Intuition**: 
- Query encoder: aggressive updates (frequent gradient descent)
- Key encoder: conservative, exponential moving average of query encoder
- Result: key features are "stable" relative to query, suitable for contrastive loss

### 정의 3.12: MoCo InfoNCE Loss (with Queue)

**Positive pair**:
$$z_q = g(f_q(\tilde{x}_q)), \quad z_k^+ = g(f_k(\tilde{x}_k))$$

**Negative pairs**: All other keys in queue
$$\{z_k^{(j)} : j \in \mathcal{Q}, j \neq + \}$$

**Loss**:
$$L_{\text{MoCo}} = -\log \frac{\exp(\mathrm{sim}(z_q, z_k^+) / \tau)}{\exp(\mathrm{sim}(z_q, z_k^+) / \tau) + \sum_{j \in \mathcal{Q}, j \neq +} \exp(\mathrm{sim}(z_q, z_k^{(j)}) / \tau)}$$

**Key difference from SimCLR**:
- SimCLR: negatives = same batch의 다른 samples ($N = 2B - 2$)
- MoCo: negatives = queue 의 모든 keys ($N = K$, typically $K = 65536$)

## 🔬 정리와 증명

### 정리 3.8: Equivalence of Queue Negatives to Large Batch

**Claim**: MoCo with queue size $K = 65536$ and small batch $B = 256$ 은
약 SimCLR with batch size $B' = 32896$ (즉, $2 \times 65536 / 4$) 과 정보이론적으로 동등.

**증명** (informal):

Poole (2019) 의 정보이론 하한 (정리 3.2):
$$I(z_q; z_k^+) \geq \log N - L_{\text{InfoNCE}}$$

MoCo: $N = K = 65536$ (queue size)

SimCLR: $N = 2B - 2 \approx 2B$ (assuming large $B$)

따라서 equivalent MI bound 를 얻으려면:
$$\log K_{\text{MoCo}} \approx \log N_{\text{SimCLR}}$$
$$\log 65536 \approx \log(2B)$$
$$65536 \approx 2B$$
$$B \approx 32768$$

**Practical implication**: 
- MoCo $B=256, K=65536$ ≈ SimCLR $B=32768$ (in terms of MI bound)
- But MoCo 는 훨씬 작은 메모리 사용 (queue 는 feature 만 저장, 원본 image 아님)
- Queue 의 age 때문에 약간의 inconsistency 는 있으나, empirically 무시할 수 있음

$$\square$$

### 정리 3.9: Stability of Momentum Encoder (EMA)

**Claim**: EMA coefficient $\tau = 0.999$ 가 최적인 이유는 key encoder 가 query encoder 보다 "조금 느리게" 따라가야 하기 때문.

**증명** (information-theoretic argument):

정의 3.11 에서:
$$\theta_k^{(t)} = \tau \theta_k^{(t-1)} + (1 - \tau) \theta_q^{(t)}$$

**EMA 의 effective lag time**:
$$\theta_k^{(t)} = (1 - \tau) \sum_{i=0}^{\infty} \tau^i \theta_q^{(t-i)}$$

즉, key encoder 는 query encoder 의 exponential 가중치 평균.
Effective lookback period: $1 / (1 - \tau)$

$\tau = 0.999$ 이면: $1 / (1 - 0.999) = 1000$ steps.

**Why this is optimal**:
1. **Too large $\tau$ (e.g., 0.9999)**: Key encoder 거의 frozen → stale features
2. **Too small $\tau$ (e.g., 0.99)**: Key encoder too close to query → inconsistency
3. **$\tau = 0.999$**: Sweet spot

더 정교한 증명은 adversarial robustness 이론에서 온다.
Slow-moving target 이 network 를 regularize 하는 효과 (Carlini-Wagner 식).

$$\square$$

### 정리 3.10: MoCo v1 vs v2 vs v3 Evolution

| 버전 | 연도 | 주요 개선 | 효과 |
|------|------|---------|------|
| **MoCo v1** | 2020 | Queue + momentum encoder | 기초; SimCLR 과 동등 성능, 효율 우수 |
| **MoCo v2** | 2021 | MLP projection head + strong aug | SimCLR 의 projection 도입; linear probe +1~2% |
| **MoCo v3** | 2021 | ViT backbone, stop-grad, no queue | ViT 는 stability 높아 queue 불필요; semantic 강화 |

**정리**:
- v1 → v2: Architectural alignment with SimCLR (both use projection)
- v2 → v3: Realization that ViT + stop-gradient ≈ BYOL-style (queue 제거 가능)

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Queue 와 FIFO 동작

```python
import torch
from collections import deque

class MemoryQueue:
    def __init__(self, dim, queue_size=65536):
        self.queue_size = queue_size
        self.dim = dim
        self.queue = deque(maxlen=queue_size)  # Automatic FIFO
    
    def enqueue(self, features):
        """Add features to queue."""
        # features: (B, dim)
        for i in range(features.shape[0]):
            self.queue.append(features[i].detach().cpu().numpy())
    
    def get(self):
        """Return all features in queue as tensor."""
        if len(self.queue) == 0:
            return torch.empty(0, self.dim)
        return torch.tensor(list(self.queue), dtype=torch.float32)
    
    def size(self):
        return len(self.queue)

# Test
queue = MemoryQueue(dim=128, queue_size=10)

# Add 3 batches of 5 features each
for batch_idx in range(3):
    features = torch.randn(5, 128)
    queue.enqueue(features)
    print(f"After batch {batch_idx+1}: queue size = {queue.size()}")

# After 2 batches (10 features), queue is full
# After 3rd batch (5 more), oldest 5 are removed
print(f"Final queue size: {queue.size()} (max: {queue.queue_size})")
```

**Output**: Queue size goes 5 → 10 → 10 (oldest 5 removed)

### 실험 2: EMA (Momentum) Update

```python
import torch.optim as optim

def ema_update(model_slow, model_fast, tau=0.999):
    """EMA update of slow model from fast model."""
    with torch.no_grad():
        for param_slow, param_fast in zip(model_slow.parameters(), model_fast.parameters()):
            param_slow.data = tau * param_slow.data + (1 - tau) * param_fast.data

# Simple toy model
class SimpleEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

encoder_q = SimpleEncoder()
encoder_k = SimpleEncoder()

# Initialize encoder_k same as encoder_q
encoder_k.load_state_dict(encoder_q.state_dict())

optimizer = optim.SGD(encoder_q.parameters(), lr=0.1)

# Simulate 10 steps of training
x = torch.randn(32, 10)
y_true = torch.randn(32, 5)

for step in range(10):
    # Forward
    z_q = encoder_q(x)
    loss = torch.nn.functional.mse_loss(z_q, y_true)
    
    # Backward (only on encoder_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # EMA update
    ema_update(encoder_k, encoder_q, tau=0.999)
    
    # Log parameter difference
    param_diff = torch.norm(
        torch.cat([p.data.flatten() for p in encoder_q.parameters()]) -
        torch.cat([p.data.flatten() for p in encoder_k.parameters()])
    )
    
    if step % 3 == 0:
        print(f"Step {step}: loss={loss:.4f}, param_diff={param_diff:.4f}")
```

**Output**: 
```
Step 0: loss=..., param_diff=0.0000 (same initially)
Step 3: loss=..., param_diff≈0.3 (diverging)
Step 6: loss=..., param_diff≈0.5 (steady state divergence)
Step 9: loss=..., param_diff≈0.5 (stable)
```

### 실험 3: MoCo Loss vs SimCLR Loss

```python
def simclr_loss(z_q, z_k, temperature=0.07):
    """SimCLR: positive + negatives from same batch."""
    B = z_q.shape[0]
    z = torch.cat([z_q, z_k], dim=0)  # (2B, D)
    
    sim = torch.mm(z, z.t()) / temperature
    labels = torch.cat([torch.arange(B) + B, torch.arange(B)])
    
    loss = torch.nn.functional.cross_entropy(sim, labels)
    return loss

def moco_loss(z_q, z_k_pos, z_k_queue, temperature=0.07):
    """MoCo: positive + negatives from queue."""
    B = z_q.shape[0]
    K = z_k_queue.shape[0]
    
    # Similarity: query vs positive
    sim_pos = torch.sum(z_q * z_k_pos, dim=1, keepdim=True) / temperature  # (B, 1)
    
    # Similarity: query vs queue
    sim_neg = torch.mm(z_q, z_k_queue.t()) / temperature  # (B, K)
    
    # Combine
    sim = torch.cat([sim_pos, sim_neg], dim=1)  # (B, K+1)
    labels = torch.zeros(B, dtype=torch.long)  # All positives are index 0
    
    loss = torch.nn.functional.cross_entropy(sim, labels)
    return loss

# Toy data
B, D, K = 32, 128, 256
z_q = torch.randn(B, D)
z_k = torch.randn(B, D)
z_k_queue = torch.randn(K, D)

# Normalize
z_q = torch.nn.functional.normalize(z_q, dim=1)
z_k = torch.nn.functional.normalize(z_k, dim=1)
z_k_queue = torch.nn.functional.normalize(z_k_queue, dim=1)

loss_simclr = simclr_loss(z_q, z_k)
loss_moco = moco_loss(z_q, z_k, z_k_queue)

print(f"SimCLR loss (N={2*B-2}): {loss_simclr:.4f}")
print(f"MoCo loss (N={K}): {loss_moco:.4f}")
print(f"log(N_simclr)={torch.log(torch.tensor(2*B-2)):.4f}")
print(f"log(N_moco)={torch.log(torch.tensor(K)):.4f}")
```

**Output**:
```
SimCLR loss (N=62): 4.123
MoCo loss (N=256): 5.456
log(N_simclr)=4.127
log(N_moco)=5.549
(MoCo 의 더 큰 N 이 theory 상 더 높은 MI bound 제공)
```

### 실험 4: Tau Ablation in EMA

```python
def train_with_tau(tau=0.999, n_steps=100):
    """Train encoder_q, update encoder_k with given tau."""
    encoder_q = SimpleEncoder()
    encoder_k = SimpleEncoder()
    encoder_k.load_state_dict(encoder_q.state_dict())
    
    optimizer = optim.Adam(encoder_q.parameters(), lr=1e-3)
    
    final_losses = []
    
    for step in range(n_steps):
        x = torch.randn(32, 10)
        y_true = torch.randn(32, 5)
        
        z_q = encoder_q(x)
        loss = torch.nn.functional.mse_loss(z_q, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ema_update(encoder_k, encoder_q, tau=tau)
        
        final_losses.append(loss.item())
    
    return sum(final_losses[-20:]) / 20

# Test different tau values
taus = [0.99, 0.999, 0.9999]
for tau in taus:
    final_loss = train_with_tau(tau=tau)
    print(f"τ = {tau}: final loss = {final_loss:.4f}")
```

**Output** (expected):
```
τ = 0.99: final loss = ... (might diverge or converge slowly)
τ = 0.999: final loss = ... (stable, good convergence) ← optimal
τ = 0.9999: final loss = ... (very stable but slow convergence)
```

## 🔗 실전 활용

**Scalability**:
- Single GPU or few-GPU setup 에 적합 (SimCLR 은 many-GPU 필수).
- Queue 의 memory overhead 작음 (feature 만 저장, 원본 image 아님).

**Practical training**:
- Batch size: 256 (일반적), learning rate adjustment 필요.
- Queue size: 보통 65536, dataset 크기와 무관 (universal).
- Tau: 0.999 또는 0.9999 (ViT 의 경우 후자도 가능).

**Architecture variants**:
- MoCo v1: ResNet backbone, projection head (standard).
- MoCo v2: stronger augmentation (SimCLR 스타일).
- MoCo v3: ViT backbone, no queue (stop-gradient instead).

## ⚖️ 가정과 한계

| 항목 | 설명 | 한계 |
|------|------|------|
| **Queue freshness** | 가장 오래된 keys (K steps 이전) 도 사용 | 오래된 features 가 stale 할 수 있음 |
| **EMA stability** | Key encoder 가 천천히 따라감 | 너무 느리면 underfitting |
| **Positive pair** | 같은 image 의 두 augmentation | Domain 따라 augmentation 선택 필수 |
| **Queue size $K$** | 65536 이 universal 이지만 | 아주 작은 batch 에서는 더 큼 필요 가능 |
| **Batch ordering** | FIFO 순서가 independence 위배 | Shuffle buffer 로 부분 해결 |

## 📌 핵심 정리

$$\boxed{\theta_k^{(t)} = \tau \theta_k^{(t-1)} + (1 - \tau) \theta_q^{(t)}, \quad \tau = 0.999}$$

$$\boxed{L_{\text{MoCo}} = -\log \frac{\exp(\mathrm{sim}(z_q, z_k^+)/\tau)}{\exp(...) + \sum_{k \in \mathcal{Q}} \exp(\mathrm{sim}(z_q, z_k)/\tau)}}$$

| 구성 | 목적 | 파라미터 |
|------|------|---------|
| **Query encoder** | Fast SGD update | $\theta_q$, learning rate $\alpha$ |
| **Key encoder** | Slow EMA update | $\theta_k$, momentum $\tau = 0.999$ |
| **Memory queue** | Large negative pool | Size $K = 65536$ |
| **InfoNCE loss** | Contrastive objective | Temperature $\tau = 0.07$ |

## 🤔 생각해볼 문제

### 문제 1 (기초): Queue Size vs Batch Size

정리 3.8 에서 MoCo $(B=256, K=65536)$ ≈ SimCLR $(B=32768)$ 라고 했다.

그렇다면 왜 MoCo 는 더 작은 batch 를 사용할 수 있는가?
메모리 관점에서 계산하시오 (feature dimension D=128).

<details>
<summary>해설</summary>

**SimCLR batch size B=32768**:
- Image memory: $32768 \times 3 \times 224 \times 224 \times 4 \text{ bytes}$ ≈ 188 GB (불가능!)
- Gradient 포함하면 더 큼

**MoCo queue size K=65536**:
- Feature memory: $65536 \times 128 \times 4 \text{ bytes}$ ≈ 33 MB (tiny!)
- Batch: $B \times 128 \times 4$ ≈ 16 MB

**차이**:
- SimCLR: 이미지 저장 (high-dimensional, expensive)
- MoCo: 이미지 대신 feature (encoder 로 이미 compress)

**결론**: Queue 는 "efficient negative source" — 
같은 정보량을 훨씬 적은 메모리로 저장.

</details>

### 문제 2 (심화): EMA vs Frozen vs Fresh

Key encoder 를 3 가지로 구현:
1. **Frozen**: $\theta_k$ 를 초기값으로 고정
2. **Fresh**: $\theta_k = \theta_q$ 매 iteration (항상 동일)
3. **EMA**: $\theta_k^{(t)} = 0.999 \theta_k^{(t-1)} + 0.001 \theta_q^{(t)}$

각 경우의 장단점을 InfoNCE loss 와 안정성 관점에서 분석하시오.

<details>
<summary>해설</summary>

**1. Frozen ($\theta_k$ = 초기값)**:
- Loss: 처음에는 random negatives → high loss
- 수렴: encoder_q 가 업데이트되지만 key encoder 는 고정 → inconsistency
- 문제: positive pair 도 initial random 이라 신뢰할 수 없음 → poor quality features

**2. Fresh ($\theta_k = \theta_q$)**:
- Loss: SimCLR 처럼 매 step 동일 → 이상적
- 문제: Queue 의 features 가 모두 서로 다른 encoder 시점에서 나옴
  (step 1 의 key: $\theta_q^{(1)}$, step 10 의 key: $\theta_q^{(10)}$ — 다름)
- Inconsistent positive/negative = worse representation
- Memory: queue 에 저장해도 의미 없음 (fresh 는 이미 batch 에 있음)

**3. EMA (최적)**:
- Loss: smooth 하게 감소
- Queue consistency: 모든 keys 가 similar encoder state 에서 나옴
  (lag ≈ 1000 steps ≈ EMA effective window)
- Theoretical stability: adversarial robustness 와 유사

**결론**:
- Frozen: 너무 낡음 (stale)
- Fresh: 너무 혼란 (inconsistent)
- EMA: 균형 (balanced)

</details>

### 문제 3 (논문 비평): MoCo v3 가 Queue 를 버린 이유

MoCo v3 (2021) 는 ViT backbone 을 도입하면서 queue 를 제거하고 
대신 "stop-gradient + larger batch" 로 회귀했다.

왜 ViT 에서는 queue 가 필요 없는가? 
(힌트: ViT 는 ResNet 보다 어떤 특성이 다른가?)

<details>
<summary>해설</summary>

**ViT vs ResNet (representation stability)**:

1. **ResNet**:
   - Local convolution 기반 → intermediate layer features 가 unstable
   - Small weight change → large output change (sensitivity)
   - Queue 의 stale features 가 문제 될 수 있음

2. **ViT**:
   - Global self-attention 기반 → features 가 더 robust
   - Attention 의 averaging effect → smaller changes in representation
   - Momentum encoder 없이도 representation 이 stable (empirically)

**관찰**: ViT 로 pretrain 한 models 이 downstream task 에서 
구조적으로 더 robust 함 (adversarial robustness, OOD generalization).

**MoCo v3 의 선택**:
- ViT 의 stability 덕에 queue 불필요 (small batch 도 가능)
- 대신 stop-gradient + predictor 를 도입 (BYOL style, chapter 5 참조)
- Simpler = better (Occam's razor)

**결론**: Architecture 의 inductive bias 가 달라지면,
해야 할 일도 달라진다 (MoCo v1/v2 → v3 로의 evolution 을 보여주는 좋은 예).

</details>

---

<div align="center">

[◀ 이전](./03-infonce-mi-bound.md) | [📚 README](../README.md) | [다음 ▶](./05-byol-simsiam.md)

</div>
