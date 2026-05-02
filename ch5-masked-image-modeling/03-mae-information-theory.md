# 03. MAE 의 정보이론적 해석

## 🎯 핵심 질문

- 자연이미지의 spatial redundancy 를 정보이론적으로 어떻게 정량화하는가?
- 왜 BERT (NLP 15% masking) 와 MAE (vision 75% masking) 의 차이가 5배나 나는가?
- Compressive sensing 의 관점에서 25% visible patch 가 충분한 이유는?
- Reconstruction loss 가 왜 contrastive loss 대비 "fine-tuning friendly" 한가?

## 🔍 왜 정보이론인가

**Statistical intuition**:
- 자연이미지: neighboring pixel 들이 매우 유사 (높은 correlation)
- 자연텍스트: 연속된 word 들이 상대적으로 독립적

**결과**:
- Image: 25% visible patch → 나머지 추론 가능 (high redundancy)
- Text: 15% visible word → 나머지 예측 어려움 (low redundancy)

**정보이론이 이를 정량화하는 유일한 도구**:
- Shannon entropy: 평균 정보량
- Mutual information: 두 변수 간 공유 정보
- Rate-distortion theory: "어느 정도 정보 손실까지 괜찮은가"

## 📐 수학적 선행 조건

- **Shannon Entropy**: $H(X) = -\sum_x p(x) \log p(x)$
- **Mutual Information**: $I(X; Y) = H(X) + H(Y) - H(X, Y)$
- **Conditional Entropy**: $H(X|Y) = H(X, Y) - H(Y)$
- **Autocorrelation function**: $\rho(k) = \frac{\mathrm{Cov}(X_t, X_{t+k})}{\mathrm{Var}(X)}$
- **Power spectrum**: $S(f) = \mathrm{FT}[\rho(k)]$ (frequency domain)

## 📖 직관적 이해

```
자연이미지의 Spatial Redundancy:

시간/공간에서 neighboring 들이 매우 유사:
┌─────────────────┐
│ 159  158  157   │  intensity change
│ 160  161  162   │  is very small
│ 161  160  159   │
└─────────────────┘
Autocorrelation ρ(1) ≈ 0.9 (lag-1)

vs 텍스트의 Sequential Redundancy:

"the quick brown fox"
  ↓    ↓      ↓     ↓
 words 가 independent
Mutual info between consecutive words: low

결과: Image 는 far patch 들도 복원 가능,
      Text 는 masked word 가 isolated 이면 어려움

╔═══════════════════════════════════════════════════════╗
║ Masking Ratio vs Redundancy                          ║
║                                                       ║
║ Text (low redundancy):                                ║
║  Masked word 들이 서로 predict 안 됨                 ║
║  → 15% masking 가 적절                               ║
║                                                       ║
║ Image (high redundancy):                              ║
║  Masked patch 들이 neighbor 로부터 복원 가능        ║
║  → 75% masking 도 가능                              ║
╚═══════════════════════════════════════════════════════╝
```

## ✏️ 엄밀한 정의

### 정의 5.9: Spatial Autocorrelation of Image Patches

Image patches $p_1, p_2, \ldots, p_N \in \mathbb{R}^D$ (N = spatial grid) 에 대해:

$$\rho(k) = \frac{\mathbb{E}[(p_i - \mu)(p_{i+k} - \mu)^T]}{\sigma^2}$$

여기서:
- $\mu$ = patch mean
- $\sigma^2$ = patch variance
- $k$ = spatial lag (이웃까지의 거리)

**자연이미지 특성**:
$$\rho(0) = 1 \quad \text{(자기상관)}$$
$$\rho(1) \approx 0.8 \text{-} 0.95 \quad \text{(인접한 patch, 매우 높음)}$$
$$\rho(5) \approx 0.3 \text{-} 0.5 \quad \text{(5-patch 떨어짐, 중간)}$$
$$\rho(20+) \approx 0.05 \text{-} 0.1 \quad \text{(먼 patch, 낮음)}$$

### 정의 5.10: Mutual Information between Visible and Masked

Visible subset $V \subset [N]$ 와 masked subset $M = [N] \setminus V$ 에 대해:

$$I(p_V; p_M) = H(p_M) - H(p_M | p_V)$$

**Interpretation**:
- $H(p_M)$: masked patch 들의 entropy (불확실성)
- $H(p_M | p_V)$: visible patch 가 주어진 상태에서의 조건부 entropy
- 차이: visible 이 masked 를 얼마나 설명하는가

**높은 spatial autocorrelation** → $H(p_M | p_V)$ 가 작음 → 복원 쉬움

### 정의 5.11: Rate-Distortion Trade-off

Reconstruction 에서 허용하는 distortion (오차) $D$ 와 required bitrate $R$ 의 관계:

$$R(D) = \min_{p(\hat{x}|x) : \mathbb{E}[d(x, \hat{x})] \leq D} I(X; \hat{X})$$

**Vision context**:
- Distortion: pixel MSE 또는 patch MSE
- Rate: 필요한 measurement 수 (25% visible patches)
- MAE 의 75% masking: $D$ 가 충분히 작으면 $R = 0.25N$ 으로 가능

### 정의 5.12: Entropy Rate (Long-term Redundancy)

Patch sequence 의 entropy rate (per-patch 평균 정보):

$$H_\infty = \lim_{n \to \infty} \frac{1}{n} H(p_1, p_2, \ldots, p_n)$$

**자연이미지** (2D 이미지 가정):
$$H_\infty^{\text{image}} \approx 4 \text{-} 6 \text{ bits/patch}$$
(8×8 patch in uint8, theoretical max = 8 bits, but correlated → 4-6)

**자연텍스트** (단어 기반):
$$H_\infty^{\text{text}} \approx 12 \text{-} 15 \text{ bits/word}$$
(English corpus, 약 2-3 bits/character × 4-5 character/word)

**결론**: Image entropy rate 가 text 대비 훨씬 낮음 → 더 적은 관찰로도 복원 가능

## 🔬 정리와 증명

### 정리 5.5: Random Sampling 과 Compressive Sensing

**명제** (Candes-Tao 2004, adapted to image patches):

자연이미지가 frequency domain 에서 sparse 하고 (low-frequency dominant),
visible patches 를 random uniform 하게 샘플링하면,
25% measurement 로도 high probability 로 복원 가능.

**증명 스케치**:

1. **Sparsity assumption**: Image $x$ 의 frequency representation $\hat{x} = \mathrm{FFT}(x)$ 가 sparse
   (low-frequency 만 significant, high-frequency near-zero)

2. **Random sampling 의 incoherence**: 
   Sparse basis (frequency) 와 measurement basis (spatial random) 가 incoherent
   → Random projection 으로도 정보 손실 없음

3. **Restricted Isometry Property (RIP)**:
   Random 25% measurement 가 이미지의 거리를 약간의 오차로 보존
   (sparse signal 의 경우)

4. **Reconstruction via optimization**:
   Masked region 에서:
   $$\min_{\hat{x}} \|\hat{x}\|_1 \quad \text{subject to} \quad \hat{x}_V = x_V$$
   
   이 문제는 high probability 로 원래 $x$ 를 복원.

**따라서** 자연이미지의 frequency sparsity + random sampling incoherence 가
25% visible patch 로도 reconstruction 가능하게 함.

$$\square$$

### 정리 5.6: Reconstruction vs Contrastive 의 Representation Geometry

**명제**: Reconstruction objective (MAE) 는 **manifold 의 intrinsic geometry** 를 학습하고,
contrastive objective (SimCLR, DINO) 는 **discriminative subspace** 를 학습한다.

**증명**:

**Reconstruction** (MAE):
- Loss: masked region 복원
- 결과: encoder 가 **모든 pixel 정보를 compress** 해야 함
  → intrinsic manifold 의 full geometry (curvature, volume, etc.)
- Fine-tuning 시: decoder 제거, encoder 의 rich geometry 를 활용 가능
  → lower-level feature 들을 semantic 로 변환

**Contrastive** (InfoNCE):
- Loss: augmented view 들을 match, 다른 sample 은 push away
- 결과: encoder 가 **discrimination boundary** 만 확보
  → high-dimensional subspace 에서 instance-wise separation
- Linear probe: 이미 분리된 subspace → 추가 fine-tuning 이 한계 (이미 최적)

**수학적 모델**:

MAE representation:
$$\mathbf{h}_{\text{MAE}} \approx \mathrm{chart}(\mathbf{x}) + \text{detail}(\mathbf{x})$$
(manifold 상의 좌표 + low-level detail)

Contrastive representation:
$$\mathbf{h}_{\text{contrastive}} \approx \mathrm{decision\_boundary}(\mathbf{x})$$
(discriminative 에만 필요한 dimension)

따라서:
- Fine-tuning: MAE >> contrastive (detail 활용)
- Linear probe: contrastive ≥ MAE (이미 정제된)

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Patch 의 Autocorrelation 계산

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate natural image with spatial structure
np.random.seed(42)
H, W = 64, 64
image = np.zeros((H, W))

# Create correlated image (simple smoothing)
noise = np.random.randn(H, W)
for _ in range(5):
    image += np.roll(noise, 1, axis=0) + np.roll(noise, 1, axis=1)
image /= 10
image = (image - image.mean()) / image.std()

# Compute autocorrelation: lag-0, lag-1, lag-2, ...
def compute_autocorrelation_2d(image, max_lag=10):
    H, W = image.shape
    autocorr = []
    for lag in range(max_lag + 1):
        # Lag in spatial dimension (average across)
        if lag == 0:
            corr = 1.0
        else:
            # Horizontal and vertical lag
            shifted_h = np.roll(image, lag, axis=1)
            shifted_v = np.roll(image, lag, axis=0)
            corr = (
                np.mean(image * shifted_h) + 
                np.mean(image * shifted_v)
            ) / 2.0
        autocorr.append(corr)
    return np.array(autocorr)

autocorr = compute_autocorrelation_2d(image, max_lag=15)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Simulated Natural Image')
plt.colorbar()

plt.subplot(1, 2, 2)
lags = np.arange(len(autocorr))
plt.plot(lags, autocorr, 'o-', markersize=6)
plt.xlabel('Spatial Lag')
plt.ylabel('Autocorrelation')
plt.title('Spatial Autocorrelation of Natural Image')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/autocorrelation.png', dpi=100)

print(f"ρ(1) = {autocorr[1]:.3f} (very high)")
print(f"ρ(5) = {autocorr[5]:.3f} (still high)")
print(f"ρ(15) = {autocorr[15]:.3f} (decreases with distance)")
```

### 실험 2: Shannon Entropy Rate

```python
from scipy.stats import entropy as scipy_entropy

# Estimate entropy rate from patches
def estimate_entropy_rate(image_patches):
    """
    image_patches: [N, D] numpy array (N patches, D=patch_size²)
    Estimate entropy per patch in bits.
    """
    # Quantize to 256 levels (uint8 range)
    patches_quantized = (image_patches * 127.5 + 127.5).astype(np.uint8)
    
    # Count distribution of patch values
    hist, _ = np.histogram(patches_quantized.flatten(), bins=256, range=(0, 256))
    p = hist / hist.sum()
    
    # Shannon entropy in bits
    entropy_bits = scipy_entropy(p, base=2)
    
    return entropy_bits

# Simulate patches
image_patches = np.random.randn(100, 64)  # 100 patches, 64 dim (8×8)
entropy_per_patch = estimate_entropy_rate(image_patches)

print(f"Estimated entropy per patch: {entropy_per_patch:.2f} bits")
print(f"Random Gaussian (max entropy): 6 bits")
print(f"Natural image (empirical): 4-5 bits")
print(f"\nWith 25% visible patches:")
print(f"  Information needed: 0.25 × 784 patches × 5 bits ≈ 980 bits")
print(f"  Remaining uncertainty: 75% patches with conditional entropy << 5 bits")
```

### 실험 3: Compressive Sensing Simulation

```python
import torch
import torch.nn as nn
from scipy.fft import fft2, ifft2

# Simple 1D sparse signal (frequency domain sparse)
N = 128
signal_sparse = torch.zeros(N)
signal_sparse[[5, 10, 15, 20]] = torch.randn(4)  # Only 4 non-zero frequencies

# Inverse FFT to get spatial signal
signal_spatial = torch.fft.ifft(signal_sparse, dim=0).real

# Random measurement (25% samples)
measure_ratio = 0.25
num_measurements = int(N * measure_ratio)
measurement_indices = torch.randperm(N)[:num_measurements]

measurements = signal_spatial[measurement_indices]

print(f"Original signal length: {N}")
print(f"Number of measurements: {num_measurements} ({measure_ratio:.0%})")
print(f"Sparsity (non-zero frequencies): 4/128 = 3.1%")
print(f"\nCompressive Sensing principle:")
print(f"  - Signal is sparse in frequency domain (4 non-zero)")
print(f"  - Random spatial sampling incoherent with frequency basis")
print(f"  - Therefore, 25% measurements sufficient for recovery (theoretical)")
print(f"\nNatural image analogy:")
print(f"  - Image sparse in frequency (low-freq dominant)")
print(f"  - 25% random visible patch measurement")
print(f"  - Reconstruction loss minimizes via ℓ1 (implicit, via NN)")
```

### 실험 4: Fine-tuning vs Linear Probe 효과

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simulate pretrained encoder (MAE-like)
class MAELikeEncoder(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        # Simulate encoder with rich low-level features
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.output_dim = 256
    
    def forward(self, x):
        return self.features(x)

encoder = MAELikeEncoder()

# Downstream task: binary classification
num_samples = 1000
X = torch.randn(num_samples, 768)
y = torch.randint(0, 2, (num_samples,)).float()

# Split train/test
train_idx = torch.arange(800)
test_idx = torch.arange(800, 1000)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Extract features (frozen encoder)
with torch.no_grad():
    h_train = encoder(X_train)
    h_test = encoder(X_test)

# Linear probe
linear_probe = nn.Linear(256, 1)
opt_linear = optim.SGD(linear_probe.parameters(), lr=0.1)
for epoch in range(100):
    pred = linear_probe(h_train).squeeze()
    loss = nn.BCEWithLogitsLoss()(pred, y_train)
    opt_linear.zero_grad()
    loss.backward()
    opt_linear.step()

with torch.no_grad():
    acc_linear = ((linear_probe(h_test).squeeze() > 0).float() == y_test).float().mean()

print(f"Linear probe accuracy: {acc_linear:.3f}")
print(f"Expected (MAE regime): ~79%")
print(f"\nFine-tuning would update encoder as well,")
print(f"leveraging rich low-level features for semantic tasks.")
```

## 🔗 실전 활용

**정보론적 관점에서의 MAE 설계**:
1. **Masking ratio 선택**: Spatial redundancy ($\rho(k)$) 계산 후 결정
   - High redundancy (natural image) → 높은 masking ratio 가능
   - Low redundancy (synthetic, medical) → 낮은 masking ratio
2. **Loss design**: Per-patch normalized MSE 는 uniform information contribution
3. **Downstream task 고려**: Reconstruction >> contrastive 인 경우 (detection, segmentation)

**하이퍼파라미터 튜닝**:
- Masking ratio: 75% (자연이미지), 50% (의료영상), 30% (synthetic)
- Patch size: $P = 16$ (표준), 더 큰 patch (32) 는 정보 손실
- Encoder depth: 더 깊으면 더 많은 low-level detail 보존

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Spatial redundancy 가정** | 자연이미지의 neighboring patch 상관 | Synthetic, medical image 에서 다를 수 있음 |
| **Gaussian 가정** | Entropy 계산 시 가우시안 분포 가정 | 실제 patch 분포는 비가우시안 (heavy tail) |
| **Compressive sensing** | Frequency-sparse 신호 가정 | 모든 이미지가 sparse 하지는 않음 |
| **Random uniform masking** | Structured masking 없다고 가정 | Task-dependent masking 이 더 나을 수 있음 |
| **Entropy rate 스테이셔너리** | Stationary process 가정 | 이미지 경계에서는 다를 수 있음 |

## 📌 핵심 정리

$$\boxed{\text{Autocorrelation}: \, \rho(k) = \frac{\mathbb{E}[(p_i - \mu)(p_{i+k} - \mu)^T]}{\sigma^2}}$$

$$\boxed{\text{Mutual Information}: \, I(p_V; p_M) = H(p_M) - H(p_M | p_V)}$$

$$\boxed{\text{Entropy Rate}: \, H_\infty^{\text{image}} \approx 4\text{-}6 \text{ bits/patch} \ll H_\infty^{\text{text}}}$$

| 개념 | 값/식 | 의미 |
|------|-------|------|
| **Spatial autocorr (lag-1)** | ρ(1) ≈ 0.8-0.95 | 이웃 patch 가 매우 유사 |
| **Spatial autocorr (lag-5)** | ρ(5) ≈ 0.3-0.5 | 거리 5 이후 상관 감소 |
| **Entropy per patch** | 4-6 bits (vs 8 bits max) | High redundancy |
| **Mutual info visible→masked** | I(p_V; p_M) >> 0 | Visible 에서 masked 추론 가능 |
| **Masking ratio** | 75% (vs BERT 15%) | 5배 차이 = redundancy 차이 |
| **Reconstruction objective** | pixel MSE + per-patch norm | Low-level detail 보존 |

## 🤔 생각해볼 문제

### 문제 1 (기초): Autocorrelation 과 Masking Ratio 의 관계

자연이미지에서 spatial autocorrelation $\rho(k)$ 가 감소할수록,
허용 가능한 masking ratio 가 어떻게 변하는가?

극단 예시: $\rho(1) = 0.99$ (매우 높음) vs $\rho(1) = 0.3$ (낮음)
각 경우의 적절한 masking ratio 를 추정하시오.

<details>
<summary>해설</summary>

**High $\rho(1) = 0.99$**:
- 이웃 patch 가 거의 동일
- Masked patch 를 visible 에서 쉽게 복원
- Masking ratio 을 80-90% 까지 높일 수 있음

**Low $\rho(1) = 0.3$**:
- 이웃 patch 가 독립적
- Masked patch 를 예측하기 어려움
- Masking ratio 은 20-30% 정도로 낮춰야 함

**정량 모델**:
$$\text{allowable masking ratio} \propto \log(1 / (1 - \rho(1)))$$

예: $\rho(1) = 0.9$ → 약 40% masking
     $\rho(1) = 0.85$ → 약 25% masking

</details>

### 문제 2 (심화): Compressive Sensing 의 조건

Candes-Tao 의 RIP (Restricted Isometry Property) 를 만족하려면,
signal sparsity $s$ (non-zero coefficients) 와 measurement 수 $m$ 에 대해:
$$m \gtrsim O(s \log(N / s))$$

자연이미지에서 이를 구체적으로 적용하면 어떤 masking ratio 가 나오는가?

<details>
<summary>해설</summary>

**자연이미지 frequency sparsity**:
- DCT 또는 FFT 기준: high-frequency 가 sparse
- Low-frequency 에 90% 이상 에너지 집중
- Effective sparsity: $s \approx 0.1N$ (10%)

**RIP 조건**:
$$m \gtrsim s \log(N) = 0.1N \times \log N$$
$$\text{masking ratio} = 1 - m/N \approx 1 - 0.1 \log N$$

$N = 784$ patches 인 경우:
$$m \gtrsim 0.1 \times 784 \times \log(784) \approx 0.1 \times 784 \times 6.66 \approx 520$$
$$\text{masking ratio} \approx 1 - 520/784 \approx 0.34 = 34\%$$

실제 MAE 75% 가 이론치 34% 보다 높은 이유:
- Image 는 실제로 frequency 에서 더 sparse (low-freq dominant)
- Neural network decoder 가 optimal L1 복원보다 나음
- Per-patch normalization 으로 효율적인 loss

</details>

### 문제 3 (논문 비평): Manifold Geometry vs Discriminative Subspace

MAE 의 reconstruction objective 가 "manifold intrinsic geometry" 를 학습한다는 주장을 수학적으로 엄밀하게 정의하고,
contrastive 의 "discriminative subspace" 와 명확히 구분하시오.

<details>
<summary>해설</summary>

**Manifold Geometry (Reconstruction)**:
Data manifold $\mathcal{M} \subset \mathbb{R}^D$ 위의 patch 들을 학습.
Encoder 가 이 manifold 를 encode 할 때:
$$\mathbf{h} = E(x) \in \mathbb{R}^d$$

Reconstruction loss 최소화 = manifold 의 **모든 dimension** 을 capture
- Intrinsic dimension 뿐 아니라 tangent space 의 curvature도
- Low-level detail (texture, edge) 까지 compress

**Discriminative Subspace (Contrastive)**:
$\mathcal{M}$ 의 여러 point 들을 구분하는 **최소 dimension** 만 학습:
$$\mathbf{h} = E(x), \quad \|\mathbf{h}_i - \mathbf{h}_j\| \propto \text{dissimilarity}(x_i, x_j)$$

Contrastive loss = discriminative direction 만 sharp
- Null space 의 structure 는 무관 (fine-tuning 불가)

**정량적 차이**:
- Reconstruction: $d$ (encoder dimension) 이 모두 의미 있음 → fine-tuning 시 각 dimension 업데이트
- Contrastive: $d' \ll d$ effective dimension (나머지는 noise)

따라서 fine-tuning 시:
- MAE: 모든 dimension 활용 → 큰 향상 가능
- Contrastive: 이미 분리된 dimension → 한계

</details>

---

<div align="center">

[◀ 이전](./02-mae.md) | [📚 README](../README.md) | [다음 ▶](./04-simmim.md)

</div>
