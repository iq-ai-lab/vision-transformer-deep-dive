# 01. BEiT — BERT for Image

## 🎯 핵심 질문

- DALL-E 의 discrete VAE (dVAE) tokenizer 는 어떻게 이미지를 visual token sequence 로 변환하는가?
- Vision 에서 BERT 의 Masked Language Modeling 을 직접 적용하려면, 텍스트의 word token 대신 이미지의 무엇을 사용해야 하는가?
- 40% masking ratio 를 선택한 이유는? BERT 의 15% 대비 왜 2.7× 큰가?
- dVAE tokenizer 의 quality 가 downstream task performance 의 ceiling 을 결정하는가?

## 🔍 왜 이 tokenization 이 필요한가

BERT 의 성공은 **masked language modeling (MLM)** 에 있다:
1. 입력 token 의 일부를 `[MASK]` 로 가린다
2. 신경망이 masked token 의 원래 값을 예측한다
3. Discrete (문자 단위) 예측이므로 cross-entropy loss 가 명확하다

이미지에서 이를 직접 적용하려면 **discrete token** 이 필요하다.
- Pixel 은 continuous 이므로 cross-entropy 를 정의하기 어렵다
- DALL-E 의 dVAE 는 image → 8192 vocabulary 의 discrete code 로 변환
- 이제 BERT-MLM 을 이미지에 그대로 적용할 수 있다

## 📐 수학적 선행 조건

- **Variational Autoencoder (VAE)**: encoder $q_\phi(z|x)$, decoder $p_\theta(x|z)$, ELBO
- **Vector Quantization (VQ)**: continuous embedding 을 discrete codebook 으로 양자화
- **Cross-Entropy Loss**: discrete 분류 문제의 표준 손실함수
- **Masked Prediction**: BERT 의 MLM objective
- **Vision Patch 개념**: 이미지를 $P \times P$ patches 로 분할 (Ch1 복습)

## 📖 직관적 이해

```
DALL-E dVAE Tokenization Process:

┌─────────────────┐
│  원본 이미지    │ (H × W × 3)
│  224 × 224 × 3 │
└────────┬────────┘
         │
    ┌────▼─────┐
    │  Encoder  │ (conv layers with downsampling)
    │ 224→28×28 │ → 28×28 discrete tokens
    └────┬─────┘
         │
    ┌────▼──────────────────────┐
    │ VQ (Vector Quantization)  │
    │ 각 위치의 embedding       │
    │ → 가장 가까운 codebook   │
    │   entry index (0~8191)   │
    └────┬──────────────────────┘
         │
    ┌────▼──────────────────────┐
    │  Visual Token Seq         │
    │ [t₁, t₂, ..., t₇₈₄]      │
    │ vocab size = 8192         │
    └──────────────────────────┘

BEiT Masked Prediction:

[t₁] [MASK] [t₃] [MASK] [t₅] ...
  ↓    ↓      ↓     ↓      ↓
Transformer encoder
  ↓    ↓      ↓     ↓      ↓
[h₁] [h₂]   [h₃] [h₄]   [h₅]
      ↓            ↓
   predict t₂    predict t₄
   (8192-way classification)

MSA loss = -Σ log p(tᵢ | masked input)
```

**전체 흐름**: 이미지 → dVAE tokenization → discrete token sequence → BERT-style masking + prediction

## ✏️ 엄밀한 정의

### 정의 5.1: Discrete VAE (dVAE)

Encoder-decoder pair $(E_\phi, D_\theta)$ 와 codebook $\mathcal{C} = \{c_k \in \mathbb{R}^d : k \in [8192]\}$ 로 구성된다.

**Encoding step**:
$$z_q = \mathrm{Quantize}(E_\phi(x))$$

여기서 Quantize 는:
$$\mathrm{Quantize}(z) = c_{k^*}, \quad k^* = \arg\min_k \|z - c_k\|^2$$

각 spatial position 에서 가장 가까운 codebook entry 의 index 를 선택한다.

**Decoding**:
$$\hat{x} = D_\theta(z_q)$$

Output 은 reconstructed image.

### 정의 5.2: Tokenized Image

dVAE 의 encoder 출력 (quantization 전) 은 $H' \times W'$ spatial grid 에 배치된 embeddings 이다.
각 위치 $(i,j)$ 에서의 discrete token index $t_{i,j} \in [8192]$ 로, 전체 sequence 는:

$$\mathbf{t} = [t_{1,1}, t_{1,2}, \ldots, t_{H',W'}] \in [8192]^{H' \cdot W'}$$

일반적으로 input 224×224 이미지 → 28×28 grid → 784 tokens.

### 정의 5.3: BEiT Masked Image Modeling

Input token sequence $\mathbf{t}$ 에서 random subset $M \subset [784]$ (40% of positions) 를 `[MASK]` 로 대체한다:

$$\tilde{\mathbf{t}} = [\mathbf{t}^{(1)}, \ldots, \mathbf{t}^{(784)}] \quad \text{where} \quad \tilde{\mathbf{t}}^{(i)} = \begin{cases} [\mathrm{MASK}] & i \in M \\ \mathbf{t}^{(i)} & i \notin M \end{cases}$$

Vision Transformer 로 encode 한 후 masked position 의 token 을 예측:

$$L_{\mathrm{BEiT}} = -\sum_{i \in M} \log p_\theta(\mathbf{t}^{(i)} \mid \tilde{\mathbf{t}})$$

여기서 $p_\theta$ 는 softmax over 8192 vocabulary.

## 🔬 정리와 증명

### 정리 5.1: dVAE Tokenization 의 정보 손실

Original image $x \in \mathbb{R}^{H \times W \times 3}$ → tokenized $\mathbf{t} \in [8192]^{H' W'}$ → reconstructed $\hat{x}$.

**명제**: dVAE reconstruction error $\|x - \hat{x}\|_2$ 는 downstream fine-tuning 시 lower bound 를 결정한다.

**직관적 증명**: 
Tokenization 과정에서 codebook 의 discrete grid 로 인해 정보가 손실된다. 
이 손실된 정보는 어떤 decoder 도 복원할 수 없으므로,
downstream task (예: ImageNet classification) 도 이 수준 이상으로 향상할 수 없다.

더 형식적으로, information bottleneck theory (Tishby 2000) 의 관점에서:
$$I(x; \mathbf{t}) \leq H(\mathbf{t}) \leq \log 8192 \approx 13 \text{ bits per token}$$

원본 이미지의 mutual information 을 모두 capture 할 수 없으므로,
BEiT pretrain 이 아무리 완벽해도 이 정보 제약을 벗을 수 없다.

$$\square$$

### 정리 5.2: Masking Ratio 와 Prediction Difficulty

40% masking ratio 는 information-theoretic 과 empirical 사이의 균형이다.

**경험적 관찰** (Bao et al. 2022):
- 15% masking (BERT 처럼): 이웃 visible token 으로부터 거의 trivial 하게 복원 가능 → 약한 supervision
- 40% masking: 이웃 context 가 부족해 nontrivial 한 예측 필요 → 더 강한 representation 학습
- 75% masking (MAE, Ch5-02): 이미지의 spatial redundancy 로 인해 가능 (Ch5-03)

**수학적 근거**:
Vision 의 **spatial autocorrelation** (neighboring patches 가 매우 유사함) 때문에,
token-level 예측은 text 보다 이웃 정보에 더 많이 의존한다.

정보론적으로:
$$I(t_i; t_j) \approx \begin{cases} \text{high} & \text{if} \|i-j\|_1 \text{ small} \\ \text{low} & \text{if} \|i-j\|_1 \text{ large} \end{cases}$$

따라서 "비가까운" masked token 들의 비율 (40%) 이 적절한 challenge level.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: dVAE Codebook 시뮬레이션

```python
import torch
import torch.nn as nn
import numpy as np

# Simplified dVAE tokenizer skeleton
class SimpleVQEmbedding(nn.Module):
    """Minimal VQ-VAE codebook quantization."""
    def __init__(self, num_embeddings=8192, embedding_dim=512):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        
    def forward(self, z_e):
        """z_e: [B, H, W, D] → indices: [B, H, W]"""
        # Flatten spatial dimensions
        B, H, W, D = z_e.shape
        z_flat = z_e.view(-1, D)  # [B*H*W, D]
        
        # Compute distances to all codebook entries
        # ||z - c||^2 = ||z||^2 + ||c||^2 - 2 z·c
        z_norm = (z_flat ** 2).sum(dim=1, keepdim=True)  # [B*H*W, 1]
        c_norm = (self.codebook.weight ** 2).sum(dim=1)  # [8192]
        distances = z_norm + c_norm - 2 * z_flat @ self.codebook.weight.T  # [B*H*W, 8192]
        
        # Find nearest codebook index
        indices = distances.argmin(dim=1)  # [B*H*W]
        
        # Reconstruct quantized embedding
        z_q = self.codebook(indices)  # [B*H*W, D]
        z_q = z_q.view(B, H, W, D)
        
        return indices.view(B, H, W), z_q

# Test
B, H, W, D = 2, 28, 28, 512
z_e = torch.randn(B, H, W, D)
vq = SimpleVQEmbedding(num_embeddings=8192, embedding_dim=D)

indices, z_q = vq(z_e)
print(f"Input shape: {z_e.shape}")
print(f"Token indices shape: {indices.shape}, range: [{indices.min()}, {indices.max()}]")
print(f"Reconstructed shape: {z_q.shape}")
print(f"Reconstruction error (MSE): {((z_e - z_q) ** 2).mean():.4f}")
```

### 실험 2: Masked Token Prediction Loss

```python
import torch
import torch.nn.functional as F

# Simulated BEiT setup
B, N, vocab_size = 4, 784, 8192
mask_ratio = 0.4

# Mock Transformer output logits for each position
logits = torch.randn(B, N, vocab_size)  # [B, N, 8192]

# Ground truth tokens (from dVAE)
true_tokens = torch.randint(0, vocab_size, (B, N))  # [B, N]

# Create mask (40% masked)
mask = torch.rand(B, N) < mask_ratio  # [B, N] bool

# Only compute loss on masked positions
masked_logits = logits[mask]  # [num_masked, 8192]
masked_targets = true_tokens[mask]  # [num_masked]

# Cross-entropy loss
loss = F.cross_entropy(masked_logits, masked_targets)
print(f"Batch size: {B}, Sequence length: {N}, Vocab size: {vocab_size}")
print(f"Mask ratio: {mask_ratio:.1%}")
print(f"Masked positions: {mask.sum().item()}")
print(f"MLM Loss: {loss:.4f}")
```

### 실험 3: Masking Ratio 비교 (이론적)

```python
import numpy as np

# Simulated "reconstruction difficulty" as function of masking ratio
masking_ratios = np.linspace(0, 0.9, 20)

# Empirical model: difficulty ∝ sqrt(masking_ratio) * spatial_decay
spatial_correlation = 0.8  # neighboring patch correlation

def difficulty_score(mask_ratio, spatial_correlation=0.8):
    """Higher = harder to predict."""
    visible_fraction = 1 - mask_ratio
    neighbor_info_retention = spatial_correlation ** (1 / max(visible_fraction, 0.01))
    difficulty = mask_ratio / (neighbor_info_retention + 0.1)
    return difficulty

# Add BERT (15%), BEiT (40%), MAE (75%) markers
bert_ratio = 0.15
beit_ratio = 0.40
mae_ratio = 0.75

print(f"Difficulty at BERT ratio ({bert_ratio:.1%}): {difficulty_score(bert_ratio):.3f}")
print(f"Difficulty at BEiT ratio ({beit_ratio:.1%}): {difficulty_score(beit_ratio):.3f}")
print(f"Difficulty at MAE ratio  ({mae_ratio:.1%}): {difficulty_score(mae_ratio):.3f}")
print("\nBEiT 의 40% 는 BERT 의 15% 보다 약 2.7배 더 어려운 task 를 제공합니다.")
```

### 실험 4: Tokenization Quality 측정

```python
import numpy as np

def compute_reconstruction_snr(x_orig, x_recon):
    """Signal-to-Noise Ratio of reconstruction."""
    signal = (x_orig ** 2).mean()
    noise = ((x_orig - x_recon) ** 2).mean()
    snr = 10 * np.log10(signal / (noise + 1e-8))
    return snr

# Simulate reconstructed image (in practice, from dVAE decoder)
x_orig = np.random.randn(1, 3, 224, 224)
# Simulate lossy reconstruction (quantization loss)
noise_level = 0.05  # 5% quantization noise
x_recon = x_orig + noise_level * np.random.randn(1, 3, 224, 224)

snr = compute_reconstruction_snr(x_orig, x_recon)
print(f"Reconstruction SNR: {snr:.2f} dB")
print(f"이 정도의 reconstruction error 가 downstream task 의 ceiling 을 결정합니다.")
```

## 🔗 실전 활용

**BEiT Pretraining (Bao et al. 2022)**:
1. 대규모 이미지 데이터 (ImageNet-21k, LAION 등) 수집
2. DALL-E 의 dVAE tokenizer 로 이미지 → token sequence 변환
3. 40% masking, Transformer encoder, MLM loss 로 사전학습
4. Downstream task (ImageNet-1k fine-tune, detection, segmentation) 로 평가

**구현 체크리스트**:
- dVAE tokenizer: 사전학습된 weight 필요 (inference only)
- Patch masking: random uniform sampling (no clustering)
- Loss: cross-entropy over 8192 vocabulary
- Optimizer: AdamW, warmup schedule 필수
- Batch size: 2048 이상 권장 (larger better)

**하이퍼파라미터 가이드**:
- Masking ratio: 40% (고정, BERT 의 15% 와 차별화)
- Learning rate: 1.5e-4 (ImageNet-1k scale)
- Warmup: 20 epochs
- Total epochs: 300
- Model: ViT-B/16, ViT-L/16

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **dVAE 품질** | 8192 vocabulary 의 codebook 이 정보 충분 | 더 큰 vocab (16K, 32K) 이 나을 수 있음 |
| **Tokenizer 재사용** | DALL-E 의 공개된 dVAE weight 사용 | Task 특화 tokenizer 학습 시 성능 향상 가능 |
| **Discrete 가정** | Token indices 의 cross-entropy 가 유일한 objective | Continuous latent 도 동시에 학습하면 더 나음 |
| **40% 고정** | 모든 dataset 에 동일한 masking ratio | Image resolution 에 따라 조정 여지 있음 |
| **Spatial correlation 무시** | Random masking 이므로 이웃 bias 없음 | 구조화된 masking (block-wise) 도 탐색 가능 |

## 📌 핵심 정리

$$\boxed{\text{dVAE Tokenization}: \, x \in \mathbb{R}^{H \times W \times 3} \to \mathbf{t} \in [8192]^{H' W'}}$$

$$\boxed{\text{BEiT MLM}: \, L = -\sum_{i \in M} \log p_\theta(\mathbf{t}^{(i)} \mid \tilde{\mathbf{t}}), \quad |M| = 0.4 \cdot N}$$

| 개념 | 수식 | 의미 |
|------|-----|------|
| **dVAE Encoder** | $E_\phi: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{H' \times W' \times D}$ | Image → embedding grid |
| **VQ (Quantization)** | $k^* = \arg\min_k \|z - c_k\|^2$ | Nearest codebook entry |
| **Token Sequence** | $\mathbf{t} \in [8192]^{H' W'}$ | Discrete visual tokens (vocab 8192) |
| **Masking** | $\tilde{\mathbf{t}}^{(i)} = [\mathrm{MASK}]$ for $i \in M$ | 40% 의 position 을 mask |
| **MLM Loss** | Cross-entropy over vocabulary | Masked position 의 token 예측 |
| **Masking Ratio** | 40% (vs BERT 15%) | Text 보다 높은 이유: spatial correlation |

## 🤔 생각해볼 문제

### 문제 1 (기초): dVAE 와 VQ-VAE 의 차이

DALL-E 의 dVAE 는 "discrete VAE" 라고 불리는데, 일반적인 **VQ-VAE (Vector Quantized VAE)** 와의 구체적 차이점은?
특히 codebook update 메커니즘이 다른가?

<details>
<summary>해설</summary>

**VQ-VAE (van den Oord 2017)**:
- Codebook 은 learnable parameter 로, 학습 중 update
- Straight-through estimator (STE) 로 gradient 전파
- Codebook collapse 방지를 위한 exponential moving average (EMA) update
- Loss: reconstruction + codebook commitment + perplexity 항

**dVAE (DALL-E)**:
- Codebook 은 고정되거나 별도로 사전학습 (inference-only)
- Image generation 을 위한 특화된 discrete latent
- BEiT 에서는 dVAE 를 frozen 으로 사용 (tokenization 만 수행)

**결론**: dVAE 는 VQ-VAE 의 downstream task 최적화 버전.
BEiT 는 이미 학습된 dVAE 를 활용하므로, tokenization 과정은 deterministic 이고,
representation learning 만 focus.

</details>

### 문제 2 (심화): Masking Ratio 의 정보론적 정당화

왜 vision 에서는 40% masking 이 자연스럽고, NLP 에서는 15% 가 표준인가?
두 domain 의 **spatial/sequential autocorrelation** 차이를 정량적으로 설명하시오.

<details>
<summary>해설</summary>

**자연이미지의 spatial redundancy** (Bao et al. 2022):
- Neighboring patch correlation: ρ ≈ 0.8~0.9 (매우 높음)
- Reconstruction 이 이웃으로부터 거의 trivial
- 15% masking 시 주변 85% visible → over-easy task

**텍스트의 sequential redundancy**:
- Word-level correlation: ρ ≈ 0.3~0.4 (낮음)
- 한 단어가 다음 단어를 강제하지 않음
- 15% masking 시 주변 85% visible → 적절한 challenge

**정보론적 모델**:
Masked position 의 "복원 어려움" ∝ visible neighbors 로부터의 정보 loss

따라서 image 가 더 높은 mask ratio 를 감당할 수 있음.

</details>

### 문제 3 (논문 비평): dVAE Tokenizer 의 한계

BEiT 논문 (Bao et al. 2022) 은 "dVAE tokenizer quality 가 representation ceiling 을 결정한다" 고 했다.
이를 극복하는 방법 3 가지를 제시하고, 각각의 trade-off 를 분석하시오.

<details>
<summary>해설</summary>

**방법 1: 더 큰 vocabulary**
- 16K, 32K 의 codebook 으로 정보 손실 감소
- Trade-off: quantization 계산 비용 증가, training instability 위험

**방법 2: Continuous target (MAE, Ch5-02)**
- Pixel-level reconstruction 으로 discrete 제약 제거
- Trade-off: loss 가 scalar 에서 image dimension 으로 → 계산 비용 증가
- 하지만 "MAE 의 75% masking 을 가능하게 함" (spatial redundancy 활용)

**방법 3: Multi-target masking (MaskFeat, Ch5-05)**
- HOG, SIFT, CLIP feature 등 mid-level target 사용
- Token 이 아닌 feature 기반으로 더 높은 abstraction level
- Trade-off: target encoder 의 품질이 새로운 bottleneck

**권장**: 현 시점 (2024) 에서는 **MAE 의 continuous reconstruction** 이 가장 성공적.

</details>

---

<div align="center">

[◀ 이전](../ch4-dino/04-ibot-dinov2.md) | [📚 README](../README.md) | [다음 ▶](./02-mae.md)

</div>
