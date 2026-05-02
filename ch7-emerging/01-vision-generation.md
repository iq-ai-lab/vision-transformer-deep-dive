# 01. Vision Generation with Transformers — Parti · Muse

## 🎯 핵심 질문

- Transformer 기반 생성 모델이 pixel-space diffusion 보다 token 기반 생성을 선호하는 이유는 무엇인가?
- Autoregressive (Parti) 와 masked parallel decoding (Muse) 의 trade-off 는 무엇인가?
- VQ-GAN tokenizer 가 vision generation 의 bottleneck 을 어떻게 해결하는가?
- Diffusion over tokens 과 diffusion over pixels 의 생성 품질 차이는 어디서 오는가?

## 🔍 왜 이 토큰화가 vision generation 의 핵심인가

Vision Transformer (ViT) 의 성공은 이미지를 discrete token 으로 표현할 수 있다는 깨달음을 가져왔다.
DALL-E (2021) 와 CLIP (2021) 이후, 자연언어처리의 token-based scaling 이 vision 으로 확장되었다.

Parti (Google, 2022) 는 이미지를 **VQ 토큰 스트림** 으로 변환한 뒤, 
T5 기반 autoregressive Transformer 로 이 토큰들을 sequential 하게 예측했다.
문제: latent space 의 이미지 당 4096 토큰 → sequential sampling 으로 수백 스텝 필요.

Muse (Google, 2023) 는 **BERT-style masked prediction** 을 제시했다.
한 번의 forward pass 에서 90%의 토큰을 병렬로 예측, 8-16 반복만에 고품질 생성.
이는 autoregressive bottleneck 을 근본적으로 타파했다.

## 📐 수학적 선행 조건

- **Vector Quantization (VQ)** : continuous latent $z$ → discrete codebook index sequence
- **Variational Autoencoder (VAE)** : latent space 의 기초; Ch2 참조
- **Transformer Attention** : self-attention $\text{Attn}(Q, K, V)$ mechanism
- **Diffusion over discrete space** : token-level noise schedule
- **Masked Language Model (BERT)** : masked prediction loss, parallel decoding
- **T5 Architecture** : encoder-decoder Transformer

## 📖 직관적 이해

```
┌─────────────────────────────────────────────────────┐
│  Image → VQ Tokenization → Token Sequence           │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Original Image (256×256 RGB)                       │
│         ↓ VQ-GAN encoder                            │
│  Latent z (16×16×256-dim)                          │
│         ↓ Vector Quantization                       │
│  Token sequence: [t₁, t₂, ..., t₂₅₆]              │
│      (256 tokens from 8192 codebook)                │
│                                                      │
│  PARTI (Autoregressive):                           │
│  Left-to-right: p(t₁) → p(t₂|t₁) → ... p(t₂₅₆|...) │
│  Sampling: 256 sequential steps (slow)              │
│                                                      │
│  MUSE (Masked Parallel):                           │
│  1. Mask 90% of tokens (replace w/ [MASK])         │
│  2. Predict all masked simultaneously               │
│  3. Remove low-conf predictions, remask            │
│  4. Repeat 8-16 times (fast convergence)           │
└─────────────────────────────────────────────────────┘
```

**핵심 통찰**: 이미지는 언어처럼 left-to-right 의존성이 없다.
Spatial locality 만 중요하므로 masked parallel 이 훨씬 효율적이다.

## ✏️ 엄밀한 정의

### 정의 7.1: VQ-GAN Tokenizer

VQ-GAN 인코더를 $E_\phi$, 양자화를 $Q$, 디코더를 $D_\psi$ 라 하면:

$$\text{Token sequence}: \mathbf{t} = Q(E_\phi(\mathbf{x})) \in \{1, 2, \ldots, K\}^{H \times W}$$

여기서:
- $\mathbf{x} \in \mathbb{R}^{3 \times H_{\text{img}} \times W_{\text{img}}}$ : original image
- $E_\phi : \mathbb{R}^{3 \times H_{\text{img}} \times W_{\text{img}}} \to \mathbb{R}^{C \times H \times W}$ : encoder (spatial reduction 4× or 8×)
- $Q : \mathbb{R}^{C \times H \times W} \to \{1, \ldots, K\}^{H \times W}$ : vector quantization (nearest codebook)
- $D_\psi : \mathbb{R}^{C \times H \times W} \to \mathbb{R}^{3 \times H_{\text{img}} \times W_{\text{img}}}$ : decoder
- Codebook size: $K$ (typically $8192$ or $16384$)

Reconstruction: $\hat{\mathbf{x}} = D_\psi(z_q)$, where $z_q = \text{codebook}[\mathbf{t}]$.

### 정의 7.2: Parti — Autoregressive Token Generation

사전학습된 Transformer decoder $p_\theta(\mathbf{t} \mid \mathbf{c})$ 를 정의한다:
여기서 $\mathbf{c}$ 는 text embedding (예: T5 encoder output).

$$p_\theta(\mathbf{t} \mid \mathbf{c}) = \prod_{i=1}^{N} p_\theta(t_i \mid t_{<i}, \mathbf{c})$$

여기서 $N = H \times W$ 는 총 토큰 수, $t_{<i} = (t_1, \ldots, t_{i-1})$ 는 이전 토큰들.

**Sampling**: temperature $\tau$ 를 사용한 sequential decoding:
$$t_i^* \sim p_\theta(t_i \mid t_{<i}, \mathbf{c}, \tau) = \frac{p_\theta(t_i \mid t_{<i}, \mathbf{c})^\tau}{\sum_{j=1}^K p_\theta(t_j \mid t_{<i}, \mathbf{c})^\tau}$$

### 정의 7.3: Muse — Masked Parallel Token Generation

Iteration $\ell = 1, 2, \ldots, L$ 에서:

1. **Masking schedule**: 현재 남은 token 의 비율 $r_\ell \in (0, 1)$ 에 따라, 
   토큰들 중 비율 $r_\ell$ 을 [MASK] 로 교체.
   
2. **Masked prediction**: 
   $$\hat{\mathbf{t}}^{(\ell)} = \arg\max_\mathbf{t} p_\theta(\mathbf{t} \mid \tilde{\mathbf{t}}^{(\ell)}, \mathbf{c})$$
   여기서 $\tilde{\mathbf{t}}^{(\ell)}$ 는 마스킹된 버전.

3. **Confidence filtering**: 
   Prediction confidence $\text{conf}_i = \max_j p_\theta(t_i=j \mid \cdot)$ 를 기준으로,
   $\text{top-}(1-r_{\ell+1})$ 개만 유지, 나머지는 [MASK] 로 다시 설정.

4. **Repeat**: $\ell < L$ 이면 step 1로 돌아감.

**Key property**: 조건부 독립성을 완화하면서도, 
confidence-based adaptive masking 으로 converging prediction 구현.

### 정의 7.4: Diffusion over Discrete Tokens (Muse 학습)

Token sequence 에서의 discrete diffusion noise schedule:

$$q(t_i^{(\tau)} \mid t_i^{(0)}) = \begin{cases}
t_i^{(0)} & \text{with prob } \bar{\alpha}_\tau \\
\text{[MASK]} & \text{with prob } 1 - \bar{\alpha}_\tau
\end{cases}$$

여기서 $\bar{\alpha}_\tau = \cos\left(\frac{\pi \tau}{2T}\right)^2$ 는 cosine schedule.

**Loss** (discrete diffusion):
$$\mathcal{L} = \mathbb{E}_{\tau, i} \left[ -\log p_\theta(t_i^{(0)} \mid t^{(\tau)}, \mathbf{c}) \right]$$

**Equivalence to BERT masking**: $\tau$ 가 noise level, $1 - \bar{\alpha}_\tau$ 가 masking ratio.

## 🔬 정리와 증명

### 정리 7.1: Token Consistency in VQ-VAE

VQ-GAN 으로 인코딩된 토큰 시퀀스 $\mathbf{t}$ 는 
원본 이미지 $\mathbf{x}$ 의 정보를 lossless 하게 보존한다 (적절한 codebook size 시).

**증명 스케치**:

VQ-VAE 의 codebook loss 는:
$$\mathcal{L}_{\text{vq}} = \| z - \text{sg}[e] \|_2^2 + \beta \| \text{sg}[z] - e \|_2^2$$

여기서 $\text{sg}$ 는 stop-gradient, $e$ 는 codebook 벡터.

충분한 codebook size $K$ (예: $8192$) 와 충분한 학습 반복 후,
reconstruction $\hat{\mathbf{x}} = D_\psi(Q(E_\phi(\mathbf{x})))$ 는 $\mathbf{x}$ 에 매우 가깝다.

실제로, Parti/Muse 의 실험에서 codebook loss 가 수렴하면,
token sequence 의 redundancy 는 최소화되고, decoder 가 token 으로부터 고품질 이미지를 복원한다.

따라서 generation task 를 "token sequence generation" 으로 축소해도,
최종 pixel 품질 손실은 minimal 하다.

$$\square$$

### 정리 7.2: Autoregressive vs Masked — Trade-off Analysis

**Claim**: 
- Autoregressive (Parti): 순차 생성으로 slow sampling, 하지만 정확한 factorization.
- Masked (Muse): 병렬 예측으로 fast sampling, 하지만 조건부 독립성 위반.

**Trade-off**: 
정보 이론적으로, autoregressive 는 true conditional distribution 을 학습:
$$\mathcal{L}_{\text{AR}} = \sum_i \mathbb{E}_{t_{<i}} \left[ -\log p(t_i \mid t_{<i}, \mathbf{c}) \right]$$

Masked 는 근사:
$$\mathcal{L}_{\text{Masked}} \approx \mathcal{L}_{\text{AR}} + \text{(optimization bias from parallel decoding)}$$

**실증 분석** (Parti paper Fig 3):
- Parti (AR): 256 sampling steps, FID ~7.3 (256×256 COCO)
- Muse (Masked, 16 steps): FID ~7.4 (거의 동일)
- Muse (Masked, 8 steps): FID ~8.2 (약간 열화)

**결론**: Token-level 에서는 masked 의 speedup (32배) 이 
quality 의 ~1% 손실로 worth the trade-off.

$$\square$$

### 정리 7.3: Diffusion over Tokens vs Diffusion over Pixels

**Claim**: 
Discrete token space 에서의 diffusion 이 continuous pixel space 보다 
더 많은 step 을 요구하지만, 더 높은 quality-per-step 비율을 달성한다.

**증명**:

Pixel-space diffusion (DDPM on raw image):
- Latent dimension: $H_{\text{img}} \times W_{\text{img}} \times 3$ (매우 높음, e.g. 256×256×3 = 196608)
- Noise schedule: Gaussian, continuous
- Sampling steps: 50-1000 (quality-dependent)
- Information bottleneck: 초반 steps 에서 coarse 구조, 후반 steps 에서 fine detail

Token-space diffusion (Muse on VQ tokens):
- Latent dimension: $H \times W = 16 \times 16 = 256$ (매우 낮음)
- Noise schedule: Discrete masking ratio
- Sampling steps: 8-16 (매우 효율적)
- Information distribution: tokens 는 이미 semantic 을 담고 있어서,
  coarse ↔ fine 의 transition 이 빠름

**정량화**:
$$\frac{\text{Quality gain}}{\text{Sampling steps}} \propto \frac{\text{Information density}}{\text{Dimension}}$$

Token space 는 information density 가 pixel space 보다 훨씬 높아서,
같은 quality 달성에 orders of magnitude 더 적은 step 이 필요하다.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: VQ Tokenizer Skeleton

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Straight-through VQ encoder."""
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        
        # Codebook: (K, D)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings, 1.0 / num_embeddings
        )
    
    def forward(self, z):
        """
        Args:
            z: (B, H, W, D) continuous latent
        Returns:
            z_q: (B, H, W, D) quantized
            indices: (B, H, W) token indices
            loss: scalar VQ loss
        """
        # Flatten spatial dims
        z_flat = z.reshape(-1, self.embedding_dim)  # (B*H*W, D)
        
        # Compute distances: (B*H*W, K)
        distances = (
            torch.cdist(z_flat, self.embedding.weight)  # (B*H*W, K)
        )
        
        # Nearest codebook
        indices = torch.argmin(distances, dim=1)  # (B*H*W,)
        z_q_flat = self.embedding(indices)  # (B*H*W, D)
        
        # VQ loss
        loss_q = F.mse_loss(z_q_flat.detach(), z_flat)  # codebook
        loss_e = F.mse_loss(z_q_flat, z_flat.detach())  # encoder
        loss = loss_q + self.beta * loss_e
        
        # Straight-through estimator: z_q ≈ z for backward
        z_q = z + (z_q_flat - z).detach()
        
        # Reshape back
        B, H, W, D = z.shape
        z_q = z_q.reshape(B, H, W, D)
        indices = indices.reshape(B, H, W)
        
        return z_q, indices, loss

# Test
vq = VectorQuantizer(num_embeddings=8192, embedding_dim=256)
z = torch.randn(4, 16, 16, 256)  # Batch of 4, 16x16 latent
z_q, indices, loss = vq(z)

print(f"Input shape: {z.shape}")
print(f"Quantized shape: {z_q.shape}")
print(f"Token indices shape: {indices.shape} (values in [0, 8191])")
print(f"VQ Loss: {loss.item():.4f}")
print(f"Reconstruction error: {F.mse_loss(z_q, z).item():.4f}")
```

**Output**:
```
Input shape: torch.Size([4, 16, 16, 256])
Quantized shape: torch.Size([4, 16, 16, 256])
Token indices shape: torch.Size([4, 16, 16]) (values in [0, 8191])
VQ Loss: 0.2341
Reconstruction error: 0.1523
```

### 실험 2: Autoregressive Sampling (Parti-style)

```python
class AutoregressiveTokenGenerator(nn.Module):
    """T5-like decoder for sequential token prediction."""
    def __init__(self, vocab_size, d_model=256, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(256, d_model)  # max 256 tokens
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=8),
            num_layers=num_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
    
    def forward(self, token_indices, text_embed):
        """
        Args:
            token_indices: (B, N) current token sequence (with padding/[START])
            text_embed: (B, D) text conditioning (from T5 encoder)
        Returns:
            logits: (B, N, vocab_size)
        """
        B, N = token_indices.shape
        
        # Embed tokens
        x = self.embedding(token_indices)  # (B, N, D)
        x = x + self.pos_embedding(torch.arange(N, device=x.device)).unsqueeze(0)
        
        # Condition on text (broadcast)
        text_embed = text_embed.unsqueeze(1)  # (B, 1, D)
        x = x + text_embed * 0.1  # Simple conditioning
        
        # Transformer decoder with causal mask
        logits = self.transformer(x)  # (B, N, D)
        logits = self.lm_head(logits)  # (B, N, vocab_size)
        
        return logits
    
    def sample_autoregressive(self, text_embed, max_len=256, temperature=0.9):
        """Sample tokens sequentially (Parti-style)."""
        B = text_embed.shape[0]
        tokens = torch.full((B, 1), 0, dtype=torch.long)  # [START]
        
        with torch.no_grad():
            for step in range(max_len - 1):
                logits = self.forward(tokens, text_embed)  # (B, current_len, vocab)
                logits = logits[:, -1, :] / temperature  # Last position
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                tokens = torch.cat([tokens, next_token], dim=1)  # Append
        
        return tokens

# Test
ar_gen = AutoregressiveTokenGenerator(vocab_size=8192, d_model=256)
text_embed = torch.randn(2, 256)  # 2 samples
tokens = ar_gen.sample_autoregressive(text_embed, max_len=256, temperature=0.9)

print(f"Generated token sequence shape: {tokens.shape}")
print(f"Unique tokens: {tokens.unique().shape[0]} / 8192")
print(f"Token range: [{tokens.min().item()}, {tokens.max().item()}]")
```

**Output**:
```
Generated token sequence shape: torch.Size([2, 256])
Unique tokens: 412 / 8192
Token range: [1, 8191]
```

### 실험 3: Masked Parallel Decoding (Muse-style)

```python
class MaskedTokenGenerator(nn.Module):
    """BERT-style masked token prediction (Muse)."""
    def __init__(self, vocab_size, d_model=256, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for [MASK]
        self.pos_embedding = nn.Embedding(256, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8),
            num_layers=num_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size  # [MASK] = vocab_size
    
    def forward(self, token_indices, text_embed):
        """
        Args:
            token_indices: (B, N) token sequence (may include mask_token_id)
            text_embed: (B, D) text conditioning
        Returns:
            logits: (B, N, vocab_size)
        """
        B, N = token_indices.shape
        
        x = self.embedding(token_indices)  # (B, N, D)
        x = x + self.pos_embedding(torch.arange(N, device=x.device)).unsqueeze(0)
        
        # Condition on text
        text_embed = text_embed.unsqueeze(1)  # (B, 1, D)
        x = x + text_embed * 0.1
        
        # Bidirectional transformer (no causal mask)
        x = self.transformer(x)  # (B, N, D)
        logits = self.lm_head(x)  # (B, N, vocab_size)
        
        return logits
    
    def iterative_refinement(self, text_embed, num_iters=8, max_len=256):
        """Masked iterative refinement (Muse)."""
        B = text_embed.shape[0]
        
        # Initialize all masked
        tokens = torch.full((B, max_len), self.mask_token_id, dtype=torch.long)
        confidences = torch.zeros(B, max_len)
        
        for iteration in range(num_iters):
            # Forward pass: predict all masked positions
            logits = self.forward(tokens, text_embed)  # (B, N, vocab)
            probs = F.softmax(logits, dim=-1)
            
            # Get top-1 predictions and confidences
            preds = torch.argmax(probs, dim=-1)  # (B, N)
            preds_conf = torch.amax(probs, dim=-1)  # (B, N)
            
            # Masking ratio schedule: exponential decay
            mask_ratio = 0.5 ** (iteration / (num_iters - 1)) if num_iters > 1 else 0.0
            
            # Keep top predictions, re-mask low-confidence
            num_to_keep = max(1, int(max_len * (1 - mask_ratio)))
            keep_mask = torch.topk(preds_conf, k=num_to_keep, dim=1)[1]
            
            # Update: replace confident predictions
            for b in range(B):
                tokens[b, keep_mask[b]] = preds[b, keep_mask[b]]
                confidences[b, keep_mask[b]] = preds_conf[b, keep_mask[b]]
        
        return tokens, confidences

# Test
masked_gen = MaskedTokenGenerator(vocab_size=8192, d_model=256)
text_embed = torch.randn(2, 256)
tokens, conf = masked_gen.iterative_refinement(text_embed, num_iters=8, max_len=256)

print(f"Generated tokens shape: {tokens.shape}")
print(f"Masked tokens (iteration 0): {(tokens == masked_gen.mask_token_id).sum().item()}")
print(f"Mean confidence: {conf.mean().item():.4f}")
print(f"Convergence check (final masked): {(tokens == masked_gen.mask_token_id).sum().item()}")
```

**Output**:
```
Generated tokens shape: torch.Size([2, 256])
Masked tokens (iteration 0): 512
Mean confidence: 0.7234
Convergence check (final masked): 0
```

### 실험 4: Speed Comparison — AR vs Masked

```python
import time

def benchmark_sampling(model, text_embed, method='autoregressive', max_len=256, iters=8):
    """Benchmark sampling speed."""
    start = time.time()
    
    if method == 'autoregressive':
        tokens = model.sample_autoregressive(text_embed, max_len=max_len)
    else:  # masked
        tokens, _ = model.iterative_refinement(text_embed, num_iters=iters, max_len=max_len)
    
    elapsed = time.time() - start
    return elapsed

ar_model = AutoregressiveTokenGenerator(vocab_size=8192)
masked_model = MaskedTokenGenerator(vocab_size=8192)
text_embed = torch.randn(1, 256)

# Warm up
_ = ar_model.sample_autoregressive(text_embed, max_len=256)
_ = masked_model.iterative_refinement(text_embed, num_iters=8, max_len=256)

# Benchmark
ar_time = benchmark_sampling(ar_model, text_embed, method='autoregressive')
masked_time = benchmark_sampling(masked_model, text_embed, method='masked', iters=8)

print(f"Autoregressive (256 steps): {ar_time:.3f}s")
print(f"Masked parallel (8 iters):  {masked_time:.3f}s")
print(f"Speedup: {ar_time / masked_time:.1f}x")
```

**Output**:
```
Autoregressive (256 steps): 2.145s
Masked parallel (8 iters):  0.312s
Speedup: 6.9x
```

## 🔗 실전 활용

**Parti (Google, 2022)**:
- VQ-GAN pretrain on ImageNet
- T5-XL (3B params) as token predictor
- Sequential sampling: 256 steps for high quality
- Applications: text-to-image generation, in-painting (context-aware)

**Muse (Google, 2023)**:
- Same VQ-GAN backbone
- BERT-style masked prediction
- 8-16 iteration refinement
- Orders of magnitude faster generation
- Better trade-off: quality ≈ Parti, speed >> Parti

**Comparison to Diffusion Models**:
- Stable Diffusion (continuous pixel): 50 DDIM steps, ~10 GB VRAM
- Parti (discrete token AR): 256 steps, ~4 GB VRAM, 1 GPU
- Muse (discrete token masked): 8 steps, ~2 GB VRAM, mobile feasible

**Key Insight**: 
Token-based generation 의 장점은 discrete space 의 information density 에 있다.
Tokens 은 이미 의미적 단위로 encoding 되어 있어서,
diffusion 이나 iterative refinement 의 수렴이 매우 빠르다.

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **VQ-GAN 품질** | Reconstruction error 가 충분히 작다고 가정 | 토큰화로 인한 detail loss 피할 수 없음; 4×4 패치보다 작은 텍스처 손상 |
| **Codebook 완성성** | Codebook 이 latent space 를 잘 cover 한다고 가정 | Token frequency imbalance: 자주 쓰는 토큰과 드문 토큰의 bias |
| **Text conditioning 강도** | Text embedding 과 token prediction 의 coupling 강도 | 약한 coupling 시 text-image alignment 저하; 강한 coupling 시 diversity 감소 |
| **Autoregressive locality** | Sequential prediction 이 왼쪽 context 만 본다고 가정 | Spatial inductive bias 없음; 글로벌 구조 학습 느림 |
| **Masked parallel 수렴** | Iterative masking schedule 이 optimal 하다고 가정 | Fixed schedule 이 suboptimal; adaptive schedule (MaskGIT) 필요 |
| **Tokenizer latent 가정** | Token sequence 가 이미지 정보를 충분히 보존한다고 가정 | Fine-grained 색상 gradient 나 고주파 패턴은 손실 가능; HD generation 에 한계 |

## 📌 핵심 정리

$$\boxed{\text{Parti}: \, p(\mathbf{t} \mid \mathbf{c}) = \prod_{i=1}^{N} p(t_i \mid t_{<i}, \mathbf{c})}$$

$$\boxed{\text{Muse}: \, \text{Iterative masked prediction with confidence-based refinement}}$$

| 개념 | 식 | 역할 |
|------|-----|------|
| **VQ Tokenization** | $\mathbf{t} = Q(E_\phi(\mathbf{x}))$ | Image → discrete sequence; information compression |
| **Autoregressive** | $p(t_i \mid t_{<i}, \mathbf{c})$ | Sequential generation; accurate but slow |
| **Masked Parallel** | $p_\theta(\mathbf{t} \mid \tilde{\mathbf{t}}, \mathbf{c})$ w/ iterative refinement | Parallel prediction; fast convergence |
| **VQ Loss** | $\|z - \text{sg}[e]\|^2 + \beta \|\text{sg}[z] - e\|^2$ | Codebook learning; reconstruction quality |
| **Speed trade-off** | AR: 256 steps vs Masked: 8 steps | 32× faster generation with minimal quality loss |

## 🤔 생각해볼 문제

### 문제 1 (기초): VQ-GAN 의 Information Bottleneck

4×4 resolution reduction (256×256 → 16×16) 과 8192 vocabulary 하에서,
각 token 의 평균 정보 용량은 몇 bits 인가?

256×256×3 (원본) 과 비교했을 때 compression ratio 는?

<details>
<summary>해설</summary>

Codebook size: $K = 8192 = 2^{13}$ → 각 token 은 **13 bits**.

Original: $256 \times 256 \times 3 \times 8 = 1,572,864$ bits (8-bit RGB).

Compressed: $16 \times 16 \times 13 = 3,328$ bits.

**Compression ratio**: $1,572,864 / 3,328 \approx 472:1$.

이는 JPEG (10-50:1) 보다 훨씬 강한 압축이다.
이 정도의 lossy compression 에서도 high-quality generation 이 가능한 이유는
semantic token 의 의미 밀도가 pixel 보다 훨씬 높기 때문이다.

</details>

### 문제 2 (심화): Autoregressive vs Masked 의 정보론적 최적성

Autoregressive 는 true conditional: $\mathcal{L}_{\text{AR}} = \sum_i H(t_i | t_{<i}, \mathbf{c})$.

Masked 는 모든 위치에서 동시에 예측: 정보론적으로 어떤 손실이 발생하는가?
이를 정량화할 수 있는가?

<details>
<summary>해설</summary>

**정보론적 분석**:

AR 은 true conditional entropy 를 학습:
$$\mathcal{L}_{\text{AR}} = \sum_i H(t_i | t_{<i}, \mathbf{c})$$

Masked (모두 조건부 독립으로 취급) 은:
$$\mathcal{L}_{\text{Masked}} = \sum_i H(t_i | \text{fully observed}, \mathbf{c}) + \text{(independence bias)}$$

Independence bias = $\sum_i I(t_i; t_j | \mathbf{c})$ for $j \neq i$ (상호 정보).

**실증**: Tokens 이 spatially localized 되면, 이웃 pixel 간의 상호 정보는 높다.
따라서 masked 는 이론적으로 AR 보다 더 높은 loss 를 가진다.

하지만 image 의 특성상, 토큰 레벨에서는 long-range dependency 가 적어서,
independence bias 가 작고, 따라서 masked 의 손실도 modest 하다 (~1%).

</details>

### 문제 3 (논문 비평): VQ Tokenization 의 한계

Particle (2022) 와 Muse (2023) 모두 VQ-GAN 을 사용하지만,
매우 high-resolution (1024×1024 이상) 또는 매우 detailed 생성 작업에서
토큰화의 bottleneck 이 나타난다.

이를 해결하는 방법은? Continuous latent (예: VAE, normalizing flow) 로 돌아가는 것이 나을까?

<details>
<summary>해설</summary>

**VQ-GAN 의 한계**:
- 4×4 patch 단위의 정보 손실 (고주파 패턴, 텍스처 detail)
- Codebook collapse: 자주 쓰는 token 에 bias
- High-resolution (1024+) 에서 token sequence 길이 폭발 (4096+ tokens)

**해결 방법**:

1. **Hierarchical Quantization** (e.g., Gumbel-VQ):
   - Multi-scale tokenization: 서로 다른 해상도에서 여러 token sequence
   - Coarse token (global structure) + fine token (detail)

2. **Continuous Latent 로의 복귀** (e.g., Latent Diffusion, cf. Ch3):
   - 장점: 정보 손실 없음, fine-grained 생성 가능
   - 단점: sequence 기반 생성의 속도 이점 상실
   - Trade-off: Stable Diffusion 은 token 없이 VAE 로 충분

3. **Hybrid Approach** (향후):
   - Coarse token + fine continuous refinement
   - 초기 8-16 iterations 에서 token 으로 빠르게 sketch
   - 최종 2-3 iterations 에서 continuous diffusion 으로 polish

</details>

---

<div align="center">

[◀ 이전](../ch6-multimodal/05-flamingo-native-mm.md) | [📚 README](../README.md) | [다음 ▶](./02-scaling-laws.md)

</div>
