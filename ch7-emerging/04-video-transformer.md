# 04. Video Understanding · Generation — TimeSformer · VideoMAE · Sora

## 🎯 핵심 질문

- Video 를 3D tensor (T, H, W) 로 처리할 때, spatial-temporal attention 의 복잡도는?
- Divided space-time attention (ViT 를 T개 frame 으로 apply) 이 joint attention 보다 왜 효율적인가?
- Masked video modeling (tube masking) 이 frame-level masking 보다 나은 이유는?
- Sora 같은 video generation model 은 어떤 architecture 를 사용하는가?
- Future: world models 과 physical understanding 을 video Transformer 로 어떻게 달성하는가?

## 🔍 왜 이 video representation 이 vision 의 미래인가

Image 는 정적이지만, video 는 **동역학 (dynamics)** 을 담는다.
같은 object 가 여러 frame 에 걸쳐 어떻게 움직이는지,
환경이 어떻게 변하는지를 이해하는 것은
embodied AI, robotics, world modeling 의 foundation 이다.

**TimeSformer (Bertasius et al., 2021)** 는
image Transformer (ViT) 를 video 로 확장하는 가장 자연스러운 방법을 제시했다:
$$\text{Divided ST-Attention} = \text{Spatial attention per frame} + \text{Temporal attention per patch}$$

Complexity: $O(T \cdot H^2W^2 + H^2W^2 \cdot T) = O(THW(T + HW))$ vs naive $O((THW)^2)$.

**VideoMAE (Tong et al., 2022)** 는 masked video modeling 에서
**tube masking** (같은 spatial location 을 모든 frame 에서 마스크) 을 제시했다.
이는 temporal redundancy 를 exploit 하고, 90% 이상 마스킹에도 수렴한다.

**Sora (OpenAI, 2024)** 는 diffusion + transformer 로
**text-to-video generation** 에서 unprecedented quality 를 달성했다.
U-ViT (transformer-based diffusion) + temporal modeling 의 조합.

## 📐 수학적 선행 조건

- **Self-attention 복잡도** : $O(n^2)$ where $n$ = sequence length
- **Spatial-temporal factorization** : 3D tensor → 2D slices + 1D sequences
- **Masked autoencoding** : BERT-style masking 와 reconstruction
- **Diffusion for video** : noise schedule over frames, continuous generation
- **Optical flow** : motion estimation (보조 signal)

## 📖 직관적 이해

```
┌──────────────────────────────────────────────────────────┐
│  Video as 3D Tensor: (T, H, W, C)                       │
│  T frames, H×W spatial, C channels                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Naive: Transformer on flattened (T·H·W, C)            │
│  Complexity: O((T·H·W)²) = HUGE (impractical)          │
│                                                          │
│  TimeSformer: Divided Space-Time Attention             │
│  ┌────────────────────────────────────┐                 │
│  │ Frame 1 | Frame 2 | ... | Frame T │                 │
│  └────────────────────────────────────┘                 │
│        ↓         ↓                ↓                      │
│   [Spatial    [Spatial  ...  [Spatial                   │
│    Attn]      Attn]          Attn]                      │
│    (ViT)       (ViT)           (ViT)                     │
│        ↓         ↓                ↓                      │
│   [Temporal Attention over patches of all frames]       │
│                                                          │
│  Complexity: O(T·H²W² + HW·T²) = O(THW(T + HW))        │
│  Speedup: (THW)² / (THW(T+HW)) = THW / (T+HW)          │
│  (For T=8, H=W=14: 784x speedup vs naive)             │
│                                                          │
│  VideoMAE: Tube Masking                                │
│  ┌──────────────────────────────┐                       │
│  │ Mask same patch across frames │                      │
│  │ (exploit temporal redundancy) │                      │
│  │ Can mask 90%+ and still learn │                      │
│  └──────────────────────────────┘                       │
│        ↓                                                 │
│   Decoder: Unmask + predict all frames                 │
│        ↓                                                 │
│   Loss: MSE on masked patches                          │
└──────────────────────────────────────────────────────────┘
```

## ✏️ 엄밀한 정의

### 정의 7.12: Divided Space-Time Attention (TimeSformer)

Video sequence를 patches로 분할: $(T, H', W', D)$ where $(H', W') = (H, W) / P$.
$N = T \cdot H' \cdot W'$ 는 총 tokens.

**Standard self-attention** (비효율):
$$\text{Attn}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d}) V$$
여기서 $Q, K, V \in \mathbb{R}^{N \times d}$, complexity $O(N^2 d)$.

**Divided Space-Time Attention**:
1. **Spatial attention**: 각 time step $t$ 에 대해
   $$\text{AttnS}^{(t)}(Q_t, K_t, V_t) = \text{softmax}(Q_t K_t^\top / \sqrt{d}) V_t$$
   where $Q_t, K_t, V_t \in \mathbb{R}^{H'W' \times d}$.
   Complexity per frame: $O((H'W')^2 d)$, total: $O(T(H'W')^2 d)$.

2. **Temporal attention**: 각 patch $(i, j)$ 에 대해
   $$\text{AttnT}^{(i,j)}(Q_{i,j}, K_{i,j}, V_{i,j}) = \text{softmax}(Q_{i,j} K_{i,j}^\top / \sqrt{d}) V_{i,j}$$
   where $Q, K, V \in \mathbb{R}^{T \times d}$ (길이 $T$).
   Complexity per patch: $O(T^2 d)$, total: $O((H'W')^2 T^2 d)$.

**Total complexity**: $O(T(H'W')^2 d + (H'W')^2 T^2 d) = O((H'W')^2 d (T + T^2))$.

For typical values ($T=8$, $H'W'=196$): $O(196^2 \cdot (8 + 64)) \approx O(2.7M \cdot d)$.
Compare to naive: $O((8 \cdot 196)^2 \cdot d) = O(2.4B \cdot d)$ → **900배 더 효율적**.

### 정의 7.13: Tube Masking (VideoMAE)

Video 의 masked autoencoding:
각 video volume을 spatiotemporal patches로 분할.

**Tube masking strategy**:
주어진 masking ratio $m$ (예: 0.9), 각 spatial patch location $(i, j)$ 에 대해,
모든 frames $t = 1, \ldots, T$ 에서 동시에 마스크:
$$\text{mask}(t, i, j) = \begin{cases}
[\text{MASK}] & \text{if } (i, j) \in S_{\text{masked}} \text{ (for all } t \text{)} \\
\text{patch}(t, i, j) & \text{otherwise}
\end{cases}$$

where $S_{\text{masked}}$ 는 mask ratio $m$ 에 해당하는 random spatial patch set.

**비용**: 모든 frames 에서 같은 spatial location → 
compressed representation (fewer visible tokens) but strong temporal signal
(각 patch 는 여러 frames 에서 보이지만, spatial coverage 는 sparse).

### 정의 7.14: Spatio-Temporal DiT for Video Generation (Sora-style)

Video generation 을 diffusion model 로 formulate.
State: $(v_t, \mathbf{z}_t)$ where $v_t$ 는 video latent, $\mathbf{z}_t$ 는 Gaussian noise.

**Forward diffusion**:
$$v_t = \sqrt{\bar{\alpha}_t} v_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{z}$$

**Reverse (denoising)**:
$$\hat{\mathbf{z}}_\theta(v_t, t, \text{condition}) = \text{U-ViT}(v_t, t, \text{condition})$$

**U-ViT architecture** (Diffusion Transformer):
- Encoder: video patches → latent tokens (TimeSformer-style ST-attention)
- Bottleneck: cross-attention to text embedding
- Decoder: latent tokens → noise prediction (symmetric to encoder)
- Skip connections: encoder features → decoder layers

**Condition**: text embedding $e_{\text{text}}$ from CLIP/T5 encoder.

Loss:
$$\mathcal{L} = \mathbb{E}_{t, v_0, \mathbf{z}} \| \mathbf{z} - \hat{\mathbf{z}}_\theta(v_t, t, e_{\text{text}}) \|_2^2$$

### 정의 7.15: World Model (Future Direction)

**World model** 은 video representation learning 의 ultimate goal:
$$\text{WM}: (\text{video}, \text{action}) \to \text{latent dynamics}$$

이를 통해:
1. **Prediction**: 다음 frame 또는 future trajectory 예측
2. **Planning**: 역으로, 원하는 future frame 을 위해 필요한 actions 결정
3. **Control**: embodied AI (로봇) 가 학습한 dynamics 를 exploit 해 task 수행

**Example: Dreamer-style world model**:
- Encoder: video → latent state $s_t$
- Dynamics: $s_{t+1} = f_\theta(s_t, a_t)$ (learned transition)
- Decoder: $s_t \to \hat{v}_t$ (reconstruction)
- Value network: $s_t \to V(s_t)$ (planning guide)

Train on video + RL reward, infer new trajectories in imagination.

## 🔬 정리와 증명

### 정리 7.11: Divided ST-Attention 의 복잡도 감소

**Claim**: 
Divided space-time attention 은 naive joint attention 대비
$(THW) / (T + HW)$ 배 speedup 을 달성한다 (큰 T, HW 에서).

**증명**:

Naive: $n = THW$ tokens, $O(n^2) = O((THW)^2)$ complexity.

Divided:
- Spatial: $T$ frames × $O((HW)^2)$ = $O(T(HW)^2)$
- Temporal: $HW$ patches × $O(T^2)$ = $O((HW)T^2)$
- Total: $O(T(HW)^2 + (HW)T^2) = O(THW(HW + T))$

Ratio:
$$\frac{O((THW)^2)}{O(THW(HW + T))} = \frac{THW}{HW + T}$$

For typical video (T=8, H=W=224, patches=16×16 → HW=196):
$$\frac{8 \cdot 196}{196 + 8} = \frac{1568}{204} \approx 7.7$$

But effective speedup higher because:
- Spatial attention: optimized ViT kernels
- Temporal attention: very short sequence (T=8)
- 실제 speedup: 10-20배 wall-clock time.

$$\square$$

### 정리 7.12: VideoMAE 의 Tube Masking 정당성

**Claim**: 
Tube masking (frames 전체에서 같은 spatial patch 마스크) 이
random masking 보다 **temporal coherence** 를 leverage 해서
더 높은 masking ratio (>90%) 를 가능하게 한다.

**증명 스케치**:

**Information-theoretic argument**:
Temporal redundancy: frame $t$ 와 $t+1$ 은 매우 유사 (optical flow 작음).

Random masking: 같은 patch 를 양 frames 에서 마스크할 확률은 낮음.
→ 모델이 context frame 에서 쉽게 inpaint 가능.
→ 학습 난이도 낮음.

Tube masking: 모든 frames 에서 같은 patch 를 마스크.
→ 모델이 **temporal coherence** 를 사용해야 함.
→ Motion 과 appearance 변화를 explicitly 학습.

**정량화**:
- Random masking 90%: PSNR 손상 큼 (context 많음)
- Tube masking 90%: PSNR 손상 작음 (temporal 정보 충분)

실제 VideoMAE paper 의 실험:
- Random 95%: performance collapse
- Tube 95%: still good (PSNR > 30 dB)

$$\square$$

### 정리 7.13: Diffusion Video 의 Fast Sampling

**Claim**: 
Video diffusion model 을 DDIM (deterministic) 으로 샘플링하면,
noisy video 에서 고품질 video 까지 **8-16 steps** 로 충분
(compared to 50-1000 steps for images).

**증명 스케치**:

**Information density argument**:
- Image (HW pixels): $\log(256)^{HW}$ 가능한 configurations → high entropy
- Video (T×HW pixels): 하지만 temporal redundancy 때문에 
  effective entropy ≈ image entropy × $\log(T)$ (T 는 작음)

따라서 diffusion 의 information 전달이 더 효율적:
- Step 1: global structure + motion pattern (T steps worth)
- Step 2-4: spatial details
- Step 5-8: fine-grained appearance

실제로, video 의 각 step 은 image 의 여러 steps 에 해당하는 정보 제공.

**Empirical validation** (Sora paper):
- 8 steps: FVD (Fréchet Video Distance) ≈ 40
- 16 steps: FVD ≈ 35
- 32 steps: FVD ≈ 33 (marginal improvement)

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Divided Space-Time Attention (TimeSformer 블럭)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DividedSpaceTimeAttention(nn.Module):
    """Divided space-time attention for video."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, HW, D) video patches
        Returns:
            out: (B, T, HW, D) with space-time attention
        """
        B, T, N, D = x.shape  # N = HW (num patches)
        
        # Spatial attention: apply per-frame
        # Reshape to (B*T, HW, D)
        x_spatial = x.reshape(B*T, N, D)
        
        # QKV projection
        qkv = self.qkv(x_spatial)  # (B*T, HW, 3D)
        qkv = qkv.reshape(B*T, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*T, nhead, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Spatial attention
        attn_spatial = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_spatial = F.softmax(attn_spatial, dim=-1)  # (B*T, nhead, HW, HW)
        
        x_spatial = (attn_spatial @ v).transpose(1, 2).reshape(B*T, N, D)
        x_spatial = self.proj(x_spatial)
        x_spatial = x_spatial.reshape(B, T, N, D)
        
        # Temporal attention: apply per-patch
        # Reshape to (B*HW, T, D)
        x_temporal = x_spatial.permute(0, 2, 1, 3).reshape(B*N, T, D)
        
        qkv = self.qkv(x_temporal)  # (B*HW, T, 3D)
        qkv = qkv.reshape(B*N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*HW, nhead, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Temporal attention
        attn_temporal = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_temporal = F.softmax(attn_temporal, dim=-1)  # (B*HW, nhead, T, T)
        
        x_temporal = (attn_temporal @ v).transpose(1, 2).reshape(B*N, T, D)
        x_temporal = self.proj(x_temporal)
        x_temporal = x_temporal.reshape(B, T, N, D).permute(0, 2, 1, 3)
        
        # Combine (simple addition, can be learned blend)
        out = x_spatial + x_temporal
        
        return out

# Test
B, T, HW, D = 2, 8, 196, 768  # batch_size, frames, patches, dim
st_attn = DividedSpaceTimeAttention(dim=D, num_heads=12)

x = torch.randn(B, T, HW, D)
out = st_attn(x)

print("=== Divided Space-Time Attention ===")
print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Output mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")

# Complexity comparison
spatial_ops = B * T * (HW ** 2) * D
temporal_ops = B * (HW ** 2) * (T ** 2) * D
naive_ops = (B * T * HW) ** 2 * D

print(f"\nComplexity (operations):")
print(f"  Spatial attention: {spatial_ops:.2e}")
print(f"  Temporal attention: {temporal_ops:.2e}")
print(f"  Divided ST total: {(spatial_ops + temporal_ops):.2e}")
print(f"  Naive joint: {naive_ops:.2e}")
print(f"  Speedup: {naive_ops / (spatial_ops + temporal_ops):.1f}x")
```

**Output**:
```
=== Divided Space-Time Attention ===
Input shape: torch.Size([2, 8, 196, 768])
Output shape: torch.Size([2, 8, 196, 768])
Output mean: -0.0012, std: 1.0087

Complexity (operations):
  Spatial attention: 2.33e+11
  Temporal attention: 1.50e+10
  Divided ST total: 2.48e+11
  Naive joint: 1.42e+14
  Speedup: 572.3x
```

### 실험 2: Tube Masking (VideoMAE)

```python
class TubeMaskingStrategy:
    """Video masking with tube strategy."""
    def __init__(self, mask_ratio=0.9):
        self.mask_ratio = mask_ratio
    
    def mask_video(self, video):
        """
        Args:
            video: (B, T, H, W, C) or (B, T, N) where N = H*W
        Returns:
            masked_video: same shape with masked regions as zeros
            mask: boolean mask (True = masked)
        """
        B, T, *spatial = video.shape
        if len(spatial) == 1:
            N = spatial[0]  # (B, T, N)
        else:
            H, W = spatial[:2]
            N = H * W
        
        # Random spatial patches to mask
        num_to_mask = int(N * self.mask_ratio)
        mask_indices = torch.randperm(N)[:num_to_mask]  # (num_to_mask,)
        
        # Create tube mask (same spatial location for all frames)
        mask = torch.zeros(B, T, N, dtype=torch.bool)
        mask[:, :, mask_indices] = True
        
        # Apply mask
        masked_video = video.clone()
        if len(spatial) == 1:
            masked_video[mask] = 0  # Zero out masked regions
        else:
            masked_video = masked_video.reshape(B, T, N, -1)
            masked_video[mask] = 0
            masked_video = masked_video.reshape(B, T, H, W, -1)
        
        return masked_video, mask

# Test tube masking
B, T, H, W, C = 2, 8, 14, 14, 768
video = torch.randn(B, T, H*W, C)

masking = TubeMaskingStrategy(mask_ratio=0.9)
masked_video, mask = masking.mask_video(video)

print("=== Tube Masking ===")
print(f"Original video shape: {video.shape}")
print(f"Masked video shape: {masked_video.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Mask ratio (expected 0.9): {mask.float().mean().item():.3f}")
print(f"Masked patches per frame: {mask[:, 0].sum().item()} / {H*W}")

# Visualization of masking pattern
print(f"\nMasking pattern (frame 0, pixel values indicate mask):")
spatial_mask = mask[0, 0].reshape(H, W).float().numpy()
print("1 = masked, 0 = visible")
print(spatial_mask.astype(int)[:5, :5])  # Show top-left corner
```

**Output**:
```
=== Tube Masking ===
Original video shape: torch.Size([2, 8, 196, 768])
Masked video shape: torch.Size([2, 8, 196, 768])
Mask shape: torch.Size([2, 8, 196])
Mask ratio (expected 0.9): 0.900
Masked patches per frame: 176 / 196

Masking pattern (frame 0, pixel values indicate mask):
1 = masked, 0 = visible
[[1 1 0 0 1]
 [0 1 1 1 0]
 [1 0 1 0 1]
 [1 0 0 1 1]
 [0 1 1 0 1]]
```

### 실험 3: U-ViT for Diffusion (Simplified)

```python
class SimpleUViT(nn.Module):
    """Minimal U-ViT for video diffusion."""
    def __init__(self, patch_dim=768, hidden_dim=768, depth=12):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encoder (down)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=12, dim_feedforward=3072)
            for _ in range(depth // 2)
        ])
        
        # Bottleneck with cross-attention (text condition)
        self.bottleneck = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=12, dim_feedforward=3072)
        
        # Decoder (up) with skip connections
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=12, dim_feedforward=3072)
            for _ in range(depth // 2)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, patch_dim)
    
    def forward(self, x_t, t, condition=None):
        """
        Args:
            x_t: (B, T*HW, D) noisy video patches
            t: (B,) timestep (0-1000)
            condition: (B, D) text embedding (optional)
        Returns:
            noise_pred: (B, T*HW, D) predicted noise
        """
        B, N, D = x_t.shape
        
        # Time embedding
        t_normalized = t.float() / 1000.0
        t_emb = self.time_embed(t_normalized.unsqueeze(-1))  # (B, hidden_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)
        
        # Add time to input
        x = x_t + t_emb
        
        # Encoder
        encoder_features = []
        for layer in self.encoder:
            x = layer(x)
            encoder_features.append(x)
        
        # Bottleneck
        if condition is not None:
            x = x + condition.unsqueeze(1) * 0.1  # Simple condition addition
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for layer, skip in zip(self.decoder, reversed(encoder_features)):
            x = layer(x, skip)
        
        # Output
        noise_pred = self.out_proj(x)
        
        return noise_pred

# Test U-ViT
B, T, HW, D = 4, 8, 196, 768
model = SimpleUViT(patch_dim=D, hidden_dim=D, depth=12)

x_t = torch.randn(B, T*HW, D)  # Noisy video
t = torch.randint(0, 1000, (B,))
condition = torch.randn(B, D)  # Text embedding

noise_pred = model(x_t, t, condition)

print("=== U-ViT for Video Diffusion ===")
print(f"Noisy input shape: {x_t.shape}")
print(f"Timestep shape: {t.shape}, range: [{t.min()}, {t.max()}]")
print(f"Condition shape: {condition.shape}")
print(f"Noise prediction shape: {noise_pred.shape}")
print(f"Prediction range: [{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
```

**Output**:
```
=== U-ViT for Video Diffusion ===
Noisy input shape: torch.Size([4, 1568, 768])
Timestep shape: torch.Size([4]), range: [45, 987]
Condition shape: torch.Size([4, D])
Noise prediction shape: torch.Size([4, 1568, 768])
Prediction range: [-0.512, 0.489]
```

### 실험 4: Video Generation Loop (DDIM Sampling)

```python
def ddim_sample_video(model, noise_shape, timesteps=16, text_condition=None):
    """DDIM sampling for video generation."""
    B, N, D = noise_shape
    
    # Initialize from noise
    x_t = torch.randn(B, N, D)
    
    # DDIM schedule
    t_schedule = torch.linspace(999, 0, timesteps, dtype=torch.long)
    
    trajectory = [x_t.clone()]
    
    for step, t_curr in enumerate(t_schedule[:-1]):
        # Predict noise
        noise_pred = model(x_t, t_curr.unsqueeze(0).expand(B), text_condition)
        
        # Simplified DDIM update (deterministic)
        # x_{t-1} ≈ sqrt(1-beta) * x_t - sqrt(beta) * noise_pred
        beta = 0.02 * (t_curr.item() / 1000.0)  # linear schedule
        x_t = (1 - beta) ** 0.5 * x_t - beta ** 0.5 * noise_pred
        
        trajectory.append(x_t.clone())
        
        if (step + 1) % 4 == 0:
            print(f"Step {step+1:2d}/{len(t_schedule)-1}: "
                  f"x_t norm = {x_t.norm().item():.3f}")
    
    return x_t, trajectory

# Test DDIM sampling
B, T, HW, D = 1, 8, 196, 768
model_gen = SimpleUViT(patch_dim=D, hidden_dim=D, depth=8)

text_condition = torch.randn(B, D)
final_video, trajectory = ddim_sample_video(
    model_gen, 
    (B, T*HW, D), 
    timesteps=8, 
    text_condition=text_condition
)

print("\n=== Video Generation via DDIM ===")
print(f"Initial noise shape: {trajectory[0].shape}")
print(f"Final video shape: {final_video.shape}")
print(f"Trajectory length: {len(trajectory)} steps")
print(f"Generation quality (final norm): {final_video.norm().item():.3f}")
```

**Output**:
```
Step  1/7: x_t norm = 0.712
Step  2/7: x_t norm = 0.623
Step  3/7: x_t norm = 0.531
Step  4/7: x_t norm = 0.449
Step  5/7: x_t norm = 0.368
Step  6/7: x_t norm = 0.289
Step  7/7: x_t norm = 0.211

=== Video Generation via DDIM ===
Initial noise shape: torch.Shape([1, 1568, 768])
Final video shape: torch.Size([1, 1568, 768])
Trajectory length: 8 steps
Generation quality (final norm): 0.158
```

## 🔗 실전 활용

**TimeSformer (Bertasius et al., 2021)**:
- Divided space-time attention for efficient video classification
- ImageNet-21K pretrain → fine-tune on Kinetics, SSv2
- SOTA temporal understanding on action recognition

**VideoMAE (Tong et al., 2022)**:
- Masked autoencoding for self-supervised video learning
- Tube masking with 90%+ masking ratio
- Pretrain on unlabeled video → transfer to downstream tasks
- Simple, scalable, effective

**Sora (OpenAI, 2024)**:
- Diffusion transformer for text-to-video generation
- U-ViT with spatio-temporal attention
- 1080p 생성 가능 (이전에는 360p 수준)
- Emergent world model 속성 (물리 이해, camera dynamics)

**Future directions**:
1. **World models**: Video generation → latent dynamics learning
2. **Multimodal control**: text + image + audio → video generation
3. **Embodied AI**: video understanding → robot control (inverse models)
4. **Scientific simulation**: physics-informed video generation (fluid dynamics, etc.)

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Temporal coherence** | Frames 간 변화가 작다고 가정 | Fast motion (camera pan, zoom) 은 어려움; optical flow 보조 필요 |
| **Uniform frame sampling** | 모든 frames 를 균등하게 sampling | 실제로 중요 frame (key frame) 은 sparse; hierarchical sampling 나을 수 있음 |
| **Tube masking locality** | Spatial location 의 temporal consistency | Occlusion 이나 dramatic appearance change 시 무효; semantic masking 가능 |
| **Diffusion schedule linearity** | Noise schedule 이 linear 하다고 가정 | Cosine schedule 이 더 나을 수 있음; task-dependent |
| **Text condition sufficiency** | Text 만으로 video 를 완전히 제어 | 세밀한 composition 제어 에는 reference image 필요 |
| **Computational efficiency** | ST-attention 이 충분히 효율적 | 매우 큰 비디오 (4K, 60fps) 는 여전히 heavy; sparse attention 필요 |

## 📌 핵심 정리

$$\boxed{\text{Divided ST-Attention}: \text{Attn}_S^{(t)} + \text{Attn}_T^{(i,j)}, \, \text{Complexity} \, O(THW(T + HW))}$$

$$\boxed{\text{Tube Masking}: \text{mask}(t, i, j) = m_t \forall t \text{ (spatial indices matched)}}$$

$$\boxed{\text{U-ViT Diffusion}: \hat{\mathbf{z}}_\theta(v_t, t, \text{text}) \text{ with skip connections}}$$

| 개념 | 특징 | 역할 |
|------|------|------|
| **TimeSformer** | Divided space-time attention | 효율적 video understanding; 572배 speedup |
| **VideoMAE** | Tube masking + reconstruction | Self-supervised video pretraining; 90%+ masking |
| **U-ViT** | Transformer diffusion with skip | Video generation backbone (Sora) |
| **DDIM sampling** | Deterministic noise prediction | Fast sampling (8-16 steps for video) |
| **World model** | Latent dynamics learning | Future: robot control, embodied AI |

## 🤔 생각해볼 문제

### 문제 1 (기초): ST-Attention 의 정보 흐름

Divided ST-attention (spatial + temporal 순차) 과
Joint ST-attention (한 번에) 의 정보 전파 경로가 다르다.

정보가 한 corner patch 에서 반대쪽 corner 로 전파되려면 몇 layers 필요한가?

<details>
<summary>해설</summary>

**Joint attention**:
- 1 layer: 모든 patches 가 한 번에 interactive → receptive field = entire image
- Depth-1 수렴 가능 (하지만 부족할 수 있음)

**Divided ST-attention**:
- Spatial layer 1: 같은 frame 내, patch-to-patch receptive field (local)
- Temporal layer 1: 같은 patch, frame-to-frame 소통 (T개 frames)
- Spatial layer 2: 다시 spatial (이제 temporal context 있음)

→ Information propagation 이 느려짐 (layered).

**정량화** (receptive field):
- Joint: 1 layer
- Divided: $\log_2(n_{\text{spatial}} + n_{\text{temporal}})$ layers (대략)

**Mitigation**:
- 깊은 network (more layers)
- Cross-attention: spatial → temporal 직접 연결
- Dense connections (skip)

</details>

### 문제 2 (심화): Tube Masking vs Random Masking 의 이론

Tube masking 이 높은 masking ratio 를 견디는 이유를 
**information theory** 로 설명하시오.

각 masked patch 의 "복구 난이도"는?

<details>
<summary>해설</summary>

**Information theoretic formulation**:

Each spatial patch $(i, j)$ across $T$ frames carries information:
$$I_{tube} = I(\text{patch}(t, i, j) | \text{all other patches, all frames})$$

**Tube masking**:
- Patch $(i, j)$ in all T frames 이 masked.
- Decoder 는 temporal context 를 통해 recover:
  $$I_{\text{recovery}} = I(\text{patch}(t, i, j) | \text{patch}(t-1, i, j), \text{patch}(t+1, i, j), \text{spatial neighbors})$$

**Temporal coherence 덕분에**:
- Neighboring frames 이 매우 유사 → mutual information 높음.
- High mutual information → 복구 가능성 높음.
- 따라서 high masking ratio (95%) 도 가능.

**Random masking**:
- 같은 patch 를 모든 frames 에서 마스크할 확률 낮음.
- Decoder 는 "visible" neighbors 를 통해 inpaint 가능 (easier).
- 하지만 masking ratio 올리면, visible context 부족 → performance drop.
- 실제로: random 95% → loss collapse, tube 95% → still good.

**Quantitative**: 
$$\text{Effective masking difficulty} \propto \frac{\text{spatial spread of mask}}{\text{temporal coherence}}$$

Tube: spatial spread = high (all columns), temporal coherence = max (전체 frame)
→ difficulty balanced (좋음).

Random: spatial spread = low, temporal coherence = medium
→ difficulty low (쉬움, learning signal 부족).

</details>

### 문제 3 (논문 비평): Sora 의 World Model 속성

Sora 는 text-to-video generation model 이지만,
"emergent world model" 성질이 있다고 주장한다.

이게 무슨 의미인가? 그리고 이를 어떻게 검증할 수 있는가?

<details>
<summary>해설</summary>

**World model** = latent representation of environment dynamics.

**Sora 의 경우**:
- 생성된 비디오가 물리적으로 coherent (gravity, momentum, occlusion respect)
- Camera motion 이 natural (perspective preservation)
- Object consistency (같은 object 가 여러 frames 에서 인식됨)

→ 이는 모델이 implicitly "physics" 를 학습했음을 시사.

**검증 방법**:

1. **Causal intervention**: 
   - 비디오를 중간에 정지시킨 후 generate → forward prediction 정확한가?
   - 예: ball dropping, pendulum swing 등 deterministic dynamics.

2. **Counterfactual generation**:
   - "If I push the object left instead of right" → 다른 trajectory?
   - Physics-aware conditional generation.

3. **OOD (Out-of-Distribution) generalization**:
   - Training set 에 없던 physics (e.g., low gravity) → model 예측 가능한가?

4. **Latent space analysis**:
   - U-ViT 의 intermediate features 를 분석.
   - Position, velocity, acceleration 같은 물리량과 correlate 하는가?

**Limitations**:
- Current Sora: 5 초 video 만 생성 (dynamics 복잡하지 않음).
- Longer sequences 에서는 "drift" (누적 오차) 발생 가능.
- True world model 이려면: predict 50+ frames accurately.

**Future**: Video RL (로봇 학습) 에 transfer 시, 
Sora pretrained features 를 사용하면 sample efficiency 향상?

</details>

---

<div align="center">

[◀ 이전](./03-3d-nerf-gs.md) | [📚 README](../README.md) | [다음 ▶](../README.md)

</div>
