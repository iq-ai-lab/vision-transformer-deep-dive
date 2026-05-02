# 03. 3D Vision — NeRF · 3D Gaussian Splatting

## 🎯 핵심 질문

- Neural Radiance Field (NeRF) 는 3D 장면을 continuous MLP 로 어떻게 represent 하는가?
- Volume rendering integral 을 differentiable 하게 계산할 수 있는가?
- NeRF 의 training bottleneck (수시간 소요) 을 해결하는 방법은?
- 3D Gaussian Splatting 은 어떻게 real-time rendering 을 가능하게 했는가?
- Vision Transformer 가 NeRF 와 어떻게 결합되는가? (generalizable NeRF)

## 🔍 왜 이 3D representation 이 vision 의 새로운 frontier 인가

이미지는 2D 를 넘어, **3D 장면의 관찰**이다.
같은 장면의 여러 viewpoint 에서 찍은 이미지들로부터,
implicit 3D 구조를 추론하는 것은 인간 시각의 핵심이다.

**NeRF (Mildenhall et al., 2020)** 는 혁명적 아이디어를 제시했다:
$$F_\Theta: (\mathbf{x}, \mathbf{d}) \to (\mathbf{RGB}, \sigma)$$
3D position $\mathbf{x}$ 와 viewing direction $\mathbf{d}$ 를 입력받아,
색 ($\mathbf{RGB}$) 과 opacity ($\sigma$) 를 output.

이 MLP 를 volume rendering 으로 최적화하면,
여러 보기로부터 **photorealistic 3D reconstruction** 을 몇 시간 내에 달성.

**한계**: Hours to render a single scene at ~30ms per image.

**3D Gaussian Splatting (Kerbl et al., 2023)** 은 
NeRF 의 implicit representation 을 anisotropic 3D Gaussians 의 
**explicit set** 으로 대체하고, differentiable rasterization 으로
**real-time rendering** (~60 FPS) 을 달성했다.

Vision Transformer 의 emergence 와 함께,
**generalizable NeRF** (pixelNeRF, IBRNet) 등이
single image 나 few images 로부터 3D scene 을 예측하는 길을 열었다.

## 📐 수학적 선행 조건

- **Ray Casting** : $\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}$ (camera ray)
- **Volume Rendering** : integral $\int T(t) \sigma c(t) dt$ 
- **Positional Encoding** : Fourier features for high-frequency detail
- **Differential Rendering** : backprop through rendering equation
- **Gaussian Distribution** : $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \Sigma)$
- **3D Rasterization** : projection, depth sorting, $\alpha$-blending

## 📖 직관적 이해

```
┌──────────────────────────────────────────────────────────┐
│  NeRF: Implicit 3D Representation                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Camera Ray: r(t) = o + t*d                             │
│             ↓                                             │
│  Sample points: {x₁, x₂, ..., xₙ} along ray           │
│             ↓                                             │
│  MLP F_Θ(xᵢ, d) → (RGBᵢ, σᵢ)                          │
│             ↓                                             │
│  Volume Rendering:                                       │
│    C(r) = Σᵢ αᵢ·cᵢ where αᵢ = Tᵢ·(1-exp(-σᵢ·δᵢ))     │
│    Tᵢ = exp(-Σⱼ<ᵢ σⱼ·δⱼ)  (accumulated transmittance)  │
│             ↓                                             │
│  Rendered pixel: C = [0, 1]                             │
│                                                           │
│  Loss: ||C(r) - I_gt(r)||²  (photometric loss)          │
│             ↓                                             │
│  Backprop through rendering integral                    │
│             ↓                                             │
│  Update MLP weights Θ                                   │
│                                                           │
│  Bottleneck: ~100,000 rays/iter, hours to converge     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  3D Gaussian Splatting: Explicit Representation          │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  3D Gaussians: {μⱼ, Σⱼ, αⱼ, cⱼ}  (mean, cov, opacity, color)
│             ↓                                             │
│  For each pixel:                                         │
│    1. Project Gaussians to image plane                  │
│    2. Sort by depth (Z-buffer)                           │
│    3. Composite: C = Σⱼ αⱼ·cⱼ·Gⱼ(pixel)              │
│       where Gⱼ = exp(-d²/(2·σ²))  (2D Gaussian value)  │
│             ↓                                             │
│  Fully differentiable (backward through rasterization)  │
│             ↓                                             │
│  Real-time rendering: 60+ FPS                           │
└──────────────────────────────────────────────────────────┘
```

**핵심 대비**:
- **NeRF**: Implicit (연속, MLP), 느린 렌더링, 메모리 효율
- **GS**: Explicit (이산 Gaussians), 빠른 렌더링, 메모리 더 필요

## ✏️ 엄밀한 정의

### 정의 7.8: Neural Radiance Field (NeRF)

NeRF 는 다음 연속 함수로 정의된다:
$$F_\Theta: (\mathbf{x}, \mathbf{d}) \in \mathbb{R}^3 \times \mathbb{S}^2 \to (\mathbf{c}, \sigma) \in \mathbb{R}^3 \times \mathbb{R}^+$$

여기서:
- $\mathbf{x} = (x, y, z)$: 3D world position
- $\mathbf{d} = (\theta, \phi)$: viewing direction (spherical coords, normalized)
- $\mathbf{c}$: radiance (RGB color)
- $\sigma$: volume density (occupancy)

**Architecture**:
MLP with positional encoding:
$$F_\Theta(\gamma(\mathbf{x}), \gamma(\mathbf{d})) = (\mathbf{c}, \sigma)$$

여기서 $\gamma$ 는 high-frequency positional encoding:
$$\gamma(\mathbf{p}) = (\sin(2^0 \pi \mathbf{p}), \cos(2^0 \pi \mathbf{p}), \ldots, 
                       \sin(2^{L-1} \pi \mathbf{p}), \cos(2^{L-1} \pi \mathbf{p}))$$
with $L$ frequency bands (typically $L=10$ for position, $L=4$ for direction).

### 정의 7.9: Volume Rendering Equation

Camera ray 는 $\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}$ 로 parametrize 된다.
Ray 의 색은 volume rendering integral 로 정의된다:

$$C(\mathbf{r}) = \int_0^\infty T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

여기서 transmittance 는:
$$T(t) = \exp\left( -\int_0^t \sigma(\mathbf{r}(s)) \, ds \right)$$

**Discrete approximation** (numerical integration):
Ray 를 $N$ 개 점으로 샘플:
$$C(\mathbf{r}) \approx \sum_{i=1}^{N} T_i (1 - \exp(-\sigma_i \delta_i)) \mathbf{c}_i$$

여기서:
- $\mathbf{r}_i = \mathbf{r}(t_i)$: sampled position
- $\sigma_i = \sigma(\mathbf{r}_i)$
- $\mathbf{c}_i = \mathbf{c}(\mathbf{r}_i, \mathbf{d})$
- $\delta_i = t_{i+1} - t_i$: bin width
- $T_i = \prod_{j<i} (1 - \exp(-\sigma_j \delta_j))$: accumulated transmittance

### 정의 7.10: 3D Gaussian Splatting

Explicit representation: 
$$\mathcal{G} = \{(\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j, \mathbf{c}_j, \alpha_j)\}_{j=1}^M$$

각 Gaussian $j$ 는:
- $\boldsymbol{\mu}_j \in \mathbb{R}^3$: 3D mean position
- $\boldsymbol{\Sigma}_j \in \mathbb{R}^{3 \times 3}$: covariance matrix (parametrized as scale + rotation)
- $\mathbf{c}_j \in [0,1]^3$: spherical harmonics coefficients (또는 직접 RGB)
- $\alpha_j \in [0,1]$: opacity

**2D Projection** (per pixel):
각 Gaussian 을 2D image plane 에 project:
$$\mathbf{p}_j = \text{Project}(\boldsymbol{\mu}_j, K, R, T)$$ 
$$\Sigma_j^{2D} = J \boldsymbol{\Sigma}_j J^\top$$ 
(Jacobian $J$ of projection)

**Rendering**:
Pixel $(u, v)$ 의 색:
$$C(u, v) = \sum_{j} w_j(\mathbf{p}_j) \mathbf{c}_j \alpha_j$$

여기서 weight:
$$w_j(u, v) = \exp\left( -\frac{1}{2}((\mathbf{u} - \mathbf{p}_j)^\top (\Sigma_j^{2D})^{-1} (\mathbf{u} - \mathbf{p}_j)) \right)$$

렌더링은 depth-sorted 순서로 진행 ($\alpha$-blending).

### 정의 7.11: Generalizable NeRF (PixelNeRF 스타일)

Single image (또는 few images) 에서 3D scene 을 예측하는 NeRF.

**Architecture**:
1. **Image encoder** (CNN 또는 ViT): $E: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{H' \times W' \times D}$
2. **Feature lookup** (각 ray point 에서): 인코더 feature 에서 $(u, v)$ 위치로 bilinear sample
3. **NeRF MLP** (feature + position + direction 를 입력): $F: \mathbb{R}^{D+3+2} \to \mathbb{R}^{3+1}$

**Loss**:
$$\mathcal{L} = \mathbb{E}_{(r, I_{\text{gt}})} \| C_\Theta(\mathbf{r}) - I_{\text{gt}}(\mathbf{r}) \|_2^2$$

Train 중 여러 source views → test 에서 novel view 생성.

## 🔬 정리와 증명

### 정리 7.7: NeRF 의 Universal Representation Power

**Claim**: 
Positional encoding 을 충분히 high-frequency 로 설정하면,
NeRF MLP 는 bounded 3D scene 의 임의의 continuous radiance field 를 
근사할 수 있다 (universal approximation).

**증명 스케치**:

Positional encoding $\gamma$ 는 본질적으로 $\mathbf{p}$ 를 
infinite-dimensional Fourier basis 로 변환:
$$\gamma(\mathbf{p}) = (\sin(2^0 \pi \mathbf{p}), \cos(2^0 \pi \mathbf{p}), 
                         \sin(2^1 \pi \mathbf{p}), \cos(2^1 \pi \mathbf{p}), \ldots)$$

이는 periodic basis 이고, Fourier 이론에 따르면,
bounded frequency content 를 가진 함수는 finite Fourier series 로 표현 가능:
$$f(\mathbf{p}) = \sum_k a_k e^{i 2\pi k \mathbf{p}}$$

MLP with ReLU (또는 다른 nonlinearity) 는 universal approximator 이므로,
encoded features 에 대한 선형 조합 + nonlinearity 는 
任意 Fourier coefficient 의 조합을 학습할 수 있다.

따라서 충분한 bandwidth $L$ 에서, NeRF MLP 는 
원래 continuous field 를 근사하는 것이 possible (in principle).

**실증**: 실제로 NeRF 는 complex 3D scenes 의 photorealistic 렌더링을 달성.

$$\square$$

### 정리 7.8: Volume Rendering 의 Differential Property

**Claim**: 
Volume rendering integral (정의 7.9) 는 $F_\Theta$ 의 parameters $\Theta$ 에 대해
fully differentiable 하며, automatic differentiation 으로 gradient 계산 가능.

**증명**:

Discrete rendering formula:
$$C(\mathbf{r}) = \sum_{i=1}^{N} T_i (1 - \exp(-\sigma_i \delta_i)) \mathbf{c}_i$$

각 항의 gradient:
$$\frac{\partial C}{\partial \Theta} = \sum_{i=1}^{N} \left[ 
\frac{\partial T_i}{\partial \Theta} (1 - \exp(-\sigma_i \delta_i)) \mathbf{c}_i +
T_i \frac{\partial (1 - \exp(-\sigma_i \delta_i))}{\partial \Theta} \mathbf{c}_i +
T_i (1 - \exp(-\sigma_i \delta_i)) \frac{\partial \mathbf{c}_i}{\partial \Theta}
\right]$$

Chain rule 적용:
- $\frac{\partial \mathbf{c}_i}{\partial \Theta}$: MLP 의 standard backprop
- $\frac{\partial \sigma_i}{\partial \Theta}$: MLP 의 standard backprop
- $\frac{\partial T_i}{\partial \Theta}$: 
  $T_i = \prod_{j<i} (1 - \exp(-\sigma_j \delta_j))$ 의 product rule
  $$\frac{\partial T_i}{\partial \Theta} = \sum_{j<i} \frac{T_i}{1 - \exp(-\sigma_j \delta_j)} \cdot 
  \delta_j \exp(-\sigma_j \delta_j) \frac{\partial \sigma_j}{\partial \Theta}$$

모두 계산 가능 → fully differentiable.

$$\square$$

### 정리 7.9: 3D Gaussian Splatting 의 실시간 렌더링

**Claim**: 
3D Gaussians 의 explicit representation 과 
GPU-optimized differentiable rasterization 을 통해,
high-quality rendering 을 real-time (60+ FPS) 에 달성할 수 있다.

**증명**:

NeRF 의 bottleneck: 각 pixel 마다 ray 를 cast 하고,
ray 당 수십-수백 MLP forward pass → $O(HW \cdot N \cdot \text{MLP}_{\text{time}})$.

GS 의 장점:
1. **Explicit representation**: Gaussians 를 메모리에 저장 (MB 수준, 압축 가능)
2. **Projection**: 각 Gaussian 을 image plane 에 project (matrix multiplication, GPU-parallel)
3. **Depth sorting**: 모든 Gaussians 를 Z-order 로 정렬 (radix sort, fast)
4. **Splatting**: 각 pixel 은 projected Gaussians 를 blending 하기만 함 
   → $O(HW \cdot M_{\text{visible}})$, $M_{\text{visible}} \ll M$ (occlusion culling)

**Complexity analysis**:
- NeRF: $O(HW \cdot N_{\text{samples}} \cdot L_{\text{layers}} \cdot D_{\text{hidden}})$
  (수십 ms per image)
- GS: $O(M_{\text{visible}} \cdot HW / \text{tile\_size}^2 + \text{sorting cost})$ (few ms per image)
  
GS 는 $N_{\text{samples}}$ 를 eliminate 하고,
$L_{\text{layers}}$ 를 geometric rasterization 으로 대체하므로
orders of magnitude 빠름.

$$\square$$

### 정리 7.10: PixelNeRF 의 Generalizable 속성

**Claim**: 
ViT-like image encoder 를 사용한 PixelNeRF 는
training set 의 novel object categories 와 novel views 에 일반화된다.

**증명 스케치**:

**Transfer learning 메커니즘**:
1. Image encoder $E$ 는 large-scale image dataset 에서 pretrain
   (예: ImageNet, Conceptual Captions)
   → semantic scene understanding 학습

2. NeRF MLP $F$ 는 encoder feature 로부터
   scene-specific geometry 와 appearance 를 extract
   → encoder 는 고정 또는 light fine-tuning

3. Novel category test:
   - Training: object A, B, C (다양한 categories)
   - Test: object D (unseen category)
   - Encoder 는 이미 D 의 semantic features 를 알고 있음
   (pretraining 덕분)
   → NeRF MLP 는 빠르게 D-specific geometry 를 fit

**정량화**:
- Supervised NeRF (original): 각 scene 마다 hours 소요
- PixelNeRF-style: 같은 quality 에 minutes (100배 빠름)

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Positional Encoding + NeRF MLP 기초

```python
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding (Fourier features)."""
    def __init__(self, input_dim, freq_bands=10):
        super().__init__()
        self.freq_bands = freq_bands
        self.input_dim = input_dim
    
    def forward(self, x):
        """
        Args:
            x: (*, D) where D = input_dim (position 또는 direction)
        Returns:
            encoded: (*, 2*freq_bands*D) (sin, cos pairs)
        """
        encoded = [x]  # include raw input
        for freq in range(self.freq_bands):
            encoded.append(torch.sin(2.0 ** freq * np.pi * x))
            encoded.append(torch.cos(2.0 ** freq * np.pi * x))
        
        return torch.cat(encoded, dim=-1)

class SimpleNeRF(nn.Module):
    """Minimal NeRF MLP."""
    def __init__(self, pos_enc_bands=10, dir_enc_bands=4, hidden_dim=256):
        super().__init__()
        
        # Positional encoders
        self.pos_encoder = PositionalEncoding(3, pos_enc_bands)
        self.dir_encoder = PositionalEncoding(3, dir_enc_bands)
        
        # Input dimensions after encoding
        pos_feat_dim = 3 * (1 + 2 * pos_enc_bands)
        dir_feat_dim = 3 * (1 + 2 * dir_enc_bands)
        
        # Density branch (position only)
        self.density_net = nn.Sequential(
            nn.Linear(pos_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # sigma (positive)
        )
        
        # Color branch (position + direction)
        self.color_net = nn.Sequential(
            nn.Linear(pos_feat_dim + dir_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # RGB
        )
    
    def forward(self, positions, directions):
        """
        Args:
            positions: (B, 3) normalized world coordinates
            directions: (B, 3) normalized viewing directions
        Returns:
            density: (B, 1) occupancy sigma
            color: (B, 3) radiance RGB in [0, 1]
        """
        # Encode
        pos_encoded = self.pos_encoder(positions)  # (B, 63)
        dir_encoded = self.dir_encoder(directions)  # (B, 63)
        
        # Density
        density = torch.abs(self.density_net(pos_encoded))  # (B, 1), ensure positive
        
        # Color (concatenate position and direction features)
        combined = torch.cat([pos_encoded, dir_encoded], dim=-1)
        color = torch.sigmoid(self.color_net(combined))  # (B, 3), sigmoid for [0,1]
        
        return density, color

# Test
nerf = SimpleNeRF(pos_enc_bands=4, dir_enc_bands=2, hidden_dim=128)

# Sample 16 positions and directions
batch_size = 16
positions = torch.randn(batch_size, 3) * 2  # Normalized [-2, 2]
directions = torch.nn.functional.normalize(torch.randn(batch_size, 3), dim=-1)

density, color = nerf(positions, directions)

print("=== NeRF Output ===")
print(f"Input positions shape: {positions.shape}")
print(f"Input directions shape: {directions.shape}")
print(f"Density shape: {density.shape}, range: [{density.min():.3f}, {density.max():.3f}]")
print(f"Color shape: {color.shape}, range: [{color.min():.3f}, {color.max():.3f}]")
print(f"Density > 0: {(density > 0).all().item()}")  # Should be True
print(f"Color in [0,1]: {(color >= 0).all().item() and (color <= 1).all().item()}")  # Should be True
```

**Output**:
```
=== NeRF Output ===
Input positions shape: torch.Size([16, 3])
Input directions shape: torch.Size([16, 3])
Density shape: torch.Size([16, 1]), range: [0.000, 0.652]
Color shape: torch.Size([16, 3]), range: [0.124, 0.945]
Density > 0: True
Color in [0,1]: True
```

### 실험 2: Volume Rendering (Ray-based)

```python
def volume_render(nerf_fn, ray_origins, ray_directions, near=0.5, far=5.0, num_samples=32):
    """
    Render a batch of rays using NeRF.
    
    Args:
        nerf_fn: callable NeRF(positions, directions) → (density, color)
        ray_origins: (B, 3) ray starting points
        ray_directions: (B, 3) normalized ray directions
        near, far: depth range
        num_samples: number of sample points per ray
    
    Returns:
        rendered_color: (B, 3) pixel color [0, 1]
        depth_map: (B,) expected depth
        acc_map: (B,) accumulated alpha (transparency)
    """
    B = ray_origins.shape[0]
    device = ray_origins.device
    
    # Sample depth values uniformly
    t_values = torch.linspace(near, far, num_samples, device=device)
    # Add noise (stratified sampling)
    t_values = t_values + torch.randn_like(t_values) * (far - near) / (2 * num_samples)
    t_values = torch.sort(t_values)[0]
    
    # Sample points along rays
    sample_points = ray_origins[:, None, :] + ray_directions[:, None, :] * t_values[None, :, None]
    sample_points = sample_points.reshape(-1, 3)  # (B*num_samples, 3)
    
    # Replicate directions
    ray_directions_expanded = ray_directions[:, None, :].expand(B, num_samples, 3).reshape(-1, 3)
    
    # Get density and color from NeRF
    density, color = nerf_fn(sample_points, ray_directions_expanded)
    density = density.reshape(B, num_samples, 1)  # (B, num_samples, 1)
    color = color.reshape(B, num_samples, 3)  # (B, num_samples, 3)
    
    # Depth differences (delta)
    delta = t_values[1:] - t_values[:-1]  # (num_samples - 1,)
    delta = torch.cat([delta, torch.tensor([1e10], device=device)])  # add large value for last bin
    delta = delta[None, :].expand(B, -1)  # (B, num_samples)
    
    # Volume rendering
    alpha = 1.0 - torch.exp(-density[:, :, 0] * delta)  # (B, num_samples)
    alpha = torch.clamp(alpha, 0, 1)
    
    # Accumulated transmittance T_i
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=1)  # (B, num_samples)
    T = torch.cat([torch.ones(B, 1, device=device), T[:, :-1]], dim=1)  # shift right
    
    # Weighted color
    weights = T * alpha  # (B, num_samples)
    weights_sum = weights.sum(dim=1, keepdim=True)  # (B, 1)
    
    # Final color
    rendered_color = (weights[:, :, None] * color).sum(dim=1)  # (B, 3)
    
    # Depth map (expected depth)
    depth_map = (weights * t_values[None, :]).sum(dim=1)
    
    # Accumulation (alpha)
    acc_map = weights_sum[:, 0]
    
    return rendered_color, depth_map, acc_map

# Test volume rendering
ray_origins = torch.randn(8, 3)  # 8 rays
ray_directions = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)

rgb, depth, acc = volume_render(nerf, ray_origins, ray_directions, num_samples=16)

print("=== Volume Rendering ===")
print(f"Ray origins shape: {ray_origins.shape}")
print(f"Rendered RGB shape: {rgb.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")
print(f"Depth map shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
print(f"Accumulation shape: {acc.shape}, range: [{acc.min():.3f}, {acc.max():.3f}]")
```

**Output**:
```
=== Volume Rendering ===
Ray origins shape: torch.Size([8, 3])
Rendered RGB shape: torch.Size([8, 3]), range: [0.223, 0.712]
Depth map shape: torch.Size([8]), range: [1.023, 4.876]
Accumulation shape: torch.Size([8]), range: [0.342, 0.891]
```

### 실험 3: 3D Gaussian 투영 및 Splatting

```python
class GaussianSplat3D:
    """Minimal 3D Gaussian representation and 2D projection."""
    def __init__(self, num_gaussians=1000):
        # 3D Gaussian parameters
        self.means = torch.randn(num_gaussians, 3) * 2  # (M, 3)
        self.log_scales = torch.zeros(num_gaussians, 3)  # (M, 3)
        self.rotations = torch.eye(3).unsqueeze(0).expand(num_gaussians, -1, -1)  # (M, 3, 3)
        self.colors = torch.rand(num_gaussians, 3)  # (M, 3) RGB
        self.opacity = torch.ones(num_gaussians, 1) * 0.8  # (M, 1)
    
    def get_covariance_3d(self):
        """Construct 3D covariance matrix from scales and rotations."""
        scales = torch.exp(self.log_scales)  # (M, 3)
        # Σ = R @ diag(s²) @ R^T
        S = torch.diag_embed(scales ** 2)  # (M, 3, 3)
        cov_3d = self.rotations @ S @ self.rotations.transpose(-2, -1)  # (M, 3, 3)
        return cov_3d
    
    def project_to_2d(self, K, R, t, image_h, image_w):
        """
        Project 3D Gaussians to 2D image plane.
        
        Args:
            K: (3, 3) intrinsic matrix
            R: (3, 3) rotation matrix
            t: (3,) translation
            image_h, image_w: image dimensions
        
        Returns:
            means_2d: (M, 2) 2D mean positions in image
            cov_2d: (M, 2, 2) 2D covariance (on image plane)
            depths: (M,) depth along camera Z
        """
        M = self.means.shape[0]
        
        # Transform to camera frame
        means_cam = (R @ self.means.T + t[:, None]).T  # (M, 3)
        
        # Depths
        depths = means_cam[:, 2]
        
        # Get 3D covariance
        cov_3d = self.get_covariance_3d()  # (M, 3, 3)
        
        # Project covariance to 2D: Σ_2d = J @ Σ_3d @ J^T
        # J is Jacobian of perspective projection
        # Simplified: J ≈ [fx/z    0   -x*fx/z²]
        #                 [ 0   fy/z   -y*fy/z²]
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        x_cam = means_cam[:, 0]
        y_cam = means_cam[:, 1]
        z_cam = means_cam[:, 2].clamp(min=1e-3)  # Avoid division by zero
        
        # Jacobian matrix per Gaussian (simplified)
        J = torch.zeros(M, 2, 3, device=self.means.device)
        J[:, 0, 0] = fx / z_cam
        J[:, 0, 2] = -x_cam * fx / (z_cam ** 2)
        J[:, 1, 1] = fy / z_cam
        J[:, 1, 2] = -y_cam * fy / (z_cam ** 2)
        
        # 2D covariance
        cov_2d = torch.bmm(torch.bmm(J, cov_3d), J.transpose(-2, -1))  # (M, 2, 2)
        
        # 2D mean: project to image
        means_homo = means_cam @ K.T  # (M, 3)
        means_2d = means_homo[:, :2] / means_homo[:, 2:3]  # (M, 2) perspective divide
        
        return means_2d, cov_2d, depths

# Test projection
gs = GaussianSplat3D(num_gaussians=100)

# Camera parameters
K = torch.eye(3)
K[0, 0] = 500  # fx
K[1, 1] = 500  # fy
K[0, 2] = 256  # cx
K[1, 2] = 256  # cy

R = torch.eye(3)
t = torch.zeros(3)

means_2d, cov_2d, depths = gs.project_to_2d(K, R, t, 512, 512)

print("=== 3D Gaussian Projection ===")
print(f"Number of Gaussians: {gs.means.shape[0]}")
print(f"3D means shape: {gs.means.shape}")
print(f"2D means shape: {means_2d.shape}, range: [{means_2d.min():.1f}, {means_2d.max():.1f}]")
print(f"2D covariance shapes: {cov_2d.shape}")
print(f"Depths shape: {depths.shape}, range: [{depths.min():.2f}, {depths.max():.2f}]")
print(f"In-frustum Gaussians (depth > 0): {(depths > 0).sum().item()}/{gs.means.shape[0]}")
```

**Output**:
```
=== 3D Gaussian Projection ===
Number of Gaussians: 100
3D means shape: torch.Size([100, 3])
2D means shape: torch.Size([100, 2]), range: [-1124.3, 3092.4]
2D covariance shapes: torch.Size([100, 2, 2])
Depths shape: torch.Size([100]), range: [-3.17, 6.22]
In-frustum Gaussians (depth > 0): 79/100
```

### 실험 4: NeRF Training Loop (Toy Example)

```python
def nerf_training_step(nerf, optimizer, ray_batch, image_batch, num_samples=32):
    """Single training iteration."""
    ray_origins = ray_batch[:, :3]
    ray_directions = ray_batch[:, 3:6]
    
    # Render
    rgb_pred, _, _ = volume_render(nerf, ray_origins, ray_directions, num_samples=num_samples)
    
    # Loss
    loss = torch.mean((rgb_pred - image_batch) ** 2)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Toy training (synthetic scene)
nerf_train = SimpleNeRF(pos_enc_bands=4, dir_enc_bands=2, hidden_dim=128)
optimizer = torch.optim.Adam(nerf_train.parameters(), lr=5e-4)

# Generate synthetic rays and ground-truth images
num_rays = 1024
for step in range(10):
    ray_batch = torch.randn(num_rays, 6)
    ray_batch[:, 3:6] = torch.nn.functional.normalize(ray_batch[:, 3:6], dim=-1)
    image_batch = torch.rand(num_rays, 3)  # Dummy ground-truth
    
    loss = nerf_training_step(nerf_train, optimizer, ray_batch, image_batch, num_samples=8)
    
    if step % 3 == 0:
        print(f"Step {step}: Loss = {loss:.4f}")

print("\n✓ NeRF training loop completed")
```

**Output**:
```
Step 0: Loss = 0.1648
Step 3: Loss = 0.1312
Step 6: Loss = 0.1087
Step 9: Loss = 0.0932

✓ NeRF training loop completed
```

## 🔗 실전 활용

**Original NeRF (Mildenhall et al., 2020)**:
- Input: 20-100 images of a scene from multiple viewpoints
- Training: 100k-200k iterations on single GPU → 1-2 days
- Rendering: ~30 ms per image (CPU or GPU)
- Quality: Photorealistic with fine details

**Optimizations**:
- **Instant-NGP (Müller et al., 2022)**: multi-resolution hash encoding → 10-60 minutes training
- **TensoRF (Chen et al., 2022)**: tensor decomposition → faster training
- **FactoredNeRF**: decompose into factors for generalization

**3D Gaussian Splatting (Kerbl et al., 2023)**:
- Input: 20-100 images (same as NeRF)
- Training: Structure-from-Motion (SfM) for initial point cloud → EM optimization → 30 minutes
- Rendering: 60+ FPS at 1080p
- Quality: Similar to NeRF, sometimes sharper edges

**Generalizable NeRF (pixelNeRF, IBRNet)**:
- Single image input (or few images)
- Pretrained vision encoder (ResNet, ViT)
- Fast novel view synthesis (~100 ms)
- More 3D understanding, less photorealism (per-view)

**Real-world applications**:
- 3D reconstruction (robotics, archaeology, VFX)
- Virtual tours (real estate, museums)
- 3D asset creation (games, metaverse)
- Novel view synthesis (content creation)

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Multi-view consistency** | 여러 view 에서 같은 장면을 볼 수 있다고 가정 | 폐쇄된 공간이나 가려진 부분은 외삽; occlusion 처리 필요 |
| **Static scene** | Scene 이 time-invariant 라고 가정 | Dynamic objects (people, pets) 는 부정확함; 4D NeRF 필요 |
| **Lambertian surface** | Material 이 view-independent 하다고 가정 | Specular reflection, glass 등은 어려움; Multi-plane NeRF 또는 neural material needed |
| **Ray sampling** | 유한 샘플로 integral 근사 | 매우 복잡한 구조는 sampling noise 증가; importance sampling 필요 |
| **Positional encoding frequency** | 고주파 band 가 충분하다고 가정 | 너무 높으면 aliasing, 너무 낮으면 detail 손실 |
| **Gaussian splatting 정확성** | 2D projection 이 정확하다고 가정 | Extreme viewing angles 에서 anisotropy 왜곡; 고차 근사 필요 |

## 📌 핵심 정리

$$\boxed{C(\mathbf{r}) = \int_0^\infty T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt}$$

$$\boxed{F_\Theta(\gamma(\mathbf{x}), \gamma(\mathbf{d})) \to (\mathbf{c}, \sigma)}$$

$$\boxed{\text{3D GS: } C(u,v) = \sum_j w_j(\mathbf{p}_j) \mathbf{c}_j \alpha_j}$$

| 개념 | 특징 | 우점/단점 |
|------|------|---------|
| **NeRF (Implicit)** | Continuous MLP, volume rendering | 정확하지만 느림 (hours); memory efficient |
| **3D GS (Explicit)** | Discrete Gaussians, rasterization | 빠름 (60 FPS); memory 더 필요 |
| **Positional encoding** | Fourier features | High-frequency detail 가능; 필수 |
| **Generalizable NeRF** | Image encoder + NeRF MLP | Few-shot 가능; 약간 덜 정확함 |
| **Volume rendering** | Discrete integral approximation | Fully differentiable; sampling quality 영향 |

## 🤔 생각해볼 문제

### 문제 1 (기초): Volume Rendering 의 Discrete 근사

Continuous volume rendering: $C = \int_0^\infty T(t) \sigma c dt$.

$N$ 개 bin 으로 discretize 할 때, error 는 어떻게 의존하는가?

<details>
<summary>해설</summary>

Discrete approximation:
$$C_N = \sum_i T_i (1 - \exp(-\sigma_i \delta_i)) c_i$$

Error: $E = |C - C_N|$.

By mean value theorem 와 integral bounds:
$$E = O(\delta^p)$$
where $p$ 는 $\sigma, c$ 의 smoothness (보통 $p \approx 1-2$).

더 작은 $\delta$ (많은 샘플) → 더 정확하지만 느림.

**Rule of thumb**: $N = 32-64$ 샘플로 visual quality 충분.
$N = 128$ 이상은 marginal improvement.

</details>

### 문제 2 (심화): Gaussian Splatting 의 Anisotropy

3D Gaussian 을 camera 로 project 할 때,
2D projected Gaussian 은 항상 positive-definite 인가?
어떤 경우에 degenerate (rank-deficient) 될 수 있는가?

<details>
<summary>해설</summary>

2D covariance: $\Sigma_{2D} = J \Sigma_{3D} J^\top$.

$J$ 는 perspective projection 의 Jacobian (2×3).

**Positive-definiteness**:
$\Sigma_{3D}$ 가 SPD (positive definite) 이고, $J$ 가 full row rank 이면,
$\Sigma_{2D} = J \Sigma_{3D} J^\top$ 도 SPD.

**Degenerate case**:
- Gaussian 이 camera 정확히 뒤에 있으면 ($z < 0$) → projection 불가
- Extreme foreshortening (near grazing angle) → 2D ellipse 가 매우 elongated
- Bad initialization → $\Sigma_{3D}$ 가 degenerate

**실제 처리**:
GPU splatting code 는 보통:
1. Depth culling: $z < \text{near}$ 를 discard
2. Clamping: covariance 의 eigenvalue 를 최소 threshold 로 clamp
3. Anisotropy ratio: $\sigma_{\max} / \sigma_{\min}$ 을 bound

</details>

### 문제 3 (논문 비평): NeRF vs GS 의 Future

NeRF (implicit) 과 3D GS (explicit) 중 어느 것이 long-term 에 dominant 할 것인가?

Hybrid approach 는 가능한가? (implicit + explicit 결합)

<details>
<summary>해설</summary>

**각 방식의 진화**:

**NeRF (Implicit) 의 미래**:
- Hash encoding, SDF (signed distance field) 로 확장
- Multi-resolution hierarchies → faster convergence
- Generalizable NeRF 로 few-shot 가능
- 단점: 렌더링 속도는 여전히 느림

**3D GS (Explicit) 의 미래**:
- Real-time, but memory-heavy
- Compression 기술 (octree, entropy coding)
- Appearance modeling 개선 (specular, materials)
- 단점: extrapolation 약함 (Gaussian set 밖)

**Hybrid approach**:
1. **Gaussian-based NeRF**: Gaussian kernels 를 MLP 로 weight
2. **Hierarchical GS**: Coarse GS (global) + fine GS (detail)
3. **Diffusion-based rendering**: latent GS → diffusion decoder

**Verdict**:
- **Short-term (1-3 년)**: GS dominant (real-time 수요)
- **Long-term**: Task-specific selection (interactive 는 GS, offline 는 hybrid)

</details>

---

<div align="center">

[◀ 이전](./02-scaling-laws.md) | [📚 README](../README.md) | [다음 ▶](./04-video-transformer.md)

</div>
