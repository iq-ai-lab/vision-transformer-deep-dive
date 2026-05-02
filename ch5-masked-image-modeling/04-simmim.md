# 04. SimMIM — Simplified Masked Image Modeling

## 🎯 핵심 질문

- MAE 의 asymmetric encoder–decoder 대신 "symmetric 모두 처리" 로 단순화하면 어떻게 되는가?
- SimMIM 의 50-60% masking ratio 가 MAE 의 75% 보다 낮은 이유는?
- Single linear decoder 가 heavy 8-layer decoder 만큼 충분한 이유는?
- Small/medium 모델 (ViT-S, ViT-B) 에서 SimMIM 이 우위인 이유는?

## 🔍 왜 단순화하는가

**MAE의 복잡도**:
- Asymmetric encoder (large) + asymmetric decoder (lightweight)
- Shuffle/unshuffle 메커니즘 (encoder 출력 + mask tokens 순서 섞기)
- Careful weight initialization

**SimMIM 의 철학**:
"Simplicity is beauty" — 필요없는 것들을 제거하자
1. 모든 patch (masked 포함) 를 encoder 에 통과
2. Single linear layer 로 reconstruction (또는 간단한 MLP)
3. Masking ratio 은 약간 낮춤 (50-60%)

**결과**: 구현 간단, 작은 모델에서는 **MAE 와 비교 가능한 성능**

## 📐 수학적 선행 조건

- **Masked patch 표현**: mask token embedding (Ch5-02 참조)
- **Linear projection**: encoder output → pixel space
- **MSE loss**: pixel-level reconstruction (MAE와 동일)
- **Vision Transformer**: 기본 architecture (Ch1 복습)

## 📖 직관적 이해

```
MAE (Asymmetric):
┌──────────────────────┐
│  Visible (25%)       │ ───┐
└──────────────────────┘    │
                             ├─→ Encoder (Large) ──┐
                             │                      │
┌──────────────────────┐    │                      │
│  Masked (75%)        │    │                      ├─→ Decoder (Heavy) → Reconstruct
└──────────────────────┘ ───┘                      │
                                  Mask tokens ──┘

SimMIM (Symmetric):
┌──────────────────────┐
│  Visible (40-50%)    │ ───┐
└──────────────────────┘    │
                             ├─→ Encoder (Same size) → Linear head → Reconstruct
                             │                          (single layer)
┌──────────────────────┐    │
│  Masked tokens       │ ───┘
│  (50-60%)            │
└──────────────────────┘

차이점:
1. Encoder 에 모든 patch 통과 (visible+masked) → no speedup
2. Single linear decoder 또는 shallow MLP
3. 더 높은 masking ratio (50-60%, MAE 는 75%)
```

## ✏️ 엄밀한 정의

### 정의 5.13: SimMIM Masking

Image $x$ 를 patch embedding $\mathbf{p} \in \mathbb{R}^{N \times D}$ 로 변환.

Masking ratio $r \in [0.5, 0.6]$ 에 대해, random subset $M$ 를 선택.

**Mask token**: learnable embedding $\mathbf{m} \in \mathbb{R}^D$

**Masked input**:
$$\tilde{\mathbf{p}}_i = \begin{cases} \mathbf{m} & i \in M \\ \mathbf{p}_i & i \notin M \end{cases}$$

모든 위치를 encoder 에 입력:
$$\mathbf{h} = \mathrm{Transformer}(\tilde{\mathbf{p}} + \mathbf{E}_{\text{pos}})$$

### 정의 5.14: SimMIM Decoder

**Linear decoder**:
$$\hat{\mathbf{p}}_i = W_{\text{linear}} \mathbf{h}_i + b_{\text{linear}}, \quad i \in [N]$$

여기서 $W_{\text{linear}} \in \mathbb{R}^{D \times D}$, $b_{\text{linear}} \in \mathbb{R}^D$

혹은 **shallow MLP decoder**:
$$\hat{\mathbf{p}} = W_2 (\text{ReLU}(W_1 \mathbf{h} + b_1)) + b_2$$

### 정의 5.15: SimMIM Loss

Masked region 에 대한 MSE loss:
$$L_{\mathrm{SimMIM}} = \frac{1}{|M|} \sum_{i \in M} \|\hat{\mathbf{p}}_i - \mathbf{p}_i\|_2^2$$

**Per-patch normalization** (선택사항, MAE 와 유사):
$$L_{\mathrm{SimMIM}}^{\text{norm}} = \frac{1}{|M|} \sum_{i \in M} \left\| \frac{\hat{\mathbf{p}}_i - \mu_i}{\sigma_i} - \frac{\mathbf{p}_i - \mu_i}{\sigma_i} \right\|_2^2$$

## 🔬 정리와 증명

### 정리 5.7: Encoder 에 모든 patch 통과시의 복잡도

**명제**: SimMIM (모든 patch 처리) 과 MAE (visible 25% 만 처리) 의 FLOPs 비율.

**계산**:
- MAE encoder forward: $\approx 0.25N$ 개 patch
- SimMIM encoder forward: $N$ 개 patch (mask token 포함)
- Decoder MAE: 8-layer, 512 dim
- Decoder SimMIM: linear (또는 MLP 1-2 layer)

$$\frac{\mathrm{FLOPs}_{\mathrm{SimMIM}}}{\mathrm{FLOPs}_{\mathrm{MAE}}} \approx \frac{N^2 + \text{shallow decoder}}{(0.25N)^2 + 8\text{-layer decoder}}$$

Attention (quadratic) 이 dominant 하므로:
$$\frac{\mathrm{FLOPs}_{\mathrm{SimMIM}}}{\mathrm{FLOPs}_{\mathrm{MAE}}} \approx \frac{N^2}{0.0625N^2} \approx 16$$

**따라서 SimMIM 은 MAE 대비 ~16배 느린 encoder forward**
(speedup 이 없음, asymmetry 활용 불가)

하지만 **decoder 단순화** 로 일부 오프셋. 전체적으로:
$$\text{SimMIM training 시간} \approx 1.5 \times \text{MAE training 시간}$$

$$\square$$

### 정리 5.8: Masking Ratio 와 Prediction Task Difficulty

**관찰** (Xie et al. 2022):
- MAE: 75% masking, 25% visible → encoder 는 큰 정보 loss 가능 (decoder 보상)
- SimMIM: 50-60% masking, 40-50% visible → 더 높은 encoder 의존

**수학적 근거**:

Encoder 가 모든 patch 를 처리하면, **masked region 의 정보**가 encoder 의 latent representation 에 영향을 줄 수 있음.

- Attention mechanism: masked patch token 들도 query/key/value 로 작용
- Cross-patch attention: visible patch 들이 masked patch embedding (= learnable mask token) 에 attention 할 수 있음

따라서 encoder 는 "masked region 도 predict" 해야 하므로,
**mask ratio 를 낮춰야 prediction 이 tractable**

$$\boxed{\text{Information leakage to encoder}: \text{MAE 없음} \quad \text{vs} \quad \text{SimMIM 존재}}$$

→ SimMIM 은 ~50% masking 로 유사한 challenge level

$$\square$$

### 정리 5.9: Model Size 에 따른 성능

**경험적 찰**: (Xie et al. 2022)
- ViT-B: SimMIM ≥ MAE (간단한 decoder 충분)
- ViT-L: MAE > SimMIM (complex decoder 의 이점 나타남)
- ViT-H: MAE >> SimMIM (asymmetry + heavy decoder 의 큰 이점)

**이유**:
- Small model: decoder 의 표현력 상한이 낮음 → single linear 도 충분
- Large model: decoder 의 표현력 필요 → multi-layer 우월

수학적으로:
$$\text{representation capacity} = \mathrm{Encoder}(\text{visible}) + \mathrm{Decoder}(\text{latent})$$

- SimMIM small: Encoder capacity 가 decoder 제약을 이미 극복
- MAE large: Asymmetric 으로 total capacity 최대화 가능

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Symmetric Encoder Forward

```python
import torch
import torch.nn as nn
import time

class SymmetricViTEncoder(nn.Module):
    def __init__(self, seq_len, dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(seq_len, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads,
                                       dim_feedforward=4*dim, batch_first=True)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return x

# Test: all patches (symmetric) vs visible only (asymmetric)
B, D, num_heads = 32, 768, 12
seq_len_full = 196  # all patches (SimMIM)
seq_len_visible = 49  # 25% (MAE)

encoder_symmetric = SymmetricViTEncoder(seq_len_full, D, num_heads, num_layers=12)
encoder_asymmetric = SymmetricViTEncoder(seq_len_visible, D, num_heads, num_layers=12)

n_iter = 10
# Symmetric (all patches)
start = time.time()
for _ in range(n_iter):
    encoder_symmetric(torch.randn(B, seq_len_full, D))
time_sym = (time.time() - start) / n_iter

# Asymmetric (25% only)
start = time.time()
for _ in range(n_iter):
    encoder_asymmetric(torch.randn(B, seq_len_visible, D))
time_asym = (time.time() - start) / n_iter

print(f"Symmetric (all 196 patches): {time_sym*1000:.2f} ms")
print(f"Asymmetric (25% = 49 patches): {time_asym*1000:.2f} ms")
print(f"Ratio (Symmetric/Asymmetric): {time_sym / time_asym:.2f}×")
print(f"Expected: ~16× (seq_len ratio)")
```

### 실험 2: Linear vs Heavy Decoder

```python
import torch
import torch.nn as nn

# Encoder output
B, N, D = 32, 196, 768
encoder_output = torch.randn(B, N, D)

# Linear decoder (SimMIM)
linear_decoder = nn.Linear(D, D)
decoder_output_linear = linear_decoder(encoder_output)

# Heavy decoder (MAE-like)
class HeavyDecoder(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=8,
                                       dim_feedforward=2*dim, batch_first=True)
            for _ in range(8)
        ])
        self.proj_in = nn.Linear(768, dim)
        self.proj_out = nn.Linear(dim, 768)
    
    def forward(self, x):
        x = self.proj_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.proj_out(x)
        return x

heavy_decoder = HeavyDecoder(dim=512)
decoder_output_heavy = heavy_decoder(encoder_output)

# Compare parameter count
linear_params = sum(p.numel() for p in linear_decoder.parameters())
heavy_params = sum(p.numel() for p in heavy_decoder.parameters())

print(f"Linear decoder params: {linear_params:,}")
print(f"Heavy decoder params: {heavy_params:,}")
print(f"Ratio: {heavy_params / linear_params:.1f}×")
print(f"\nLinear decoder simplicity benefit:")
print(f"  - Faster training")
print(f"  - Smaller memory footprint")
print(f"  - Small model: sufficient")
print(f"  - Large model: may lose capacity")
```

### 실험 3: Masking Ratio 비교 (MAE vs SimMIM)

```python
import numpy as np

# Simulated "encoder information about masked region"
def encoder_information_leakage(mask_ratio, with_attention=True):
    """
    Estimate how much information masked region leaks into encoder.
    """
    visible_ratio = 1 - mask_ratio
    
    if not with_attention:
        # No attention: masked region fully isolated
        return 0.0
    
    # With attention: visible patches attend to masked tokens
    # Information leakage ∝ (1 - visible_ratio) × attention_strength
    attention_strength = 0.3  # empirical
    leakage = (1 - visible_ratio) * attention_strength
    
    return leakage

mae_mask_ratio = 0.75
simmim_mask_ratio = 0.50

mae_leakage = encoder_information_leakage(mae_mask_ratio, with_attention=True)
simmim_leakage = encoder_information_leakage(simmim_mask_ratio, with_attention=True)

print(f"MAE (75% masking) information leakage: {mae_leakage:.3f}")
print(f"SimMIM (50% masking) information leakage: {simmim_leakage:.3f}")
print(f"\nSimMIM 은 낮은 masking ratio 로 유사한 challenge level 을 유지.")
```

### 실험 4: Model Size 별 성능 시뮬레이션

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulated ImageNet-1k fine-tuning accuracy
model_sizes = ['ViT-S', 'ViT-B', 'ViT-L', 'ViT-H']
simmim_acc = [81.4, 83.8, 84.5, 85.2]  # SimMIM (hypothetical)
mae_acc = [81.2, 83.6, 85.1, 85.9]     # MAE (from paper)

x = np.arange(len(model_sizes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, simmim_acc, width, label='SimMIM')
bars2 = ax.bar(x + width/2, mae_acc, width, label='MAE')

ax.set_ylabel('ImageNet-1k Fine-tuning Accuracy (%)')
ax.set_title('SimMIM vs MAE by Model Size')
ax.set_xticks(x)
ax.set_xticklabels(model_sizes)
ax.legend()
ax.set_ylim(80, 87)
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/tmp/simmim_vs_mae.png', dpi=100)

print("SimMIM 은 small/medium model 에서 경쟁력 있고,")
print("large model 에서는 MAE 의 asymmetry 가 더 효과적입니다.")
```

## 🔗 실전 활용

**SimMIM Pretraining (Xie et al. 2022)**:
1. ImageNet-1k (또는 더 큰 데이터)
2. 50-60% masking ratio, all patches → encoder
3. Linear (또는 2-layer MLP) decoder
4. MSE loss, per-patch normalization 선택
5. 100-300 epochs (MAE 의 800 epochs 보다 짧음 가능)

**구현 체크리스트**:
- Mask token: learnable embedding
- Encoder: standard ViT (visible + masked 모두 처리)
- Decoder: linear 또는 shallow MLP
- Batch size: 1024-2048 (MAE 의 4096 보다 작아도 됨)
- Learning rate: 1e-4

**하이퍼파라미터**:
- Masking ratio: 50-60% (data/model 에 맞춰 조정)
- Encoder: ViT-S/B 기준 (ViT-L 이상은 MAE 권장)
- Decoder: linear
- Epochs: 300 (MAE 의 800 대비 빠름)

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **모든 patch 처리** | Masked region 이 encoder 에 노출 | Information leakage 가능 |
| **Single linear decoder** | Small model 에 적합 | Large model 에서는 bottleneck |
| **50-60% masking** | MAE 의 75% 보다 낮음 | Dataset/task 에 따라 조정 필요 |
| **정보 누수** | Masked token embedding 이 encoder 의 hidden 상태에 영향 | Biased representation 가능성 |
| **Simplicity trade-off** | 구현이 간단 | 큰 모델에서는 성능 손실 |

## 📌 핵심 정리

$$\boxed{\text{SimMIM}: \, \text{All patches} \to \text{Encoder} \to \text{Linear decoder} \to \text{Reconstruct}}$$

$$\boxed{\text{Masking ratio}: 50\text{-}60\% \quad (\text{vs MAE } 75\%)}$$

| 개념 | SimMIM | MAE |
|------|--------|-----|
| **Encoder input** | 모든 patch (visible + masked) | Visible 만 (25%) |
| **Encoder speedup** | 없음 | 4× |
| **Decoder** | Linear 또는 shallow MLP | 8-layer Transformer |
| **Masking ratio** | 50-60% | 75% |
| **ViT-B 성능** | ~83.8% | ~83.6% |
| **ViT-L 성능** | ~84.5% | ~85.1% |
| **구현 복잡도** | 매우 낮음 | 중간 |

## 🤔 생각해볼 문제

### 문제 1 (기초): Mask Token 의 역할

SimMIM 에서 masked position 에 learnable mask token embedding 을 주입하면,
encoder 가 이를 통해 "masked region 에 대한 정보" 를 암묵적으로 획득할 수 있다.
이것이 "정보 누수" 인가, 아니면 "학습 신호" 인가?

<details>
<summary>해설</summary>

**정보 누수 관점**:
- Masked position 의 embedding 이 learnable → encoder 가 이를 점진적으로 학습
- Attention 을 통해 visible patch 가 masked position 에 attend
- 결과: encoder 가 masked region 을 **부분적으로 "보고 있음"**

**학습 신호 관점**:
- Mask token 은 random initialization (정보 없음)
- Encoder 가 mask token 을 처리하면서 "masked region 의 feature" 를 learn
- 이것이 **representation 학습의 일부**

**결론**: 둘 다 맞음.
- MAE 는 정보 누수 최소화 (encoder 에 visible 만 입력)
- SimMIM 은 정보 누수 허용 (대신 masking ratio 낮춤)

각각의 trade-off:
- MAE: clean separation, 높은 masking 가능, 4× speedup
- SimMIM: simple, 작은 모델에 충분, 구현 용이

</details>

### 문제 2 (심화): Decoder 단순화의 한계

왜 large model (ViT-L, ViT-H) 에서는 linear decoder 가 insufficient 할까?
**representation bottleneck** 의 관점에서 수학적으로 설명하시오.

<details>
<summary>해설</summary>

**Encoder output dimension**: $D = 1024$ (ViT-L)
**Linear decoder**: $\mathbb{R}^{1024} \to \mathbb{R}^{1024}$ (same dimension)

문제: reconstruction task 의 복잡도가 encoder 의 single output dimension 으로는 capture 불가능

**이유**:
Encoder 는 모든 patch 의 정보를 compress 해야 함:
- Visible patch: 자신의 정보 + masked region context
- 결과: encoder output $\mathbf{h}_i$ 는 "aggregate feature" (low-level detail 손실)

Linear decoder 로는 이 aggregate feature 로부터 pixel-level detail 복원 불가능:
$$\text{rank}(W_{\text{linear}}) = D \quad \text{(fixed)}$$

Deep decoder (8-layer) 는 이 정보를 **점진적으로 decompile**:
$$\text{effective rank} = \text{accumulated from multiple layers}$$

**정량적**:
- ViT-B: encoder 의 768D 가 196 patch 의 정보 충분히 contain
  → linear decoder 로도 복원 가능 (각 patch 약 4D)
- ViT-L: encoder 의 1024D 는 196 patch 에 대해 부족
  → multi-layer decoder 필수

</details>

### 문제 3 (논문 비평): MAE vs SimMIM 의 원칙적 차이

MAE (asymmetric, clean masking) vs SimMIM (symmetric, information leakage) 은
단순한 "구현 선택의 차이" 가 아니라, **representation learning 의 철학적 차이**를 반영한다.

각 접근이 암묵적으로 가정하는 "좋은 representation 의 정의" 를 논의하시오.

<details>
<summary>해설</summary>

**MAE 의 철학**: "Compression + Reconstruction"
- Encoder: visible patch 로부터 masked region 을 compress
- Decoder: compressed latent 에서 masked region 을 reconstruction
- 가정: **"작은 latent space 로 큰 정보를 효율적으로 담는 능력"** = good representation

**SimMIM 의 철학**: "Prediction in deep space"
- Encoder: 모든 input 을 처리하면서 masked region 도 고려
- Decoder: deep encoder 의 latent 로부터 직접 prediction
- 가정: **"깊은 encoding 에서 low-level detail 을 추출 가능"** = good representation

**구체적 차이**:

MAE:
- 정보론: 최소한의 정보로 maximum reconstruction
- 장점: asymmetric 이용해 4× speedup, clean objective
- 한계: decoder 의존도 높음

SimMIM:
- 학습론: encoder 가 충분히 rich 하면 simple decoder 도 가능
- 장점: symmetric, 구현 간단, small model 에 neutral
- 한계: 정보 누수, 큰 모델에서 성능 손실

**결론**: 
- MAE → "Efficiency + Information theory focused"
- SimMIM → "Simplicity + Empirical sufficiency focused"

현실: 두 접근 모두 유효. Task 와 model size 에 따라 선택.

</details>

---

<div align="center">

[◀ 이전](./03-mae-information-theory.md) | [📚 README](../README.md) | [다음 ▶](./05-maskfeat-mvp.md)

</div>
