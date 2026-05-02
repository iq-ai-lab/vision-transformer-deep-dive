# 03. BLIP · BLIP-2 — Bootstrapping Language-Image Pretraining

## 🎯 핵심 질문

- CLIP 은 matching 만 하는데, BLIP 은 왜 captioning 까지 하는가?
- 세 가지 손실 (ITC, ITM, LM) 을 동시에 optimize 하면 균형은?
- Web data 의 noisy captions 를 어떻게 필터링하는가? (CapFilt)
- BLIP-2 의 Q-Former 가 frozen ViT 와 frozen LLM 을 왜 연결해야 하는가?

## 🔍 왜 이것이 더 나은 비전-언어 모델인가

CLIP 은 **대비 학습** (matching) 만 한다. 두 modality 의 embedding 을 align.

BLIP 은 **세 가지 신호**를 동시에 학습한다:
1. **ITC** (Image-Text Contrastive): CLIP 처럼 alignment
2. **ITM** (Image-Text Matching): Binary classification (일치/불일치)
3. **LM** (Language Modeling): 이미지로부터 caption 생성

이렇게 하면 CLIP 보다 더 rich 한 representation 을 얻는다.

BLIP-2 는 더 진화해서, **frozen pretrained models** (ViT, LLM) 의 사이에 가벼운 Q-Former adapter 를 넣어서
극도로 효율적인 vision-language alignment 를 만들었다.

## 📐 수학적 선행 조건

- **Cross-entropy**: Binary classification, language modeling
- **Contrastive loss**: InfoNCE (이전 장)
- **Momentum contrast**: MoCo 의 queue mechanism
- **Qwen-series**: Large language model backbone
- **Transformer decoder** (cross-attention layers)

## 📖 직관적 이해

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    BLIP 구조

        이미지 X              텍스트 T
           ↓                    ↓
     ViT 이미지 인코더    Transformer 텍스트 인코더
           ↓                    ↓
    Image features z_I    Text features z_T
    (256 tokens)          (sequence)
           ↓                    ↓
         ┌──────────────────────┐
         │   세 가지 Task       │
         ├──────────────────────┤
         │ 1. ITC (대비):       │
         │    cos_sim(z_I,z_T)  │
         │    → InfoNCE loss    │
         │                      │
         │ 2. ITM (매칭):       │
         │    concat(z_I,z_T)   │
         │    → MLP → binary    │
         │                      │
         │ 3. LM (생성):        │
         │    z_I → decoder     │
         │    → caption tokens  │
         │    → LM loss         │
         └──────────────────────┘
               ↓
    Loss = w1*L_ITC + w2*L_ITM + w3*L_LM
    (typical: all equal weights)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BLIP-2: Frozen components + Q-Former

     Frozen ViT          Frozen LLM
   (image encoder)     (text decoder)
         ↓                  ↓
   Image tokens         Query tokens
   (e.g., 144)         (32 queries)
         ↓                  ↓
     ┌─────────────────────────┐
     │ Q-Former               │
     │ (Transformer encoder)  │
     │ - Cross-attention ↓    │
     │   to frozen ViT        │
     │ - Self-attention       │
     │ 32 learnable queries   │
     │ → 32 output features   │
     └─────────────────────────┘
              ↓
        Prefix to LLM
        (visual prompt)
              ↓
        LLM decoder
        (generates caption)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## ✏️ 엄밀한 정의

### 정의 6.11: Image-Text Contrastive (ITC)

CLIP 과 유사:
$$L_{ITC} = -\log \frac{e^{S_{ii} / \tau}}{\sum_j e^{S_{ij} / \tau}} - \log \frac{e^{S_{ii} / \tau}}{\sum_j e^{S_{ji} / \tau}}$$

여기서 $S_{ij} = f_I(I_i)^T f_T(T_j)$ (image-text 인코더로부터).

### 정의 6.12: Image-Text Matching (ITM)

이미지와 텍스트가 matching 되는지 binary classification:
$$p_{ITM} = \sigma(\mathrm{MLP}([z_I; z_T]))$$

여기서 $[\cdot; \cdot]$ 는 concatenation, $\sigma$ 는 sigmoid.

$$L_{ITM} = -[y \log p + (1-y) \log(1-p)]$$

$y=1$ 이면 positive pair, $y=0$ 이면 negative pair (hard negatives).

### 정의 6.13: Language Modeling (LM)

이미지에서 caption 생성:
$$L_{LM} = -\sum_{t=1}^T \log p_\theta(w_t \mid I, w_{<t})$$

여기서 $w_t$ 는 caption 의 t-번째 token, decoder 는 cross-attention (image features).

### 정의 6.14: CapFilt (Cleaning Noisy Web Data)

Web data 의 noisy captions 를 필터링:
$$\text{score}_c = (p_{ITC} + p_{ITM}) / 2$$

Threshold 이상의 (image, caption) 쌍만 retain.

### 정의 6.15: Q-Former (BLIP-2)

Frozen ViT 로부터 image tokens $\mathbf{z}_{ViT} \in \mathbb{R}^{N \times D}$ 를 받음
(e.g., $N=144$ tokens, $D=1024$).

Q-Former 는:
- 32 learnable query vectors $\mathbf{q} \in \mathbb{R}^{32 \times D}$
- Cross-attention: $\mathbf{q}$ → attends to $\mathbf{z}_{ViT}$
- Self-attention: within 32 queries
- Output: $\mathbf{z}_{out} \in \mathbb{R}^{32 \times D}$ (compact visual prompt)

Frozen LLM 에 prefix 로 들어감.

## 🔬 정리와 증명

### 정리 6.5: Multi-Task Learning 의 Gradient 균형

**명제**: BLIP 의 세 손실 (ITC, ITM, LM) 을 같은 weight 로 학습하면,
각 task 의 gradient magnitude 는 조정이 필요할 수 있다.

**증명**:

(1) 각 손실의 scale 다름:
- $L_{ITC}$: typically $\sim 1.4$ (log B, where B~256)
- $L_{ITM}$: binary classification, $\sim 0.5$-$1.0$ 범위
- $L_{LM}$: token-wise, much longer sequence (100+ tokens)

따라서 $L_{LM}$ 의 총합이 다른 두 손실보다 훨씬 클 수 있다.

(2) Gradient flow:
$$\frac{\partial L_{\text{total}}}{\partial \theta} = \frac{\partial L_{ITC}}{\partial \theta} + \frac{\partial L_{ITM}}{\partial \theta} + \frac{\partial L_{LM}}{\partial \theta}$$

만약 $L_{LM}$ scale 이 크면, gradient 가 LM task 쪽으로 biased.

(3) 실무 조정:
BLIP 논문에서는 각 손실에 importance weight 를 주거나,
batch normalization 형태의 gradient clipping 으로 균형.

예: $L = w_1 \cdot L_{ITC} + w_2 \cdot L_{ITM} + w_3 \cdot L_{LM}$
where $w_1, w_2$ = 1, $w_3 = 1$ (normalized by sequence length).

$$\square$$

### 정리 6.6: CapFilt 의 유효성

**명제**: CapFilt (quality filtering) 는 web data 의 50% 를 제거하면서도
모델 성능을 향상시킨다.

**증명**:

(1) Web data noise:
- LAION 같은 large web-scraped dataset 은 image-text 쌍이 일치하지 않을 수 있음
- Alt text, metadata 가 부정확

(2) CapFilt mechanism:
- 초기 BLIP 으로 fine-tune score 계산
- ITC score: image-text alignment 정도
- ITM score: hard negative 와의 구분
- $(S_{ITC} + S_{ITM}) / 2$ 로 rank

(3) 효과:
- 상위 50% captions 만 retain
- 실제로 downstream task (retrieval, captioning) 성능 UP
- Noisy supervision 제거로 더 clean 한 signal

$$\square$$

### 정리 6.7: Q-Former 의 효율성

**명제**: Q-Former 는 frozen ViT (1B+ params) 와 frozen LLM (7B+ params) 을
수십만 개의 learnable parameters 로 연결할 수 있어서,
full fine-tuning 대비 1000배 이상 효율적이다.

**증명**:

(1) Bottleneck design:
- ViT: 수백 개 tokens (예: 144)
- Q-Former: 32 queries (bottleneck compression)
- LLM: 수십 개 prefix tokens

따라서 information flow 가 극도로 narrowing.

(2) Parameter count:
- Q-Former: ~100-200M (relatively small)
- Frozen ViT, LLM: ~1B + ~7B (not updated)
- Total trainable: <500M

vs. Full fine-tuning: ~8B 모두 update → memory 10배 이상

(3) 성능:
- Zero-shot VQA: LLaVA 수준의 정확도 달성
- Training time: 2 주 (A100 8개)
- vs. Full model training: 몇 달

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: BLIP Multi-Task 손실

```python
import torch
import torch.nn.functional as F

def blip_loss(image_emb, text_emb, match_labels,
              caption_logits, caption_targets,
              temperature=0.07):
    """
    image_emb: (B, D) normalized
    text_emb: (B, D) normalized
    match_labels: (B,) binary (1 = match, 0 = no match)
    caption_logits: (B, T, V) logits for caption tokens
    caption_targets: (B, T) target token indices
    """
    # 1. ITC loss (symmetric)
    S = image_emb @ text_emb.T / temperature  # (B, B)
    labels = torch.arange(len(image_emb), device=image_emb.device)
    loss_i2t = F.cross_entropy(S, labels)
    loss_t2i = F.cross_entropy(S.T, labels)
    loss_itc = (loss_i2t + loss_t2i) / 2
    
    # 2. ITM loss (binary classification)
    # Assume ITM head: concat(image_emb, text_emb) → MLP → sigmoid
    matched_embs = torch.cat([image_emb, text_emb], dim=1)
    itm_logits = torch.randn(len(image_emb), 1)  # mock: would come from MLP head
    loss_itm = F.binary_cross_entropy_with_logits(itm_logits.squeeze(-1), 
                                                   match_labels.float())
    
    # 3. LM loss (language modeling)
    # Reshape: (B, T, V) → (B*T, V), (B, T) → (B*T)
    batch_size, seq_len, vocab_size = caption_logits.shape
    loss_lm = F.cross_entropy(caption_logits.reshape(-1, vocab_size),
                              caption_targets.reshape(-1),
                              ignore_index=-100)
    
    # Combined loss (equal weights)
    loss_total = loss_itc + loss_itm + loss_lm
    return loss_total, {"ITC": loss_itc.item(), 
                        "ITM": loss_itm.item(), 
                        "LM": loss_lm.item()}

# Test
B, D, T, V = 4, 256, 50, 10000
image_emb = F.normalize(torch.randn(B, D), dim=1)
text_emb = F.normalize(torch.randn(B, D), dim=1)
match_labels = torch.tensor([1, 1, 0, 0])
caption_logits = torch.randn(B, T, V)
caption_targets = torch.randint(0, V, (B, T))

loss, breakdown = blip_loss(image_emb, text_emb, match_labels,
                            caption_logits, caption_targets)
print(f"Total loss: {loss.item():.4f}")
print(f"Breakdown: {breakdown}")
```

### 실험 2: CapFilt Scoring

```python
def capfilt_score(image_emb, text_emb, itm_prob):
    """
    Combine ITC and ITM scores for quality filtering
    """
    # ITC: cosine similarity
    itc_score = F.cosine_similarity(image_emb, text_emb)  # (B,)
    
    # ITM: already a probability
    itm_score = itm_prob  # (B,)
    
    # Combined score
    combined = (itc_score + itm_score) / 2
    return combined

# Mock data
B = 1000
image_embs = F.normalize(torch.randn(B, 256), dim=1)
text_embs = F.normalize(torch.randn(B, 256), dim=1)
itm_probs = torch.sigmoid(torch.randn(B))

scores = capfilt_score(image_embs, text_embs, itm_probs)

# Threshold: keep top 50%
threshold = torch.quantile(scores, 0.5)
keep_mask = scores >= threshold
num_kept = keep_mask.sum().item()

print(f"Original data: {B}")
print(f"Kept (threshold={threshold:.3f}): {num_kept} ({100*num_kept/B:.1f}%)")
print(f"Removed: {B - num_kept}")
```

### 실험 3: Q-Former (간단한 버전)

```python
class SimpleQFormer(torch.nn.Module):
    def __init__(self, hidden_dim=256, num_queries=32, seq_len=144):
        super().__init__()
        # 32 learnable queries
        self.queries = torch.nn.Parameter(torch.randn(num_queries, hidden_dim))
        
        # Cross-attention (simplified: just weighted sum)
        self.cross_attn_weight = torch.nn.Linear(hidden_dim, seq_len)
        
    def forward(self, vit_features):
        # vit_features: (B, seq_len, hidden_dim)
        # Output: (B, num_queries, hidden_dim)
        B = vit_features.shape[0]
        
        # Expand queries
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, 32, D)
        
        # Simple attention: queries attend to vit_features
        attn_weights = torch.softmax(
            self.cross_attn_weight(queries) @ vit_features.transpose(1, 2) / 16,
            dim=-1
        )  # (B, 32, seq_len)
        
        # Weighted sum
        output = attn_weights @ vit_features  # (B, 32, D)
        return output

# Test
q_former = SimpleQFormer(hidden_dim=256, num_queries=32, seq_len=144)
vit_features = torch.randn(2, 144, 256)  # (B=2, seq_len, D)

output = q_former(vit_features)
print(f"Q-Former output shape: {output.shape}")  # (2, 32, 256)
print(f"Learnable params (queries): {32 * 256} = 8K")
```

### 실험 4: BLIP vs CLIP 비교

```python
def evaluate_tasks(model_name, model, test_data):
    """
    model_name: "CLIP" or "BLIP"
    Evaluate on different downstream tasks
    """
    results = {}
    
    # 1. Zero-shot classification (like CLIP)
    results["zero_shot"] = eval_zero_shot(model, test_data)
    
    # 2. Image-text retrieval
    results["retrieval"] = eval_retrieval(model, test_data)
    
    # 3. Caption generation (BLIP advantage)
    if model_name == "BLIP":
        results["captioning"] = eval_captioning(model, test_data)
    
    return results

# Pseudocode results
clip_results = {
    "zero_shot": 0.760,
    "retrieval": 0.680,
}

blip_results = {
    "zero_shot": 0.765,      # slightly better
    "retrieval": 0.695,       # better alignment
    "captioning": 0.385,      # CIDEr (new task!)
}

blip2_results = {
    "zero_shot": 0.768,
    "retrieval": 0.700,
    "captioning": 0.405,      # better generation
}

print("Task                 CLIP    BLIP   BLIP-2")
print("─" * 45)
print(f"Zero-shot class    {clip_results['zero_shot']:.3f}   {blip_results['zero_shot']:.3f}   {blip2_results['zero_shot']:.3f}")
print(f"Image-text ret     {clip_results['retrieval']:.3f}   {blip_results['retrieval']:.3f}   {blip2_results['retrieval']:.3f}")
print(f"Caption gen          —      {blip_results['captioning']:.3f}   {blip2_results['captioning']:.3f}")
```

## 🔗 실전 활용

**BLIP (Li et al. 2022)**:
- Multi-task learning: alignment + matching + generation
- CapFilt for data cleaning
- Image-text retrieval, VQA, image captioning

**BLIP-2 (Li et al. 2023)**:
- Q-Former bridge between frozen ViT and LLM
- Parameter efficient: trainable 200M (vs 8B total)
- Works with any frozen ViT + any frozen LLM
- Combines ViT-L/14 + Flan-T5-XL or LLaMA

**실무 applications**:
1. E-commerce: product search + captioning
2. Accessibility: image-to-text for blind users
3. Content moderation: image understanding + reasoning
4. Semantic search: combining visual + textual features

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Multi-task balance** | 세 손실의 scale 이 다를 수 있음 | 가중치 조정이나 normalization 필요 |
| **CapFilt threshold** | 50% cutoff 는 arbitrary | Dataset quality 에 따라 조정 필요 |
| **Q-Former bottleneck** | 32 queries 로 모든 visual info 압축 | Fine-grained vision tasks 에 제한 |
| **Frozen models** | LLM/ViT 업데이트 안 함 | Alignment 만 가능; model adaptation 불가 |
| **Instruction tuning** | BLIP-2 를 instruction-tune 하면? | Parameter 증가; memory tradeoff |

## 📌 핵심 정리

$$\boxed{L_{\text{BLIP}} = L_{ITC} + L_{ITM} + L_{LM}}$$

$$\boxed{L_{\text{ITC}} = -\log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}} - \log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ji}/\tau}}}$$

| 개념 | 정의 | 역할 |
|------|------|------|
| **ITC Loss** | Symmetric contrastive (CLIP-style) | Image-text alignment |
| **ITM Loss** | Binary matching classification | Hard negative distinction |
| **LM Loss** | Token-wise language modeling | Caption generation ability |
| **CapFilt** | Quality scoring + filtering | Remove noisy web data |
| **Q-Former** | 32 learnable queries + cross-attention | Frozen model bridge |
| **Efficiency** | 200M trainable / 8B+ total | 1000× fewer params than full-tune |

## 🤔 생각해볼 문제

### 문제 1 (기초): 왜 ITM 손실이 필요한가?

ITC (contrastive) 와 LM (generation) 만으로는 부족한가?  
ITM 이 추가적으로 학습하는 것은 무엇인가?

<details>
<summary>해설</summary>

ITC 는: "같은 대상을 가리키는 이미지-텍스트는 embedding 이 가깝다"  
LM 은: "이미지로부터 텍스트를 생성할 수 있다"

ITM 이 추가하는 것: "주어진 이미지-텍스트 쌍이 실제로 match 하는지 **판단**"

예: 
- 이미지: dog photo
- 텍스트 1: "a cute dog" (match)
- 텍스트 2: "a forest landscape" (no match)

ITC 만으로: 텍스트 1 과 2 의 embedding 을 different space 에 놓으려고 함  
LM 만으로: dog image → caption 생성만 함 (retrieval 능력 없음)

ITM: "이 pair 가 match 하는가?" 를 explicitly 학습 → retrieval 성능 개선.

실무: Image-text retrieval task 에서 ITM 이 없으면 성능 떨어짐.

</details>

### 문제 2 (심화): Q-Former 의 32 queries 는 충분한가?

ViT 가 144개 tokens 을 출력하는데, 32 queries 로 압축하면 정보 손실 안 되나?

<details>
<summary>해설</summary>

정보 압축은 맞음. 144 → 32 는 4.5배 축소.

하지만:
1. **Redundancy**: ViT 의 tokens 이 redundant 할 수 있음
   - Patch embeddings 의 일부는 texture 같은 low-level
   - Object center 와 periphery 중복
   
2. **Query 학습**: 32 queries 는 learnable 이고, **중요한 정보에 집중**
   - Self-attention: queries 끼리 interact
   - Cross-attention: ViT 의 informative patches 에 attend

3. **LLM 의 수용성**: LLM 은 long sequence 처리 cost 많음
   - 32 tokens (visual prefix) 는 합리적
   - 144 tokens 이면 LLM generation 느려짐
   
4. **Empirical**: BLIP-2 가 LLaVA (ViT 전체 사용) 와 비슷한 성능
   - 즉, 32 queries 에서 충분한 정보 압축 가능

Better approach: Hierarchical queries 나 adaptive number (시간 따라).

</details>

### 문제 3 (논문 비평): CapFilt 는 정말 필요한가?

Web data 를 manually clean 하는 것과, CapFilt 의 수렴성을 비교하시오.

<details>
<summary>해설</summary>

Manual cleaning:
- 비용: 엔지니어 인건비 (매우 비쌈)
- 규모: 400M 쌍은 불가능
- Quality: 일관성 문제

CapFilt (automatic):
- 비용: 컴퓨팅 (한 번만)
- 규모: 전체 dataset 처리 가능
- Quality: BLIP 모델의 편향 반영

**수렴성 argument**:
- CapFilt 는 BLIP 을 fine-tune 해서 quality score 계산
- 이것이 circular dependency 가 아닌가? (model quality 의존적)

하지만 실제로:
- 초기 BLIP 도 bad captions 보다 good captions 를 선호
- 반복 (iterate): filter → train → filter 하면 수렴
- 결과: 50% data 로 original 대비 같거나 더 나은 성능

**결론**: Manual labeling 대비 automation 으로도 충분. 
하지만 iterative refinement 가 있으면 더 좋음.

</details>

---

<div align="center">

[◀ 이전](./02-clip-zero-shot.md) | [📚 README](../README.md) | [다음 ▶](./04-llava.md)

</div>
