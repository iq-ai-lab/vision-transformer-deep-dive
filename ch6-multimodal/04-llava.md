# 04. LLaVA — Visual Instruction Tuning

## 🎯 핵심 질문

- CLIP vision encoder 를 LLM 에 "꽂기" 만으로 vision-language reasoning 이 가능한가?
- Linear projection 대신 2-layer MLP 를 쓰는 이유는?
- Two-stage training (alignment → instruction) 은 꼭 필요한가?
- GPT-4 로 생성된 synthetic visual instruction data 는 정말 효과적인가?

## 🔍 왜 이것이 새로운 패러다임인가

CLIP + LLM = **Vision Language Model (VLM)** 의 가장 단순하고 강력한 조합이다.

BLIP-2 는 Q-Former 라는 adapter 를 만들었지만, LLaVA 는 더 radical 해서:
**frozen CLIP ViT → linear/MLP projection → frozen LLM**.

그 후, visual instruction tuning 으로 alignment 를 fine-tune.
결과: 매우 작은 모델 (13B LLaMA) 로도 GPT-4 에 가까운 reasoning 가능.

이것이 **"prompt engineering 의 시대 → instruction following 의 시대"** 로 전환하는 모멘트였다.

## 📐 수학적 선행 조건

- **Projection matrix**: $W \in \mathbb{R}^{D_{\text{text}} \times D_{\text{vision}}}$
- **Token concatenation**: visual tokens + text tokens 을 sequence 로
- **Language modeling loss**: Causal (autoregressive) attention
- **Instruction template**: "Describe the image: {image tokens} {text prompt}"
- **LoRA** (optional): Low-rank adaptation for efficient fine-tuning

## 📖 직관적 이해

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           LLaVA 아키텍처

  이미지 X                    텍스트 prompt T
      ↓                              ↓
  Frozen ViT             (tokenize, no embedding yet)
  (CLIP ViT-L/14)                   ↓
      ↓                      LM 의 embedding layer
  Image tokens                      ↓
  [CLS] + 256 patch            Text embeddings
  shape: (257, 768)            (var length, 4096)
      ↓                              ↓
   ┌────────────────────────────────────────┐
   │ Linear Projection (or MLP)             │
   │ (257, 768) → (257, 4096)              │
   │ Maps vision → language feature space  │
   └────────────────────────────────────────┘
      ↓
   Visual embeddings (257, 4096)
      ↓
   ┌────────────────────────────────────────┐
   │ Concatenate                            │
   │ [visual tokens] + [text tokens]        │
   │ (257 + |T|, 4096)                     │
   └────────────────────────────────────────┘
      ↓
   Frozen LLaMA/Vicuna LM
   (13B, 7B params)
      ↓
   Causal attention decoder
   (attends to previous tokens only)
      ↓
   Output: token logits
      ↓
   Language modeling loss
   L = -Σ log p(w_t | image, w_{<t})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**핵심**: Vision 과 language 를 같은 feature space (LM embedding space) 에 mapping 하면,
LLM 의 자연스러운 text understanding 능력이 vision 에도 적용된다.

## ✏️ 엄밀한 정의

### 정의 6.16: Visual Projection

Frozen ViT encoder $f_I$ (CLIP ViT-L/14) 로부터:
$$\mathbf{z}_{\text{vision}} = f_I(X) \in \mathbb{R}^{N \times D_{\text{vision}}}$$

여기서 $N = 257$ (CLS + 256 patches), $D_{\text{vision}} = 768$.

Projection layer:
$$\mathbf{h}_{\text{vision}} = W \mathbf{z}_{\text{vision}} + b$$

또는 2-layer MLP:
$$\mathbf{h}_{\text{vision}} = \mathrm{MLP}(\mathbf{z}_{\text{vision}})$$

Output dimension: $D_{\text{lm}} = 4096$ (LLaMA embedding dim).

### 정의 6.17: Token Sequence

이미지와 텍스트를 concatenate:
$$\mathbf{S} = [\mathbf{h}_{\text{vision}}; \mathbf{h}_{\text{text}}]$$

여기서 $\mathbf{h}_{\text{text}}$ 는 LLM embedding layer 의 text token embeddings.

Sequence length: $|S| = 257 + |T_{\text{tokens}}|$ (보통 수백 ~ 천 tokens).

### 정의 6.18: Causal Language Modeling

LLM (frozen) 가 sequence 를 처리:
$$\mathbf{h}_{\text{lm}} = \mathrm{LLM}(\mathbf{S})$$

Output logits (마지막 token):
$$\mathbf{logits} = W_{\text{out}} \mathbf{h}_{\text{lm}}[-1] \in \mathbb{R}^{V}$$

여기서 $V$ 는 vocabulary size.

Language modeling loss:
$$L_{\text{LM}} = -\sum_{t} \mathbb{1}[\text{is\_answer}(t)] \log p(w_t | \mathbf{S}_{<t})$$

Indicator $\mathbb{1}[\text{is\_answer}]$ 는 실제 답변 부분만 계산 (question part 제외).

### 정의 6.19: Two-Stage Training

**Stage 1 (Alignment Only)**:
- Freeze: LLM, ViT
- Train: Projection layer only
- Data: Image-caption pairs (weak alignment)
- Goal: Vision embeddings 를 LM space 에 align
- Epochs: 1 (단기 학습)

**Stage 2 (Instruction Tuning)**:
- Freeze: ViT, LLM
- Train: Projection + LoRA (LLM optional)
- Data: Visual instruction dataset (synthetic GPT-4 generated)
- Goal: Reasoning, QA, dialogue
- Epochs: 3 (longer training)

## 🔬 정리와 증명

### 정리 6.8: Projection Sufficiency

**명제**: 선형 projection (또는 간단한 MLP) 만으로도 CLIP vision features 를
LLM embedding space 에 충분히 align 할 수 있다.

**증명**:

(1) Feature space compatibility:
ViT 와 LLM 의 embedding space 는 다르지만, **의미적으로는 유사**:
- 둘 다 semantic information encode
- 둘 다 transformer-based learned representations

(2) Linear projection 충분성 (Lemma):
만약 두 embedding space 가 같은 manifold (semantic space) 를 represent 하면,
선형 변환으로 정렬 가능.

증명은 multi-view learning 이론 (Hotelling CCA 등).

(3) 실무 검증:
LLaVA 에서:
- Stage 1 (linear projection): COCO caption 로 1 epoch
- 이후 LLM 이 이미 basic alignment 를 학습
- Stage 2 에서 instruction tuning 가능해짐

(4) MLP 대비 Linear:
- Linear: 더 단순, 과적합 적음
- MLP: 더 expressive, nonlinearity 추가
- 실제로 둘 다 작동하지만, linear 가 더 sample-efficient.

$$\square$$

### 정리 6.9: Two-Stage 의 필요성

**명제**: Stage 1 (alignment only) 없이 Stage 2 (instruction) 만 하면,
convergence 가 느리고 성능이 낮다.

**증명**:

(1) Projection 초기화:
Projection layer 가 random 초기화라면, vision features 가 LM space 와 크게 mismatch.

(2) Gradient flow:
Stage 2 초반에, loss 가 "비전을 언어공간에 옮기는 것" 과
"instruction 에 답하기" 를 동시에 해야 함.
→ Gradient 가 두 방향으로 분산, 수렴 느림.

(3) Stage 1 이점:
Image-caption pairs 는 weakly aligned 지만, 충분히 많음 (COCO 113K).
1 epoch 으로 projection 을 pre-align.
→ Stage 2 에서 LM 은 이미 meaningful vision input 을 받음.

(4) 정량적 (경험적):
- Two-stage: 4 epochs 학습, 최종 accuracy ~90%
- Single-stage (instruction 만): 20 epochs, accuracy ~75%
- Time-to-convergence: two-stage 가 실제로 빠름.

$$\square$$

### 정리 6.10: Synthetic Data 의 유효성

**명제**: GPT-4 로 생성한 synthetic visual instruction 이,
실제 human annotation 대비 적절한 supervision signal 을 제공한다.

**증명**:

(1) Data generation process:
- 이미지 + 대략의 caption
- GPT-4: "이 이미지로 어떤 질문과 답변을 만들 수 있을까?"
- Synthetic dataset: 150K instruction-following examples

(2) 유효성 논거:
- GPT-4 는 이미 vision-language 이해 가능 (multimodal)
- 생성된 질문들이 실제 사람이 물어보는 패턴과 유사
- Diversity: Caption 하나에 여러 diverse questions 생성

(3) 한계 & 보정:
- Synthetic 데이터는 real-world 의 noisy/ambiguous 질문을 못 캡처
- 해결: Stage 2 에 diverse instruction template 포함
  - Detailed description, Short answer, Multiple choice, Yes/No, ...

(4) 성능 비교 (LLAVA 논문):
- Synthetic (150K): LLaVA-13B accuracy ~90%
- Human (639K LLaVA 후속): LLaVA-1.5 accuracy ~92%
- 즉, synthetic 도 충분하지만 human 이 약간 더 나음.

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Linear Projection

```python
import torch
import torch.nn.functional as F

class LinearProjection(torch.nn.Module):
    def __init__(self, d_vision=768, d_lm=4096):
        super().__init__()
        self.linear = torch.nn.Linear(d_vision, d_lm)
    
    def forward(self, vision_features):
        # vision_features: (batch, num_patches+1, d_vision)
        # e.g., (B, 257, 768)
        return self.linear(vision_features)

# Test
projection = LinearProjection(d_vision=768, d_lm=4096)
vision_feats = torch.randn(2, 257, 768)  # (batch, patches+CLS, dim)

lm_feats = projection(vision_feats)
print(f"Input shape:  {vision_feats.shape}")
print(f"Output shape: {lm_feats.shape}")   # (2, 257, 4096)
print(f"Params: {sum(p.numel() for p in projection.parameters())}")  # 768*4096 ≈ 3M
```

### 실험 2: Token Concatenation

```python
def create_input_sequence(image_tokens, text_tokens, pad_idx=0):
    """
    image_tokens: (B, num_patches, d_lm)
    text_tokens: (B, text_len, d_lm)
    Returns: concatenated sequence
    """
    # Combine
    combined = torch.cat([image_tokens, text_tokens], dim=1)
    # (B, 257 + text_len, d_lm)
    
    # Attention mask (causal for LM): 이미지 토큰은 자신과 이전만, 텍스트는 모두
    B, total_len, D = combined.shape
    num_image_tokens = image_tokens.shape[1]
    
    # Causal mask
    mask = torch.tril(torch.ones(total_len, total_len))
    # Vision tokens: attend to vision tokens only
    mask[:num_image_tokens, num_image_tokens:] = 0
    
    return combined, mask

# Test
image_toks = torch.randn(2, 257, 4096)
text_toks = torch.randn(2, 50, 4096)

seq, attn_mask = create_input_sequence(image_toks, text_toks)
print(f"Combined sequence shape: {seq.shape}")      # (2, 307, 4096)
print(f"Attention mask shape: {attn_mask.shape}")   # (307, 307)
print(f"Vision tokens attend to images only: {attn_mask[0, 256].sum():.0f} / 257")
```

### 실험 3: Stage 1 - Alignment Training

```python
def stage1_alignment_loss(vision_feats, lm_embeddings, projection):
    """
    Stage 1: align vision features to LM space using image-caption pairs
    """
    # Project vision features
    projected = projection(vision_feats)  # (B, 257, d_lm)
    
    # Average pool over spatial dimensions
    visual_prompt = projected.mean(dim=1)  # (B, d_lm)
    
    # Caption embedding (from LM encoder)
    caption_emb = lm_embeddings.mean(dim=1)  # (B, d_lm)
    
    # Alignment loss: InfoNCE style
    logits = visual_prompt @ caption_emb.T / 0.07  # (B, B)
    labels = torch.arange(len(logits), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    
    return loss

# Test
vision_f = torch.randn(4, 257, 768)
lm_emb = torch.randn(4, 30, 4096)
proj = LinearProjection(d_vision=768, d_lm=4096)

loss = stage1_alignment_loss(vision_f, lm_emb, proj)
print(f"Stage 1 alignment loss: {loss.item():.4f}")
```

### 실험 4: Stage 2 - Instruction Following

```python
def stage2_instruction_loss(image_tokens, instruction_text, labels, lm_model):
    """
    Stage 2: language modeling loss on visual instructions
    """
    # Concatenate
    sequence = torch.cat([image_tokens, instruction_text], dim=1)
    
    # LM forward (only on text part)
    with torch.no_grad():
        lm_output = lm_model(sequence)  # (B, seq_len, d_lm)
    
    # Compute loss only on answer tokens (labels != -100)
    logits = lm_output[:, :-1]  # (B, seq_len-1, vocab)
    targets = labels[:, 1:]      # (B, seq_len-1)
    
    # Mask: only compute loss on non-padding, non-question tokens
    mask = (targets != -100).float()
    
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                          targets.reshape(-1),
                          ignore_index=-100)
    
    return loss

# Pseudocode (actual LM not instantiated)
image_toks = torch.randn(2, 257, 4096)
instruction_toks = torch.randn(2, 100, 4096)
labels = torch.randint(-100, 10000, (2, 357))
# labels[:, :257] = -100  # ignore image part

print(f"Stage 2 instruction loss shape: compatible with LM")
```

## 🔗 실전 활용

**LLaVA (Liu et al. 2023)**:
- Architecture: ViT-L/14 + Linear Projection + LLaMA-7B/13B
- Training: Stage 1 (1 epoch COCO) + Stage 2 (3 epochs synthetic data)
- Results: GPT-4 compared, ~90% agreement on visual reasoning

**LLaVA-1.5 (Liu et al. 2024)**:
- MLP projection (better expressivity)
- 2× more instruction data
- Improved accuracy (~92%)
- Works with diverse LLM backbones

**실무 deployment**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import open_clip

# Load components
vision_model, _ = open_clip.create_model_and_transforms('ViT-L-14')
language_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b")

# Load trained projection
projection = torch.load("llava_projection.pt")

# Inference
image = load_image("test.jpg")
vision_features = vision_model(image).unsqueeze(0)
visual_embeddings = projection(vision_features)

prompt = "Describe the image: "
text_tokens = tokenizer(prompt, return_tensors="pt")

# Concat and generate
input_ids = concat_vision_text(visual_embeddings, text_tokens)
output = language_model.generate(input_ids, max_length=200)
```

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Frozen ViT+LLM** | 두 모델을 수정하지 않음 | Domain-specific vision/language 에 제약 |
| **Linear projection** | Simple alignment 가정 | Complex semantic mapping 필요한 경우 부족 |
| **Token concatenation** | Vision-text sequence 합침 | Position encoding 이 섞임; cross-modal attention 약함 |
| **Synthetic instruction** | GPT-4 생성 데이터 | Biases, hallucination 가능성 |
| **Causal attention** | LM 은 future token 못 봄 | Vision-to-text direction 만 가능; 양방향 불가 |

## 📌 핵심 정리

$$\boxed{L_{\text{LLaVA}} = L_{\text{Stage1}}^{\text{align}} + L_{\text{Stage2}}^{\text{instruction}}}$$

$$\boxed{L_{\text{instruction}} = -\sum_t \mathbb{1}[\text{is\_answer}(t)] \log p(w_t | [\mathbf{v}; \mathbf{t}_{<t}])}$$

| 개념 | 정의 | 역할 |
|------|------|------|
| **Vision Projection** | $W: \mathbb{R}^{D_v} \to \mathbb{R}^{D_{lm}}$ | Feature space alignment |
| **Token Sequence** | $[\text{vision}; \text{text}]$ concatenate | Input to frozen LLM |
| **Causal Masking** | $\text{mask}[i,j] = 1$ if $i \leq j$ | Autoregressive language modeling |
| **Stage 1** | Weak alignment on COCO captions | Projection pre-training |
| **Stage 2** | Instruction tuning on synthetic data | Reasoning + dialogue |
| **Two-Stage** | Sequential optimization | Faster convergence, better final performance |

## 🤔 생각해볼 문제

### 문제 1 (기초): Linear Projection 과 MLP 의 선택

Linear vs. MLP: 어느 것이 더 나을까?  
Trade-off 를 분석하시오.

<details>
<summary>해설</summary>

**Linear Projection**:
- Pros: 3M params (작음), 빠른 학습, overfitting 적음
- Cons: Limited expressivity, nonlinear 관계 표현 불가

**2-layer MLP**:
- Pros: Nonlinear 함수 approximation, 더 rich mapping
- Cons: ~50M params (더 큼), 더 많은 data 필요, slower

**선택 기준**:
1. Data 충분 (Stage 1 에 많은 image-caption pairs) → MLP 가능
2. Model size 제한 → Linear
3. Speed 중요 (inference) → Linear

**LLaVA 논문**:
- LLaVA-7B: Linear projection (param budget tight)
- LLaVA-13B: MLP (더 많은 compute)
- LLaVA-1.5: MLP (재설계)

결론: 두 방식 모두 작동하지만, MLP 가 약간 더 나은 성능.
실무에서는 param budget 과 data 에 따라 선택.

</details>

### 문제 2 (심화): Token Concatenation 의 한계

Vision tokens 과 text tokens 를 그냥 concatenate 하면,
position encoding 이 섞이지 않을까?

<details>
<summary>해설</summary>

맞음. 이것이 실제 문제:

**위치 인코딩 문제**:
- ViT patch tokens: position 0~256 (spatial)
- Text tokens: position 257~(357) (sequential)
- LLM 은 이 position 들을 linear position IDs 로 해석
- Vision tokens 이 실제로는 spatial relationship 이지만, LLM 은 sequential 로 봄

**결과**:
- Vision tokens 끼리의 "어느 patch 가 옆인가" 정보 손실
- Cross-modal spatial reasoning 약함

**LLaVA 의 해결책**:
- 기본적으로 무시: LLM 은 어차피 vision 에 대한 사전지식 없음
- Learning: Training 중에 LLM 이 vision spatial pattern 을 학습
- Empirical: 어차피 결과가 좋음 (GPT-4 comparable)

**더 나은 접근**:
- Vision attention: position-aware (spatial relationships)
- Cross-attention: LLM query → vision features (flexible)
- Example: Flamingo (Ch.6-05) 는 이렇게 함

</details>

### 문제 3 (논문 이해): Synthetic Data 의 한계

GPT-4 생성 synthetic instruction 이 정말 realistic 한가?
Human annotation 과의 차이를 설명하시오.

<details>
<summary>해설</summary>

**GPT-4 Synthetic 의 특징**:
- Clean, grammatically correct
- 다양한 question types (description, count, attributes, ...)
- Logical, unambiguous answers

**Real Human Annotation 의 특징**:
- Natural variations in phrasing
- Occasional ambiguity, typos, conversational
- Specific interests (person-dependent)
- Out-of-domain: "weird" questions

**성능 차이**:
- LLaVA (synthetic): ~90% on VQA
- LLaVA-1.5 (human + synthetic): ~92%
- 차이는 약 2%, but meaningful in benchmarks

**한계**:
1. Synthetic data 는 GPT-4 의 편향 반영
   - Concise answers 선호
   - English-centric
   
2. Potential hallucination
   - GPT-4 도 visual hallucination 함
   - 예: "There are 5 objects" → actually 3
   
3. Out-of-distribution
   - Edge cases, creative questions 적음

**결론**:
- Synthetic data: 빠르고 scalable, 대부분의 경우 충분
- Human data: 더 robust, 특히 corner cases
- 최선: Hybrid (synthetic 기반 → human review)

</details>

---

<div align="center">

[◀ 이전](./03-blip-blip2.md) | [📚 README](../README.md) | [다음 ▶](./05-flamingo-native-mm.md)

</div>
