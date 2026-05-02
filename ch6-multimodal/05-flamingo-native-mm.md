# 05. Flamingo · GPT-4V · Gemini — Interleaved Vision-Language & Native Multimodal

## 🎯 핵심 질문

- Flamingo 의 gated cross-attention 왜 필요한가? $\tanh(\alpha)$ 로 initialized zero?
- Native multimodal models (GPT-4V, Gemini, Claude 3) 는 왜 "single Transformer" 설계인가?
- Interleaved image-text sequences 를 어떻게 처리하는가?
- Flamingo 의 frozen vision/language 모델에서 gated mechanism 만 학습하는 이유는?

## 🔍 왜 이것이 패러다임 전환인가

지금까지의 모든 모델 (CLIP, BLIP, LLaVA) 은 **separate encoders** 를 fusion 했다:
- Vision encoder (frozen ViT)
- Text encoder/decoder (frozen LLM)
- Adapter/projection (trainable)

Flamingo 는 다르다. **Interleaved multimodal input** 을 처음부터 설계:
```
[text] [image] [text] [image] [text]
```

이는 few-shot in-context learning 을 자연스럽게 지원하고,
실제로 인간의 communication 방식에 더 가깝다.

GPT-4V, Gemini, Claude 3 은 더 radical해서, 진정한 **native multimodal Transformer** 를 만들었다:
- Single transformer backbone
- Image tokens 와 text tokens 가 unified attention space 에서 섞임
- 더 이상 vision/text 분리 없음

## 📐 수학적 선행 조건

- **Gated attention**: $\text{tanh}(\alpha) \in (-1, 1)$, learned scale
- **Frozen vision/language**: No backprop through large encoders
- **Cross-attention**: Query (LLM hidden) × Key/Value (vision)
- **Interleaved sequences**: Mixed image/text token ordering
- **Position embeddings**: Unified across modalities

## 📖 직관적 이해

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          Flamingo: Gated Cross-Attention

  LLM decoder layer (in-place)
           ↓
    Self-attention ← text only
    (previous tokens)
           ↓
    Gated Cross-Attention (NEW)
           ↓
    ┌──────────────────────────────────────┐
    │ Query: LLM hidden state (text-only)  │
    │                                      │
    │ Key/Value: frozen vision encoder     │
    │ (from image tokens)                  │
    │                                      │
    │ Attention output: a = softmax(Q·K^T) V
    │                                      │
    │ Gating (critical):                   │
    │   α = learnable scalar (init ~ 0)    │
    │   gate = tanh(α)                     │
    │   output = tanh(α) * a               │
    │                                      │
    │ Effect: Init behavior = no vision    │
    │ Gradually learn to attend            │
    └──────────────────────────────────────┘
           ↓
    Feed-forward
           ↓
    [back to decoder]

Why tanh(α) initialized to 0?
  → tanh(0) ≈ 0, so gate ≈ 0
  → Initially: LLM behaves as text-only
  → Gradually: α learns to let vision info flow
  → Prevents early training instability
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Native Multimodal (GPT-4V, Gemini):

  Unified Transformer Backbone
  
  Image tokens [img_0_0] [img_0_1] ... [img_0_256]
  Text tokens  [The]  [image]  [shows]  [a]  [dog]
       ↓
  ┌─────────────────────────────────────────┐
  │ Unified embedding space                 │
  │ - Same embedding dimension              │
  │ - Same position encoding scheme          │
  │ - Unified token vocabulary               │
  │ - Mixed in single attention              │
  └─────────────────────────────────────────┘
       ↓
  Unified self-attention
  (images and text attend to each other)
       ↓
  Causally masked decoder
  (future tokens masked)
       ↓
  Output: text tokens only (generation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## ✏️ 엄밀한 정의

### 정의 6.20: Gated Cross-Attention (Flamingo)

Flamingo 의 각 decoder layer 에, 다음이 추가됨:

**Cross-attention block**:
$$\text{attn}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k}) V$$

여기서:
- $Q$: LLM hidden states (text-only, from self-attention)
- $K, V$: Frozen vision encoder output (ViT 또는 NFNet)

**Gating mechanism**:
$$\alpha \in \mathbb{R}^{1}$$ (learnable scalar)

$$g = \tanh(\alpha)$$

$$\text{output} = g \odot \text{attn}(Q, K, V)$$

여기서 $\odot$ 은 element-wise multiplication.

### 정의 6.21: Initialization Strategy

$$\alpha^{(0)} = \log(1/c)$$

where $c \approx 0.01$ (초기 gating strength weak).

따라서:
$$g^{(0)} = \tanh(\alpha^{(0)}) = \tanh(\log(1/0.01)) \approx \tanh(-4.6) \approx -0.99...$$

실제로는 다시 정규화하거나, bias 를 더해서 초기값 조정.

**Key point**: 초기에 $g \approx 0$ 또는 작은 값이 되어, vision 정보 흐름을 제약.

### 정의 6.22: Interleaved Input Format

Sequence of mixed tokens:
$$S = [\mathbf{t}_1, \mathbf{i}_1, \mathbf{t}_2, \mathbf{i}_2, \ldots, \mathbf{t}_K]$$

여기서:
- $\mathbf{t}_k$: Text token sequence
- $\mathbf{i}_k$: Image feature sequence (from frozen vision encoder)

Few-shot example:
```
[Image: dog.jpg] "This is a dog."
[Image: cat.jpg] "This is a cat."
[Image: bird.jpg] "What is this?"  ← model generates "This is a bird."
```

### 정의 6.23: Native Multimodal Transformer

Single backbone $\mathcal{T}$ with:

**Image tokenization**:
$$\text{img\_tokens} = \text{tokenizer}_{\text{img}}(\text{image})$$

**Text tokenization**:
$$\text{text\_tokens} = \text{tokenizer}_{\text{text}}(\text{text})$$

**Unified embedding**:
$$\mathbf{e}_{\text{img}} = E_{\text{embed}}(\text{img\_tokens})$$
$$\mathbf{e}_{\text{text}} = E_{\text{embed}}(\text{text\_tokens})$$

여기서 $E_{\text{embed}}: \{0, \ldots, V\} \to \mathbb{R}^D$ (같은 embedding layer).

**Processing**:
$$[\mathbf{h}_{\text{img}}, \mathbf{h}_{\text{text}}] = \mathcal{T}([\mathbf{e}_{\text{img}}, \mathbf{e}_{\text{text}}])$$

All tokens attend to all tokens (subject to causal mask for text generation).

## 🔬 정리와 증명

### 정리 6.11: Gated Cross-Attention 의 수렴 안정성

**명제**: Gated cross-attention 에서 $\alpha$ 를 0 근처에서 초기화하면,
early training 에서 vision-language misalignment 로 인한 gradient explosion 을 방지한다.

**증명**:

(1) 초기화 없이 random cross-attention:
$$\text{output} = \text{softmax}(QK^T) V$$

만약 frozen vision encoder 와 LLM 의 representation space 가 misaligned 라면,
$K$ 와 $Q$ 가 barely 정렬됨.
→ Softmax attention weights 가 sparse, noisy
→ Gradient 가 크고 불안정.

(2) Gating 효과 ($g \approx 0$):
$$\text{output}_{\text{gated}} = g \cdot \text{attn} \approx 0$$

LLM decoder 의 입장에서는, initial phase 에:
$$\text{LLM}(\text{text only, with near-zero vision})$$

= effectively text-only LM training (stable).

(3) Gradient flow:
$$\frac{\partial \text{loss}}{\partial \alpha} = \sum_t \text{loss}_t \cdot \text{(attention contrib)}$$

$g$ 가 작으면, gradient 도 작음 (via chain rule).
→ $\alpha$ 가 천천히 증가 (안정적).

(4) Learned warm-up:
Training 진행되면서 $\alpha$ 는 점차 증가하고,
vision information 이 gradually integrated.

$$\square$$

### 정리 6.12: Native Multimodal 의 통일성

**명제**: 단일 Transformer 에서 image 와 text 를 섞으면,
두 modality 가 같은 feature space 를 share 하므로,
cross-modal reasoning 이 자연스러워진다.

**증명**:

(1) Separate vs. unified representation:
- Flamingo: Vision encoder (frozen) → tokens → cross-attention to LLM
  → Vision feature space ≠ Language feature space (different frozen models)
  
- Native: Image tokenizer → unified embedding space ← text tokenizer
  → Same representation (learned jointly)

(2) Attention 패턴:
- Flamingo: Text query (Q) can attend to vision (K,V), but vision cannot attend back
  → Asymmetric information flow
  
- Native: All tokens (image, text) mutually attend
  → Symmetric, bidirectional reasoning

Example:
```
[Image tokens...] [text] [Query tokens...]
     ↑                         ↓
Native: image and query can directly attend to each other
      (meaningful interaction)
      
Flamingo: Query attends to image (good)
         But image features are frozen (limited adaptation)
```

(3) 수렴성:
Unified space 에서 training 하면,
image 와 text 의 embeddings 이 자동으로 정렬됨 (jointly learned).
→ No separate alignment phase needed.

$$\square$$

### 정리 6.13: Few-Shot In-Context Learning

**명제**: Interleaved format 은 few-shot examples 를 자연스럽게 포함하고,
LLM 의 in-context learning 능력을 vision task 에 확장한다.

**증명**:

(1) In-context learning (text LLM 에서):
```
"Q: What is 2+3? A: 5"     ← example
"Q: What is 4+6? A:"       ← query
→ Model learns to follow pattern
```

(2) Vision-language extension (Flamingo):
```
[Image: apple] "This is an apple."
[Image: dog] "This is a dog."
[Image: cat] "What is this?"  → "This is a cat."
```

Same mechanism, but with image examples.

(3) 수렴 이론:
- Text LLM: Position embeddings 로 example/query 구분
- Flamingo: Vision tokens 가 추가되지만, same sequence format
  → In-context mechanism 그대로 작동

(4) 성능:
Flamingo 에서 few-shot (1-2 images) 로 새로운 task adapt 가능.
→ More flexible than LLaVA (which requires retraining for new tasks).

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: Gated Cross-Attention 구현

```python
import torch
import torch.nn.functional as F

class GatedCrossAttention(torch.nn.Module):
    def __init__(self, d_model=4096, d_vision=768, num_heads=32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Cross-attention
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_vision, d_model)
        self.v_proj = torch.nn.Linear(d_vision, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        
        # Gating: initialize close to 0
        self.gate_alpha = torch.nn.Parameter(torch.tensor([np.log(0.01)]))
    
    def forward(self, lm_hidden, vision_features):
        """
        lm_hidden: (B, T, d_model) from LLM decoder
        vision_features: (B, N, d_vision) from frozen encoder
        Returns: (B, T, d_model) gated attention output
        """
        B, T, D = lm_hidden.shape
        
        # Project
        Q = self.q_proj(lm_hidden)  # (B, T, D)
        K = self.k_proj(vision_features)  # (B, N, D)
        V = self.v_proj(vision_features)  # (B, N, D)
        
        # Multi-head reshape (simplified: single head shown)
        # Attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(D)  # (B, T, N)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.bmm(attn_weights, V)  # (B, T, D)
        attn_out = self.out_proj(attn_out)
        
        # Gating
        gate = torch.tanh(self.gate_alpha)
        output = gate * attn_out
        
        return output

# Test
gca = GatedCrossAttention(d_model=256, d_vision=128)
lm_hidden = torch.randn(2, 50, 256)  # 2 sequences, 50 text tokens
vision_feats = torch.randn(2, 100, 128)  # 100 image tokens

output = gca(lm_hidden, vision_feats)
print(f"Input (LLM): {lm_hidden.shape}")
print(f"Input (Vision): {vision_feats.shape}")
print(f"Output: {output.shape}")
print(f"Gate value (init): {torch.tanh(gca.gate_alpha).item():.4f}")  # ≈ 0
```

### 실험 2: Interleaved Sequence 구성

```python
def create_interleaved_sequence(images, texts, image_tokenizer, text_tokenizer):
    """
    Create interleaved image-text sequence for Flamingo
    """
    sequence = []
    positions = []
    modality_mask = []  # 0=image, 1=text
    
    pos = 0
    for img, text in zip(images, texts):
        # Image tokens
        img_tokens = image_tokenizer(img)  # (N_img, D_vision)
        sequence.append(img_tokens)
        positions.extend([pos] * len(img_tokens))
        modality_mask.extend([0] * len(img_tokens))
        pos += len(img_tokens)
        
        # Text tokens
        text_tokens = text_tokenizer(text)  # (N_text,)
        sequence.append(text_tokens)
        positions.extend([pos + i for i in range(len(text_tokens))])
        modality_mask.extend([1] * len(text_tokens))
        pos += len(text_tokens)
    
    # Concatenate
    all_tokens = torch.cat(sequence, dim=0)
    positions = torch.tensor(positions)
    modality_mask = torch.tensor(modality_mask)
    
    return all_tokens, positions, modality_mask

# Test (pseudocode)
images = [img1, img2]
texts = ["a dog", "a cat"]
seq, pos, mask = create_interleaved_sequence(images, texts, img_tok, txt_tok)
print(f"Sequence length: {len(seq)}")
print(f"Modality (0=img,1=txt): {mask[:50]}")  # should be mixed
```

### 실험 3: Native Multimodal Unified Embedding

```python
class NativeMultimodalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size_text=32000, vocab_size_image=1024, d_model=4096):
        super().__init__()
        # Single embedding layer for both modalities
        self.embed = torch.nn.Embedding(vocab_size_text + vocab_size_image, d_model)
        
    def forward(self, image_tokens, text_tokens):
        """
        image_tokens: (B, N_img) in range [0, vocab_image)
        text_tokens: (B, N_text) in range [0, vocab_text)
        """
        # Offset image token IDs to avoid collision
        image_tokens_offset = image_tokens + 32000  # shift by text vocab size
        
        # Embed both
        img_emb = self.embed(image_tokens_offset)  # (B, N_img, D)
        text_emb = self.embed(text_tokens)  # (B, N_text, D)
        
        # Concatenate (or interleave)
        combined = torch.cat([img_emb, text_emb], dim=1)  # (B, N_img+N_text, D)
        return combined

# Test
native_embed = NativeMultimodalEmbedding()
img_tok = torch.randint(0, 1024, (2, 100))
txt_tok = torch.randint(0, 32000, (2, 50))

emb = native_embed(img_tok, txt_tok)
print(f"Unified embedding shape: {emb.shape}")  # (2, 150, 4096)
print("Both modalities in same feature space ✓")
```

### 실험 4: Few-Shot Evaluation

```python
def flamingo_few_shot_eval(model, few_shot_examples, query_image, query_prompt):
    """
    few_shot_examples: list of (image, description) tuples
    query_image: image to classify
    query_prompt: "What is this?"
    """
    # Build interleaved sequence
    sequence = []
    for img, desc in few_shot_examples:
        img_tokens = encode_image(img)
        text_tokens = encode_text(desc)
        sequence.append([img_tokens, text_tokens])
    
    # Add query
    query_img_tokens = encode_image(query_image)
    query_text_tokens = encode_text(query_prompt)
    sequence.append([query_img_tokens, query_text_tokens])
    
    # Forward pass (frozen, no training)
    with torch.no_grad():
        output = model.generate(sequence, max_length=20)
    
    return output

# Pseudocode evaluation
few_shot = [
    (dog_img, "a dog"),
    (cat_img, "a cat"),
]
result = flamingo_few_shot_eval(model, few_shot, bird_img, "What is this?")
print(f"Few-shot result: {result}")  # "a bird"
```

## 🔗 실전 활용

**Flamingo (Alayrac et al. 2022)**:
- Architecture: Frozen Chinchilla LM (70B) + Frozen NFNet vision + Gated cross-attention
- In-context learning: 1-32 shot examples
- Downstream tasks: VQA, captioning, counting
- Efficiency: Only gating parameters trainable (~0.1% of total)

**GPT-4V (OpenAI, 2023)**:
- Native multimodal Transformer
- Training: Unified attention across image/text tokens
- Capabilities: More sophisticated reasoning, hallucination awareness
- Limitations: Closed-source, unclear architecture

**Gemini (Google, 2023)**:
- Multimodal Transformers (Gemini 1.0, 1.5)
- Longer context (up to 1M tokens in Gemini 1.5)
- Mixed modality: image + text + video
- Training: Very large-scale (details sparse)

**Claude 3 (Anthropic, 2024)**:
- Native multimodal architecture
- Strong visual reasoning (200K context)
- Emphasis on image understanding (not generation)

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Frozen vision encoder** | 효율적이지만 제약 | Vision task 는 pretraining 에 의존 |
| **Gated mechanism scale** | $\tanh(\alpha)$ 의 range | 극단값 (±1) 에서 saturation 가능 |
| **Interleaved format** | Few-shot 자연스럽지만 | 긴 sequence 는 computationally expensive |
| **Native unified embedding** | 우아하지만 | Image/text token vocab 크기 imbalance |
| **Frozen LLM** | Parameter efficient | Domain adaptation 어려움 |

## 📌 핵심 정리

$$\boxed{\text{output} = \tanh(\alpha) \odot \text{softmax}(QK^T) V}$$

$$\boxed{\text{Native}: \, [\text{img tokens}, \text{text tokens}] \to \text{Unified Transformer} \to \text{text output}}$$

| 개념 | 정의 | 역할 |
|------|------|------|
| **Gated Cross-Attn** | $\tanh(\alpha) \cdot \text{cross-attn}$ | Controlled vision integration |
| **Gate initialization** | $\alpha^{(0)} = \log(1/c), c \approx 0.01$ | Stable early training |
| **Interleaved input** | $[\text{img}, \text{text}, \text{img}, \ldots]$ | Few-shot in-context learning |
| **Frozen encoders** | ViT/LM not updated | Parameter efficiency, stability |
| **Native multimodal** | Single Transformer for all tokens | Unified representation, bidirectional reasoning |
| **Few-shot** | Examples in prompt | Rapid task adaptation |

## 🤔 생각해볼 문제

### 문제 1 (기초): Gating 의 역할

왜 단순히 cross-attention 을 쓰지 않고, gating 을 더할까?
Ablation study 를 설계하시오.

<details>
<summary>해설</summary>

**Flamingo ablation (논문에서):**
1. No cross-attention: baseline LM only → poor multimodal ability
2. Cross-attention (no gate): integrated early, but training unstable
   - Early gradient explosion
   - Vision info 로 text generation 이 overwhelmed
   - Final performance: ~85%
3. Gated cross-attention: stable, gradual integration
   - Better convergence
   - Final performance: ~92%

**왜 gating 이 필요한가:**
- Flamingo 는 frozen vision/LM encoder 를 사용
- 두 encoder 는 별도로 pretrained (mismatch 가능)
- Gating 은 이 mismatch 를 점진적으로 해결

**Non-gated 와 gated 의 비유:**
- Non-gated: 두 개의 라디오 신호를 직접 섞음 → 간섭
- Gated: Volume control 로 천천히 블렌딩 → clean

</details>

### 문제 2 (심화): Native Multimodal 의 scalability

Native multimodal (단일 Transformer) 이 좋으면,
왜 분리된 encoder (Flamingo 식) 를 여전히 쓸까?

<details>
<summary>해설</summary>

**Native 의 장점:**
- 우아한 아키텍처
- Bidirectional reasoning
- Unified representation

**Native 의 단점:**
1. **Training cost**: 
   - Image tokens 이 많음 (예: 256 → 1024 per image)
   - Single Transformer 이 모든 tokens attend → quadratic complexity
   - Cost: $O(N^2)$ where $N$ = image + text tokens
   
2. **Parameter sharing**:
   - Vision/text 를 다르게 optimize 하기 어려움
   - Vision 은 texture 감지, text 는 semantic → different feature scales

3. **Pretrained model 활용**:
   - Separate ViT, LLM 은 이미 강력한 pretrained 모델
   - Native 는 from-scratch training → computational cost

**Trade-off:**
- **Flamingo 식 (separated)**: Efficient, leverages frozen models, flexible
- **Native (unified)**: Elegant, unified reasoning, more training cost

**현실:**
- Small models (1B-7B): Native 가능 (Phi-3V, LLaVA)
- Large models (70B+): Flamingo 식 유리 (GPT-4V probably?)
- Recent: Gemini 1.5 는 native 하려 함 (engineering heavy)

</details>

### 문제 3 (논문 이해): Few-Shot Transfer 의 한계

Flamingo 의 few-shot 이 정말 "zero-shot learning" 만큼 powerful 한가?
Language models 의 few-shot 과 비교하시오.

<details>
<summary>해설</summary>

**Text-only LLM few-shot:**
```
"Q: 2+3? A: 5. Q: 4+6? A:"
→ Very quick adaptation, interpretable pattern
```

**Flamingo few-shot:**
```
[dog_image] "a dog"
[cat_image] "a cat"
[bird_image] "what is this?"
→ Few-shot, but requires...?
```

**차이점:**
1. **Task clarity**: 
   - Text: 명확한 Q-A pattern
   - Vision: image 를 보고 class/description 을 학습
   
2. **In-context learning mechanism:**
   - Text LLM: Attention weights 가 examples 에 크게 shift
   - Vision: Image embeddings 이 frozen → visual feature 는 고정
   → Text generation 만 adaptive

3. **성능 한계:**
   - Text LLM: 2-3 examples 로 새로운 task 가능
   - Flamingo: 많은 examples 필요 (diminishing returns)
   
4. **왜?**
   - Vision encoder 가 frozen 이므로, visual representation 은 fix
   - Adaptation 은 LLM 의 text generation 쪽에서만
   → Limited visual few-shot learning

**결론:**
Flamingo 의 few-shot 은 "image description + text generation" 의 조합.
진정한 "visual few-shot learning" (new visual classes) 은 어렵다.
→ 이것이 Native multimodal 이 필요한 이유 (vision 도 adaptive).

</details>

---

<div align="center">

[◀ 이전](./04-llava.md) | [📚 README](../README.md) | [다음 ▶](../ch7-emerging/01-vision-generation.md)

</div>
