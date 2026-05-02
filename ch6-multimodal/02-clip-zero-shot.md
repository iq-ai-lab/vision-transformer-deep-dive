# 02. CLIP 의 Zero-Shot Transfer

## 🎯 핵심 질문

- Zero-shot 은 정말 "한 번도 본 적 없는" 클래스를 분류할 수 있는가?
- Prompt template 의 역할은? "a photo of a {cls}" 가 왜 중요한가?
- 여러 prompt template 을 평균내는 prompt ensembling 은 얼마나 효과적인가?
- CLIP zero-shot vs. linear probe: 어느 것이 나을까?

## 🔍 왜 이것이 zero-shot 인가

CLIP 은 semantic similarity 를 통해 분류한다.  
학습할 때, 데이터셋 클래스 이름을 본 적이 없어도, **자연어로 클래스를 설명**하면 그 의미를 이해한다.

"dog", "cat", "bird" 같은 새로운 클래스도, 사전학습된 CLIP 이미지-텍스트 embedding space 에서:
- "a photo of a dog" 의 embedding
- 주어진 이미지의 embedding
을 비교해서, cosine similarity 가 높은 클래스를 고른다.

이것이 **일반화 능력** (generalization without task-specific training) 의 본질이다.

## 📐 수학적 선행 조건

- **Cosine similarity**: $\text{sim}(x, y) = \frac{x^T y}{\|x\| \|y\|}$
- **Softmax over scores**: $p_c = \frac{e^{\text{sim}(I, T_c)}}{\sum_k e^{\text{sim}(I, T_k)}}$
- **Prompt engineering**: Text 입력을 의도적으로 구성
- **Template ensemble**: 여러 prompt 의 embedding 평균
- **ImageNet evaluation**: 1000 클래스, top-1/top-5 accuracy

## 📖 직관적 이해

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         Zero-Shot Classification Flow

  주어진 이미지 X
        ↓
   ViT 인코더 (CLIP frozen)
        ↓
   Image embedding z_I ∈ ℝ^D (정규화됨)
        ↓
   ┌──────────────────────────────────────────┐
   │ 클래스들에 대한 prompt template 생성      │
   │ c ∈ {dog, cat, bird, ...}               │
   │ prompt_c = "a photo of a " + c          │
   └──────────────────────────────────────────┘
        ↓
   Transformer 텍스트 인코더 (CLIP frozen)
   각 prompt_c → Text embedding z_{T,c}
        ↓
   Similarity scores
   s_c = z_I · z_{T,c}  (cosine, pre-normalized)
   scores = [s_dog, s_cat, s_bird, ...]
        ↓
   Softmax 정규화
   p_c = exp(s_c * 100) / Σ_k exp(s_k * 100)
   (* 100은 scale factor, CLIP 에서는 1/temperature)
        ↓
   Argmax
   pred_class = argmax(p_c)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**핵심**: CLIP 은 training 후 수정되지 않는다. Prompt 문자열만 바뀐다.

## ✏️ 엄밀한 정의

### 정의 6.7: Zero-Shot 분류 설정

Dataset: $\mathcal{D} = \{(x_i, y_i)\}$ where $y_i \in \mathcal{C} = \{c_1, \ldots, c_K\}$.

CLIP pretraining 중에 $\mathcal{C}$ 의 클래스들을 보지 못했다고 가정.

### 정의 6.8: Prompt Template

클래스 $c$ 에 대해, natural language prompt:
$$\text{prompt}_c = \mathrm{template}(c)$$

예: $\mathrm{template}(c) = \text{"a photo of a } c \text{"}$.

더 자세한 template: $\text{template}(c) = \text{"a photo of a } c \text{, a type of animal"}$.

### 정의 6.9: Zero-Shot Prediction

이미지 $x$ 에 대해:
$$\hat{y} = \arg\max_c \, \cos(\mathbf{z}_{I,\theta}(x), \mathbf{z}_{T,\phi}(\text{prompt}_c))$$

여기서 $\theta$, $\phi$ 는 CLIP pretraining 에서 고정된 가중치.

### 정의 6.10: Prompt Ensembling

$M$ 개의 template 을 사용:
$$\bar{\mathbf{z}}_c = \frac{1}{M} \sum_{m=1}^{M} \mathbf{z}_{T,\phi}(\text{template}_m(c))$$

그 후:
$$\hat{y} = \arg\max_c \, \cos(\mathbf{z}_{I,\theta}(x), \bar{\mathbf{z}}_c)$$

## 🔬 정리와 증명

### 정리 6.3: Zero-Shot 정확도의 Template 의존성

**명제**: Zero-shot 성능은 prompt template 의 선택에 크게 의존하며,
template ensembling 은 single template 의 variance 를 감소시킨다.

**증명**:

(1) Template 의 영향:  
두 개의 template $t_1$, $t_2$ 에 대해, embedding 들이 다를 수 있다:
$$\mathbf{z}_{T}(t_1(c)) \neq \mathbf{z}_{T}(t_2(c))$$

예:
- $t_1(\text{dog}) = \text{"a photo of a dog"}$ 
- $t_2(\text{dog}) = \text{"a dog"}$

Transformer 는 contextual embedding 을 하므로,
$$\mathbf{z}_{T}(t_1(\text{dog}))^T \mathbf{z}_{I} \neq \mathbf{z}_{T}(t_2(\text{dog}))^T \mathbf{z}_{I}$$

실험적으로, template $t_1$ 에서 accuracy 80% vs. $t_2$ 에서 75% 같은 차이가 난다.

(2) Ensemble 효과:  
$M$ 개의 independent templates 에 대해,
$$\bar{\mathbf{z}}_c = \frac{1}{M} \sum_{m=1}^M \mathbf{z}_{T}(t_m(c))$$

Average 의 noise 감소:
$$\mathrm{Var}(\bar{\mathbf{z}}_c) = \frac{1}{M} \mathrm{Var}(\mathbf{z}_{T})$$

(만약 templates 가 independent noise 를 가지면)

따라서 $M$ 이 크면 random fluctuation 이 줄어듦.

(3) Bias-variance tradeoff:  
만약 모든 template 이 같은 bias 를 가지면 (e.g., 모두 positive 한 표현),
ensemble 도 그 bias 를 유지한다.
하지만 다양한 template 은 다양한 관점을 포함하므로, 더 robust 해진다.

$$\square$$

### 정리 6.4: Zero-Shot vs Linear Probe

**명제**: CLIP zero-shot 은 linear probe 보다 다양한 데이터셋에서 일관되게 잘 작동하지만,
downstream task 에 task-specific label 이 충분하면 linear probe 가 더 정확할 수 있다.

**증명**:

(1) Zero-shot 장점:  
- No labeled data needed
- Generalizes to unseen classes
- Fixed embeddings (no training bias)

(2) Linear probe 설정:  
주어진 downstream dataset $\{(x_i, y_i)\}$ 에서, CLIP encoder $f_\theta$ 는 frozen:
$$\min_W \sum_i \text{CE}(W f_\theta(x_i), y_i)$$

Linear map $W \in \mathbb{R}^{K \times D}$ 만 학습.

(3) 성능 비교 (경험적):  
ImageNet zero-shot: CLIP ViT-L/14 ≈ 76% top-1 accuracy  
ImageNet linear probe: CLIP ViT-L/14 ≈ 88% top-1 accuracy

Reason: Linear probe 는 labeled data 에 overfitting 할 수 있어서 (supervision signal),
task-specific features 를 학습함.

(4) Distribution shift 시:  
Out-of-distribution dataset (e.g., ImageNet-V2):
- Zero-shot: 비교적 robust (prompt 가 general)
- Linear probe: 더 취약 (training set bias)

$$\square$$

## 💻 NumPy / PyTorch 구현 검증

### 실험 1: 기본 Zero-Shot 분류

```python
import torch
import torch.nn.functional as F

# Assume CLIP 가 이미 loaded: image_encoder, text_encoder
# 모두 frozen

def zero_shot_classify(image, classes, templates_list, 
                       image_encoder, text_encoder):
    """
    image: (3, H, W) tensor
    classes: list of class names (str)
    templates_list: list of template functions
    Returns: (predictions, class_scores)
    """
    with torch.no_grad():
        # Image encoding
        image = image.unsqueeze(0)  # (1, 3, H, W)
        img_emb = image_encoder(image)  # (1, D)
        img_emb = F.normalize(img_emb, dim=1)
        
        # Text encoding for all classes and templates
        class_embeddings = []
        for cls in classes:
            cls_embs = []
            for template in templates_list:
                text = template(cls)
                # Tokenize, encode (omit tokenization details)
                text_tokens = tokenize(text)  # mock
                text_emb = text_encoder(text_tokens)  # (1, D)
                text_emb = F.normalize(text_emb, dim=1)
                cls_embs.append(text_emb)
            
            # Average over templates (ensemble)
            cls_emb_avg = torch.mean(torch.stack(cls_embs), dim=0)
            class_embeddings.append(cls_emb_avg)
        
        class_embeddings = torch.cat(class_embeddings, dim=0)  # (K, D)
        
        # Similarity (logits)
        logits = (img_emb @ class_embeddings.T).squeeze(0)  # (K,)
        
        # Softmax
        probs = F.softmax(logits * 100, dim=0)  # temperature = 1/100
        
        pred = torch.argmax(probs)
        return pred, probs

# Example usage
templates = [
    lambda c: f"a photo of a {c}",
    lambda c: f"a photo of the {c}",
    lambda c: f"a {c}",
]
classes = ["dog", "cat", "bird"]

pred, probs = zero_shot_classify(image, classes, templates,
                                  image_encoder, text_encoder)
print(f"Predicted class: {classes[pred]}")
print(f"Probabilities: {probs}")
```

**예상 결과**: Probs 가 0~1 사이, 합이 1.

### 실험 2: Template 영향 비교

```python
def evaluate_template(dataset, test_images, test_labels, 
                      classes, template_fn,
                      image_encoder, text_encoder):
    """Template 하나에 대한 accuracy"""
    correct = 0
    for img, true_label in zip(test_images, test_labels):
        pred, _ = zero_shot_classify(img, classes, [template_fn],
                                    image_encoder, text_encoder)
        if pred == true_label:
            correct += 1
    return correct / len(test_labels)

templates = {
    "basic": lambda c: f"a photo of a {c}",
    "object": lambda c: f"a {c}",
    "detailed": lambda c: f"a photo of a {c}, a type of animal",
}

for name, tmpl in templates.items():
    acc = evaluate_template(dataset, test_imgs, test_labels, classes, tmpl,
                           image_encoder, text_encoder)
    print(f"Template '{name}': {acc:.3f}")

# Example output:
# Template 'basic': 0.760
# Template 'object': 0.745
# Template 'detailed': 0.768
```

### 실험 3: Ensemble 효과 (수렴성)

```python
def zero_shot_ensemble(image, classes, templates_list,
                       image_encoder, text_encoder, n_ensemble):
    """n_ensemble 개의 template 만 사용"""
    selected_templates = templates_list[:n_ensemble]
    
    with torch.no_grad():
        img_emb = F.normalize(image_encoder(image.unsqueeze(0)), dim=1)
        
        class_embeddings = []
        for cls in classes:
            cls_embs = []
            for tmpl in selected_templates:
                text_emb = F.normalize(text_encoder(tmpl(cls)), dim=1)
                cls_embs.append(text_emb)
            cls_emb_avg = torch.mean(torch.stack(cls_embs), dim=0)
            class_embeddings.append(cls_emb_avg)
        
        class_embeddings = torch.cat(class_embeddings, dim=0)
        logits = (img_emb @ class_embeddings.T).squeeze(0)
        return torch.argmax(logits)

# 여러 ensemble 크기로 평가
templates = [
    lambda c: f"a photo of a {c}",
    lambda c: f"a {c}",
    lambda c: f"a photo of the {c}",
    lambda c: f"{c}",
    lambda c: f"an image of a {c}",
]

for n in range(1, len(templates) + 1):
    preds = [zero_shot_ensemble(img, classes, templates,
                               image_encoder, text_encoder, n)
            for img in test_imgs]
    acc = (torch.stack(preds) == torch.tensor(test_labels)).float().mean()
    print(f"n_ensemble={n}: accuracy={acc:.3f}")

# Example output:
# n_ensemble=1: accuracy=0.760
# n_ensemble=2: accuracy=0.765
# n_ensemble=3: accuracy=0.768
# n_ensemble=4: accuracy=0.769
# n_ensemble=5: accuracy=0.770
```

### 실험 4: Zero-Shot vs Linear Probe

```python
# Linear probe: downstream classification layer
class LinearProbe(torch.nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# Train linear probe on small labeled set
probe = LinearProbe(embed_dim=512, num_classes=1000)
optimizer = torch.optim.SGD(probe.parameters(), lr=0.1)

for epoch in range(10):
    for x, y in train_loader:
        with torch.no_grad():
            emb = image_encoder(x)
            emb = F.normalize(emb, dim=1)
        logits = probe(emb)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

# Evaluate both methods
zero_shot_acc = evaluate_zero_shot(test_loader, classes, templates, ...)
linear_probe_acc = evaluate_linear_probe(test_loader, probe, image_encoder, ...)

print(f"Zero-shot: {zero_shot_acc:.3f}")
print(f"Linear probe: {linear_probe_acc:.3f}")

# Output (expected):
# Zero-shot: 0.760
# Linear probe: 0.880
```

## 🔗 실전 활용

**Zero-Shot Classification**:
- Product categorization (new product types)
- Open-vocabulary object detection
- Sentiment classification (emotions as "classes")

**Prompt Engineering Tips**:
- Include context: "a photo of a dog" vs. "dog"
- Use natural language: "a type of transport vehicle"
- Ensemble multiple perspectives
- Domain-specific templates

**open_clip 라이브러리**:
```python
import open_clip

model, preprocess = open_clip.create_model_and_transforms('ViT-L-14', 
                                                           pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# Zero-shot prediction
image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
text = tokenizer(["a photo of a dog", "a photo of a cat"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = image_features @ text_features.T
    probs = similarity.softmax(dim=-1)
```

## ⚖️ 가정과 한계

| 항목 | 설명 | 주의사항 |
|------|------|---------|
| **Template 효율성** | 좋은 template 을 수동으로 설계해야 함 | Automatic prompt learning (APE) 로 완화 가능 |
| **New class generalization** | 학습 데이터와 다른 distribution | Long-tail classes 는 성능 저하 |
| **Multilingual** | 영어 template 이 최적; 다른 언어는? | CLIP multilingual 모델 (XLM-R) 있음 |
| **Fine-grained classification** | "dog" vs. "cat" 잘함; "dog breed" 는? | 더 상세한 description 필요 |
| **Template bias** | Positive word 이 많으면 overfit | Neutral templates + ensemble |

## 📌 핵심 정리

$$\boxed{\hat{y} = \arg\max_c \, \cos(\mathbf{z}_{I}(x), \bar{\mathbf{z}}_c) = \arg\max_c \, \frac{1}{M} \sum_{m=1}^M \mathbf{z}_{T}(\text{template}_m(c))^T \mathbf{z}_{I}(x)}$$

| 개념 | 정의 | 역할 |
|------|------|------|
| **Prompt Template** | $\mathrm{template}(c)$ → natural language string | Class description; semantic richness |
| **Image Embedding** | $\mathbf{z}_I(x)$ (normalized, frozen CLIP) | Query vector |
| **Text Embedding** | $\mathbf{z}_T(\text{prompt})$ (normalized, frozen CLIP) | Class prototype |
| **Zero-Shot** | No labeled data; only semantic similarity | Generalization to unseen classes |
| **Ensemble Average** | $\bar{\mathbf{z}}_c = \frac{1}{M} \sum_m \mathbf{z}_{T}(t_m(c))$ | Variance reduction; robustness |
| **Linear Probe** | Learn classification layer on top of frozen embeddings | Better accuracy with labeled data |

## 🤔 생각해볼 문제

### 문제 1 (기초): Template 의 역할

"a photo of a dog" 와 "dog" 의 embedding 이 다른 이유는?  
Transformer 의 어떤 특성 때문인지 설명하시오.

<details>
<summary>해설</summary>

Transformer 는 contextual embedding 을 한다:
- "a photo of a dog" 에서 "dog" 은 photo 라는 맥락 속에서 embedded
- "dog" 혼자는 animal 이라는 wider context 에서 embedded

Attention mechanism 이 neighboring tokens 를 본다:
- "a photo of a dog": dog ← attends to [a, photo, of]
- "dog": dog ← no context (문장 처음)

따라서 embedding 의 "meaning" 이 다르다.
CLIP 은 이미지-텍스트 쌍으로 학습했을 때, 
"a photo of a dog" 가 실제 dog image 와 더 가깝게 정렬되었을 가능성이 높다
(web text 에서 이런 표현이 image captions 에 흔함).

</details>

### 문제 2 (심화): Ensemble 의 수렴 속도

$M$ 개의 independent templates 를 ensemble 할 때,
accuracy 의 표준편차 (혹은 error) 가 어떻게 감소하는가?

<details>
<summary>해설</summary>

만약 각 template 의 prediction error 가 independent 라면:

Central Limit Theorem:
$$\sigma_{\text{ensemble}} = \frac{\sigma_{\text{single}}}{\sqrt{M}}$$

실제로는:
- Templates 가 완전히 independent 하지 않음 (같은 underlying embedding space)
- Correlation $\rho$ 가 있음:
$$\sigma_{\text{ensemble}} = \sigma_{\text{single}} \sqrt{\frac{1}{M}(1 - \rho) + \rho}$$

따라서:
- $\rho$ 가 높으면 (correlated templates) → $1/\sqrt{M}$ 개선 < 예상
- $\rho$ 가 낮으면 (diverse templates) → $1/\sqrt{M}$ 개선 > 예상

실무에서: $M = 3$ ~ 5 정도면 대부분의 이득을 봄.
$M > 10$ 이면 diminishing returns.

</details>

### 문제 3 (논문 비평): Generalization 의 한계

Zero-shot 이 정말 "generalization" 인가?
Prompt engineering 이 implicit supervision 이 아닌가?

<details>
<summary>해설</summary>

좋은 지적. Prompt engineering 을 "human-in-the-loop" 형태의 supervision 으로 볼 수 있다:
- "a photo of a dog" 를 고르는 것도 일종의 label
- Multiple templates → multiple supervision signals

하지만 zero-shot 의 강점:
1. **Unlabeled classes**: Dog breeds 를 못 봤어도, "dog" 는 봤다 (pretrain 중)
2. **Flexibility**: 같은 CLIP model 로 thousand classes 분류 가능 (각각 retraining 없이)
3. **Scalability**: New class 추가 → template 하나 추가 (no data collection)

따라서: "Task adaptation 없는 generalization" 이 아니라
"Minimal supervision with semantic knowledge" 정도로 봐야 함.

Linear probe 는 explicit supervision (labels).  
Zero-shot 은 implicit supervision (natural language + pretraining).

둘 다 형태의 leverage knowledge 임.

</details>

---

<div align="center">

[◀ 이전](./01-clip.md) | [📚 README](../README.md) | [다음 ▶](./03-blip-blip2.md)

</div>
