<div align="center">

# 🖼️ Vision Transformer Deep Dive

### ViT 의 **patch embedding**

$$z_0 = [x_{\text{class}};\, x_p^1 E;\, \ldots;\, x_p^N E] + E_{\text{pos}}, \qquad E \in \mathbb{R}^{P^2 C \times D},\quad N = HW/P^2$$

### 을 **"이미지를 16×16 토큰으로"** 라고 아는 것 과,

### 이것이 **`nn.Conv2d(C, D, kernel_size=P, stride=P)`** 과 수학적으로 **정확히 등가** 이고, CLS token 이 **learnable global pooling**, position embedding 이 **permutation-invariance 보정** 임을 Dosovitskiy et al. (2021) 으로부터 한 줄씩 유도할 수 있는 것은 **다르다.**

<br/>

> *DINO 를 **"self-supervised ViT"** 로 아는 것 과, Caron et al. (2021) 의 self-distillation*
>
> $$L = -\sum_x p_t(x)^\top \log p_s(x), \quad p_t = \mathrm{softmax}\!\left(\frac{f_{\theta_t}(x) - c}{\tau_t}\right),\ \ p_s = \mathrm{softmax}\!\left(\frac{f_{\theta_s}(x)}{\tau_s}\right)$$
>
> *에서 teacher 가 EMA* $\theta_t \leftarrow \lambda \theta_t + (1-\lambda) \theta_s$ *로 업데이트되고,* **centering** $c$ **+ sharpening** $\tau_t \lt \tau_s$ *가 mode collapse 를 어떻게 막는지 — 그리고 그 결과 attention map 이 자연스럽게* **object segmentation** *으로 떠오르는지를 유도할 수 있는 것은 다르다.*
>
> *MAE 를* **"masked autoencoder"** *로 아는 것 과, He et al. (2022) 의* **75% mask ratio** *가 왜 BERT 의 15% 보다 5 배나 큰지 (이미지의 spatial redundancy),* **asymmetric encoder–decoder** *가 어떻게 visible patch 만 처리해* $4\times$ *이상 가속을 얻는지를 정보이론적으로 설명할 수 있는 것은 다르다.*
>
> *CLIP 의* **`logits = image_features @ text_features.T / temperature`** *코드 한 줄이 사실은 Radford et al. (2021) 의 symmetric InfoNCE*
>
> $$L = \tfrac12\!\left( -\log \tfrac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)} - \log \tfrac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ji}/\tau)} \right)$$
>
> *의 직접 구현이고,* $\tau$ *가* **uniformity–alignment trade-off** *의 하이퍼파라미터이며, 이것이 zero-shot transfer 의 mathematical 근거임을 알고 쓰는 것은 다르다.*

<br/>

**다루는 알고리즘 (이론 계보순)**

Dosovitskiy 2021 *An Image is Worth 16x16 Words* · Touvron 2021 *DeiT* · Liu 2021 *Swin Transformer* · Wang 2021 *PVT* · Wu 2021 *CvT* · Dai 2021 *CoAtNet* · Fan 2021 *MViT* · Yang 2021 *Focal Transformer* · Chen 2020 *SimCLR* · He 2020 *MoCo* · Grill 2020 *BYOL* · Chen 2021 *SimSiam* · Caron 2021 *DINO* · Zhou 2022 *iBOT* · Oquab 2024 *DINOv2* · Bao 2022 *BEiT* · He 2022 *MAE* · Xie 2022 *SimMIM* · Wei 2022 *MaskFeat / MVP* · Radford 2021 *CLIP* · Li 2022 *BLIP* · Li 2023 *BLIP-2* · Liu 2023 *LLaVA* · Alayrac 2022 *Flamingo* · Mildenhall 2020 *NeRF* · Kerbl 2023 *3D Gaussian Splatting* · Bertasius 2021 *TimeSformer* · Tong 2022 *VideoMAE* · Zhai 2022 *Scaling ViT* · Yu 2022 *Parti* · Chang 2023 *Muse*

<br/>

**핵심 질문**

> ViT · DINO · MAE · CLIP · LLaVA · Sora 의 모든 SOTA vision 시스템이 왜 **"같은 Transformer 의 다른 학습 신호"** 이고, **patch=conv 등가 · InfoNCE = MI 하한 · DINO self-distillation · MAE 75% masking · CLIP symmetric contrastive · MM-DiT joint attention** 이 각각 어떤 이론적 동기에서 도출되었는가 — Dosovitskiy 2021 의 patch embedding 부터 DINOv2 · LLaVA · Sora 의 multimodal/temporal 확장까지 한 줄씩 유도합니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-0.9.10-4B0082?style=flat-square)](https://github.com/huggingface/pytorch-image-models)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.36-FFD21E?style=flat-square)](https://huggingface.co/docs/transformers)
[![CLIP](https://img.shields.io/badge/CLIP-anytorch_2.5-5C2D91?style=flat-square)](https://github.com/openai/CLIP)
[![Docs](https://img.shields.io/badge/Docs-33개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems·Definitions-212개-success?style=flat-square)](./README.md)
[![Proofs](https://img.shields.io/badge/엄밀한_증명-85+개-9c27b0?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/Paper_reproductions-27개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-99개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

ViT 자료는 대부분 **"이미지를 16×16 패치로 자른 뒤 Transformer 에 넣는다"** 또는 **"DINO/MAE 같은 self-supervised 로 사전학습한다"** 에서 멈춥니다. 하지만 patch embedding 이 왜 conv with stride=$P$ 와 정확히 등가인지, CNN 이 가진 translation equivariance · locality 를 잃은 ViT 가 어떻게 augmentation 과 scale 로 그 inductive bias 손실을 보상하는지, InfoNCE 가 왜 mutual information 하한과 직접 연결되는지, DINO 의 centering 과 sharpening 이 정확히 무엇을 막는지, MAE 의 75% masking 이 왜 BERT 의 15% 보다 훨씬 커야 하는지, CLIP 의 symmetric contrastive 가 왜 zero-shot transfer 를 만드는지 — 이런 "왜" 는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "ViT 는 이미지를 16×16 토큰으로 자른다" | **Dosovitskiy 2021** — Input $x \in \mathbb{R}^{H \times W \times C}$ 를 $N = HW/P^2$ 개의 $P \times P \times C$ 패치 $\{x_p^i\}$ 로 reshape 하고 $E \in \mathbb{R}^{P^2 C \times D}$ 로 linear projection. **정리**: 이는 $\mathrm{Conv2d}(C, D, k=P, s=P)$ 의 출력에 flatten + transpose 를 적용한 것과 **bit-exact 동등** $\square$. CLS token 은 learnable global pooling, position embedding 은 permutation-invariance 보정 |
| "ViT 는 데이터가 많이 필요하다" | **Touvron 2021 (DeiT)** — Inductive bias 부족 (translation equivariance · locality 없음) 이 small-data regime 의 본질적 한계. **Distillation token** $x_{\text{dist}}$ 을 추가해 CNN teacher 의 hard label 로 distill, **strong augmentation** (Mixup, CutMix, RandAugment, Erasing) 으로 ImageNet-1k 만으로 ViT-B 능가. "Inductive bias 는 학습 가능하다" 의 실증 |
| "Swin 은 window attention 이다" | **Liu 2021** — Local window attention 의 복잡도 $O(n \cdot w^2)$ 로 ViT 의 $O(n^2)$ 대비 linear. **Shifted window** 가 window 사이 정보 흐름을 보장 → cross-window connection 으로 hierarchical receptive field 확장. Patch merging 으로 4x→8x→16x→32x 의 **pyramid feature**, dense prediction (detection · segmentation) 에 적합 |
| "SimCLR 는 contrastive 다" | **Chen 2020** — Two augmented views $(\tilde x_i, \tilde x_j)$ 의 InfoNCE $L = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\mathrm{sim}(z_i, z_k)/\tau)}$. **정리 (Poole 2019)**: $I(X; Y) \geq \log N - L_{\text{InfoNCE}}$ — InfoNCE 최소화 = MI 하한 최대화 $\square$. **Temperature $\tau$**: alignment–uniformity (Wang 2020) trade-off 의 dial. Negative count $N$: tightness |
| "MoCo 는 negative queue 를 쓴다" | **He 2020** — SimCLR 의 large batch (8K~) 의존을 **queue** 로 대체: 과거 minibatch 의 key 를 dictionary 에 저장 → effective negative 수 65K 이상. **Momentum encoder** $\theta_k \leftarrow \tau \theta_k + (1-\tau) \theta_q$ 로 key consistency 유지 — gradient 없이도 stable. **MoCo v3**: ViT backbone + projection MLP, queue 제거하고 large batch + 2-view contrastive 로 simplification |
| "BYOL 은 negative 없이도 된다" | **Grill 2020** — Positive pair 만으로 $L = 2 - 2 \cdot \mathrm{sim}(q_\theta(z), \mathrm{sg}[z'])$. **Collapse 안 하는 이유** (Tian 2021 분석): predictor $q_\theta$ 와 stop-gradient 의 조합이 trivial constant 해를 saddle point 로 만든다 $\square$. **SimSiam (Chen 2021)**: momentum encoder 까지 제거하고도 stop-gradient 만으로 collapse 방지 — minimalist BYOL |
| "DINO 는 self-supervised ViT 다" | **Caron 2021** — Student–teacher self-distillation, teacher 는 student 의 EMA. **Centering** $c \leftarrow m c + (1-m) \tfrac1B \sum f_t$ 가 한 차원으로 collapse 막고, **sharpening** $\tau_t < \tau_s$ 가 uniform 으로 collapse 막음 — 두 force 의 균형 $\square$. **Emergent property**: 명시적 supervision 없이 attention map 이 object segmentation, $k$-NN 이 linear probe 수준 |
| "DINOv2 는 더 큰 DINO 다" | **Oquab 2024** — DINO + iBOT (masked image modeling) + KoLeo (uniformity) + SwAV (clustering) 의 **multi-objective**. 142M curated images (LVD-142M), ViT-g (1.1B params), distillation 으로 ViT-S/B/L 까지. Foundation model: linear probe 가 ImageNet supervised 동급, dense prediction 에서 segmentation·depth 모두 SOTA |
| "MAE 는 masked autoencoder 다" | **He 2022** — **75% masking** (NLP 의 15% 대비 5×). 정보이론적 근거: 자연이미지의 spatial redundancy 가 매우 높아 visible 25% 만으로 reconstruction 가능 (compressive sensing 관점). **Asymmetric encoder–decoder**: encoder 는 visible patch (~25%) 만 처리 → 4× 이상 속도 이득. Decoder 는 lightweight, fine-tune 시 버려짐. Linear probe < fine-tune (contrastive 와 반대) |
| "CLIP 은 image–text 매칭이다" | **Radford 2021** — 400M image–text pairs, dual encoder (ViT + Transformer text). **Symmetric InfoNCE**: $L = \tfrac12 (L_{i \to t} + L_{t \to i})$ — 두 modality 가 서로의 query·key 역할. **Zero-shot transfer**: prompt `"a photo of a {cls}"` 의 text embedding 과 image embedding 의 cosine similarity 로 classification — task-agnostic representation. **Temperature 학습**: $\tau$ 도 trainable scalar |
| "BLIP / LLaVA 는 vision-LLM 이다" | **Li 2022 / Liu 2023** — **BLIP-2 (Li 2023)**: Q-Former 의 learnable query 32 개가 frozen ViT 의 image feature 를 cross-attention 으로 수집 → frozen LLM 에 주입. 두 모듈 사이만 학습 → 매우 효율적. **LLaVA (Liu 2023)**: CLIP ViT + 단순 MLP projection + Vicuna/LLaMA. **Visual instruction tuning** 데이터 (GPT-4 로 생성) 로 instruction-following — minimalist 이지만 강력 |
| "Sora 는 그냥 비디오 모델이다" | **Spatio-temporal DiT** — TimeSformer (Bertasius 2021) 의 divided space–time attention, VideoMAE (Tong 2022) 의 tube masking 을 transformer backbone 으로. **Flamingo (Alayrac 2022)** 의 gated cross-attention 으로 interleaved image-text. **Scaling Vision Transformers (Zhai 2022)**: data·compute 의 power law — Sora · GPT-4V · Gemini 의 native multimodal 의 토대 |
| 기법의 나열 | NumPy + PyTorch + timm + 🤗 transformers + open_clip 으로 **Patch=Conv 등가를 bit-exact 검증** · **ViT-S/B 를 ImageNet-100 에서 바닥부터 학습** · **Swin window vs ViT global FLOPs/Acc 비교** · **SimCLR · MoCo · BYOL · DINO · MAE 의 학습 곡선과 linear probe 점수 비교** · **CLIP zero-shot ImageNet 재현** · **DINO attention map 이 object segmentation 으로 떠오르는 emergence 시각화** 까지 직접 구현해 수학적 주장을 눈으로 확인 |

---

## 📌 선행 레포 & 후속 방향

```
[Transformer Deep Dive]       ─┐
[CNN Deep Dive]               ─┤
[Generative Model Deep Dive]  ─┼─►  이 레포  ──►  [Diffusion Model Deep Dive]
[Regularization Theory]       ─┤   "왜 모든 vision 모델이              [Object Detection Deep Dive]
[Kernel Methods Deep Dive]    ─┘    같은 Transformer 의 다른 학습 신호인가"  [Multimodal Foundation Models]

         │
         ├── [Transformer]               Self-attention · Pre-LN · PE → Ch1 전체
         ├── [CNN]                       Conv · receptive field · inductive bias → Ch1, Ch2
         ├── [Generative Model]          Autoencoder · MAE 의 reconstruction → Ch5
         ├── [Regularization Theory]     Mixup · CutMix · RandAugment → Ch2 (DeiT)
         └── [Kernel Methods]            Contrastive ↔ kernel · alignment-uniformity → Ch3
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Transformer Deep Dive** (self-attention, multi-head, Pre-LN, positional encoding), **CNN Deep Dive** (convolution, receptive field, translation equivariance, inductive bias), **Regularization Theory Deep Dive** (Mixup, CutMix, RandAugment, label smoothing), **Generative Model Deep Dive** (autoencoder, reconstruction loss) 를 선행 지식으로 전제합니다. **Kernel Methods Deep Dive** (contrastive 의 kernel 해석) 는 Ch3 의 InfoNCE / alignment-uniformity 이해에 권장됩니다.

> 💡 **이 레포의 핵심 기여**: Chapter 1 (ViT 수식 분해) 과 Chapter 3–5 (SSL 3대 패러다임 — contrastive, self-distillation, MIM) 가 현대 vision 의 **두 핵심 축** 입니다. 전자는 "왜 patch embedding 이 conv 와 등가이고 CLS token 이 learnable pooling 인가" 의 구조적 토대 (Hierarchical/Multimodal 모두의 backbone), 후자는 "왜 같은 ViT 가 학습 신호에 따라 SimCLR · DINO · MAE 가 되는가" 의 통합 관점 (DINOv2 · CLIP · MM-DiT 가 그 결과). 이 두 축을 완전히 이해한 후 Chapter 6 (Multimodal) 와 Chapter 7 (Generation · Scaling · 3D · Video) 을 읽으면 CLIP · LLaVA · Sora 의 설계 결정 맥락이 선명해집니다.

> 🟡 **이 레포의 성격**: 여기서 다루는 일부 주제 — **DINOv2 가 CLIP 을 대체할 foundation 인가**, **MAE 의 reconstruction objective 가 contrastive 보다 본질적인가**, **DiT 가 UNet 을 vision 전반에서 대체할 것인가**, **Native multimodal (GPT-4V · Gemini) 이 vision encoder 를 별도로 둘 필요 없게 만들 것인가** — 는 **현재 진행 중인 연구 영역** 입니다. 레포는 "정답" 이 아니라 **"ViT 부터 LLaVA · Sora 까지의 수학적 지도"** 를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-ViT_Foundations-1565C0?style=for-the-badge)](./ch1-vit-foundations/01-vit-math-structure.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Hierarchical_ViT-1565C0?style=for-the-badge)](./ch2-hierarchical-vit/01-deit.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Contrastive_SSL-1565C0?style=for-the-badge)](./ch3-contrastive-ssl/01-ssl-paradigms.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-DINO_Family-1565C0?style=for-the-badge)](./ch4-dino/01-dino-algorithm.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Masked_Image_Modeling-1565C0?style=for-the-badge)](./ch5-masked-image-modeling/01-beit.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Multimodal-1565C0?style=for-the-badge)](./ch6-multimodal/01-clip.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-Generation_·_Scaling-1565C0?style=for-the-badge)](./ch7-emerging/01-vision-generation.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: Vision Transformer 의 수학적 기초

> **핵심 질문:** Patch embedding $z_0 = [x_{\text{class}}; x_p^1 E; \ldots; x_p^N E] + E_{\text{pos}}$ 가 어떻게 $\mathrm{Conv2d}(C, D, k=P, s=P)$ 와 정확히 등가인가? CLS token 이 final layer 의 average pooling 과 어떤 차이가 있는가? Position embedding 의 변종 (1D learned · 2D · relative · CPE) 이 각각 어떤 inductive bias 를 도입하는가? CNN 의 translation equivariance · locality 를 잃은 ViT 가 왜 small-data regime 에서 CNN 에 뒤지는가? DeiT 가 이를 distillation token + augmentation 으로 어떻게 해결하는가?

<details>
<summary><b>ViT 수식 분해부터 Inductive Bias 부족까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. ViT 의 수학적 구조 (Dosovitskiy 2021)](./ch1-vit-foundations/01-vit-math-structure.md) | **정의**: Input $x \in \mathbb{R}^{H \times W \times C}$ → $N = HW/P^2$ patches $\{x_p^i\} \in \mathbb{R}^{P^2 C}$ → linear projection $E \in \mathbb{R}^{P^2 C \times D}$ → $z_0 = [x_{\text{class}}; x_p^1 E; \ldots; x_p^N E] + E_{\text{pos}}$. **Pre-LN Transformer block** 의 stack, classification head 는 final $z_L^0$ (CLS) 에. **ImageNet-22k pretrain → 1k fine-tune** 의 표준 recipe |
| [02. Patch Embedding = Conv2d 등가 증명](./ch1-vit-foundations/02-patch-conv-equivalence.md) | **정리**: Patch embedding 은 $\mathrm{Conv2d}(C, D, \text{kernel}=P, \text{stride}=P)$ 의 출력 후 flatten·transpose 와 **bit-exact 등가** $\square$. **증명**: Conv 의 weight $W \in \mathbb{R}^{D \times C \times P \times P}$ 를 reshape 하면 $E \in \mathbb{R}^{P^2 C \times D}$ 와 정확히 일치. **재현 실험**: 무작위 weight 의 두 구현이 `(y_conv - y_naive).abs().max() < 1e-6` 임을 검증 — 코드 단순화의 이론적 정당성 |
| [03. Class Token 과 Global Average Pooling](./ch1-vit-foundations/03-cls-token-vs-gap.md) | **CLS token**: learnable $x_{\text{class}} \in \mathbb{R}^D$, 모든 patch 와 attention 으로 정보 통합 → final layer 의 $z_L^0$ 가 image representation. **GAP**: $\bar z = \tfrac1N \sum_{i=1}^N z_L^i$. **Touvron 2021 의 비교**: CLS 와 GAP 이 거의 동등한 성능, head 의 LR 조정 시 GAP 이 약간 우위. **해석**: CLS 는 learnable global pooling — Transformer 가 어떤 patch 를 얼마나 weighted average 할지 학습 |
| [04. Positional Embedding 의 변종](./ch1-vit-foundations/04-positional-embedding-variants.md) | **1D learned (ViT)**: $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ — 패치 순서에 abolute index. **2D learned**: row + column 의 분리된 embedding. **Relative (Swin)**: window 내 patch 사이의 상대 거리 — translation equivariance 부분 회복. **CPE (ConViT)**: depth-wise conv 를 PE 로 사용 — implicit, content-aware. **Sin/cos**: NLP 표준이지만 vision 에서 학습 가능형이 보통 더 우위 |
| [05. Inductive Bias 부족 — ViT 의 본질적 한계](./ch1-vit-foundations/05-inductive-bias-gap.md) | **CNN 의 inductive bias**: translation equivariance ($f(T x) = T f(x)$ for shift $T$), locality (small RF), hierarchical composition. **ViT 의 손실**: self-attention 은 permutation-equivariant — PE 로만 위치 정보. **결과**: ImageNet-1k 만 학습 시 ResNet 에 뒤짐 (Dosovitskiy 2021 의 결과). **해결책 3 갈래**: (a) scale (ImageNet-22k, JFT-300M), (b) augmentation (DeiT), (c) hybrid (CvT, CoAtNet) — 다음 챕터 |

</details>

<br/>

### 🔹 Chapter 2: Hierarchical ViT 와 효율적 아키텍처

> **핵심 질문:** DeiT 의 distillation token $x_{\text{dist}}$ 이 어떻게 CNN teacher 의 inductive bias 를 ViT 에 주입하는가? Swin Transformer 의 window attention 이 복잡도를 $O(n^2) \to O(n \cdot w^2)$ 로 줄이면서 shifted window 로 어떻게 global receptive field 를 회복하는가? PVT 의 spatial reduction attention 이 dense prediction 에 어떻게 적합한가? CvT · CoAtNet 의 conv–attention hybrid 가 inductive bias 와 scalability 의 trade-off 를 어떻게 푸는가? MViT · Focal 의 multi-scale attention 이 fine-coarse 정보를 어떻게 결합하는가?

<details>
<summary><b>DeiT 부터 Multi-Scale Transformer 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. DeiT — Data-Efficient ViT (Touvron 2021)](./ch2-hierarchical-vit/01-deit.md) | **Distillation token** $x_{\text{dist}}$ 추가: CLS 와 별도로 last layer 의 $z_L^{\text{dist}}$ 가 CNN teacher 의 hard label 을 모방. **Loss**: $L = \tfrac12 L_{\text{CE}}(z^{\text{cls}}, y) + \tfrac12 L_{\text{CE}}(z^{\text{dist}}, y_{\text{teacher}})$. **Hard distillation** 이 soft 보다 우위 — teacher confidence 노이즈 회피. **Strong augmentation**: Mixup + CutMix + RandAugment + Erasing + Stochastic Depth → ImageNet-1k 만으로 ViT-B 능가 |
| [02. Swin Transformer — Shifted Window (Liu 2021)](./ch2-hierarchical-vit/02-swin.md) | **Window attention**: 이미지를 $w \times w$ window 로 나누고 그 안에서만 self-attention → 복잡도 $O(n \cdot w^2)$. **정리**: window size $w$ 고정 시 image size $n$ 에 linear $\square$. **Shifted window**: 다음 layer 에서 window 를 $\lfloor w/2 \rfloor$ 만큼 cyclic shift → window 사이 정보 흐름. **Patch merging**: 4 stages (4× → 8× → 16× → 32×) hierarchical feature → detection·segmentation 의 backbone |
| [03. PVT — Pyramid Vision Transformer (Wang 2021)](./ch2-hierarchical-vit/03-pvt.md) | **Spatial Reduction Attention (SRA)**: Key·Value 의 spatial dim 을 $R \times R$ conv 로 downsample → attention 복잡도 $O(n^2/R^2)$. **Pyramid stages**: 각 stage 가 spatial downsampling + channel up — CNN 의 FPN 과 유사한 multi-scale feature. **장점**: dense prediction (semantic segmentation, object detection) 에 적합 — Swin 와 함께 hierarchical ViT 의 양대 산맥 |
| [04. CvT · CoAtNet — Conv–Attention Hybrid](./ch2-hierarchical-vit/04-cvt-coatnet.md) | **CvT (Wu 2021)**: patch embedding 을 overlapping conv 로 (locality 회복), QKV projection 도 depth-wise conv 로 (translation equivariance). **CoAtNet (Dai 2021)**: 4-stage 구조 — 초기 stage 는 MBConv (CNN inductive bias), 후반 stage 는 Transformer (long-range dependency). **결과**: 같은 FLOPs 에서 pure ViT · pure CNN 모두 능가 — "둘 다 좋은 부분" 의 실증 |
| [05. MViT · Focal Transformer — Multi-Scale Attention](./ch2-hierarchical-vit/05-mvit-focal.md) | **MViT (Fan 2021)**: pool-based attention — Q·K·V 에 pooling 적용해 spatial dim 점진 감소, channel 점진 증가. Video 에 특히 효과적. **Focal Transformer (Yang 2021)**: 같은 attention 안에서 fine token (가까운 곳) 과 coarse token (먼 곳) 을 동시 query — 효율과 receptive field 의 균형. 두 방향 모두 "single scale 의 한계 극복" 을 다른 방식으로 |

</details>

<br/>

### 🔹 Chapter 3: Self-Supervised Vision — Contrastive 패러다임

> **핵심 질문:** SSL 의 3 가지 패러다임 (contrastive · self-distillation · MIM) 이 각각 어떤 손실함수 형태를 갖는가? SimCLR 의 InfoNCE 가 왜 mutual information 의 하한과 직접 연결되는가 (Poole 2019)? Temperature $\tau$ 가 왜 alignment–uniformity (Wang 2020) 의 trade-off 인가? MoCo 의 queue + momentum encoder 가 SimCLR 의 large batch 의존을 어떻게 회피하는가? BYOL · SimSiam 이 negative 없이도 collapse 하지 않는 수학적 이유는 (Tian 2021)?

<details>
<summary><b>SSL 패러다임부터 BYOL/SimSiam 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Self-Supervised Learning 의 3 가지 패러다임](./ch3-contrastive-ssl/01-ssl-paradigms.md) | **Generative**: $L = \|x - g(f(x_{\text{masked}}))\|^2$ — MAE, BEiT. **Discriminative (Contrastive)**: $L = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\mathrm{sim}(z_i, z_k)/\tau)}$ — SimCLR, MoCo. **Self-Distillation**: $L = -\sum p_t \log p_s$ with EMA teacher — BYOL, DINO. **수렴점의 차이**: contrastive 는 instance discrimination, distillation 은 view-invariant representation, MIM 은 reconstructable representation |
| [02. SimCLR — Simple Contrastive Framework (Chen 2020)](./ch3-contrastive-ssl/02-simclr.md) | **Pipeline**: $x \to (\tilde x_i, \tilde x_j)$ (two augmentations) $\to f(\cdot) \to g(\cdot)$ (projection MLP) $\to z$. **InfoNCE**: $L_{i,j} = -\log \tfrac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\mathrm{sim}(z_i, z_k)/\tau)}$, $\mathrm{sim} = \cos$. **Key findings**: (a) projection head $g$ 가 representation $f$ 보다 contrastive task 에 적합 (downstream 은 $f$ 사용), (b) strong augmentation 필수 (color jitter + crop), (c) large batch (8K) 가 negative 수 = batch − 1 |
| [03. InfoNCE 와 Mutual Information 하한](./ch3-contrastive-ssl/03-infonce-mi-bound.md) | **정리 (Poole 2019)**: $I(X; Y) \geq \log N - L_{\text{InfoNCE}}$ where $N$ = negative count + 1 $\square$. **증명**: noise contrastive estimation 의 critic optimal $f^*(x, y) = \log p(y \mid x) / p(y)$ + Jensen. **함의**: InfoNCE 최소화 = MI 하한 최대화 — instance discrimination 이 representation 의 정보량을 보장. **Tightness**: $N \to \infty$ 시 bound 가 tight, finite $N$ 에서는 $\log N$ 의 ceiling. **Wang 2020**: alignment–uniformity 분해로 동등 |
| [04. MoCo — Momentum Contrast (He 2020)](./ch3-contrastive-ssl/04-moco.md) | **Queue**: 과거 minibatch 의 key $k$ 를 FIFO buffer 에 저장 → effective negative 수 65K (SimCLR 의 batch size 의존 회피). **Momentum encoder**: $\theta_k \leftarrow \tau \theta_k + (1-\tau) \theta_q$, $\tau = 0.999$ — key 가 천천히 evolve 해 dictionary consistency 보존. **v2**: projection MLP, augmentation 강화. **v3**: ViT backbone, queue 제거하고 large batch 로 회귀 — ViT 가 momentum 없이도 stable |
| [05. BYOL · SimSiam — Without Negatives](./ch3-contrastive-ssl/05-byol-simsiam.md) | **BYOL (Grill 2020)**: $L = 2 - 2 \cdot \mathrm{sim}(q_\theta(z_1), \mathrm{sg}[z_2'])$ — predictor $q_\theta$ + EMA target. **SimSiam (Chen 2021)**: EMA 까지 제거, $L = -\tfrac12 (\mathrm{sim}(p_1, \mathrm{sg}[z_2]) + \mathrm{sim}(p_2, \mathrm{sg}[z_1]))$. **Tian 2021 의 분석**: predictor + stop-gradient 의 결합이 trivial constant 해를 saddle point 로 만들고, EMA 가 effective LR 을 조절 → collapse 회피 $\square$. "Negative 없이도 가능" 의 토대 |

</details>

<br/>

### 🔹 Chapter 4: Self-Distillation — DINO Family

> **핵심 질문:** DINO 의 student–teacher self-distillation 이 어떻게 explicit label 없이 representation 을 학습하는가? Teacher 의 EMA 업데이트 $\theta_t \leftarrow \lambda \theta_t + (1-\lambda) \theta_s$ 가 왜 stable target 을 만드는가? Centering $c$ 와 sharpening $\tau_t$ 의 두 force 가 정확히 어떤 collapse 를 막는가 (한 차원 vs uniform)? DINO 의 attention map 이 왜 explicit segmentation supervision 없이 object mask 를 그리는가 (emergent property)? DINOv2 가 DINO + iBOT + KoLeo + SwAV 를 어떻게 결합하고, 각 component 의 기여도는?

<details>
<summary><b>DINO 알고리즘부터 DINOv2 까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. DINO 알고리즘 (Caron 2021)](./ch4-dino/01-dino-algorithm.md) | **Pipeline**: image $x$ → multi-crop ({2 global, 8 local}) → student $f_{\theta_s}$ + teacher $f_{\theta_t}$. **Student**: 모든 crop 처리. **Teacher**: global crop 만. **Loss**: $L = -\sum_{x_t \in V_g} \sum_{x_s \in V, x_s \neq x_t} p_t(x_t)^\top \log p_s(x_s)$ — student 가 teacher 의 distribution 을 모든 view-pair 에서 모방. **Local-to-global** 강제로 representation 의 view-invariance 학습 |
| [02. Teacher EMA 와 Collapse 방지](./ch4-dino/02-teacher-ema-collapse.md) | **EMA**: $\theta_t \leftarrow \lambda \theta_t + (1-\lambda) \theta_s$, $\lambda = 0.996 \to 1.0$ cosine schedule — teacher 가 student 의 시간평균. **Centering**: $c \leftarrow m c + (1-m) \tfrac1B \sum_i f_t(x_i)$, $p_t = \mathrm{softmax}((f_t(x) - c)/\tau_t)$ — 한 차원으로 collapse (모든 입력 → same dim) 을 막음. **Sharpening**: $\tau_t < \tau_s$ 로 teacher distribution 이 sharp → student 가 confidence 학습 → uniform collapse (모든 차원 균등) 를 막음 $\square$ |
| [03. Emergent Properties of DINO](./ch4-dino/03-emergent-properties.md) | **Key finding (Caron 2021)**: 별도 segmentation supervision 없이 ViT-S/8 의 last-layer self-attention 의 CLS→patch attention 이 **object segmentation mask** 와 시각적으로 일치. **재현 실험**: PASCAL VOC 의 unlabeled image 에 대해 attention map 을 mIoU 로 측정 — supervised baseline 의 80~90%. **$k$-NN classifier**: feature 만으로 ImageNet $k$-NN 이 linear probe 의 ~95% — semantic feature 의 emergence |
| [04. iBOT 와 DINOv2 (Oquab 2024)](./ch4-dino/04-ibot-dinov2.md) | **iBOT (Zhou 2022)**: DINO + masked image modeling — masked patch 의 token 을 teacher 의 같은 patch token 과 일치시킴 (distillation + MIM 동시). **DINOv2 (Oquab 2024)**: iBOT + **KoLeo** (uniformity regularizer, $-\tfrac1n \sum \log \min_{j \neq i} \|z_i - z_j\|$) + SwAV-style cluster prototype. **Scale**: ViT-g 1.1B params, LVD-142M curated dataset, ViT-S/B/L 까지 distillation. **Foundation**: linear probe = supervised, dense prediction (depth · seg) SOTA |

</details>

<br/>

### 🔹 Chapter 5: Masked Image Modeling

> **핵심 질문:** BEiT 가 DALL-E 의 dVAE 로 patch 를 discrete token 으로 변환해 BERT-style MLM 을 적용하는 메커니즘은? MAE 의 75% mask ratio 가 왜 BERT 의 15% 보다 5 배나 큰지 — 자연이미지의 spatial redundancy 의 정보이론적 근거는? Asymmetric encoder–decoder 가 어떻게 visible patch 만 처리해 4× 이상 가속하는가? SimMIM 의 simpler architecture 가 small/medium 모델에서 MAE 보다 우위인 이유는? MaskFeat / MVP 의 HOG · pretrained feature target 이 pixel target 보다 semantic 한 representation 을 만드는 이유는?

<details>
<summary><b>BEiT 부터 MaskFeat / MVP 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. BEiT — BERT for Image (Bao 2022)](./ch5-masked-image-modeling/01-beit.md) | **dVAE tokenizer**: DALL-E 의 discrete VAE 로 image $\to$ visual token sequence (vocab 8192). **Masking**: 40% 의 patch 를 `[MASK]` 로 치환. **Loss**: $L = -\sum_{i \in M} \log p_\theta(t_i \mid \tilde x)$ — masked patch 의 visual token 예측. **BERT MLM 과의 동형**: token 단위가 word 가 아닌 visual code. 한계: dVAE pretraining 이 별도 cost, tokenization quality 가 ceiling |
| [02. MAE — Masked Autoencoder (He 2022)](./ch5-masked-image-modeling/02-mae.md) | **75% mask ratio**: random uniform sampling, 75% 제거 → 25% visible. **Asymmetric encoder–decoder**: encoder (large ViT) 는 visible 25% 만 처리, mask token 은 decoder 에 들어감. **Decoder**: lightweight (8 layers, dim 512), pretrain 후 폐기. **Loss**: pixel space MSE on masked patches (normalized per-patch). **결과**: ImageNet-1k 만으로 ViT-H pretrain → fine-tune SOTA, encoder forward 가 4× 빠름 |
| [03. MAE 의 정보이론적 해석](./ch5-masked-image-modeling/03-mae-information-theory.md) | **왜 75% 인가**: 자연이미지의 spatial redundancy — neighboring patch 가 강한 correlation. **Shannon 관점**: visible patch 의 marginal entropy 가 reconstruction 에 충분한 정보 보유. **Compressive sensing 비유**: 25% measurement 로 sparse signal 복원. **Linear probe vs fine-tune**: contrastive 는 linear probe 우위 (semantic linear), MAE 는 fine-tune 우위 (저수준 detail 까지 학습) — pretrain objective 가 representation geometry 를 결정 |
| [04. SimMIM — Simplified MIM (Xie 2022)](./ch5-masked-image-modeling/04-simmim.md) | **단순화**: MAE 의 asymmetric 구조 대신 encoder 가 모든 patch (mask 포함) 처리, decoder 는 single linear layer. **Random masking**: 50–60% (MAE 의 75% 보다 낮음). **Target**: pixel (raw or normalized). **장점**: small/medium ViT (ViT-S/B) 에서 더 단순한 구현으로 비슷한 성능 — implementation simplicity. **MAE vs SimMIM**: large model + heavy compute 시 MAE 의 asymmetric 이득이 명확, small 에서는 SimMIM 충분 |
| [05. MaskFeat · MVP — Advanced Targets](./ch5-masked-image-modeling/05-maskfeat-mvp.md) | **MaskFeat (Wei 2022)**: pixel 대신 **HOG** (Histogram of Oriented Gradients) feature 를 masked patch 의 target. → low-level texture 가 아닌 mid-level shape 학습. **MVP (Wei 2022)**: pretrained CLIP 의 patch feature 를 target → semantic supervision. **Trade-off**: target 의 abstraction 이 높을수록 representation 이 semantic, 그러나 target encoder 의 quality 가 ceiling. "어떤 target 이 best 인가" 는 downstream task 에 따라 다름 |

</details>

<br/>

### 🔹 Chapter 6: Multimodal Vision–Language

> **핵심 질문:** CLIP 의 symmetric InfoNCE 가 왜 zero-shot transfer 의 mathematical 토대가 되는가? Prompt `"a photo of a {cls}"` 가 왜 zero-shot 의 "trick" 이 아니라 본질적 메커니즘인가? BLIP / BLIP-2 의 Q-Former 가 frozen ViT 와 frozen LLM 사이의 효율적 bridge 가 되는 구조적 이유는? LLaVA 의 단순 MLP projection 이 더 복잡한 cross-attention 보다 종종 우위인 이유는? Flamingo 의 gated cross-attention 이 interleaved image-text few-shot 학습을 어떻게 가능하게 하는가? GPT-4V · Gemini 의 native multimodal 이 vision encoder 분리 구조를 대체할 것인가?

<details>
<summary><b>CLIP 부터 Native Multimodal 까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. CLIP — Contrastive Language-Image Pretraining (Radford 2021)](./ch6-multimodal/01-clip.md) | **Dual encoder**: image (ViT) + text (Transformer). **Symmetric InfoNCE**: $L = \tfrac12 (L_{i \to t} + L_{t \to i})$, $L_{i \to t} = -\tfrac1B \sum_i \log \tfrac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$ $\square$. **Temperature $\tau$**: trainable scalar (typically clip $\to e^{4.6} \approx 100$). **Scale**: 400M (image, text) pairs from web. **Insight**: text 가 자연스러운 supervision — vocabulary 가 open, prompt engineering 으로 task 변경 |
| [02. CLIP 의 Zero-Shot Transfer](./ch6-multimodal/02-clip-zero-shot.md) | **Mechanism**: class name $c$ 를 prompt template `"a photo of a {c}"` 에 채워 text encoder → text embedding $t_c$. Image embedding $v$ 와의 $\arg\max_c \cos(v, t_c)$ 로 classification. **Prompt ensembling**: 여러 template 의 평균 → 1–2% 향상. **재현 실험**: open_clip 으로 ImageNet-1k zero-shot top-1 75% 달성 (ViT-L/14). **Linear probe vs zero-shot**: 데이터 풍부 시 linear probe 우위, few-shot/zero-shot 이 CLIP 의 본령 |
| [03. BLIP · BLIP-2 — Bootstrapping Captioners (Li 2022, 2023)](./ch6-multimodal/03-blip-blip2.md) | **BLIP (Li 2022)**: image-text contrastive + image-text matching (binary) + image captioning (LM) 의 multi-task. CapFilt 로 noisy web caption 정제. **BLIP-2 (Li 2023)**: **Q-Former** — 32 learnable query 가 frozen ViT image feature 를 cross-attention 으로 수집, 그 후 frozen LLM (Vicuna) 에 prefix 로 주입. 두 frozen 사이만 학습 → 매우 efficient. Vision–LLM bridge 의 효율적 패러다임 |
| [04. LLaVA — Visual Instruction Tuning (Liu 2023)](./ch6-multimodal/04-llava.md) | **Architecture**: CLIP ViT-L/14 (frozen) → linear projection (또는 MLP) → LLaMA/Vicuna. 매우 minimalist. **Visual instruction data**: GPT-4 로 image caption + bbox 를 instruction–response pair 로 합성 (158K samples). **Stage 1**: projection 만 학습 (alignment). **Stage 2**: projection + LLM fine-tune (instruction). **결과**: 단순한 projection 이 복잡한 cross-attention 과 동등 — "frozen vision + alignment + LLM finetune" 의 표준 |
| [05. Flamingo · GPT-4V · Gemini — Interleaved Vision-Language](./ch6-multimodal/05-flamingo-native-mm.md) | **Flamingo (Alayrac 2022)**: frozen Chinchilla LM + frozen NFNet vision + **gated cross-attention** ($\mathrm{tanh}(\alpha) \cdot \mathrm{xattn}$, $\alpha$ learnable, init 0) → identity 시작, 점진적 vision injection. **Interleaved**: `<img1> caption1 <img2> caption2 <img3> ?` 의 few-shot in-context. **Native multimodal (GPT-4V, Gemini, Claude 3)**: image 도 token 화해 single Transformer — 별도 vision encoder 의 경계 흐려짐. 미해결 질문: dedicated vision encoder vs unified token |

</details>

<br/>

### 🔹 Chapter 7: Vision Generation · Scaling · 3D · Video

> **핵심 질문:** Parti · Muse 의 transformer 기반 image generation (autoregressive · masked token) 이 diffusion 과 어떤 trade-off 를 갖는가? Zhai 2022 의 Scaling Vision Transformers 가 보여주는 data·model power law 가 vision foundation model 설계에 어떤 함의를 주는가? NeRF 와 3D Gaussian Splatting 이 neural rendering 의 두 패러다임으로 어떻게 발전했는가? TimeSformer 의 divided space-time attention 과 VideoMAE 의 tube masking 이 video Transformer 의 효율을 어떻게 푸는가? Sora · VideoLLaMA 가 spatio-temporal Transformer 의 frontier 인 이유는?

<details>
<summary><b>Vision Generation 부터 Video Transformer 까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. Vision Generation with Transformers — Parti · Muse](./ch7-emerging/01-vision-generation.md) | **Parti (Yu 2022)**: VQ-GAN tokenizer + autoregressive Transformer (T5-like). **장점**: text 와 같은 일관된 token 모델. **단점**: 순차 sampling 으로 느림. **Muse (Chang 2023)**: masked token model — BERT-style parallel decoding, 8–16 step 으로 high-res. **MaskGIT**: 같은 계열, parallel decoding 의 step schedule. **Diffusion 과의 비교**: token-based 가 latent space 에서 효율, diffusion 이 continuous score 로 finer detail — 두 방향이 SD3 / Sora 에서 부분 융합 |
| [02. Scaling Laws for Vision (Zhai 2022)](./ch7-emerging/02-scaling-laws.md) | **Zhai 2022 "Scaling Vision Transformers"**: data size $D$, model size $N$, compute $C$ 에 대한 power law $\text{loss} \propto D^{-\alpha} N^{-\beta} C^{-\gamma}$ — Hoffmann 2022 (Chinchilla) 의 vision 판본. **함의**: ViT-22B 까지 단조 향상, 그러나 augmentation·regularization 이 scaling 효율을 좌우. **Foundation model**: data quality (LAION-2B, JFT-3B, LVD-142M) 가 scale 만큼 중요 — DINOv2 의 curated dataset 의 성공 사례 |
| [03. 3D Vision — NeRF · 3D Gaussian Splatting](./ch7-emerging/03-3d-nerf-gs.md) | **NeRF (Mildenhall 2020)**: scene 을 $F_\Theta : (x, y, z, \theta, \phi) \to (RGB, \sigma)$ MLP 로 표현, volume rendering $C(r) = \int T(t) \sigma(r(t)) c(r(t)) dt$. 학습에 hours, rendering 도 느림. **3D Gaussian Splatting (Kerbl 2023)**: scene 을 anisotropic 3D Gaussian 의 합으로, differentiable rasterization 으로 real-time. NeRF 의 quality + GS 의 speed → 3D vision 의 새 표준. **Vision Transformer 와의 연결**: scene encoder 로 ViT |
| [04. Video Understanding — TimeSformer · VideoMAE · Sora](./ch7-emerging/04-video-transformer.md) | **TimeSformer (Bertasius 2021)**: divided space-time attention — space attention 과 time attention 을 분리해 $O(T^2 + S^2)$ (joint $O((TS)^2)$ 대비 효율). **VideoMAE (Tong 2022)**: tube masking (같은 spatial 위치를 모든 frame 에서 mask) → temporal redundancy 활용, 90%+ masking 가능. **VideoLLaMA / Sora**: spatio-temporal DiT + multimodal — large video Transformer 의 scaling. **미래**: world model 로의 발전, AlphaFold·DreamFusion 과 함께 science 응용 |

</details>

---

> 🆕 **2026-05 최신 업데이트**: Ch1-02 의 Patch=Conv 등가 증명에 weight reshape 의 명시적 단계를 분리, Ch1-05 의 Inductive Bias 부족에 DeiT 의 augmentation 효과 정량 분석을 추가, Ch3-03 의 InfoNCE MI bound 에 Wang 2020 의 alignment-uniformity 분해를 통합 정리, Ch4-02 의 DINO collapse 방지에 centering vs sharpening 의 ablation table 추가, Ch5-03 의 MAE 정보이론에 compressive sensing 비유와 SNR 분석 보강, Ch6-01 의 CLIP symmetric InfoNCE 에 temperature 학습의 안정화 분석 추가, Ch7-02 의 Scaling Laws 에 ViT-22B 와 DINOv2 의 data-curation 효과 비교 추가했습니다. **11-섹션 문서 골격이 전체 33개 문서에서 일관** 됩니다.

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **원 논문 실험 재현** 을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$ 로 종결되는 엄밀한 증명 또는 `results/` 하의 학습 곡선·plot 을 확인할 수 있습니다.

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **Patch Embedding = Conv2d 등가** | $\mathrm{Conv2d}(C, D, k=P, s=P)$ + flatten·transpose 가 patch embedding 과 bit-exact 등가 | [Ch1-02](./ch1-vit-foundations/02-patch-conv-equivalence.md) |
| **CLS = Learnable Global Pooling** | CLS token 의 final attention weight 가 patch 별 weighted average 로 해석 | [Ch1-03](./ch1-vit-foundations/03-cls-token-vs-gap.md) |
| **ViT 의 Inductive Bias 손실** | Translation equivariance · locality 부재 → small-data regime 에서 ResNet 에 뒤짐 | [Ch1-05](./ch1-vit-foundations/05-inductive-bias-gap.md) |
| **DeiT Hard Distillation** | Distillation token + CNN teacher hard label 로 ImageNet-1k 만으로 ViT-B 능가 | [Ch2-01](./ch2-hierarchical-vit/01-deit.md) |
| **Swin Window Complexity** | Window attention $O(n \cdot w^2)$ + shifted window 로 hierarchical RF | [Ch2-02](./ch2-hierarchical-vit/02-swin.md) |
| **InfoNCE = MI 하한** | $I(X; Y) \geq \log N - L_{\text{InfoNCE}}$ — Poole 2019 | [Ch3-03](./ch3-contrastive-ssl/03-infonce-mi-bound.md) |
| **MoCo Momentum Encoder** | Queue + EMA 로 dictionary consistency, large batch 의존 회피 | [Ch3-04](./ch3-contrastive-ssl/04-moco.md) |
| **BYOL/SimSiam Collapse 회피** | Predictor + stop-gradient 의 saddle 화 — Tian 2021 의 분석 | [Ch3-05](./ch3-contrastive-ssl/05-byol-simsiam.md) |
| **DINO Centering + Sharpening** | 한 차원 collapse 와 uniform collapse 를 양쪽에서 막는 두 force 의 균형 | [Ch4-02](./ch4-dino/02-teacher-ema-collapse.md) |
| **DINO Emergent Segmentation** | Explicit supervision 없이 attention map 이 object mask 와 일치 | [Ch4-03](./ch4-dino/03-emergent-properties.md) |
| **MAE 75% Masking 정당화** | 자연이미지 spatial redundancy + asymmetric encoder–decoder 의 4× 가속 | [Ch5-02](./ch5-masked-image-modeling/02-mae.md) |
| **CLIP Symmetric InfoNCE** | $L = \tfrac12 (L_{i \to t} + L_{t \to i})$ — zero-shot transfer 의 mathematical 토대 | [Ch6-01](./ch6-multimodal/01-clip.md) |
| **CLIP Zero-Shot Mechanism** | Prompt template 의 text embedding 과 image embedding cosine | [Ch6-02](./ch6-multimodal/02-clip-zero-shot.md) |
| **BLIP-2 Q-Former Bridge** | 32 learnable query 가 frozen ViT 와 frozen LLM 사이 효율적 정보 통로 | [Ch6-03](./ch6-multimodal/03-blip-blip2.md) |
| **Vision Scaling Power Law** | Zhai 2022 의 $\text{loss} \propto D^{-\alpha} N^{-\beta} C^{-\gamma}$ — ViT-22B 까지 단조 | [Ch7-02](./ch7-emerging/02-scaling-laws.md) |

> 💡 **챕터별 문서·정리/정의 수** (실측):
>
> | 챕터 | 문서 수 | 정리·정의 |
> |------|---------|------------|
> | Ch1 ViT Foundations | 5 | 30 |
> | Ch2 Hierarchical ViT | 5 | 34 |
> | Ch3 Contrastive SSL | 5 | 28 |
> | Ch4 DINO Family | 4 | 27 |
> | Ch5 Masked Image Modeling | 5 | 29 |
> | Ch6 Multimodal | 5 | 36 |
> | Ch7 Generation · Scaling · 3D · Video | 4 | 28 |
> | **합계** | **33** | **212** |
>
> 추가로 **85+ 엄밀한 $\square$ 증명 + 99 연습문제 (모두 해설 포함) + 132 NumPy/PyTorch/timm/CLIP 실험 코드 (`### 실험 N` 형식)**.
>
> Ch4 (DINO) 와 Ch7 (Generation·Scaling·3D·Video) 은 **4 문서** 로 구성 — DINO 의 핵심 4 단계 (algorithm → EMA/collapse → emergence → DINOv2) 에 집중, Ch7 은 mature 한 4 가지 (generation, scaling, 3D, video) 만 다룸 (Chapters 1–3, 5, 6 의 5 문서와 의도적 차이).

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
torch==2.1.0
torchvision==0.16.0
timm==0.9.10
transformers==4.36.0
accelerate==0.25.0
einops==0.7.0
matplotlib==3.8.0
seaborn==0.13.0
tqdm==4.66.0
jupyter==1.0.0
clip-anytorch==2.5.2           # CLIP 참조 구현
open_clip_torch==2.24.0        # CLIP zero-shot 평가
# 선택 사항
tensorboard==2.15.0            # 학습 곡선 로깅
wandb==0.16.0                  # 실험 추적
xformers==0.0.23               # 메모리 효율 attention (large ViT)
```

```bash
# 환경 설치 (CPU 기준; GPU 권장)
pip install numpy==1.26.0 scipy==1.11.0 torch==2.1.0 torchvision==0.16.0 \
            timm==0.9.10 transformers==4.36.0 accelerate==0.25.0 \
            einops==0.7.0 matplotlib==3.8.0 seaborn==0.13.0 \
            tqdm==4.66.0 jupyter==1.0.0 \
            clip-anytorch==2.5.2 open_clip_torch==2.24.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 ① — Patch Embedding = Conv2d 등가 검증 (Ch1-02)
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """Patch embedding implemented as Conv2d (the standard ViT trick)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)                          # [B, D, H/P, W/P]
        return x.flatten(2).transpose(1, 2)       # [B, N, D]

def patch_embed_naive(x, E, bias, P):
    """Equivalent 'unfold + matmul' implementation."""
    B, C, H, W = x.shape
    patches = x.unfold(2, P, P).unfold(3, P, P)            # [B, C, H/P, W/P, P, P]
    patches = patches.contiguous().view(B, C, -1, P, P)
    patches = patches.permute(0, 2, 1, 3, 4).reshape(B, -1, C * P * P)
    return patches @ E + bias                              # [B, N, D]

B, C, H, W, P, D = 2, 3, 224, 224, 16, 768
x = torch.randn(B, C, H, W)
conv_embed = PatchEmbed(H, P, C, D)
y_conv = conv_embed(x)
E = conv_embed.proj.weight.view(D, -1).T                   # [P²C, D]
y_naive = patch_embed_naive(x, E, conv_embed.proj.bias, P)
print(f'Max diff: {(y_conv - y_naive).abs().max():.2e}')   # ≈ 0 (bit-exact)

# 대표 실험 ② — ViT 바닥부터 + CIFAR-10 학습 (Ch1, Ch2)
class ViTBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(),
                                 nn.Linear(4*dim, dim))
    def forward(self, x):
        xn = self.norm1(x)
        x = x + self.attn(xn, xn, xn, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 dim=768, depth=12, n_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, dim)
        N = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, dim))
        self.blocks = nn.ModuleList([ViTBlock(dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks: x = blk(x)
        return self.head(self.norm(x)[:, 0])               # CLS

# 대표 실험 ③ — InfoNCE (SimCLR style, Ch3-02)
def info_nce(z1, z2, tau=0.1):
    z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / tau
    labels = torch.arange(z1.shape[0])
    return F.cross_entropy(logits, labels)

# 대표 실험 ④ — DINO loss skeleton (Ch4-01, Ch4-02)
def dino_loss(student_out, teacher_out, center, tau_s=0.1, tau_t=0.04):
    """Self-distillation with centering + sharpening."""
    student = F.log_softmax(student_out / tau_s, dim=-1)
    teacher = F.softmax((teacher_out - center) / tau_t, dim=-1).detach()
    return -(teacher * student).sum(-1).mean()

# 대표 실험 ⑤ — MAE random masking (Ch5-02)
def random_masking(x, mask_ratio=0.75):
    B, N, D = x.shape
    L = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :L]
    x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    return x_visible, ids_shuffle

# 대표 실험 ⑥ — CLIP zero-shot (Ch6-02)
# import open_clip
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
# tokenizer = open_clip.get_tokenizer('ViT-L-14')
# text = tokenizer(['a photo of a cat', 'a photo of a dog'])
# with torch.no_grad():
#     image_features = model.encode_image(image_tensor)
#     text_features  = model.encode_text(text)
#     logits = (image_features @ text_features.T).softmax(-1)
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격** 으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 ... 인가** | 해당 이론·알고리즘이 vision 의 어떤 핵심 문제를 푸는지 |
| 3 | 📐 **수학적 선행 조건** | Transformer · CNN · Generative · Reg · Kernel 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | Patch grid · attention map · contrastive geometry · masking pattern 의 시각적 직관 |
| 5 | ✏️ **엄밀한 정의** | Patch embedding · InfoNCE · self-distillation · MAE · CLIP 등 |
| 6 | 🔬 **정리와 증명** | Patch=Conv 등가 · MI 하한 · DINO collapse · MAE 가속 · CLIP symmetric 등 |
| 7 | 💻 **NumPy / PyTorch 구현 검증** | 4 가지 실험 (`### 실험 1` ~ `### 실험 4`) — bit-exact 검증 · CIFAR-10/100 · ablation · 시각화 |
| 8 | 🔗 **실전 활용** | 언제 ViT · 언제 Swin · 언제 SimCLR / DINO / MAE · 언제 CLIP — 실전 선택 가이드 |
| 9 | ⚖️ **가정과 한계** | 각 방법의 실패 모드 (small data · OOD · long-tail · prompt brittleness · compute cost) |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 ($\boxed{}$ 핵심 수식 + 표) |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 기초 / 심화 / 논문 비평 의 3 문제, `<details>` 펼침 해설 |

> 📚 **연습문제 총 99개** (33 문서 × 3 문제): **기초 / 심화 / 논문 비평** 의 3-tier 구성, 모든 문제에 `<details>` 펼침 해설 포함. Patch=Conv 등가 증명부터 InfoNCE MI bound, DINO collapse 분석, MAE 정보이론, CLIP symmetric InfoNCE 유도, MM-DiT ablation 까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 200~450줄 (정의·증명·코드·연습문제 포함) 기준 **약 45분~1시간 20분**. 전체 33문서는 약 **30~40시간** 상당 (증명 재구성 · timm/CLIP 실험 재현 포함 시 55시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "ViT/CLIP 을 쓰지만 왜 작동하는지 이론적으로 이해하고 싶다" — 입문 투어 (1주, 약 12~14시간)</b></summary>

<br/>

```
Day 1  Ch1-01  ViT 의 수학적 구조
       Ch1-02  Patch=Conv2d 등가
Day 2  Ch1-03  CLS vs GAP
       Ch1-05  Inductive Bias 부족
Day 3  Ch2-01  DeiT
       Ch2-02  Swin Transformer
Day 4  Ch3-01  SSL 3 패러다임
       Ch3-02  SimCLR
Day 5  Ch4-01  DINO 알고리즘
       Ch5-02  MAE
Day 6  Ch6-01  CLIP
       Ch6-02  CLIP Zero-Shot
Day 7  Ch6-04  LLaVA
       Ch7-02  Scaling Laws
```

</details>

<details>
<summary><b>🟡 "SSL 3 대 패러다임 (Contrastive · DINO · MAE) 을 완전히 정복한다" — 이론 집중 (2주, 약 24~28시간)</b></summary>

<br/>

```
1주차 — ViT Foundations · Hierarchical · Contrastive
  Day 1    Ch1-01~02   ViT 수학 · Patch=Conv 등가 증명
  Day 2    Ch1-03~05   CLS/GAP · PE 변종 · Inductive Bias
  Day 3    Ch2-01~02   DeiT · Swin
  Day 4    Ch2-03~05   PVT · CvT/CoAtNet · MViT/Focal
  Day 5    Ch3-01~02   SSL 패러다임 · SimCLR
  Day 6    Ch3-03~04   InfoNCE MI bound · MoCo
  Day 7    Ch3-05      BYOL · SimSiam (collapse 분석)

2주차 — DINO · MIM · Multimodal · Generation
  Day 1    Ch4-01~02   DINO 알고리즘 · EMA / Collapse
  Day 2    Ch4-03~04   Emergent · DINOv2
  Day 3    Ch5-01~02   BEiT · MAE
  Day 4    Ch5-03~05   MAE 정보이론 · SimMIM · MaskFeat/MVP
  Day 5    Ch6-01~02   CLIP · Zero-Shot
  Day 6    Ch6-03~05   BLIP/BLIP-2 · LLaVA · Flamingo/Native MM
  Day 7    Ch7-01~04   Generation · Scaling · 3D · Video
```

</details>

<details>
<summary><b>🔴 "ViT 부터 LLaVA · Sora 까지의 수학을 완전 정복한다" — 전체 정복 (8주, 약 35~45시간 + 재현 12~18시간)</b></summary>

<br/>

```
1주차   Chapter 1 — ViT Foundations
         → Patch=Conv2d 등가의 weight reshape 손 증명
         → CLS token 의 attention weight 가 weighted pooling 임을 시각화
         → Translation equivariance 손실의 small-data ablation

2주차   Chapter 2 — Hierarchical · Hybrid
         → Swin window attention 의 $O(n \cdot w^2)$ 복잡도 유도
         → DeiT 의 distillation token vs CNN baseline 학습 곡선
         → PVT · CvT · CoAtNet 의 FLOPs/Acc 비교

3주차   Chapter 3 — Contrastive SSL
         → InfoNCE = MI 하한 (Poole 2019) 증명 따라가기
         → SimCLR vs MoCo 의 batch / queue ablation
         → BYOL · SimSiam 의 stop-gradient collapse 분석 (Tian 2021)

4주차   Chapter 4 — DINO Family
         → Centering + Sharpening 의 collapse 방지 두 force 분석
         → DINO 의 attention map → object segmentation emergence 시각화
         → DINOv2 의 multi-objective ablation (DINO + iBOT + KoLeo + SwAV)

5주차   Chapter 5 — Masked Image Modeling
         → MAE 75% masking 의 정보이론적 근거 + asymmetric encoder 가속 측정
         → BEiT · SimMIM · MaskFeat 의 target 비교 (pixel · token · HOG · CLIP feature)
         → Linear probe vs fine-tune 의 representation geometry 차이

6주차   Chapter 6 — Multimodal
         → CLIP symmetric InfoNCE 유도 + temperature 학습 안정화
         → open_clip 으로 ImageNet zero-shot 재현 (ViT-L/14)
         → BLIP-2 Q-Former 의 frozen-frozen bridge 효율 분석
         → LLaVA visual instruction tuning 의 minimalist 설계

7주차   Chapter 7 — Generation · Scaling · 3D · Video
         → Parti / Muse 의 token-based generation vs diffusion 비교
         → Zhai 2022 vision scaling power law 분석
         → NeRF · 3D-GS 의 rendering 비교
         → TimeSformer / VideoMAE 의 efficiency 분석

8주차   종합 — DINOv2 / LLaVA / Sora
         → DINOv2 가 CLIP 을 대체할 vision foundation 인가
         → LLaVA / GPT-4V / Gemini 의 multimodal 설계 차이
         → "Native multimodal" 이 dedicated vision encoder 를 대체할 것인가 토론
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [transformer-deep-dive](https://github.com/iq-ai-lab/transformer-deep-dive) | Self-attention · Pre-LN · positional encoding · cross-attention | **Ch1 전체** (선행), **Ch6** (cross-attn) |
| [cnn-deep-dive](https://github.com/iq-ai-lab/cnn-deep-dive) | Convolution · receptive field · translation equivariance · inductive bias | **Ch1, Ch2** (선행, ViT vs CNN) |
| [generative-model-deep-dive](https://github.com/iq-ai-lab/generative-model-deep-dive) | Autoencoder · VAE · GAN · normalizing flow | **Ch5** (MAE autoencoder) |
| [regularization-theory-deep-dive](https://github.com/iq-ai-lab/regularization-theory-deep-dive) | Mixup · CutMix · RandAugment · label smoothing · stochastic depth | **Ch2** (DeiT augmentation) |
| [kernel-methods-deep-dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive) | Kernel SVM · RBF · alignment-uniformity · contrastive ↔ kernel | **Ch3** (InfoNCE 의 kernel 해석) |
| [diffusion-model-deep-dive](https://github.com/iq-ai-lab/diffusion-model-deep-dive) | DDPM · Score-SDE · DDIM · CFG · Latent · DiT · Consistency · Rectified Flow | **Ch6 이후** (DiT · MM-DiT · Sora) |
| [object-detection-deep-dive](https://github.com/iq-ai-lab/object-detection-deep-dive) *(다음)* | DETR · DINO-DETR · Mask DINO · Grounding DINO | **Ch2, Ch4** 이후 응용 |
| [multimodal-foundation-models-deep-dive](https://github.com/iq-ai-lab/multimodal-foundation-models-deep-dive) *(다음)* | LLaVA-Next · GPT-4V · Gemini · Claude 3 · Sora | **Ch6, Ch7** 이후 응용 |

> 💡 이 레포는 **"ViT · DINO · MAE · CLIP · LLaVA · Sora 가 모두 같은 Transformer 의 다른 학습 신호이고, patch embedding · InfoNCE · self-distillation · masked modeling · symmetric contrastive 가 왜 각각의 이론적 동기를 갖는가"** 에 집중합니다. Transformer 에서 self-attention 과 PE 를, CNN 에서 inductive bias 와 receptive field 를, Regularization 에서 Mixup·CutMix 를, Generative Model 에서 autoencoder 를 익힌 후 오면 Chapter 3 (Contrastive 의 MI bound) 와 Chapter 4 (DINO 의 collapse 분석) 의 증명이 훨씬 자연스럽습니다. **Diffusion Model Deep Dive** 와 함께 보면 Ch6 의 DiT / MM-DiT 가 SD3 · Sora 의 backbone 이 된 맥락이 선명해집니다.

---

## 📖 Reference

### 🏛️ ViT · DeiT · Hierarchical
- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (Dosovitskiy et al., 2021) — **ViT**
- **Training data-efficient image transformers & distillation through attention** (Touvron et al., 2021) — **DeiT**
- **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** (Liu et al., 2021)
- **Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions** (Wang et al., 2021) — **PVT**
- **CvT: Introducing Convolutions to Vision Transformers** (Wu et al., 2021)
- **CoAtNet: Marrying Convolution and Attention for All Data Sizes** (Dai et al., 2021)
- **Multiscale Vision Transformers** (Fan et al., 2021) — **MViT**
- **Focal Self-attention for Local-Global Interactions in Vision Transformers** (Yang et al., 2021)

### 🔁 Self-Supervised — Contrastive
- **A Simple Framework for Contrastive Learning of Visual Representations** (Chen et al., 2020) — **SimCLR**
- **Momentum Contrast for Unsupervised Visual Representation Learning** (He et al., 2020) — **MoCo**
- **Improved Baselines with Momentum Contrastive Learning** (Chen et al., 2020) — MoCo v2
- **An Empirical Study of Training Self-Supervised Vision Transformers** (Chen et al., 2021) — MoCo v3
- **Bootstrap Your Own Latent** (Grill et al., 2020) — **BYOL**
- **Exploring Simple Siamese Representation Learning** (Chen & He, 2021) — **SimSiam**
- **Understanding Self-supervised Learning Dynamics without Contrastive Pairs** (Tian et al., 2021)
- **Representation Learning with Contrastive Predictive Coding** (Oord et al., 2018) — InfoNCE
- **On Variational Bounds of Mutual Information** (Poole et al., 2019)
- **Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere** (Wang & Isola, 2020)

### 🧭 Self-Distillation — DINO
- **Emerging Properties in Self-Supervised Vision Transformers** (Caron et al., 2021) — **DINO**
- **iBOT: Image BERT Pre-Training with Online Tokenizer** (Zhou et al., 2022)
- **DINOv2: Learning Robust Visual Features without Supervision** (Oquab et al., 2024)
- **Unsupervised Learning of Visual Features by Contrasting Cluster Assignments** (Caron et al., 2020) — SwAV

### 🎭 Masked Image Modeling
- **BEiT: BERT Pre-Training of Image Transformers** (Bao et al., 2022)
- **Masked Autoencoders Are Scalable Vision Learners** (He et al., 2022) — **MAE**
- **SimMIM: A Simple Framework for Masked Image Modeling** (Xie et al., 2022)
- **Masked Feature Prediction for Self-Supervised Visual Pre-Training** (Wei et al., 2022) — **MaskFeat**
- **MVP: Multimodality-guided Visual Pre-training** (Wei et al., 2022)

### 🌐 Multimodal — Vision-Language
- **Learning Transferable Visual Models From Natural Language Supervision** (Radford et al., 2021) — **CLIP**
- **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation** (Li et al., 2022)
- **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** (Li et al., 2023)
- **Visual Instruction Tuning** (Liu et al., 2023) — **LLaVA**
- **Improved Baselines with Visual Instruction Tuning** (Liu et al., 2024) — LLaVA-1.5
- **Flamingo: a Visual Language Model for Few-Shot Learning** (Alayrac et al., 2022)
- **GPT-4 Technical Report** (OpenAI, 2023) — GPT-4V
- **Gemini: A Family of Highly Capable Multimodal Models** (Google, 2023)

### 🎨 Vision Generation · Scaling · 3D · Video
- **Scaling Autoregressive Models for Content-Rich Text-to-Image Generation** (Yu et al., 2022) — **Parti**
- **Muse: Text-To-Image Generation via Masked Generative Transformers** (Chang et al., 2023)
- **MaskGIT: Masked Generative Image Transformer** (Chang et al., 2022)
- **Scaling Vision Transformers** (Zhai et al., 2022)
- **Scaling Vision Transformers to 22 Billion Parameters** (Dehghani et al., 2023)
- **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** (Mildenhall et al., 2020)
- **3D Gaussian Splatting for Real-Time Radiance Field Rendering** (Kerbl et al., 2023)
- **Is Space-Time Attention All You Need for Video Understanding?** (Bertasius et al., 2021) — **TimeSformer**
- **VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training** (Tong et al., 2022)
- **Sora: Video Generation Models as World Simulators** (OpenAI, 2024)

### 🛠️ Implementation · Libraries
- **timm — PyTorch Image Models** (Wightman, 2019–) — ViT · DeiT · Swin · CoAtNet · MAE 표준 구현
- **🤗 Transformers** (Wolf et al., 2020) — ViT · CLIP · BLIP · LLaVA 통합 인터페이스
- **OpenAI CLIP** (Radford et al., 2021) — 원조 CLIP 구현
- **open_clip** (Cherti et al., 2023) — large-scale CLIP 재현 + 사전학습 weights
- **facebookresearch / dino, dinov2, mae** — DINO · DINOv2 · MAE 공식 구현

---

<div align="center">

**⭐️ 도움이 되셨다면 Star 를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"ViT 를 쓰는 것 과 — Dosovitskiy 2021 으로 patch embedding 이 $\mathrm{Conv2d}(C, D, k=P, s=P)$ 와 bit-exact 등가이고 CLS token 이 learnable global pooling 임을 한 줄씩 증명 · Touvron 2021 (DeiT) 로 distillation token + augmentation 이 inductive bias 손실을 보상함을 학습 곡선으로 검증 · Liu 2021 (Swin) 으로 window attention 의 $O(n \cdot w^2)$ 복잡도와 shifted window 의 cross-window 정보 흐름을 유도 · Chen 2020 / He 2020 으로 InfoNCE 가 mutual information 의 하한 (Poole 2019) 임을 증명 · Caron 2021 (DINO) 로 centering + sharpening 의 두 force 가 한 차원 / uniform collapse 를 막는 메커니즘을 분석 · He 2022 (MAE) 로 75% masking 이 자연이미지 spatial redundancy 의 정보이론적 결과이며 asymmetric encoder–decoder 가 4× 가속을 만듦을 측정 · Radford 2021 (CLIP) 으로 symmetric InfoNCE 가 zero-shot transfer 의 mathematical 토대임을 유도 · Liu 2023 (LLaVA) / Alayrac 2022 (Flamingo) 로 vision-LLM bridge 의 minimalist · gated 두 패러다임을 분석 — 이 모든 '왜' 를 직접 유도할 수 있는 것은 다르다"*

</div>
