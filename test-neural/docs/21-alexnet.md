# 21 — AlexNet: Deep CNN on ImageNet

> **Navigation** ← [20 — VGG](20-vgg.md) | [22 — EfficientNet →](22-efficientnet.md)

**Paper**: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *NeurIPS*.

---

## Level 1 — Concepts

### The moment that changed everything

In 2012, AlexNet entered the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) and achieved a top-5 error of 15.3% — nearly half the error of the second-best entry (26.2%) using traditional vision methods. This result demonstrated that deep convolutional networks trained on GPUs could solve large-scale vision problems and ignited the modern deep learning revolution.

### Key innovations of AlexNet

1. **ReLU activations throughout** — the paper (section 3.1) reports that ReLU networks train 6× faster than Tanh networks for the same accuracy. This was one of the first papers to demonstrate ReLU's training speed advantage at scale.

2. **GPU training** — AlexNet was split across two GTX 580 GPUs (3GB VRAM each) and trained over 5–6 days. The GPU-parallel training infrastructure it pioneered is now standard.

3. **Dropout (rate 0.5)** — applied in the first two FC layers. Without dropout, AlexNet overfit significantly. This was one of the first large-scale demonstrations of dropout as a regularizer.

4. **Data augmentation** — random crops (227×227 from 256×256 images), horizontal flipping, and PCA-based color jittering. This expanded the effective dataset size by a factor of ~2048.

5. **Local Response Normalization (LRN)** — a form of lateral inhibition that normalized activations across channels. Since superseded by Batch Normalization (this library uses BatchNorm instead).

### Architecture (227×227 color input)

```
Input:    3 × 227 × 227

Conv1:    Conv 11×11, 96 maps, stride 4  → 96 × 55 × 55
          LRN (→ BatchNorm in library)
          MaxPool 3×3, stride 2          → 96 × 27 × 27

Conv2:    Conv 5×5, 256 maps, pad 2      → 256 × 27 × 27
          LRN (→ BatchNorm in library)
          MaxPool 3×3, stride 2          → 256 × 13 × 13

Conv3:    Conv 3×3, 384 maps, pad 1      → 384 × 13 × 13
Conv4:    Conv 3×3, 384 maps, pad 1      → 384 × 13 × 13
Conv5:    Conv 3×3, 256 maps, pad 1      → 256 × 13 × 13
          MaxPool 3×3, stride 2          → 256 × 6 × 6

FC6:      9216 → 4096 + ReLU + Dropout(0.5)
FC7:      4096 → 4096 + ReLU + Dropout(0.5)
FC8:      4096 → 1000 → Softmax
```

Total: ~60M parameters, mostly in the FC layers.

### Build variants in this library

`AlexNetConfig` provides three build paths:

| Preset | Input size | Use case |
|--------|-----------|----------|
| `imagenet()` / `build_full()` | ≥200×200 | Original ImageNet task |
| `build_medium()` | 64–199px | Transfer, smaller datasets |
| `build_mini()` / `cifar10()` | <64px (32×32) | CIFAR-10, MNIST |

---

## Level 2 — Mathematics

### Parameter count

**Conv layers**:

$$\text{Conv1}: (11^2 \times 3 + 1) \times 96 = 34{,}944$$
$$\text{Conv2}: (5^2 \times 96 + 1) \times 256 = 614{,}656$$
$$\text{Conv3}: (3^2 \times 256 + 1) \times 384 = 885{,}120$$
$$\text{Conv4}: (3^2 \times 384 + 1) \times 384 = 1{,}327{,}488$$
$$\text{Conv5}: (3^2 \times 384 + 1) \times 256 = 884{,}992$$
$$\text{Total conv}: \approx 3{,}747{,}200$$

**FC layers**:

$$\text{FC6}: (256 \times 6^2 + 1) \times 4096 = 37{,}752{,}832$$
$$\text{FC7}: (4096 + 1) \times 4096 = 16{,}781{,}312$$
$$\text{FC8}: (4096 + 1) \times 1000 = 4{,}097{,}000$$
$$\text{Total FC}: \approx 58{,}631{,}144$$

**Grand total**: ~62.4M parameters. FC layers dominate (94%).

### Local Response Normalization (LRN)

The original paper used LRN to implement **lateral inhibition** — reducing activations of neurons surrounded by highly active neighbors across channels:

$$b_{x,y}^i = a_{x,y}^i \left(k + \alpha \sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)} (a_{x,y}^j)^2\right)^{-\beta}$$

with $k=2, \alpha=10^{-4}, \beta=0.75, n=5$ in the paper. This was partially motivated by neuroscience (inhibitory interneurons in V1). Subsequent work showed LRN provides marginal benefit and is much harder to implement efficiently than BatchNorm. The library uses BatchNorm instead.

### Dropout as an ensemble

With $n = 4096$ units and dropout rate $p = 0.5$, the first FC layer has $2^{4096}$ possible sub-networks. Each forward pass samples one of these sub-networks. The training procedure jointly optimizes the parameters of all $2^{4096}$ networks with shared weights — a form of extreme ensemble learning.

The test-time prediction uses all neurons with weights scaled by $(1-p) = 0.5$, approximating the geometric mean of all sub-network predictions (see [07 — Dropout](07-dropout.md) for the formal ensemble interpretation).

### Why large filters in Conv1?

The 11×11 filters in Conv1 (stride 4) were a practical compromise: at 227×227 input, the computational cost of a Conv layer is $O(k^2 \times C_{in} \times C_{out} \times H_{out} \times W_{out})$, and stride 4 reduces $H_{out}, W_{out}$ by 4×. This made training feasible on 2012 GPU hardware. Modern architectures (VGG, ResNet) replaced large-filter stems with stacks of 3×3 convolutions, which is more parameter-efficient (see [20 — VGG](20-vgg.md) §3×3 stacking).

### Data augmentation — PCA color perturbation

The paper's color augmentation adds multiples of the principal components of all training pixel values:

$$[p_1 | p_2 | p_3] [\alpha_1 \lambda_1, \alpha_2 \lambda_2, \alpha_3 \lambda_3]^T$$

where $p_i$ and $\lambda_i$ are the $i$-th eigenvector and eigenvalue of the $3 \times 3$ RGB covariance matrix, and $\alpha_i \sim \mathcal{N}(0, 0.1)$ are random scalings drawn once per image. This perturbs the illumination and color balance while preserving content, imposing invariance to natural illumination changes.

**References**:
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). [Original paper above.]
- Russakovsky, O., et al. (2015). ImageNet Large Scale Visual Recognition Challenge. *IJCV*. [Full description of ILSVRC benchmark.]
