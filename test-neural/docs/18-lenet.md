# 18 — LeNet-5

> **Navigation** ← [17 — Autograd](17-autograd.md) | [19 — ResNet →](19-resnet.md)

**Paper**: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

---

## Level 1 — Concepts

### Historical significance

LeNet-5, published in 1998, is the architecture that demonstrated that end-to-end gradient-based learning of convolutional networks could solve real industrial problems. It was deployed in the 1990s to read handwritten postal codes and bank cheques, processing over 10% of all cheques in the United States at its peak.

Before LeNet, most machine learning systems required hand-crafted feature extractors (SIFT, HOG, etc.) followed by a simple classifier. LeNet-5 showed that the features themselves could be learned from data using backpropagation through convolutional and subsampling layers — the core idea of modern deep learning.

### What makes LeNet-5 distinctive

1. **Convolutional layers** — shared weights detect features regardless of position.
2. **Subsampling (pooling) layers** — reduce spatial resolution, increasing robustness.
3. **Tanh activations** — the non-linearity used in the original (this library also supports modern variants with ReLU and BatchNorm).
4. **All layers trained end-to-end** with backpropagation, not in stages.
5. **Relatively small** (~33,000 parameters) — tractable on 1990s hardware.

### Architecture summary (28×28 MNIST input)

```
Input:  1 × 28 × 28

C1:  Conv 5×5, 6 maps   → 6 × 24 × 24     (156 params)
S2:  AvgPool 2×2        → 6 × 12 × 12     (  0 params)
C3:  Conv 5×5, 16 maps  → 16 × 8 × 8      (2,416 params)
S4:  AvgPool 2×2        → 16 × 4 × 4      (  0 params)
C5:  Conv 4×4, 120 maps → 120 × 1 × 1     (30,840 params)
                                           ─────────────
Total conv params:                          ~33,412

FC6: 120 → 84           Tanh              (10,164 params)
FC7: 84 → 10            Softmax/RBF       (  850 params)
─────────────────────────────────────────────────────────
Total:                                     ~44,426 params
```

### Adaptations in this library

The library's `LeNet5Config` provides three presets:
- **`.mnist()`** — 28×28 input, Tanh, no BatchNorm (faithful to the 1998 paper).
- **`.original()`** — 32×32 input (as in the original paper, which used padded 32×32 images).
- **`.modern()`** — 28×28 input, ReLU activations, BatchNorm after each conv layer.

The C5 kernel adapts automatically: 4×4 for 28×28 input (to produce 1×1 output), 5×5 for 32×32 input.

---

## Level 2 — Mathematics

### The original subsampling layer (S2, S4)

The original LeNet-5 subsampling layers are not pure average pooling. Each output is:

$$y_{n,c,h,w} = \tanh\!\left(w_c \cdot \frac{1}{4}\sum_{i,j \in 2\times 2} x_{n,c,2h+i,2w+j} + b_c\right)$$

where $w_c$ and $b_c$ are per-map **learnable** scale and bias parameters. This gives each channel a trainable sensitivity. The library simplifies this to standard AvgPool2D (no per-channel weights), which is the universally adopted modern interpretation.

### C3 partial connectivity

In the original paper, C3 does not connect all 16 output maps to all 6 input channels. Instead, specific maps connect to specific combinations of 3 or 4 input channels:

| Output map | Connected input maps |
|------------|---------------------|
| 0–5 | 3 consecutive maps |
| 6–11 | 4 consecutive maps |
| 12–14 | non-consecutive 4 maps |
| 15 | all 6 maps |

This asymmetric connectivity was designed to break symmetry and allow maps to detect complementary features. The library uses full $6 \to 16$ connectivity (the standard modern convention), which is simpler and equally effective.

### Parameter count derivation

**C1**: $6 \times (5^2 \times 1 + 1) = 6 \times 26 = 156$

**C3**: $16 \times (5^2 \times 6 + 1) = 16 \times 151 = 2{,}416$

**C5**: $120 \times (4^2 \times 16 + 1) = 120 \times 257 = 30{,}840$

(4×4 kernel because $8 - 4 + 1 = 5$... wait, input to C5 is $16 \times 4 \times 4$ from S4, and kernel is 4×4, so output is $1\times1$. For 28×28 input: after C3 and S4 we have $16 \times 4 \times 4$, so C5 uses a $4\times 4$ kernel to produce $120 \times 1 \times 1$.)

**FC6**: $84 \times (120 + 1) = 10{,}164$

**FC7**: $10 \times (84 + 1) = 850$

### The RBF output layer (original paper)

The original LeNet-5 used **Euclidean Radial Basis Functions** as the output layer rather than a standard softmax:

$$y_j = \sum_i (x_i - w_{ij})^2$$

where $w_{ij}$ are fixed (not trained) binary patterns representing the 10 digit classes. The training loss was:

$$E = \frac{1}{P} \sum_{p=1}^{P} \left( y_{D^p} + \log\!\left( e^{-j} + \sum_j e^{-y_j} \right) \right)$$

where $D^p$ is the correct class for sample $p$. Modern implementations replace this with a standard FC + Softmax + Cross-Entropy, which trains faster and is simpler.

### Why LeNet-5 still matters

- It introduced the core architectural vocabulary: conv → pool → conv → pool → dense.
- It demonstrated that weight sharing drastically reduces parameters without losing representational power.
- Its training methodology (data augmentation with elastic distortions, carefully tuned LeCun initialization) remained relevant for decades.
- The MNIST benchmark it established remains a standard sanity-check for new architectures.

**References**:
- LeCun, Y., et al. (1998). [Original paper above.]
- LeCun, Y., Bottou, L., Orr, G. B., & Müller, K. R. (1998). Efficient BackProp. In *Neural Networks: Tricks of the Trade*. [Implementation details.]
