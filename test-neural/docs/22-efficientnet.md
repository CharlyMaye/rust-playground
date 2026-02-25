# 22 — EfficientNet: Compound Scaling for CNN

> **Navigation** ← [21 — AlexNet](21-alexnet.md) | [README →](README.md)

**Paper**: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.

---

## Level 1 — Concepts

### The scaling problem

Given a well-designed baseline network, how do you make it more accurate? Prior work scaled along only one dimension:
- **Wider** (more channels per layer): NASNet, WideResNet.
- **Deeper** (more layers): ResNet-50/101/152, DenseNet.
- **Higher resolution** (larger input images): GPipe.

Each of these helps, but with diminishing returns. Arbitrarily increasing one dimension while keeping the others fixed is inefficient.

### Compound scaling

EfficientNet's key insight: **depth, width, and resolution should be scaled together** using a fixed ratio derived from a small grid search. If you have a compute budget multiplier $\phi$:

$$\text{Depth}: d = \alpha^\phi, \quad \text{Width}: w = \beta^\phi, \quad \text{Resolution}: r = \gamma^\phi$$

subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (i.e., doubling compute means doubling FLOPs proportionally across all dimensions), with $\alpha, \beta, \gamma \geq 1$.

The baseline EfficientNet-B0 was found by Neural Architecture Search (NAS). EfficientNet-B1 through B7 apply increasing $\phi$.

### MBConv: the core building block

Each EfficientNet stage uses **Mobile Inverted Bottleneck Convolution** (MBConv), which was originally proposed in MobileNetV2 (Sandler et al., 2018). It consists of four sub-operations:

1. **Expansion**: a 1×1 Conv increases channels by `expand_ratio` (e.g., ×6). Called "inverted" because it expands first (unlike traditional bottlenecks that compress first).
2. **Depthwise convolution**: a $k \times k$ DepthwiseConv (see [16 — Depthwise Convolution](16-depthwise-conv.md)) processes each channel independently.
3. **Squeeze-and-Excitation (SE)**: a small attention mechanism that re-weights channels based on global context.
4. **Projection**: a 1×1 Conv compresses channels back to the output size.

A skip connection is added when input and output shapes match (stride = 1, same channels).

### Squeeze-and-Excitation

SE learns **which channels are most important for each input** and scales them accordingly. Given a feature map $X$:
1. **Squeeze**: GlobalAvgPool → one scalar per channel.
2. **Excitation**: FC(ratio=4) → ReLU → FC → Sigmoid → per-channel weights in (0,1).
3. **Scale**: multiply $X$ channel-wise by the learned weights.

With SE ratio 0.25, the excitation network has $C/4$ hidden units — very few parameters but a powerful attention effect.

---

## Level 2 — Mathematics

### Compound scaling derivation

FLOPs of a CNN scale approximately as:

$$\text{FLOPs} \propto d \cdot w^2 \cdot r^2$$

(linear in depth, quadratic in width since both input and output channels scale, quadratic in resolution).

Doubling the overall compute budget ($2\times$ FLOPs) can be achieved by any combination satisfying $\tilde{d} \cdot \tilde{w}^2 \cdot \tilde{r}^2 = 2$. The compound constraint with the grid-searched exponents $(\alpha, \beta, \gamma)$ ensures that doubling $\phi$ approximately doubles FLOPs:

$$d \cdot w^2 \cdot r^2 = \alpha^\phi (\beta^\phi)^2 (\gamma^\phi)^2 = (\alpha \beta^2 \gamma^2)^\phi \approx 2^\phi$$

For EfficientNet-B0: $\alpha = 1.2, \beta = 1.1, \gamma = 1.15$ (found by grid search), giving $\alpha \beta^2 \gamma^2 = 1.2 \times 1.21 \times 1.3225 \approx 1.92 \approx 2$.

### MBConv forward pass

Let $x \in \mathbb{R}^{N \times C \times H \times W}$ be the input. With `expand_ratio` $e$ and output channels $C'$:

**Step 1 — Expansion** (skipped if $e = 1$):

$$h = \text{SiLU}(\text{BN}(\text{Conv}_{1 \times 1}(x))) \in \mathbb{R}^{N \times eC \times H \times W}$$

**Step 2 — Depthwise convolution**:

$$h = \text{SiLU}(\text{BN}(\text{DepthwiseConv}_{k \times k, s}(h))) \in \mathbb{R}^{N \times eC \times H' \times W'}$$

**Step 3 — Squeeze-and-Excitation**:

$$z = \text{GlobalAvgPool}(h) \in \mathbb{R}^{N \times eC}$$

$$s = \sigma(\text{FC}_{eC}(\text{ReLU}(\text{FC}_{eC/r}(z)))) \in \mathbb{R}^{N \times eC}$$

$$h = h \odot s[\ldots, \text{broadcast}] \quad \text{(channel-wise scaling)}$$

where $r$ is the SE ratio (default 0.25, so $eC/r = 4eC/4 = eC/4$ hidden units for SE ratio $1/4$).

**Step 4 — Projection** (no activation):

$$y = \text{BN}(\text{Conv}_{1 \times 1}(h)) \in \mathbb{R}^{N \times C' \times H' \times W'}$$

**Skip connection** (only when $s = 1$ and $C = C'$):

$$\text{output} = y + x$$

### SiLU activation

EfficientNet uses **SiLU** (Sigmoid Linear Unit), also called Swish:

$$\text{SiLU}(z) = z \cdot \sigma(z)$$

See [02 — Activations](02-activations.md) §Swish for the formula and derivative.

### EfficientNet-B0 stage configuration

| Stage | Block | Channels in | Channels out | Kernel | Layers | Stride | Expand ratio |
|-------|-------|-------------|--------------|--------|--------|--------|-------------|
| Stem | Conv3×3 | 3 | 32 | 3 | 1 | 2 | — |
| 1 | MBConv1 | 32 | 16 | 3 | 1 | 1 | 1 |
| 2 | MBConv6 | 16 | 24 | 3 | 2 | 2 | 6 |
| 3 | MBConv6 | 24 | 40 | 5 | 2 | 2 | 6 |
| 4 | MBConv6 | 40 | 80 | 3 | 3 | 2 | 6 |
| 5 | MBConv6 | 80 | 112 | 5 | 3 | 1 | 6 |
| 6 | MBConv6 | 112 | 192 | 5 | 4 | 2 | 6 |
| 7 | MBConv6 | 192 | 320 | 3 | 1 | 1 | 6 |
| Head | Conv1×1 + GAP + FC | 320 | 1280 → classes | 1 | 1 | 1 | — |

Total: ~5.3M parameters for B0 (vs ~25M for ResNet-50 and ~138M for VGG-16), with comparable or better ImageNet accuracy.

### Accuracy–efficiency frontier

EfficientNet-B7 (largest variant in the series, $\phi=7$) achieves 84.4% top-1 accuracy on ImageNet with 66M parameters — higher accuracy than GPipe (557M parameters) at one-eighth the parameters. This places it on the **Pareto frontier** of accuracy vs parameter count.

### `EfficientNetConfig` in this library

| Preset | $\phi$ | Width mult. | Depth mult. | Resolution |
|--------|--------|-------------|-------------|------------|
| `b0()` | 1.0 | 1.0 | 1.0 | 224 |
| `b1()` | 1.1 | 1.0 | 1.1 | 240 |
| `b2()` | 1.2 | 1.1 | 1.2 | 260 |
| `cifar10()` | mini | — | — | 32×32 |

The `scale_width(m)` and `scale_depth(m)` methods apply custom multipliers, making it easy to define intermediate variants.

**References**:
- Tan, M., & Le, Q. V. (2019). [Original paper above.]
- Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*. [Introduced MBConv.]
- Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. *CVPR*. [Introduced Squeeze-and-Excitation.]
- Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861. [Depthwise separable convolutions.]
- Tan, M., et al. (2019). MnasNet: Platform-Aware Neural Architecture Search for Mobile. *CVPR*. [NAS methodology used to find EfficientNet-B0 baseline.]
