# 16 — Depthwise Convolution

> **Navigation** ← [15 — Batch Normalization](15-batchnorm.md) | [17 — Autograd →](17-autograd.md)

---

## Level 1 — Concepts

### Standard convolution is expensive

In a standard Conv2D, every output channel is computed from every input channel — the filter has shape $k \times k \times C_{in}$ and there are $C_{out}$ such filters. The total parameter count is $C_{out} \times k^2 \times C_{in}$.

When $C_{in}$ and $C_{out}$ are both large (e.g., 256 or 512 channels in deep networks), this becomes the computational bottleneck.

### The depthwise idea

**Depthwise convolution** applies a separate filter to each input channel independently — no mixing between channels. Each of the $C_{in}$ channels has its own $k \times k$ filter, for a total of $C_{in} \times k^2$ parameters.

The output has the same number of channels as the input: each channel is filtered independently. To change the number of channels, a separate **pointwise convolution** ($1 \times 1$ Conv2D) is applied afterwards. This two-step combination is called **depthwise separable convolution**.

### Why it matters

Depthwise separable convolution achieves nearly the same representation power as standard convolution at a fraction of the cost. This is the foundation of efficient architectures: **MobileNet** (Howard et al., 2017) achieves competitive classification accuracy on ImageNet using a network built entirely of depthwise separable blocks, with 8–9× fewer operations than VGG-16.

In this library, `DepthwiseConv2D` is used inside the **MBConv blocks** of EfficientNet (see [22 — EfficientNet](22-efficientnet.md)).

---

## Level 2 — Mathematics

### Depthwise convolution

For input $X \in \mathbb{R}^{N \times C_{in} \times H_{in} \times W_{in}}$ and per-channel weights $W \in \mathbb{R}^{C_{in} \times 1 \times k \times k}$:

$$Y_{n, c, h, w} = \sum_{i=0}^{k-1} \sum_{j=0}^{k-1} W_{c, 0, i, j} \cdot X_{n, c,\; hs+i-p,\; ws+j-p}$$

Each output channel $c$ depends **only** on input channel $c$ — no cross-channel mixing. The output shape is $[N, C_{in}, H_{out}, W_{out}]$.

### Pointwise convolution

A $1 \times 1$ Conv2D applies one dot product per spatial position across all channels:

$$Z_{n, c_{out}, h, w} = b_{c_{out}} + \sum_{c_{in}=0}^{C_{in}-1} V_{c_{out}, c_{in}} \cdot Y_{n, c_{in}, h, w}$$

This mixes channel information without any spatial context.

### Parameter and computation comparison

For input $C_{in}$ channels, output $C_{out}$ channels, kernel $k$, spatial output $H_{out} \times W_{out}$:

| Method | Parameters | Multiply-adds |
|--------|-----------|--------------|
| Standard Conv2D | $C_{out} \cdot k^2 \cdot C_{in}$ | $C_{out} \cdot k^2 \cdot C_{in} \cdot H_{out} W_{out}$ |
| Depthwise + Pointwise | $k^2 \cdot C_{in} + C_{in} \cdot C_{out}$ | $(k^2 + C_{out}) \cdot C_{in} \cdot H_{out} W_{out}$ |

**Reduction factor**:

$$\frac{k^2 C_{in} + C_{in} C_{out}}{k^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{k^2}$$

For $C_{out} = 256$ and $k = 3$: savings factor $\approx 1/9 = 0.11$ — **9× fewer parameters and operations**.

### He initialization for DepthwiseConv2D

Each filter has fan-in $= k^2$ (only the spatial extent, since channels are independent):

$$W_{c, 0, i, j} \sim \mathcal{N}\!\left(0,\; \frac{2}{k^2}\right)$$

This is the He formula applied to the per-channel fan-in, consistent with [05 — Weight Initialization](05-weight-init.md).

### Why no im2col for depthwise?

Standard Conv2D uses im2col to turn the nested loops into a GEMM, which benefits from BLAS-level optimization. Depthwise convolution has each channel fully independent — the "matrix" per channel would be too small to benefit from GEMM. The library therefore uses direct nested loops for `DepthwiseConv2D`, which are memory-efficient and avoid the im2col overhead for this case.

**Reference**: Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861. [Original depthwise separable factorization for neural networks.] Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. *CVPR*.
