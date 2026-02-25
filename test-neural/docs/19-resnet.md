# 19 — ResNet: Deep Residual Learning

> **Navigation** ← [18 — LeNet-5](18-lenet.md) | [20 — VGG →](20-vgg.md)

**Paper**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*. [Best Paper Award.]

---

## Level 1 — Concepts

### The depth barrier before ResNet

By 2015, the community knew that deeper networks should be more powerful (the Universal Approximation Theorem; Montufar et al. 2014 on linear regions). But in practice, simply stacking more layers made performance *worse*, even on the training set — not due to overfitting, but due to optimization difficulty.

The problem: gradients vanish as they are multiplied through dozens of layers (see [04 — Backpropagation](04-backpropagation.md)). A 56-layer plain network performed worse than its 20-layer counterpart on CIFAR-10 — a striking demonstration that depth alone was not enough.

### The residual idea

He et al. proposed a simple reformulation. Instead of asking a block of layers to learn the desired mapping $H(x)$, let them learn the **residual**:

$$F(x) = H(x) - x \implies H(x) = F(x) + x$$

The block's output is the sum of the learned residual and the original input (via a **skip connection**):

```
x ──┬──► Conv → BN → ReLU → Conv → BN ──+──► ReLU
    │                                    │
    └────────────────────────────────────┘  (identity skip)
```

**Why does this help?**
- If the optimal transformation is close to the identity, the block only needs to learn $F \approx 0$ — which is much easier than learning $H \approx \text{identity from scratch}$.
- Gradients flow directly back through the skip connection without passing through any weights — a **gradient highway** that prevents vanishing.

### Results

ResNet-152 (152 layers) won the 2015 ImageNet challenge with a top-5 error of 3.57% — surpassing human-level performance (~5%) and nearly halving the previous best. Networks up to 1000 layers were trained successfully.

### Residual block variants

| Variant | Used in | Structure |
|---------|---------|-----------|
| **BasicBlock** | ResNet-18, 34 | Conv3×3 → BN → ReLU → Conv3×3 → BN |
| **Bottleneck** | ResNet-50, 101, 152 | Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN |

The library implements BasicBlock (`ResidualBlock`), which is sufficient for MNIST, CIFAR-10, and small image tasks.

---

## Level 2 — Mathematics

### Residual learning formulation

Let $x$ be the input to a residual block and $\mathcal{F}(x; \{W_i\})$ be the residual function (the layers inside the block). The block output is:

$$y = \mathcal{F}(x; \{W_i\}) + x$$

The addition is element-wise — it requires $\mathcal{F}(x)$ and $x$ to have the same shape.

### Projection shortcut (dimension-changing blocks)

When stride $> 1$ or when the number of channels changes (e.g., 64 → 128), the identity shortcut cannot be used directly. A **projection shortcut** is applied:

$$y = \mathcal{F}(x; \{W_i\}) + W_s x$$

where $W_s \in \mathbb{R}^{C_{out} \times C_{in}}$ is a $1 \times 1$ Conv2D with stride $s$, followed by BatchNorm. This linearly maps $x$ to the same shape as $\mathcal{F}(x)$.

He et al. show that the projection shortcut is only needed at dimension-changing layers; identity shortcuts suffice everywhere else.

### Gradient flow through skip connections

The gradient of the loss with respect to an early layer is:

$$\frac{\partial \mathcal{L}}{\partial x_\ell} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_\ell}$$

For a chain of $L - \ell$ residual blocks, unrolling the recursion $x_{i+1} = x_i + \mathcal{F}(x_i)$:

$$x_L = x_\ell + \sum_{i=\ell}^{L-1} \mathcal{F}(x_i; \{W_i\})$$

Therefore:

$$\frac{\partial x_L}{\partial x_\ell} = 1 + \frac{\partial}{\partial x_\ell} \sum_{i=\ell}^{L-1} \mathcal{F}(x_i; \{W_i\})$$

The **+1 term** from the skip connection ensures the gradient contains an additive identity component regardless of the depth — it never vanishes to zero even if the residuals collapse.

### BasicBlock structure

$$\text{BasicBlock}(x) = \text{ReLU}\!\left(\text{BN}(\text{Conv}_{3\times3}(\text{ReLU}(\text{BN}(\text{Conv}_{3\times3}(x))))) + s(x)\right)$$

where $s(x)$ is the identity or projection shortcut.

**No bias in conv layers**: biases before BatchNorm are redundant — BatchNorm's $\beta$ already provides a free shift. Removing biases reduces parameters slightly.

### The library's `ResNetBuilder`

| Method | Effect |
|--------|--------|
| `.input_channels(c)` | Number of input image channels (1 for grayscale, 3 for RGB) |
| `.input_size(s)` | Spatial input size (sets whether stem uses MaxPool) |
| `.channels(&[c1, c2, c3])` | Channel counts per stage |
| `.blocks(&[n1, n2, n3])` | Number of residual blocks per stage |
| `.num_classes(k)` | Output size |
| `.stem_with_pool(false)` | For small inputs (MNIST, CIFAR): omit initial MaxPool to preserve spatial resolution |

**MNIST preset**: 1 channel, 28×28, channels=[16,32,64], blocks=[2,2,2], no stem pooling. Total ~87,000 parameters — much smaller than VGG/AlexNet yet significantly more powerful than LeNet.

### Why residuals learn the correction and not the full mapping

The authors give an intuitive analogy: it is easier to optimize a function close to zero than to optimize an arbitrary function. If the optimal mapping at a given block is the identity ($H(x) = x$), the residual path only needs to push $F \to 0$, which gradient descent does easily. Without skip connections, the layers would need to learn the identity mapping from scratch — empirically much harder.

This also explains why the initial training loss is lower for ResNets: at initialization, small random weights give $\mathcal{F}(x) \approx 0$, so $y \approx x$ — the network starts as (approximately) an identity mapping and is refined from there.

**References**:
- He, K., et al. (2016). [Original paper above.]
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *ECCV*. [Improved residual unit design: BN→ReLU→Conv order.]
- Veit, A., Wilber, M., & Belongie, S. (2016). Residual Networks Behave Like Ensembles of Relatively Shallow Networks. *NeurIPS*. [Interpretation of ResNets as implicit ensembles.]
