# 14 — Pooling Layers

> **Navigation** ← [13 — Conv2D](13-conv2d.md) | [15 — Batch Normalization →](15-batchnorm.md)

---

## Level 1 — Concepts

### Purpose of pooling

Pooling layers reduce the spatial dimensions of feature maps. They serve three roles:

1. **Computational efficiency**: halving $H$ and $W$ reduces the number of values by 4×, making subsequent layers much cheaper.
2. **Increased receptive field**: after pooling, each neuron in the next layer effectively "sees" a larger region of the original input.
3. **Local invariance**: a small translation or distortion of the input produces the same pooled output, as long as the important features stay within the same pooling window.

### MaxPool2D

Takes the **maximum** value inside each $p \times p$ window slid with stride $s$. The maximum value represents "is this feature present anywhere in this window?" — even a faint detection is preserved if it is the strongest in the region.

MaxPool is the standard choice in classical CNNs (LeNet, AlexNet, VGG, ResNet).

### AvgPool2D

Takes the **average** value inside each window. Smoother than MaxPool; used as the sub-sampling layer in the original LeNet-5 (the paper calls it "subsampling").

### GlobalAvgPool2D

Reduces the entire spatial extent of each feature map to a single number: the spatial average. Given input $[N, C, H, W]$, the output is $[N, C, 1, 1]$ regardless of $H$ and $W$.

This is used in modern architectures (ResNet, EfficientNet) instead of large fully-connected layers. It has zero parameters, prevents overfitting, and makes the network input-size invariant.

---

## Level 2 — Mathematics

### MaxPool2D

Given input $X \in \mathbb{R}^{N \times C \times H_{in} \times W_{in}}$, kernel $p$, stride $s$:

$$Y_{n, c, h, w} = \max_{0 \leq i < p,\; 0 \leq j < p} X_{n, c,\; hs+i,\; ws+j}$$

**Output size**: $H_{out} = \lfloor (H_{in} - p) / s \rfloor + 1$.

**Backward pass**: gradients are routed only to the position that achieved the maximum (the **argmax**). All other positions in the window receive zero gradient:

$$\frac{\partial \mathcal{L}}{\partial X_{n,c,i^*, j^*}} = \frac{\partial \mathcal{L}}{\partial Y_{n,c,h,w}}$$

where $(i^*, j^*)$ is the argmax position. The library stores these indices during the forward pass (in the `(output, max_indices)` tuple returned by `maxpool2d()`) so the backward pass can route gradients correctly.

### AvgPool2D

$$Y_{n, c, h, w} = \frac{1}{p^2} \sum_{i=0}^{p-1} \sum_{j=0}^{p-1} X_{n, c,\; hs+i,\; ws+j}$$

**Backward pass**: the gradient is divided equally among all $p^2$ positions in the window:

$$\frac{\partial \mathcal{L}}{\partial X_{n,c,hs+i,ws+j}} = \frac{1}{p^2} \cdot \frac{\partial \mathcal{L}}{\partial Y_{n,c,h,w}}$$

### GlobalAvgPool2D

$$Y_{n, c} = \frac{1}{H \cdot W} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} X_{n, c, h, w}$$

This is AvgPool with $p = H = W$ (covering the full spatial extent). The output has shape $[N, C]$ (or $[N, C, 1, 1]$ before the head).

**Why replace FC layers with GlobalAvgPool?**

A classic architecture (AlexNet, early VGG) feeds the last conv layer (e.g., $512 \times 7 \times 7$) into a fully-connected layer:

$$\text{FC params} = 512 \times 7 \times 7 \times 4096 = 102{,}760{,}448$$

GlobalAvgPool compresses $512 \times 7 \times 7$ → $512$ with zero parameters. A small FC head then maps $512$ → $C_{\text{classes}}$:

$$\text{HC params} = 512 \times C_{\text{classes}} \ll 100\text{M}$$

Lin et al. (2014) show this also acts as a structural regularizer: the classification layer directly interprets the spatial average of each feature map as a confidence score for each class — a much simpler inductive bias than a general FC layer.

**Reference**:
- Lin, M., Chen, Q., & Yan, S. (2014). Network In Network. *ICLR*. [Introduced GlobalAvgPool.]
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. §9.3.

### Pooling and translation invariance

MaxPool provides **local translation invariance**: if a feature is detected anywhere within the pooling window, the output is (approximately) the same. Formally, if the argmax position shifts by $\delta$ pixels but stays within the window, $Y$ is unchanged.

This is a coarser version of the equivariance provided by convolution (see [12 — CNN Basics](12-cnn-basics.md)): rather than precisely tracking the feature location, pooling discards fine-grained positional information in exchange for robustness.

**Reference**: Boureau, Y. L., Ponce, J., & LeCun, Y. (2010). A Theoretical Analysis of Feature Pooling in Visual Recognition. *ICML*.
