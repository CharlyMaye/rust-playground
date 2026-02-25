# 20 — VGG: Very Deep Convolutional Networks

> **Navigation** ← [19 — ResNet](19-resnet.md) | [21 — AlexNet →](21-alexnet.md)

**Paper**: Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*. [arXiv:1409.1556]

---

## Level 1 — Concepts

### The VGG principle: uniform small filters

AlexNet (2012) used large filters: 11×11 and 5×5 in the early layers. VGGNet (2014) made a simple but powerful observation: **two stacked 3×3 convolutions have the same receptive field as one 5×5 convolution, but use fewer parameters and apply two non-linearities instead of one.**

This leads to a design rule: use **only 3×3 convolutions** throughout the entire network, stack them, and periodically halve the spatial size with MaxPool. Double the number of channels after each spatial reduction.

The result is a very regular, easy-to-understand architecture that performs exceptionally well and whose design principles influenced almost every CNN architecture that followed.

### VGG-16 architecture (224×224 input)

```
Input:    3 × 224 × 224

Block 1:  2 × Conv3×3(64) + ReLU  → 64 × 224 × 224
          MaxPool 2×2              → 64 × 112 × 112

Block 2:  2 × Conv3×3(128) + ReLU → 128 × 112 × 112
          MaxPool 2×2              → 128 × 56 × 56

Block 3:  3 × Conv3×3(256) + ReLU → 256 × 56 × 56
          MaxPool 2×2              → 256 × 28 × 28

Block 4:  3 × Conv3×3(512) + ReLU → 512 × 28 × 28
          MaxPool 2×2              → 512 × 14 × 14

Block 5:  3 × Conv3×3(512) + ReLU → 512 × 14 × 14
          MaxPool 2×2              → 512 × 7 × 7

FC:       512×7×7 → 4096 → ReLU → Dropout(0.5)
          4096    → 4096 → ReLU → Dropout(0.5)
          4096    → 1000 → Softmax
```

### VGG variants in this library

| Variant | Conv blocks | Parameters |
|---------|-------------|------------|
| **VGG-11** | 1/1/2/2/2 convs | ~133M |
| **VGG-16** | 2/2/3/3/3 convs | ~138M |
| **VGG-19** | 2/2/4/4/4 convs | ~144M |
| **CIFAR-10** | 4 mini-blocks for 32×32 | ~15M |

The `VGGConfig` type defines the blocks as a list of `(num_convs, channels)` pairs, making it easy to create custom variants.

---

## Level 2 — Mathematics

### 3×3 stacking justification

Two stacked 3×3 convolutions (each with stride 1, same padding) have an effective receptive field of $5 \times 5$. Three stacked 3×3 convolutions have a receptive field of $7 \times 7$.

**Parameter comparison** for $C$ input and output channels:

| Equivalent receptive field | Configuration | Parameters |
|---------------------------|--------------|-----------|
| 5×5 | Single Conv5×5 | $5^2 C^2 = 25C^2$ |
| 5×5 | 2× Conv3×3 | $2 \times 3^2 C^2 = 18C^2$ |
| 7×7 | Single Conv7×7 | $7^2 C^2 = 49C^2$ |
| 7×7 | 3× Conv3×3 | $3 \times 3^2 C^2 = 27C^2$ |

Additionally, two 3×3 layers have two ReLU activations vs one for the 5×5 layer — more non-linearity increases the discriminative power of the feature hierarchy.

### Total parameter count (VGG-16)

**Convolutional layers** — representative terms:

$$\text{Block 1}: 2 \times (3^2 \times 3 \times 64 + 64) = 2 \times 1792 = 3{,}584$$
$$\text{Block 2}: 2 \times (3^2 \times 64 \times 128 + 128) = 2 \times 73{,}856 = 147{,}712$$
$$\vdots$$
$$\text{Total conv}: \approx 14{,}714{,}688$$

**Fully-connected layers**:
$$4096 \times (512 \times 7^2 + 1) + 4096 \times 4097 + 1000 \times 4097 \approx 123{,}667{,}456$$

Total: ~138M parameters, of which **89%** are in the FC layers. This is the main inefficiency of VGG: the three FC layers dominate the parameter count despite contributing less to accuracy than the convolutional feature extractor.

Modern architectures (ResNet, EfficientNet) replace the FC stack with GlobalAvgPool, reducing the parameter count from 100M to single-digit millions.

### The VGG design philosophy and its limits

VGG showed that **depth with small filters** outperforms shallow networks with large filters. It was the first network to systematically study how depth affects performance (VGG-11 vs VGG-16 vs VGG-19).

Limitations:
1. **High memory**: 138M parameters require ~550MB in `f32`. Training a batch with gradients and optimizer state requires several GBs.
2. **No skip connections**: training VGG-19 (19 layers) was already pushing the limits of gradient flow. Beyond ~20 layers, plain networks degrade (addressed by ResNet).
3. **Slow inference**: the large FC layers make inference expensive.
4. **Not input-size invariant**: the FC layer size is fixed to $512 \times 7 \times 7$, requiring exactly 224×224 input. GlobalAvgPool removes this constraint.

**References**:
- Simonyan, K., & Zisserman, A. (2015). [Original paper above.]
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. §9.7 — discusses historical context of VGGNet.
