# 12 — Introduction to Convolutional Neural Networks

> **Navigation** ← [11 — Metrics](11-metrics.md) | [13 — Conv2D →](13-conv2d.md)

---

## Level 1 — Concepts

### The problem with dense layers on images

A 28×28 grayscale image has 784 pixels. Connecting it to a dense layer of 512 neurons already requires 401,408 weights. A 224×224 color image has 150,528 inputs — one dense layer to 512 neurons needs 77 million weights, and we haven't even started going deep.

More critically, dense layers ignore the structure of images entirely: the pixel at position (0, 0) and the pixel at (0, 1) are treated as completely independent inputs. Images have two properties that dense layers cannot exploit:

1. **Local structure** — meaningful patterns (edges, corners, textures, shapes) span small spatial neighborhoods, not the whole image.
2. **Translation invariance** — an edge is an edge whether it appears at the top-left or the bottom-right. We want the same detector to fire in both cases.

### The convolutional solution

A **convolutional layer** uses a small **filter** (also called a kernel) that slides across the spatial dimensions of the input. At each position, the filter computes a dot product with the underlying patch — measuring how much that patch resembles the pattern the filter encodes.

```
Input patch:    Filter (3×3):    Output:
1  0  1         1  0 -1          1·1 + 0·0 + 1·(-1)
0  1  0    ×    0  1  0    →   + 0·0 + 1·1 + 0·0    = 2
1  0  0        -1  0  1        + 1·(-1) + 0·0 + 0·1
```

The same filter is applied at every spatial position. This gives two major advantages:

- **Local connectivity**: each output value depends on only a small patch ($k \times k$ pixels) of the input.
- **Weight sharing**: the 9 filter weights (for a 3×3 kernel) are reused $H_{out} \times W_{out}$ times. A Conv2D layer with 64 filters of size 3×3 has only $64 \times (9 \times C_{in} + 1)$ parameters regardless of image size.

### The CNN data pipeline

A typical CNN transforms the input through a hierarchy of spatial operations:

```
Image (H × W × C)
  ──► Conv + ReLU       → low-level features (edges, gradients)
  ──► MaxPool           → spatial downsampling (×2)
  ──► Conv + ReLU       → mid-level features (textures, parts)
  ──► MaxPool           → spatial downsampling (×2)
  ──► Conv + ReLU       → high-level features (objects, faces)
  ──► GlobalAvgPool     → one feature value per channel
  ──► Dense + Softmax   → class probabilities
```

As we go deeper: spatial resolution decreases, number of channels increases, and features become more abstract and semantically meaningful.

### NCHW memory layout

The `cma-cnn` crate stores all feature tensors in **NCHW** format:

```
Shape: [N, C, H, W]
  N = batch size
  C = channels (feature maps)
  H = height
  W = width
```

This means all values of one channel for one image are stored contiguously (channel-major), which is efficient for the inner loops of convolution.

---

## Level 2 — Mathematics

### Output spatial dimensions

Given an input of spatial size $H_{in} \times W_{in}$, a Conv2D with kernel size $k$, stride $s$, and padding $p$ produces:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1, \qquad W_{out} = \left\lfloor \frac{W_{in} + 2p - k}{s} \right\rfloor + 1$$

Special cases:
- **Valid** ($p = 0$, $s = 1$): $H_{out} = H_{in} - k + 1$ (shrinks by $k-1$).
- **Same** ($p = \lfloor k/2 \rfloor$, $s = 1$): $H_{out} = H_{in}$ (preserves spatial size).
- **Stride 2** ($p = 1$, $s = 2$): $H_{out} = \lceil H_{in}/2 \rceil$ (halves spatial size).

### Parameter count

A Conv2D layer with $C_{in}$ input channels, $C_{out}$ output channels, kernel $k \times k$, with bias:

$$\text{params} = C_{out} \cdot (k^2 \cdot C_{in} + 1)$$

Representative values:

| Layer | $C_{in}$ | $k$ | $C_{out}$ | Parameters |
|-------|---------|-----|---------|------------|
| AlexNet Conv1 | 3 | 11 | 96 | 34,944 |
| VGG Conv (typical) | 256 | 3 | 256 | 590,080 |
| ResNet Conv3×3 | 64 | 3 | 64 | 36,928 |
| Dense equiv. (32×32 spatial) | — | — | — | 33,554,432 (×906×) |

### Receptive field growth

The **effective receptive field** (ERF) of a neuron at layer $l$ is the region of the original input it depends on. For stacked Conv2D layers with kernel $k_i$ and stride $s_i$:

$$\text{ERF}_l = 1 + \sum_{i=1}^{l} (k_i - 1) \prod_{j=1}^{i-1} s_j$$

For $l$ stacked 3×3 convolutions with stride 1:

$$\text{ERF}_l = 1 + 2l$$

So 3 stacked 3×3 convolutions have $\text{ERF} = 7$, identical to one 7×7 convolution, but use $3 \times 9 = 27$ parameters per input–output channel pair vs $49$, and apply three non-linearities instead of one.

### Translation equivariance (formal)

Define the translation operator $T_\delta$ that shifts a 2D signal by $\delta$ pixels. A convolution $f_W$ is **equivariant** to translation:

$$f_W(T_\delta(x)) = T_\delta(f_W(x))$$

This follows directly from the sliding-window definition of convolution. The same filter response is produced at every location — the feature map shifts exactly as the input shifts.

**Important**: equivariance means the *location* of a detected feature is preserved in the feature map. It does not mean the output is unchanged when the input shifts. True **invariance** (same output regardless of position) requires pooling or global operations.

**Reference**: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. [The foundational CNN paper.] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Chapter 9.
