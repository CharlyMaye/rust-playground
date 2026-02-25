# 13 — Conv2D: The 2D Convolution Layer

> **Navigation** ← [12 — CNN Basics](12-cnn-basics.md) | [14 — Pooling →](14-pooling.md)

---

## Level 1 — Concepts

### What Conv2D computes

A Conv2D layer holds $C_{out}$ learnable filters, each of shape $k \times k \times C_{in}$. For every position $(h, w)$ in the output:
1. Extract the $k \times k \times C_{in}$ patch of the input centered (or starting) at $(h, w)$.
2. Compute the dot product between this patch and each of the $C_{out}$ filters.
3. Add a learnable bias and apply an activation.

The result is a **feature map** of shape $C_{out} \times H_{out} \times W_{out}$: at every spatial position, $C_{out}$ numbers describe which features were detected and how strongly.

### Padding

- **Valid** (no padding): the filter cannot be placed at the edges. The output is smaller than the input.
- **Same**: the input is padded with zeros around the border so the output is the same spatial size as the input (for stride 1). This is the default in most modern architectures.
- **Fixed(n)**: explicit padding amount, used for asymmetric cases.

### Stride

The stride $s$ controls how far the filter moves between positions. Stride 2 halves the spatial dimensions, performing a form of learned downsampling (more expressive than max pooling alone).

### im2col + GEMM: how the library accelerates convolution

Naïve convolution using nested loops (batch × output channels × output height × output width × kernel height × kernel width × input channels) is very slow. The library uses the **im2col** ("image to columns") transformation to turn convolution into a single large matrix multiplication (GEMM), which is highly optimized by modern linear algebra libraries.

---

## Level 2 — Mathematics

### Cross-correlation definition

The operation deep learning calls "convolution" is technically **cross-correlation** (filters are not flipped). For a single input channel and output channel:

$$y_{h,w} = \text{bias} + \sum_{i=0}^{k-1} \sum_{j=0}^{k-1} W_{i,j} \cdot x_{h \cdot s + i - p,\; w \cdot s + j - p}$$

where out-of-bounds positions are zero (padding). Summing over all $C_{in}$ input channels and extending to $C_{out}$ filters:

$$Y_{c_{out}, h, w} = b_{c_{out}} + \sum_{c_{in}=0}^{C_{in}-1} \sum_{i=0}^{k-1} \sum_{j=0}^{k-1} W_{c_{out}, c_{in}, i, j} \cdot X_{c_{in},\; h \cdot s + i - p,\; w \cdot s + j - p}$$

For a batch of $N$ images, this extends independently over $n = 0, \ldots, N-1$.

### im2col transformation

The **im2col** algorithm reshapes the input tensor so that all patches that the filter visits become columns (or rows) of a 2D matrix. Then the entire convolution becomes a single matrix multiplication.

Let $M = H_{out} \times W_{out}$ (number of filter positions) and $K = k^2 \times C_{in}$ (patch size). im2col produces:

$$\text{col} \in \mathbb{R}^{M \times K}$$

where row $m$ contains the flattened $k \times k \times C_{in}$ patch at position $m$.

The weight matrix is reshaped as:

$$\text{W\_mat} \in \mathbb{R}^{C_{out} \times K}$$

Then the full convolution output (for one image) is:

$$\text{Y\_mat} = \text{W\_mat} \cdot \text{col}^T \in \mathbb{R}^{C_{out} \times M}$$

which is reshaped back to $\mathbb{R}^{C_{out} \times H_{out} \times W_{out}}$.

**Complexity**: naïve convolution is $O(N \cdot C_{out} \cdot H_{out} \cdot W_{out} \cdot k^2 \cdot C_{in})$. im2col+GEMM has the same asymptotic complexity but benefits from BLAS GEMM implementations that are cache-optimized and SIMD-vectorized — typically 10–100× faster in practice.

**Memory cost**: im2col allocates an intermediate matrix of size $M \times K = H_{out} W_{out} \times k^2 C_{in}$, which can be large. This is a space–time trade-off.

**Reference**: Chellapilla, K., Puri, S., & Simard, P. (2006). High Performance Convolutional Neural Networks for Document Processing. *IWFHR*.

### Fast path: zero-padding optimization

The library implements a separate fast inner loop for the common case `padding = 0`: it avoids bounds-checking for out-of-bounds accesses in the innermost loop, giving an additional constant-factor speedup on valid-padded convolutions.

### He initialization for Conv2D

Since a Conv2D filter connects to $k^2 \times C_{in}$ input values (the fan-in of each output neuron), He initialization sets:

$$W_{c_{out}, c_{in}, i, j} \sim \mathcal{N}\!\left(0,\; \frac{2}{k^2 \cdot C_{in}}\right)$$

This ensures that, at network initialization, the variance of the activations is preserved through the conv layer (assuming ReLU activations). See [05 — Weight Initialization](05-weight-init.md) for the derivation.

Biases are initialized to 0 (as standard for layers immediately before or after BatchNorm; see [15 — BatchNorm](15-batchnorm.md)).

### Conv2D backward pass (gradient computation)

For backpropagation, three gradients are needed:

**Gradient w.r.t. weights** (used for the optimizer update):

$$\frac{\partial \mathcal{L}}{\partial W_{c_{out}, c_{in}, i, j}} = \sum_{h,w} \delta_{c_{out}, h, w} \cdot X_{c_{in},\, hs+i-p,\, ws+j-p}$$

In matrix form: $\partial \mathcal{L}/\partial \text{W\_mat} = \delta\_\text{mat} \cdot \text{col}$

**Gradient w.r.t. bias**:

$$\frac{\partial \mathcal{L}}{\partial b_{c_{out}}} = \sum_{h,w} \delta_{c_{out}, h, w}$$

**Gradient w.r.t. input** (passed to the previous layer):

$$\frac{\partial \mathcal{L}}{\partial X_{c_{in}, h', w'}} = \sum_{c_{out}} \sum_{i,j} W_{c_{out}, c_{in}, i, j} \cdot \delta_{c_{out},\, \lfloor(h'+p-i)/s\rfloor,\, \lfloor(w'+p-j)/s\rfloor}$$

In matrix form: $\text{d\_col} = \text{W\_mat}^T \cdot \delta\_\text{mat}$, then **col2im** reverses the im2col transformation to scatter gradients back to the correct input positions.

The `cma-autograd` crate implements this full backward pass for tracked training of CNN models.

**Reference**: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. §9.3 — pooling and convolution forward/backward.
