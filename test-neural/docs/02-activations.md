# 02 — Activation Functions

> **Navigation** ← [01 — Architecture](01-architecture.md) | [03 — Loss Functions →](03-loss-functions.md)

---

## Level 1 — Concepts

### Why do we need activations?

Without a non-linear activation, stacking multiple layers is useless: a composition of linear transformations is still just a linear transformation. A 10-layer network without activations collapses to a single matrix multiplication. Non-linear activations break this collapse and allow the network to represent complex, curved decision boundaries.

### Choosing an activation

The choice of activation function has a large impact on training dynamics. Key considerations:
- **Hidden layers**: prefer activations that do not saturate (do not output the same value for very large or very small inputs), so gradients can flow back during training.
- **Output layer**: match the activation to the task. Sigmoid for binary probabilities, Softmax for multi-class probabilities, Linear for unbounded regression.

### The 15 activations in this library

| Name | Range | Shape | Typical use |
|------|-------|-------|-------------|
| **Sigmoid** | (0, 1) | S-curve | Binary output, gates |
| **Tanh** | (−1, 1) | Centered S-curve | Hidden layers (older architectures), RNNs |
| **ReLU** | [0, ∞) | Ramp | Default for most hidden layers |
| **LeakyReLU** | (−∞, ∞) | Leaky ramp | Avoids dead neurons |
| **ELU** | (−α, ∞) | Smooth leaky | Faster learning than ReLU |
| **SELU** | (−λα, ∞) | Self-normalizing | Deep networks without BatchNorm |
| **Swish** | (−0.28, ∞) | Smooth bump | Modern hidden layers |
| **GELU** | (~−0.17, ∞) | Smooth bump | Transformers, BERT, GPT |
| **Mish** | (~−0.31, ∞) | Smooth bump | Object detection (YOLOv4) |
| **Softplus** | (0, ∞) | Smooth ramp | Smooth ReLU alternative |
| **Softsign** | (−1, 1) | Bounded smooth | Alternative to Tanh |
| **HardSigmoid** | [0, 1] | Piecewise linear | Faster mobile inference |
| **HardTanh** | [−1, 1] | Clipped linear | Efficient quantized models |
| **Softmax** | (0, 1) vector | Probabilities sum to 1 | Multi-class output layer |
| **Linear** | (−∞, ∞) | Identity | Regression output |

### The saturation problem

Sigmoid and Tanh both **saturate**: for very large or very small inputs, the function becomes nearly flat (slope ≈ 0). During backpropagation, this flat slope causes gradients to shrink as they are multiplied layer by layer — the **vanishing gradient problem**. ReLU and its variants avoid saturation in the positive regime, which is why they typically train faster and deeper.

---

## Level 2 — Mathematics

### Notation

For all activations below:
- $z \in \mathbb{R}$ denotes the **pre-activation** (input to the function).
- $f(z)$ denotes the **activation output**.
- $f'(z)$ denotes the **derivative** with respect to the pre-activation.

The library implements `derivative_from_preactivation(z)` for all activations (not from the post-activation $a$), which avoids redundant recomputation and is the mathematically cleaner interface for backpropagation.

---

### Sigmoid

$$f(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

$$f'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)$$

Range: $(0, 1)$. Saturates for $|z| \gg 0$. Maximum gradient $0.25$ at $z = 0$.

**Vanishing gradient**: since $f'(z) \leq 0.25$, multiplying $L$ gradients gives at most $0.25^L \to 0$ exponentially.

**Reference**: Standard sigmoid; formalization in the context of perceptrons by Rumelhart et al. (1986).

---

### Tanh

$$f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

$$f'(z) = 1 - \tanh^2(z)$$

Range: $(-1, 1)$. Zero-centered (unlike Sigmoid), which helps gradient flow. Saturates similarly.

Note: $\tanh(z) = 2\sigma(2z) - 1$, so Tanh and Sigmoid have identical expressivity.

---

### ReLU — Rectified Linear Unit

$$f(z) = \max(0, z)$$

$$f'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

Range: $[0, +\infty)$. Does not saturate for $z > 0$, enabling gradient flow through deep networks. Introduced to deep learning at scale by Nair & Hinton (2010); popularized by Krizhevsky et al. (2012).

**Dead neuron problem**: if a neuron's pre-activation is always negative, $f'(z) = 0$ always — the neuron never updates. Addressed by LeakyReLU and ELU.

**References**:
- Nair, V., & Hinton, G. E. (2010). Rectified Linear Units Improve Restricted Boltzmann Machines. *ICML*.
- Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep Sparse Rectifier Neural Networks. *AISTATS*.

---

### LeakyReLU

$$f(z) = \begin{cases} z & z > 0 \\ 0.01 \cdot z & z \leq 0 \end{cases}$$

$$f'(z) = \begin{cases} 1 & z > 0 \\ 0.01 & z \leq 0 \end{cases}$$

The fixed negative slope $0.01$ prevents dead neurons. The library uses this fixed value; the general parametric version PReLU learns the slope.

**Reference**: Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier Nonlinearities Improve Neural Network Acoustic Models. *ICML* Workshop.

---

### ELU — Exponential Linear Unit

$$f(z) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}, \quad \alpha = 1$$

$$f'(z) = \begin{cases} 1 & z > 0 \\ f(z) + \alpha & z \leq 0 \end{cases}$$

Range: $(-\alpha, +\infty)$. For $\alpha = 1$, negative values approach $-1$, pushing mean activations toward zero (faster convergence). Smooth at $z = 0$ (unlike ReLU/LeakyReLU).

**Reference**: Clevert, D. A., Unterthiner, T., & Hochreiter, S. (2015). Fast and Accurate Deep Network Learning by Exponential Linear Units. *ICLR*.

---

### SELU — Scaled Exponential Linear Unit

$$f(z) = \lambda \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \leq 0 \end{cases}$$

with fixed constants $\lambda = 1.0507009873554804$ and $\alpha = 1.6732632423543772$.

$$f'(z) = \lambda \begin{cases} 1 & z > 0 \\ \alpha e^z & z \leq 0 \end{cases}$$

These constants are derived so that a layer with SELU activation is a **contractive mapping on the space of mean-variance statistics**: if inputs have mean 0 and variance 1, the pre-activations have mean 0 and variance approximately 1 after the transformation (self-normalization).

Requires **LeCun normal initialization** (see [05 — Weight Initialization](05-weight-init.md)) for the self-normalization property to hold. Used without BatchNorm in very deep networks.

**Reference**: Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). Self-Normalizing Neural Networks. *NeurIPS*.

---

### Swish

$$f(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

$$f'(z) = f(z) + \sigma(z)(1 - f(z))$$

Range: $(-0.278, +\infty)$. Non-monotone: has a small valley around $z \approx -1.28$. Outputs are not strictly positive, which may help the network represent negative feature responses. Discovered via automated search.

**Reference**: Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for Activation Functions. *ICLR* Workshop.

---

### GELU — Gaussian Error Linear Unit

$$f(z) = z \cdot \Phi(z)$$

where $\Phi(z)$ is the CDF of the standard normal distribution.

**Approximation used in practice** (and in this library):

$$f(z) \approx 0.5 \, z \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(z + 0.044715 \, z^3\right)\right)\right)$$

$$f'(z) = 0.5\tanh(c) + \left(0.5 z \cdot \text{sech}^2(c)\right)\sqrt{\tfrac{2}{\pi}}(1 + 3 \cdot 0.044715 \, z^2) + 0.5$$

where $c = \sqrt{2/\pi}(z + 0.044715 z^3)$.

Stochastic interpretation: $f(z) = z \cdot P(\text{keep}(z))$ where the Bernoulli keep probability is $\Phi(z)$. Inputs with high pre-activation are kept with probability near 1; near-zero inputs are stochastically dropped.

**Reference**: Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units. arXiv:1606.08415.

---

### Mish

$$f(z) = z \cdot \tanh\!\bigl(\ln(1 + e^z)\bigr) = z \cdot \tanh(\text{Softplus}(z))$$

$$f'(z) = \text{sech}^2(\text{Softplus}(z)) \cdot z \cdot \sigma(z) + \frac{f(z)}{z}$$

(simplified form appears in the library's implementation)

Range: $(-0.31, +\infty)$. Smooth, non-monotone, unbounded above. The bounded below value $\approx -0.31$ provides a small negative slope that helps gradient flow.

**Reference**: Misra, D. (2019). Mish: A Self Regularized Non-Monotonic Activation Function. arXiv:1908.08681.

---

### Softplus

$$f(z) = \ln(1 + e^z)$$

$$f'(z) = \sigma(z)$$

Range: $(0, +\infty)$. Smooth, differentiable everywhere approximation of ReLU. Derivative is exactly Sigmoid. Used as a building block inside Mish.

---

### Softsign

$$f(z) = \frac{z}{1 + |z|}$$

$$f'(z) = \frac{1}{(1 + |z|)^2}$$

Range: $(-1, 1)$. Similar to Tanh but with polynomial rather than exponential tails — gradients decay more slowly.

**Reference**: Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.

---

### HardSigmoid

$$f(z) = \text{clamp}(0.2z + 0.5, \, 0, \, 1)$$

$$f'(z) = \begin{cases} 0.2 & -2.5 < z < 2.5 \\ 0 & \text{otherwise} \end{cases}$$

Piecewise linear approximation of Sigmoid. No exponential computation — useful in inference-optimized or quantized models.

---

### HardTanh

$$f(z) = \text{clamp}(z, -1, 1)$$

$$f'(z) = \begin{cases} 1 & -1 < z < 1 \\ 0 & \text{otherwise} \end{cases}$$

---

### Softmax

$$f(z)_i = \frac{e^{z_i - \max_j z_j}}{\sum_{j} e^{z_j - \max_j z_j}}$$

The $\max_j z_j$ subtraction is **numerically critical**: without it, $e^{z_i}$ overflows for large $z_i$. Subtracting the max does not change the output (constants cancel in numerator and denominator) but keeps values in a stable range.

The Jacobian is:

$$\frac{\partial f_i}{\partial z_j} = f_i(\delta_{ij} - f_j)$$

where $\delta_{ij}$ is the Kronecker delta. In practice, Softmax is always paired with Categorical Cross-Entropy loss, and the combined gradient simplifies dramatically (see [03 — Loss Functions](03-loss-functions.md)).

**Interpretation**: Softmax is the $\arg\max$ operator "softened" by temperature — as the inputs are scaled by a factor $T \to 0$, Softmax converges to a one-hot vector; as $T \to \infty$, outputs become uniform.

---

### Linear (Identity)

$$f(z) = z, \qquad f'(z) = 1$$

Used on the output layer for regression tasks where the output is an unbounded real number.

---

### Automatic initialization selection

The library's `WeightInit::for_activation(act)` encodes the following research-backed mapping:

| Activation | Recommended Init | Reason |
|------------|-----------------|--------|
| Sigmoid, Tanh, Softmax, Linear, Softsign, HardSigmoid, HardTanh | Xavier / Glorot | Preserves variance through symmetric activations |
| ReLU, LeakyReLU, ELU, GELU, Swish, Mish, Softplus | He / Kaiming | Compensates for half-zero outputs of ReLU-family |
| SELU | LeCun | Required for self-normalization property |

This is detailed in [05 — Weight Initialization](05-weight-init.md).
