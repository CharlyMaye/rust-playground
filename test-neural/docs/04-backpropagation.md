# 04 — Backpropagation

> **Navigation** ← [03 — Loss Functions](03-loss-functions.md) | [05 — Weight Initialization →](05-weight-init.md)

---

## Level 1 — Concepts

### The core question

After the forward pass produces a prediction and we compute the loss, we need to know: **how should we adjust each weight to reduce the loss?** This is answered by computing the **gradient** of the loss with respect to every weight in the network.

### The chain rule, intuitively

Think of the network as a chain of operations:

```
input → layer 1 → layer 2 → … → output → loss
```

Each operation depends on the previous one. If a small change in weight $w$ causes a small change in layer 1's output, which causes a change in layer 2's output, …, which ultimately changes the loss, then the total effect on the loss is the *product* of all those individual effects — that is the chain rule.

### Forward and backward pass

1. **Forward pass**: propagate the input through all layers, storing the intermediate values at each layer.
2. **Backward pass** (backprop): traverse the network in reverse, computing how much each intermediate value contributed to the loss, then using that to compute weight gradients.

The key insight is that once you know the gradient at layer $l+1$, you can efficiently compute the gradient at layer $l$ — no redundant computation. The algorithm runs in time proportional to the forward pass (roughly $O(W)$ where $W$ is the number of weights).

### Gradient accumulation in mini-batches

For a mini-batch of $B$ samples, the library:
1. Computes per-sample gradients in the batch (possibly in parallel with the `parallel` feature).
2. Sums all gradients.
3. Divides by $B$.
4. Applies the averaged gradient to the optimizer.

Pre-allocated buffers hold the accumulated gradients — no allocation happens per sample during training.

---

## Level 2 — Mathematics

### Notation recap

- $n_l$ — number of neurons in layer $l$, with $L$ total layers.
- $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$, $b^{(l)} \in \mathbb{R}^{n_l}$ — parameters of layer $l$.
- $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$ — pre-activation.
- $a^{(l)} = f^{(l)}(z^{(l)})$ — post-activation.
- $\mathcal{L}$ — scalar loss.

### The delta (error signal) at each layer

Define the **delta** (also called the error signal or backpropagated gradient) at layer $l$:

$$\delta^{(l)} \triangleq \frac{\partial \mathcal{L}}{\partial z^{(l)}} \in \mathbb{R}^{n_l}$$

**Output layer delta** ($l = L$):

$$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial z^{(L)}}$$

For common combinations, this simplifies (see [03 — Loss Functions](03-loss-functions.md)):
- Sigmoid + BCE: $\delta^{(L)} = \hat{y} - y$
- Softmax + CCE: $\delta^{(L)} = \hat{y} - y$
- Linear + MSE: $\delta^{(L)} = \frac{2}{n}(\hat{y} - y) \odot f'^{(L)}(z^{(L)}) = \frac{2}{n}(\hat{y} - y)$ (since $f' = 1$)

**Hidden layer delta** via the chain rule:

$$\delta^{(l)} = \left(W^{(l+1)T} \delta^{(l+1)}\right) \odot f'^{(l)}(z^{(l)})$$

where $\odot$ denotes element-wise multiplication.

**Derivation**:

$$\delta^{(l)}_j = \frac{\partial \mathcal{L}}{\partial z^{(l)}_j} = \sum_k \frac{\partial \mathcal{L}}{\partial z^{(l+1)}_k} \cdot \frac{\partial z^{(l+1)}_k}{\partial z^{(l)}_j}$$

Since $z^{(l+1)}_k = \sum_i W^{(l+1)}_{ki} a^{(l)}_i + b^{(l+1)}_k$ and $a^{(l)}_i = f^{(l)}(z^{(l)}_i)$:

$$\frac{\partial z^{(l+1)}_k}{\partial z^{(l)}_j} = W^{(l+1)}_{kj} \cdot f'^{(l)}(z^{(l)}_j)$$

Therefore:

$$\delta^{(l)}_j = \left(\sum_k W^{(l+1)}_{kj} \delta^{(l+1)}_k\right) f'^{(l)}(z^{(l)}_j) = \left(W^{(l+1)T} \delta^{(l+1)}\right)_j \cdot f'^{(l)}(z^{(l)}_j)$$

### Gradients with respect to parameters

Once all deltas $\delta^{(1)}, \ldots, \delta^{(L)}$ are computed (in reverse order), the parameter gradients follow:

**Weight gradient**:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^T \in \mathbb{R}^{n_l \times n_{l-1}}$$

**Bias gradient**:

$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)} \in \mathbb{R}^{n_l}$$

**For a mini-batch** of $B$ samples (indexed by $b$):

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{B} \sum_{b=1}^{B} \delta^{(l,b)} \left(a^{(l-1,b)}\right)^T$$

### Dropout interaction with backpropagation

Dropout zeroes a fraction of activations during the forward pass and stores the mask $m^{(l)} \in \{0,1\}^{n_l}$. During backpropagation, the same mask is applied:

$$\delta^{(l)}_{\text{masked}} = \delta^{(l)} \odot m^{(l)}$$

This ensures gradients do not flow through dropped neurons, consistent with inverted dropout scaling.

### Vanishing and exploding gradients

The hidden delta recursion is:

$$\delta^{(l)} = \left(W^{(l+1)T} \delta^{(l+1)}\right) \odot f'^{(l)}(z^{(l)})$$

The product of $L-l$ such operations can cause gradients to:
- **Vanish** exponentially: if $\|W^{(l)}\| \cdot |f'^{(l)}(z)| < 1$ systematically.
- **Explode** exponentially: if $\|W^{(l)}\| \cdot |f'^{(l)}(z)| > 1$ systematically.

**Vanishing** was the key barrier to training deep networks before ReLU and careful initialization. Sigmoid and Tanh saturate ($|f'| \leq 0.25$), causing gradients to shrink layer by layer.

**Mitigations** (implemented in this library):
- **ReLU family** activations: $|f'| = 1$ in the active regime — no gradient shrinkage.
- **Careful initialization** (He, Xavier): ensures $\|W^{(l)}\|$ is scaled to keep variance stable.
- **BatchNorm** (in `cma-cnn`): normalizes pre-activations to unit variance, keeping $f'(z)$ away from saturation.
- **Skip connections** (ResNet in `cma-models`): provide a gradient highway that bypasses the chain product.

**References**:
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536. *(The original backprop paper.)*
- LeCun, Y., Bottou, L., Orr, G. B., & Müller, K. R. (1998). Efficient BackProp. In *Neural Networks: Tricks of the Trade*. Springer.
- Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE TNN*.
- Hochreiter, S. (1998). The Vanishing Gradient Problem During Learning Recurrent Neural Nets and Problem Solutions. *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*.

### Computational complexity

A single forward + backward pass through a network with $L$ layers has computational cost $O\!\left(\sum_{l=1}^{L} n_l \cdot n_{l-1}\right)$, which is proportional to the number of weights $W = \sum_l n_l n_{l-1}$. This is why backpropagation is efficient: computing all $W$ gradients costs $O(W)$, the same asymptotic cost as a single forward pass.

For a mini-batch of $B$ samples: $O(B \cdot W)$. With Rayon parallelism (the `parallel` feature), the per-sample forward/backward computations are distributed, reducing wall-clock time roughly by the number of available threads.

**Reference**: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. §6.5 — full derivation of backpropagation.
