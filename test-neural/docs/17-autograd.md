# 17 — Automatic Differentiation Engine (`cma-autograd`)

> **Navigation** ← [16 — Depthwise Conv](16-depthwise-conv.md) | [18 — LeNet-5 →](18-lenet.md)

---

## Level 1 — Concepts

### Why autograd?

In `cma-neural-network` and `cma-cnn`, the backward pass is manually coded: each layer type has a hand-written gradient formula. This is fast and explicit, but requires a new gradient implementation every time a new operation is added.

**Automatic differentiation** (autograd) removes this burden. You write the mathematical operations of the forward pass once, and the engine automatically derives and executes the backward pass by applying the chain rule to every primitive operation.

### The computation graph

When you perform an operation on a **tensor** in the autograd engine (for example, `a + b` or `relu(x)`), the engine records:
- Which tensors were the inputs.
- What operation was performed.
- A function (`GradFn`) that knows how to compute the input gradients given the output gradient.

These connections form a **dynamic computation graph** (also called a define-by-run graph, or tape). It is rebuilt from scratch on every forward pass, which makes it flexible: different inputs can follow different computation paths (important for dynamic architectures, RNNs, etc.).

### Backward pass: topological sort

After the forward pass produces a scalar loss, calling `.backward()` on it triggers:

1. A **topological sort** of the computation graph in reverse (depth-first post-order traversal).
2. Starting from the loss (gradient = 1.0), each node receives the gradient of the loss with respect to its output.
3. It calls its `GradFn.backward(gradient)` to compute gradients for its inputs, which are accumulated into those tensors' `.grad` fields.

### Leaf tensors

**Leaf tensors** are the network's trainable parameters (weights, biases) — they are not the output of any operation, only inputs. Their accumulated gradients are what the optimizer uses to update the parameters.

Non-leaf tensors (intermediate results) accumulate gradients only transiently during backprop; they are discarded after the pass to save memory.

### No-gradient context

Operations inside `no_grad(|| { ... })` do not build any graph nodes — useful for inference, validation, and weight updates (which should not themselves be differentiated).

---

## Level 2 — Mathematics

### Reverse-mode automatic differentiation

There are two modes of automatic differentiation:

- **Forward mode**: propagates *tangents* $\dot{x} = \partial x / \partial \theta$ for a specific input parameter $\theta$. Cost: $O(\text{inputs})$ passes to get all gradients.
- **Reverse mode**: propagates *adjoints* $\bar{y} = \partial \mathcal{L} / \partial y$ backward from the scalar output. Cost: **one pass** to get all gradients simultaneously.

Deep learning always uses **reverse mode** (also called *backpropagation*) because the loss is scalar and we want gradients with respect to millions of parameters.

The `cma-autograd` engine implements reverse-mode AD.

### The GradFn contract

Every differentiable operation creates a `GradFn` node that implements:

$$\text{backward}(\bar{y}) \to [\bar{x}_1, \bar{x}_2, \ldots]$$

where $\bar{y} = \partial \mathcal{L} / \partial y$ is the gradient of the loss with respect to the operation's output, and $\bar{x}_i = \partial \mathcal{L} / \partial x_i$ is the gradient with respect to each input, computed via the chain rule:

$$\bar{x}_i = \frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x_i} = \bar{y} \cdot J_i$$

where $J_i = \partial y / \partial x_i$ is the local Jacobian of the operation.

### Complete GradFn table

All operations in `cma-autograd` and their backward rules:

| Operation | Forward $y = f(x)$ | Backward $\bar{x}$ |
|-----------|-------------------|-------------------|
| `add(a, b)` | $a + b$ | $\bar{a} = \text{unbroadcast}(\bar{y})$, $\bar{b} = \text{unbroadcast}(\bar{y})$ |
| `sub(a, b)` | $a - b$ | $\bar{a} = \bar{y}$, $\bar{b} = -\bar{y}$ |
| `mul(a, b)` | $a \odot b$ | $\bar{a} = \bar{y} \odot b$, $\bar{b} = \bar{y} \odot a$ |
| `mul_scalar(a, s)` | $s \cdot a$ | $\bar{a} = s \cdot \bar{y}$ |
| `neg(a)` | $-a$ | $\bar{a} = -\bar{y}$ |
| `matmul(A, B)` | $AB$ | $\bar{A} = \bar{y} B^T$, $\bar{B} = A^T \bar{y}$ |
| `transpose(A)` | $A^T$ | $\bar{A} = \bar{y}^T$ |
| `sum(a)` | $\sum_i a_i$ | $\bar{a}_i = \bar{y}$ (broadcast) |
| `sum_axis(a, d)` | $\sum$ along axis $d$ | expand $\bar{y}$ along $d$ |
| `mean(a)` | $\frac{1}{n}\sum_i a_i$ | $\bar{a}_i = \bar{y}/n$ |
| `powf(a, p)` | $a^p$ | $\bar{a} = p \cdot a^{p-1} \odot \bar{y}$ |
| `log(a)` | $\ln a$ | $\bar{a} = \bar{y} / a$ |
| `exp(a)` | $e^a$ | $\bar{a} = e^a \odot \bar{y}$ |
| `relu(a)` | $\max(0, a)$ | $\bar{a} = \bar{y} \odot \mathbf{1}[a > 0]$ |
| `sigmoid(a)` | $\sigma(a)$ | $\bar{a} = \bar{y} \odot \sigma(a)(1-\sigma(a))$ |
| `tanh_act(a)` | $\tanh(a)$ | $\bar{a} = \bar{y} \odot (1 - \tanh^2(a))$ |
| `clamp(a, lo, hi)` | $\text{clamp}(a)$ | $\bar{a} = \bar{y} \odot \mathbf{1}[lo < a < hi]$ |
| `reshape(a, s)` | reshape to $s$ | $\bar{a} =$ reshape $\bar{y}$ to original shape |
| `conv2d(...)` | conv + bias | $\bar{W} = \delta \cdot \text{col}^T$, $\bar{b} = \sum_{h,w}\delta$, $\bar{X} = \text{col2im}(W^T \delta)$ |
| `maxpool2d(...)` | max over window | route $\bar{y}$ to argmax positions only |

### Broadcast-aware unbroadcast

When two tensors of different shapes are added (via broadcasting), the gradient must be **un-broadcast**: summed along the dimensions that were broadcast. For example, adding a `[B, C]` tensor to a `[C]` bias:

$$\bar{\text{bias}} = \sum_{b=1}^{B} \bar{y}_{b,:}$$

The `unbroadcast` utility in the engine handles all cases by computing which dimensions have size 1 in the original and summing over them.

### Linear layer backward pass

The `Linear` module computes $y = x W^T + b$ where $x \in \mathbb{R}^{B \times n_{in}}$, $W \in \mathbb{R}^{n_{out} \times n_{in}}$, $b \in \mathbb{R}^{n_{out}}$.

The `LinearBackward` GradFn computes:

$$\bar{x} = \bar{y} W \in \mathbb{R}^{B \times n_{in}}$$

$$\bar{W} = \bar{y}^T x \in \mathbb{R}^{n_{out} \times n_{in}}$$

$$\bar{b} = \sum_{b} \bar{y}_{b,:} \in \mathbb{R}^{n_{out}}$$

### Thread safety and Arc<TensorInner>

Tensors are reference-counted via `Arc<TensorInner>`. When an operation has two inputs pointing to the same tensor (as in $x^2 = x \cdot x$), both gradient paths accumulate into the same `.grad` field — requiring `RwLock` for thread-safe gradient accumulation:

$$\bar{x} = \frac{\partial \mathcal{L}}{\partial (x \cdot x)} = \frac{\partial \mathcal{L}}{\partial y} \cdot x \;\;(\text{path 1}) + \frac{\partial \mathcal{L}}{\partial y} \cdot x \;\;(\text{path 2}) = 2x \bar{y}$$

This is handled automatically by the engine's accumulation loop.

### Optimizers in cma-autograd

`SGD` and `Adam` call `param.update_data(new_value)` to replace the tensor's data after computing the update — without creating any computation graph nodes (wrapped in `no_grad`). The `zero_grad()` method clears `.grad` on all parameters before the next forward pass.

**References**:
- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic Differentiation in Machine Learning: a Survey. *JMLR*, 18(153), 1–43. [Comprehensive survey; reverse-mode AD in §3.1.]
- Paszke, A., et al. (2017). Automatic Differentiation in PyTorch. *NeurIPS Autodiff Workshop*. [Design of dynamic graph autograd.]
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536. [Backpropagation as reverse-mode AD applied to neural networks.]
