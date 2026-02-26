# 01 — Network Architecture

> **Navigation** ← [00 — Introduction](00-introduction.md) | [02 — Activations →](02-activations.md)

---

## Level 1 — Concepts

### The neuron

A single neuron takes a vector of numbers as input, multiplies each one by a learnable weight, sums everything up, adds a learnable bias, and then passes the result through a non-linear function called an **activation function**. That final value is the neuron's output.

```
inputs:  x₁, x₂, …, xₙ
         ↓   ↓       ↓
weights: w₁, w₂, …, wₙ
         └───┴───┬───┘
                sum + bias  →  activation  →  output
```

The **weights** determine how much each input matters. The **bias** shifts the threshold. The **activation function** introduces non-linearity (explained in depth in [02 — Activations](02-activations.md)).

### Layers

Neurons are organized into layers. Every neuron in one layer connects to every neuron in the next — this is called a **dense** or **fully connected** layer. Layers are processed sequentially:

```
Input layer → Hidden layer 1 → Hidden layer 2 → … → Output layer
```

- **Input layer**: not a real layer of computations, just the raw data.
- **Hidden layers**: intermediate transformations; the network learns what to extract here.
- **Output layer**: produces the final prediction (a class probability, a regression value, etc.).

### The forward pass

Running data through the network from input to output is called the **forward pass**. It is entirely deterministic once weights are fixed: same input always produces the same output. No randomness happens during inference.

### Depth vs. width

- **Wider** networks (more neurons per layer) can represent more features at each level of abstraction.
- **Deeper** networks (more layers) can represent hierarchical compositions of features.

In practice, depth is usually more parameter-efficient than width: doubling depth is more powerful than doubling width per layer.

### The builder API

In this library, you describe the architecture with the `NetworkBuilder`:

```
input(784)
  .hidden(256, ReLU)
  .hidden(128, ReLU)
  .output(10, Softmax)
  .loss(CategoricalCrossEntropy)
  .optimizer(Adam { lr: 0.001 })
```

The builder validates dimensions automatically and selects the correct weight initialization strategy for each activation without requiring manual configuration.

---

## Level 2 — Mathematics

### Formal notation

Let $L$ denote the number of layers (hidden + output). Index layers $l = 1, \ldots, L$. Denote:

- $n_l$ — number of neurons in layer $l$; $n_0$ — input dimension.
- $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ — weight matrix of layer $l$.
- $b^{(l)} \in \mathbb{R}^{n_l}$ — bias vector of layer $l$.
- $f^{(l)} : \mathbb{R} \to \mathbb{R}$ — activation function of layer $l$, applied element-wise.

**Pre-activation** (linear transformation):

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} \in \mathbb{R}^{n_l}$$

**Activation** (non-linear transformation):

$$a^{(l)} = f^{(l)}(z^{(l)}) \in \mathbb{R}^{n_l}$$

with $a^{(0)} = x$ (the network input).

The full network computes:

$$\hat{y} = a^{(L)} = f^{(L)}\!\left(W^{(L)} f^{(L-1)}\!\left(\cdots f^{(1)}\!\left(W^{(1)} x + b^{(1)}\right) \cdots\right) + b^{(L)}\right)$$

### Implementation: weight matrix shape convention

In `cma-neural-network`, each `Layer` stores:
- `weights: Array2<Float>` — shape `[n_out, n_in]` (row = output neuron, column = input neuron).
- `biases: Array1<Float>` — shape `[n_out]`.

The forward pass for a single sample $x$ computes $z = Wx + b$ via an ndarray matrix-vector product (`weights.dot(&input) + &biases`).

For a mini-batch of $B$ samples stacked as rows into $X \in \mathbb{R}^{B \times n_{in}}$, the batch forward pass is:

$$Z = X W^T + \mathbf{1} b^T \in \mathbb{R}^{B \times n_{out}}$$

which is implemented element-wise over sampled in the batch loop in `trainer.rs`.

### Parameter count

For a fully connected layer mapping $n_{in} \to n_{out}$ with bias:

$$\text{params} = n_{out} \cdot n_{in} + n_{out} = n_{out}(n_{in} + 1)$$

For a network with layers of sizes $[n_0, n_1, n_2, \ldots, n_L]$:

$$\text{total params} = \sum_{l=1}^{L} n_l (n_{l-1} + 1)$$

### Depth vs. width — expressivity theory

For ReLU networks, the number of **linear regions** (pieces of the piecewise linear function computed by the network) grows as:

$$\Omega\!\left(\left(\frac{n_0}{L}\right)^{(L-1) n_0} \cdot n_0^{n_0}\right)$$

where $n_0$ is the input dimension and $L$ is the depth (Montufar et al., 2014). This grows **combinatorially with depth**, whereas a single-layer network has at most $\binom{N}{n_0}$ regions for $N$ neurons.

**Reference**: Montufar, G., Pascanu, R., Cho, K., & Bengio, Y. (2014). On the Number of Linear Regions of Deep Neural Networks. *NeurIPS*.

### VC dimension

The **Vapnik-Chervonenkis (VC) dimension** is a measure of a model family's capacity. For a network with $W$ weights and $L$ layers (using threshold activations):

$$\text{VC-dim} = O(W L \log W)$$

This means a network with many weights can fit many distinct Boolean functions — it has high capacity. Managing that capacity to avoid overfitting is the subject of [06 — Regularization](06-regularization.md) and [07 — Dropout](07-dropout.md).

**Reference**: Bartlett, P., & Maass, W. (2003). Vapnik-Chervonenkis dimension of neural nets. In *The Handbook of Brain Theory and Neural Networks*. MIT Press.

### The bias term

The bias $b^{(l)}$ is equivalent to an extra input neuron that always outputs $1$ with an additional learnable weight. It shifts the activation threshold independently of the input, giving the network the freedom to fit functions that are not centered at zero.

Without bias: the network's decision surface must pass through the origin of the feature space, a severe restriction.

### Why separate pre-activation $z$ from activation $a$?

The library explicitly stores $z^{(l)}$ during the forward pass because backpropagation requires $f'^{(l)}(z^{(l)})$, not $f'^{(l)}(a^{(l)})$ (the derivative evaluated *at the pre-activation*, not the post-activation). For activations like ReLU, Sigmoid, and Tanh the distinction is important:
- ReLU: $f'(z) = \mathbf{1}[z > 0]$ — needs $z$ to know if we are in the positive regime.
- Sigmoid: $f'(z) = \sigma(z)(1 - \sigma(z))$ — can be re-derived from $z$ or from $a = \sigma(z)$ (the library uses the pre-activation form for uniformity).

**Reference**: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
