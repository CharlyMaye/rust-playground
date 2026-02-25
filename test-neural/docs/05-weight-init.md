# 05 — Weight Initialization

> **Navigation** ← [04 — Backpropagation](04-backpropagation.md) | [06 — Regularization →](06-regularization.md)

---

## Level 1 — Concepts

### Why does initialization matter?

When you first create a network, all weights need some starting values. The choice of those values has a large impact on training:

- **All zeros**: every neuron in a layer computes the same function and receives the same gradient — they never diverge. The network fails to learn different features. This is called the **symmetry problem**.
- **Too large**: activations and gradients explode; training diverges or oscillates.
- **Too small**: activations and gradients vanish; training stalls completely.

Good initialization sets weights that are:
1. **Random** (breaking symmetry so neurons specialize).
2. **Carefully scaled** (maintaining stable signal magnitude through many layers).

### The library chooses automatically

When you call `.build()` on `NetworkBuilder`, the library automatically selects the right initialization strategy for each layer based on its activation function. You never need to configure this manually unless you want to override it.

### Summary of strategies

| Strategy | Magnitude | Best for |
|----------|-----------|----------|
| **Uniform** | $U(-1, 1)$ | Simple shallow experiments |
| **Xavier / Glorot** | $\sim \sqrt{2/(n_{in}+n_{out})}$ | Tanh, Sigmoid, Softmax, Linear |
| **He / Kaiming** | $\sim \sqrt{2/n_{in}}$ | ReLU and all ReLU-family activations |
| **LeCun** | $\sim \sqrt{1/n_{in}}$ | SELU (required for self-normalization) |

---

## Level 2 — Mathematics

### The variance propagation argument

Consider a layer with $n_{in}$ inputs, each with variance $\text{Var}(a) = \sigma_a^2$, and weights $w$ drawn independently with $\mathbb{E}[w] = 0$ and $\text{Var}(w) = \sigma_w^2$.

The pre-activation is:

$$z = \sum_{i=1}^{n_{in}} w_i a_i$$

Assuming independence and zero-mean:

$$\text{Var}(z) = n_{in} \cdot \sigma_w^2 \cdot \sigma_a^2$$

For variance to be preserved layer through layer (i.e., $\text{Var}(z) = \text{Var}(a)$), we need:

$$\sigma_w^2 = \frac{1}{n_{in}}$$

This is the **LeCun initialization** (for activations with no zeroing, such as Tanh or SELU).

For activations that zero half the inputs (such as ReLU, which outputs 0 for negative inputs), the effective variance is halved, requiring:

$$\sigma_w^2 = \frac{2}{n_{in}}$$

This is the **He / Kaiming initialization**.

---

### LeCun Normal Initialization

$$w \sim \mathcal{N}\!\left(0, \frac{1}{n_{in}}\right)$$

Originally proposed for Tanh networks. Required by SELU for its self-normalization property to hold.

**Reference**: LeCun, Y., Bottou, L., Orr, G. B., & Müller, K. R. (1998). Efficient BackProp. In *Neural Networks: Tricks of the Trade*, Springer.

---

### Xavier / Glorot Initialization

Glorot and Bengio (2010) extended the variance analysis to account for *both* the forward and the backward pass. The backward pass has a symmetric argument: to preserve gradient variance through layer $l$, we need $\sigma_w^2 = 1/n_{out}$.

Combining both constraints by their harmonic mean:

$$\sigma_w^2 = \frac{2}{n_{in} + n_{out}}$$

This gives the **Xavier normal** variant:

$$w \sim \mathcal{N}\!\left(0, \frac{2}{n_{in} + n_{out}}\right)$$

And the **Xavier uniform** variant uses the equivalent range:

$$w \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

The library uses the normal variant. Note that the analysis assumes approximately linear activations (Tanh and Sigmoid are approximately linear near zero), which is why it suits Sigmoid/Tanh best.

**Reference**: Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.

---

### He / Kaiming Initialization

He et al. (2015) derived the correct initialization for ReLU by accounting for the fact that ReLU sets half its inputs to zero:

$$\mathbb{E}[a^2] = \mathbb{E}[\max(0,z)^2] = \frac{1}{2}\mathbb{E}[z^2] = \frac{n_{in}}{2}\sigma_w^2 \sigma_a^2$$

Setting $\mathbb{E}[a^2] = \sigma_a^2$ (variance preservation) gives:

$$\sigma_w^2 = \frac{2}{n_{in}}$$

**He normal**:

$$w \sim \mathcal{N}\!\left(0, \frac{2}{n_{in}}\right)$$

Using the backward pass instead gives $\sigma_w^2 = 2/n_{out}$. The library uses the $n_{in}$ variant (forward-pass analysis), which is the most common choice.

This initialization is also called **Kaiming normal** (after the author's first name).

**Reference**: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. *ICCV*.

---

### Gaussian sampling via Box-Muller

The library generates Gaussian samples without using a standard library random distributions — it implements the **Box-Muller transform** directly in `init.rs`.

Given two independent uniform samples $u_1, u_2 \sim \mathcal{U}(0,1)$, the Box-Muller transform produces two independent standard normal samples:

$$z_1 = \sqrt{-2 \ln u_1} \cdot \cos(2\pi u_2)$$
$$z_2 = \sqrt{-2 \ln u_1} \cdot \sin(2\pi u_2)$$

To sample from $\mathcal{N}(0, \sigma^2)$, multiply by $\sigma$: $w = \sigma \cdot z_1$.

**Why not use the `rand_distr` crate?** The library uses a seeded `StdRng` for reproducibility and the Box-Muller transform to avoid an extra dependency (`rand_distr`), keeping the WASM binary lean.

**Reference**: Box, G. E. P., & Muller, M. E. (1958). A note on the generation of random normal deviates. *Annals of Mathematical Statistics*, 29(2), 610–611.

---

### The symmetry-breaking argument — formal statement

**Claim**: if all weights in a layer are initialized to the same value $c$ (including $c = 0$), all neurons in that layer are guaranteed to remain identical for all subsequent gradient updates.

**Proof sketch**: Consider two neurons $j$ and $k$ in layer $l$ with $W^{(l)}_{j,:} = W^{(l)}_{k,:}$. Then $z^{(l)}_j = z^{(l)}_k$, $a^{(l)}_j = a^{(l)}_k$, and $\delta^{(l)}_j = \delta^{(l)}_k$ for any input. The weight gradients are equal, so after an optimizer step, $W^{(l)}_{j,:} = W^{(l)}_{k,:}$ still. By induction, the layers remain symmetric for all time. Therefore, all $n_l$ neurons are computing the same function — functionally equivalent to having only 1 neuron. ∎

Random initialization breaks this symmetry by giving each neuron a slightly different starting point, causing different gradients, and allowing specialization.
