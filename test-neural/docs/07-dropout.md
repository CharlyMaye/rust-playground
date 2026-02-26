# 07 — Dropout

> **Navigation** ← [06 — Regularization](06-regularization.md) | [08 — Optimizers →](08-optimizers.md)

---

## Level 1 — Concepts

### The idea

During each training step, **Dropout randomly switches off a fraction of neurons**: each neuron is independently disabled with probability $p$ and kept with probability $1-p$. Disabled neurons output zero; their weights are not updated that step.

A typical dropout rate is $p = 0.2$ to $p = 0.5$. The output layer is never dropped.

### Why it works — the ensemble intuition

A network with $N$ neurons and dropout rate $p$ implicitly trains $2^N$ different sub-networks (one for each possible mask). These sub-networks share weights. At test time, the full network with all neurons active approximates averaging the predictions of all $2^N$ sub-networks. Ensembles are consistently more accurate and better-calibrated than single models — dropout gets this for free.

### Training vs Inference

- **Training**: apply dropout, scale surviving activations by $\frac{1}{1-p}$ (**inverted dropout**).
- **Inference**: use all neurons without any masking or scaling.

Inverted dropout makes inference straightforward: no scaling is needed at test time because the expected value of each neuron's output is already correct during training.

### Where to apply dropout

In this library, dropout is applied per hidden layer using a `DropoutConfig`. The output layer is never dropped.

---

## Level 2 — Mathematics

### Formal definition

Let $h^{(l)} = f^{(l)}(z^{(l)})$ be the activations of layer $l$ before dropout. Define the dropout mask:

$$m^{(l)}_j \sim \text{Bernoulli}(1 - p), \quad \text{i.i.d. for each neuron } j$$

The **thinned activation** is:

$$\tilde{h}^{(l)}_j = m^{(l)}_j \cdot h^{(l)}_j$$

With **inverted scaling**, the output that feeds into the next layer during training is:

$$\hat{h}^{(l)}_j = \frac{m^{(l)}_j}{1 - p} \cdot h^{(l)}_j$$

**Expected value**:

$$\mathbb{E}\!\left[\hat{h}^{(l)}_j\right] = \frac{1}{1-p} \cdot (1-p) \cdot h^{(l)}_j = h^{(l)}_j$$

The expected value of a dropped-out activation equals the original activation. Therefore, at test time, using the full network without any masking or scaling yields the same expected output — no correction is needed at inference.

### Backpropagation through dropout

The mask $m^{(l)}$ is stored during the forward pass. During the backward pass, the gradient through dropout is:

$$\frac{\partial \mathcal{L}}{\partial h^{(l)}_j} = \frac{m^{(l)}_j}{1 - p} \cdot \frac{\partial \mathcal{L}}{\partial \hat{h}^{(l)}_j}$$

Neurons that were dropped ($m^{(l)}_j = 0$) receive zero gradient — their weights are not updated, consistent with their forward-pass exclusion.

### Dropout as ensemble approximation

**Exact interpretation**: let $\mathcal{M}$ be the set of all $2^N$ masks (subsets of neurons). Each mask $m$ defines a sub-network $f_m(x; W_m)$ where $W_m$ is the weight subset active under mask $m$. Dropout training with rate $p$ optimizes a lower bound on the log-likelihood of the geometric mixture:

$$p(y | x) = \prod_m p_m(y | x, W)^{P(m)}$$

where weights are shared across all sub-networks.

At inference, the full network with weights scaled by $(1-p)$ approximates the **geometric mean** of all sub-network predictions (Srivastava et al., 2014).

**Note**: Dropout and BatchNorm (**see `cma-cnn`**) interact poorly when combined naively. BatchNorm uses batch statistics that depend on which neurons are active; dropout changes those statistics stochastically. The recommended practice is:
- Use dropout *before* BatchNorm, or
- Use only one of the two in a given architecture.
ResNet and other modern architectures typically use BatchNorm without dropout in convolutional blocks, and add dropout only in the fully-connected head.

**Reference**: Ioffe, S., & Szegedy, C. (2015). Batch Normalization. *ICML*. §3.4 discusses interaction with Dropout.

### Dropout as regularization

Dropout can also be understood as a form of adaptive L2 regularization. Under certain assumptions, training with dropout rate $p$ has a similar effect to L2 regularization with strength:

$$\lambda_{\text{eff}} \approx \frac{p}{2(1-p) \cdot n_{\text{data}}} \sum_i x_i^2$$

This depends on the input norms and is therefore *input-adaptive* — neurons that respond to rare, high-norm features are penalized more strongly.

**Reference**:
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*, 15, 1929–1958. *(The original paper.)*
- Wager, S., Wang, S., & Liang, P. (2013). Dropout Training as Adaptive Regularization. *NeurIPS*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*. *(Bayesian interpretation.)*
