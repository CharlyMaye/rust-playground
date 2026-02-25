# 15 — Batch Normalization

> **Navigation** ← [14 — Pooling](14-pooling.md) | [16 — Depthwise Convolution →](16-depthwise-conv.md)

---

## Level 1 — Concepts

### The internal covariate shift problem

As the weights in the early layers change during training, the distribution of inputs to later layers changes too. Each layer must continuously adapt to a shifting input distribution, which slows training. Ioffe and Szegedy (2015) called this phenomenon **internal covariate shift**.

A concrete symptom: deep networks with Sigmoid or Tanh activations are hard to train because activations can saturate when inputs drift to large values — and this happens all the time as weights update.

### What BatchNorm does

**Batch Normalization** normalizes the pre-activation values across the training batch at each layer, forcing them to have approximately zero mean and unit variance. It then applies a learnable per-channel scale $\gamma$ and shift $\beta$ to restore the representational capacity that normalization might have removed.

The result:
- The network can use higher learning rates without diverging.
- The network is far less sensitive to initialization.
- It acts as a regularizer — some papers report it reduces the need for Dropout.

### Training vs Evaluation

- **Training**: normalize using the current mini-batch's mean and variance; update running statistics via exponential moving average.
- **Inference**: normalize using the **running** (accumulated) mean and variance — the batch statistics from training, not from the single test sample.

### In this library

`BatchNorm2D` normalizes per-channel across the batch. It holds learnable parameters `gamma` ($\gamma$, initialized to 1) and `beta` ($\beta$, initialized to 0), and maintains `running_mean` and `running_var` for inference. The `momentum` parameter (default 0.1) controls how fast running statistics update.

Conv2D layers placed before BatchNorm should use **no bias** (`.without_bias()`): the bias is redundant because BatchNorm's $\beta$ parameter already provides a free shift.

---

## Level 2 — Mathematics

### Training-mode forward pass (per channel)

For a mini-batch of $N$ images with feature map of spatial size $H \times W$, BatchNorm2D computes statistics per channel $c$. Let $\mathcal{B}_c$ denote all values in channel $c$ across the batch and spatial dimensions ($N \times H \times W$ values total):

**Batch mean**:
$$\mu_c = \frac{1}{N H W} \sum_{n,h,w} X_{n,c,h,w}$$

**Batch variance** (Bessel-corrected for unbiased estimation when updating running stats):
$$\sigma_c^2 = \frac{1}{N H W - 1} \sum_{n,h,w} (X_{n,c,h,w} - \mu_c)^2$$

**Normalize**:
$$\hat{X}_{n,c,h,w} = \frac{X_{n,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \varepsilon}}$$

with $\varepsilon = 10^{-5}$ for numerical stability.

**Scale and shift**:
$$Y_{n,c,h,w} = \gamma_c \hat{X}_{n,c,h,w} + \beta_c$$

where $\gamma_c, \beta_c \in \mathbb{R}$ are **learnable per-channel parameters**.

**Update running statistics** via exponential moving average (EMA) with momentum $m = 0.1$:
$$\mu_c^{\text{run}} \leftarrow (1 - m)\,\mu_c^{\text{run}} + m\,\mu_c$$
$$\sigma_c^{2,\text{run}} \leftarrow (1 - m)\,\sigma_c^{2,\text{run}} + m\,\sigma_c^2$$

### Inference-mode forward pass

At inference, batch statistics are unavailable (single samples, or we want deterministic output). Running statistics are used instead:

$$\hat{X}_{n,c,h,w} = \frac{X_{n,c,h,w} - \mu_c^{\text{run}}}{\sqrt{\sigma_c^{2,\text{run}} + \varepsilon}}, \qquad Y = \gamma_c \hat{X} + \beta_c$$

The running statistics are frozen — no updates occur during inference.

### Backward pass

BatchNorm introduces dependencies between samples in the batch (through $\mu_c$ and $\sigma_c^2$), making its backward pass non-trivial. The gradient of $\mathcal{L}$ with respect to $\gamma_c$ and $\beta_c$:

$$\frac{\partial \mathcal{L}}{\partial \gamma_c} = \sum_{n,h,w} \frac{\partial \mathcal{L}}{\partial Y_{n,c,h,w}} \hat{X}_{n,c,h,w}$$

$$\frac{\partial \mathcal{L}}{\partial \beta_c} = \sum_{n,h,w} \frac{\partial \mathcal{L}}{\partial Y_{n,c,h,w}}$$

The gradient with respect to the input $X$ (needed to backpropagate to the conv layer) involves the chain rule through $\mu_c$ and $\sigma_c^2$:

$$\frac{\partial \mathcal{L}}{\partial X_{n,c,h,w}} = \frac{\gamma_c}{m\sqrt{\sigma_c^2 + \varepsilon}} \left[ m \cdot \frac{\partial \mathcal{L}}{\partial Y_{n,c,h,w}} - \frac{\partial \mathcal{L}}{\partial \beta_c} - \hat{X}_{n,c,h,w} \cdot \frac{\partial \mathcal{L}}{\partial \gamma_c} \right]$$

where $m = N \cdot H \cdot W$.

### Why $\gamma$ and $\beta$?

Without the learnable parameters, BatchNorm forces every layer's pre-activations to be $\mathcal{N}(0,1)$ — this removes the representational freedom of the network. The layer could not represent the identity transformation, which would break residual connections (see [19 — ResNet](19-resnet.md)).

With $\gamma = \sqrt{\sigma_c^2}$ and $\beta = \mu_c$, BatchNorm can reproduce the original, un-normalized distribution — so the network can learn to "undo" normalization when useful.

### Effect on optimization landscape

Ioffe and Szegedy argued BatchNorm reduces internal covariate shift. More recent theoretical work (Santurkar et al., 2018) reframed the benefit: BatchNorm **smooths the loss landscape**, making gradients more predictable and allowing larger learning rates. The network can take bigger steps without overshooting.

### Interaction with Dropout

Using BatchNorm and Dropout together in the same block can cause a mismatch: Dropout changes the variance of activations at training time in a way that BatchNorm's running variance estimate does not account for. When Dropout is disabled at inference, the variance changes again, and the running stats are off.

The recommended practice:
- Avoid using Dropout inside convolutional blocks that use BatchNorm.
- Apply Dropout only in the fully-connected head (after the last BatchNorm).

**References**:
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML*. [The original paper.]
- Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How Does Batch Normalization Help Optimization? *NeurIPS*. [Challenges the covariate shift explanation; argues landscape smoothing.]
- Li, X., et al. (2019). Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift. *CVPR*.
