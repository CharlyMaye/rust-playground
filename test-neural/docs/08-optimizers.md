# 08 — Optimizers

> **Navigation** ← [07 — Dropout](07-dropout.md) | [09 — LR Schedules →](09-lr-schedules.md)

---

## Level 1 — Concepts

### Gradient descent

After backpropagation computes the gradient $\nabla_W \mathcal{L}$ (the direction of steepest ascent in loss space), the optimizer applies an update to move the weights in the *opposite* direction — decreasing the loss. This is **gradient descent**.

The simplest rule: subtract a small portion $\alpha$ (the **learning rate**) of the gradient:

$$W \leftarrow W - \alpha \nabla_W \mathcal{L}$$

### The learning rate $\alpha$

- **Too large**: overshoots the minimum, oscillates, or diverges.
- **Too small**: training is extremely slow.
- **Just right**: converges smoothly to a good solution.

Finding a good learning rate is one of the most impactful hyperparameter choices. Learning rate schedules (see [09 — LR Schedules](09-lr-schedules.md)) adjust it during training.

### Why vanilla SGD is not always enough

Plain gradient descent has problems in high-dimensional loss landscapes:
- **Ravines**: areas where curvature differs sharply between directions — gradient descent oscillates across the steep dimensions.
- **Saddle points**: where gradients are zero but the point is not a minimum.
- **Noisy gradients**: mini-batch gradients are noisy estimates of the true gradient.

The optimizers below address these issues.

### The 5 optimizers in this library

| Optimizer | Key feature | Typical use |
|-----------|-------------|-------------|
| **SGD** | Simplest; pure gradient step | Rarely used alone |
| **Momentum** | Smooths oscillations; accelerates in consistent directions | Classical choice |
| **RMSprop** | Per-parameter adaptive learning rate | RNNs, non-stationary objectives |
| **Adam** | Combines momentum + adaptive rates; default choice | Almost everything |
| **AdamW** | Adam + decoupled weight decay | Models with L2 regularization |

---

## Level 2 — Mathematics

### State representation

The library stores optimizer state in structs `OptimizerState2D` (for weight matrices) and `OptimizerState1D` (for bias vectors). Each state contains:
- `m`: first moment estimate (momentum / mean of gradients)
- `v`: second moment estimate (variance of gradients)
- `t`: time step counter (for bias correction in Adam/AdamW)

All updates are applied **element-wise** over raw `f32` slices, which is cache-friendly and amenable to SIMD vectorization.

---

### SGD — Stochastic Gradient Descent

$$W \leftarrow W - \alpha g$$

where $g = \nabla_W \mathcal{L}_{\text{batch}}$ is the mini-batch gradient.

The word "stochastic" refers to the fact that each step uses a random mini-batch rather than the full dataset — the gradient is a noisy estimate of the true gradient.

**Convergence**: for a convex loss with learning rate $\alpha_t \propto 1/t$, SGD converges to the global minimum. For smooth non-convex losses (as in deep networks), it converges to a stationary point.

**Reference**: Robbins, H., & Monro, S. (1951). A stochastic approximation method. *Annals of Mathematical Statistics*, 22(3), 400–407.

---

### SGD with Momentum (Polyak, 1964)

Classical momentum maintains a **velocity** $v$ that accumulates past gradients:

$$v \leftarrow \beta v + g \qquad (\beta = 0.9 \text{ by default})$$

$$W \leftarrow W - \alpha v$$

**Effect**: the velocity builds up in directions where gradients consistently point (accelerating progress) and averages out oscillating gradients (damping ravines). With $\beta = 0.9$, the effective step is a weighted sum of the past $\sim 1/(1-\beta) = 10$ gradients.

Geometrically, momentum acts like a ball rolling down a hill — it accumulates speed in downhill directions and resists direction changes.

**Nesterov variant** (not in this library but often superior): computes the gradient at the *anticipated* position $W - \alpha \beta v$ rather than the current position, giving a more accurate gradient signal and faster convergence in theory.

**Reference**: Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1–17.

---

### RMSprop (Hinton, 2012)

RMSprop normalizes the gradient by its recent root-mean-square, giving each parameter its own adaptive learning rate:

$$v \leftarrow \beta v + (1-\beta) g^2 \qquad (\beta = 0.9)$$

$$W \leftarrow W - \frac{\alpha}{\sqrt{v} + \varepsilon} \odot g \qquad (\varepsilon = 10^{-8})$$

where all operations are element-wise and $v$ is initialized to 0.

**Effect**: parameters with consistently large gradients get a smaller effective learning rate; parameters with small gradients get a larger effective rate. This adapts to the local curvature.

$\varepsilon$ prevents division by zero and provides a floor on the effective learning rate.

**Note**: RMSprop was introduced in an unpublished Coursera lecture, making it unusual among widely-used algorithms:

**Reference**: Hinton, G. (2012). Neural Networks for Machine Learning, Lecture 6e (Coursera). [Unpublished but widely cited.]

---

### Adam — Adaptive Moment Estimation (Kingma & Ba, 2015)

Adam combines momentum (first moment) with RMSprop (second moment):

**Step 1** — update biased first moment estimate:
$$m \leftarrow \beta_1 m + (1 - \beta_1) g \qquad (\beta_1 = 0.9)$$

**Step 2** — update biased second moment estimate:
$$v \leftarrow \beta_2 v + (1 - \beta_2) g^2 \qquad (\beta_2 = 0.999)$$

**Step 3** — bias correction (both moments initialized at 0 are biased toward 0):
$$\hat{m} = \frac{m}{1 - \beta_1^t}, \qquad \hat{v} = \frac{v}{1 - \beta_2^t}$$

**Step 4** — parameter update:
$$W \leftarrow W - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon} \qquad (\varepsilon = 10^{-8})$$

**Bias correction derivation**: after $t$ steps, $m = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i$. Taking expectation: $\mathbb{E}[m] = \mathbb{E}[g] \cdot (1 - \beta_1^t)$. Dividing by $(1-\beta_1^t)$ gives an unbiased estimate $\hat{m} \approx \mathbb{E}[g]$.

**Effective learning rate**: the Adam update magnitude is approximately bounded by $\alpha$, regardless of gradient scale — useful property for setting the learning rate. A common default is $\alpha = 10^{-3}$.

**Convergence**: Adam has been shown to converge in the convex case; for non-convex losses, empirical performance is strong but there exist counterexamples where Adam diverges (addressed by AMSGrad). In practice, Adam is the default optimizer for most deep learning tasks.

**Reference**: Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.

---

### AdamW — Adam with Decoupled Weight Decay (Loshchilov & Hutter, 2019)

Standard Adam + L2 regularization is **not** equivalent to weight decay, because the adaptive scaling $1/(\sqrt{\hat{v}} + \varepsilon)$ is also applied to the regularization gradient $\lambda W$, making the effective weight decay magnitude depend on gradient history in a complex and undesirable way.

**AdamW** decouples weight decay from gradient adaptation:

$$W \leftarrow W(1 - \alpha \lambda) - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon} \qquad (\lambda = 0.01 \text{ default})$$

The weight decay term $(1 - \alpha\lambda)$ is applied directly to the weights, *not* through the gradient path. This restores the correct L2 regularization semantics.

**Empirical results**: AdamW outperforms Adam + L2 consistently on language models, transformers, and image classification when regularization is desired.

**Reference**: Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*.

---

### Comparison and choice guide

| Optimizer | Pros | Cons |
|-----------|------|------|
| SGD | Simple, generalization often best | Slow, sensitive to LR |
| Momentum | Faster than SGD | Still requires LR tuning |
| RMSprop | Good for RNNs | No bias correction |
| Adam | Fast convergence, good defaults | Can generalize slightly worse than SGD+momentum for vision |
| AdamW | Best of Adam + proper regularization | Default when weight decay is needed |

In practice: **start with Adam ($\alpha=10^{-3}$)** for quick experimentation. Switch to **AdamW** if using weight decay. Use **SGD + Momentum** for training final production image classifiers if best generalization is critical.
