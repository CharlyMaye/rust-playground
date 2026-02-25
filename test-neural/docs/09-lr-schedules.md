# 09 — Learning Rate Schedules

> **Navigation** ← [08 — Optimizers](08-optimizers.md) | [10 — Training Loop →](10-training-loop.md)

---

## Level 1 — Concepts

### Why change the learning rate during training?

A single fixed learning rate is a compromise. Early in training:
- The network is far from a good solution — you want **large steps** to explore quickly.

Late in training:
- The network is near a good solution — **large steps overshoot**; small steps allow fine-tuning into a precise minimum.

A **learning rate schedule** modifies $\alpha_t$ over the course of training to get the best of both phases.

### The 6 schedules in this library

| Schedule | Behavior | When to use |
|----------|----------|-------------|
| **StepLR** | Drops by factor `gamma` every N epochs | Simple baseline |
| **ReduceOnPlateau** | Drops when validation loss stops improving | Automatic without knowing training length |
| **ExponentialLR** | Smooth exponential decay | Gradual continuous reduction |
| **CosineAnnealing** | Smooth half-cosine decay to a minimum | Modern default; great generalization |
| **Warmup** | Linear ramp-up before another schedule | Large models, transformers |
| **OneCycle** | Rise then fall; aggressive and fast | Super-convergence; short training runs |

All schedules are applied via the `LearningRateScheduler` callback, which calls `.step(epoch, val_loss)` at the end of each epoch.

---

## Level 2 — Mathematics

### StepLR

$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / s \rfloor}$$

where $s$ is `step_size` (epochs) and $\gamma \in (0, 1)$ is the multiplicative factor.

Every $s$ epochs, the learning rate is multiplied by $\gamma$. For example, $\gamma = 0.1$ and $s = 30$ gives the classic schedule used in early ResNet training: $\alpha_0 \to \alpha_0/10 \to \alpha_0/100$.

**Reference**: He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*. (Uses StepLR with drop at epoch 90, 120 of a 160-epoch schedule.)

---

### ReduceOnPlateau

$$\alpha_{t+1} = \begin{cases} \alpha_t \cdot \text{factor} & \text{if val\_loss}_t > \text{best} - \delta \text{ for } p \text{ epochs} \\ \alpha_t & \text{otherwise} \end{cases}$$

where `factor` $\in (0,1)$, `patience` $p$, and `min_delta` $\delta$.

This schedules does not require knowing in advance how many epochs training will take — it adapts automatically to the learning curve. Particularly useful for exploratory training runs.

Internally, the library tracks the smoothed best validation loss and counts epochs without sufficient improvement.

---

### ExponentialLR

$$\alpha_t = \alpha_0 \cdot \gamma^t$$

where $t$ is the epoch index and $\gamma = e^{\ln(\alpha_{T}/\alpha_0)/T}$ for a target LR $\alpha_T$ after $T$ epochs.

Taking the logarithm: $\ln \alpha_t = \ln \alpha_0 + t \ln \gamma$ — the log-learning-rate decays linearly. Exponential decay is smooth and requires setting only $\gamma$.

---

### CosineAnnealing (Loshchilov & Hutter, 2017)

$$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_0 - \alpha_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T_{\max}}\right)\right)$$

where $T_{\max}$ is the period (total epochs), and $\alpha_{\min}$ is the minimum learning rate (often $0$ or $10^{-6}$).

**Properties**:
- Starts at $\alpha_0$ at $t = 0$ ($\cos 0 = 1$ → $\alpha = \alpha_0$).
- Ends at $\alpha_{\min}$ at $t = T_{\max}$ ($\cos \pi = -1$ → $\alpha = \alpha_{\min}$).
- Smooth derivatives everywhere — no abrupt drops.
- The slow initial decrease gives time to explore broadly; the rapid decrease at the end allows precise convergence.

**Warm restarts** (SGDR): the original paper extends this to restart the cosine cycle periodically (with increasing period), allowing the model to escape local minima repeatedly. The library implements a single-cycle version.

**Reference**: Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR*.

---

### Warmup

The Warmup schedule linearly increases $\alpha$ from near-zero to $\alpha_0$ over `warmup_epochs` $w$:

$$\alpha_t = \alpha_0 \cdot \frac{t}{w}, \quad t = 0, 1, \ldots, w$$

After epoch $w$, it delegates to an inner schedule (any of the above).

**Motivation**: at the very beginning of training, the randomly initialized weights produce large, noisy gradient estimates. Using a large learning rate immediately on an untrained model can cause large, destructive updates. Warming up the learning rate gradually allows the optimizer's moving averages (Adam's $m$ and $v$) to stabilize before full-speed training begins.

Warmup is especially important for:
- **Large batch training**: gradient noise is reduced by averaging over more samples, allowing higher final LR; but the initial step is correspondingly more aggressive.
- **Transformers / large models**: Adam bias correction at low $t$ values is insufficient for very large models.

**Reference**: Goyal, P., et al. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv:1706.02677. (Recommends 5-epoch linear warmup for large-batch training.) Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*. (Introduces warmup + inverse-sqrt decay for transformers.)

---

### OneCycle Policy (Smith & Topin, 2018)

The OneCycle policy is a two-phase schedule designed for "super-convergence":

**Phase 1** — ascending (from epoch 0 to $T_1 = $ `pct_start` $\times T$):

$$\alpha_t = \frac{\alpha_0}{D} + \frac{t}{T_1}\left(\alpha_{\max} - \frac{\alpha_0}{D}\right)$$

Linear increase from $\alpha_0/D$ (where $D$ is the `div_factor`, default 25) to $\alpha_{\max}$.

**Phase 2** — descending (from $T_1$ to $T$):

$$\alpha_t = \alpha_{\max} \cdot \frac{1}{2}\left(1 + \cos\!\left(\pi \cdot \frac{t - T_1}{T - T_1}\right)\right) \cdot \frac{1}{F_{\text{final}}}$$

Cosine decay from $\alpha_{\max}$ to $\alpha_{\max} / (D \cdot F_{\text{final}})$ where $F_{\text{final}}$ is `final_div_factor`.

**The OneCycle concept**: unlike traditional schedules that only decrease LR, OneCycle first increases it aggressively. The high-LR phase acts like a form of regularization (prevents settling into sharp minima early on) and allows the model to find better basins of attraction. The decreasing phase then fine-tunes into the minimum.

**Super-convergence**: under the right conditions, networks trained with OneCycle converge 5–10× faster (in epochs) than with standard step-decay schedules while achieving equal or better accuracy.

**Reference**: Smith, L. N., & Topin, N. (2018). Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. arXiv:1708.07120.
