# 06 — Regularization

> **Navigation** ← [05 — Weight Initialization](05-weight-init.md) | [07 — Dropout →](07-dropout.md)

---

## Level 1 — Concepts

### Overfitting

A model **overfits** when it learns the training data too precisely — including its noise and idiosyncrasies — and fails to generalize to new, unseen data. This is the central challenge of machine learning.

Intuitively: if you have very few examples but a network with many parameters, the network can memorize the training data perfectly (loss = 0) without learning any meaningful pattern. On new data, it performs no better than random.

### Regularization

**Regularization** is any technique that discourages overfitting. This document covers **weight regularization**: adding a penalty term to the loss function that punishes large weights.

The intuition: large weights amplify small differences in the input into large differences in the output — the network becomes overly sensitive. Keeping weights small forces the network to find smoother, more generally applicable functions.

### L1 vs L2

| | L1 (Lasso) | L2 (Ridge) |
|-|-------|------|
| Penalty shape | Sum of absolute values | Sum of squares |
| Effect on weights | Drives many weights to exactly 0 (sparse) | Shrinks all weights uniformly (small but non-zero) |
| Useful for | Feature selection, sparse models | General regularization |

**ElasticNet** combines both: you control the mix ratio $\rho$ between L1 and L2.

### The regularization hyperparameter $\lambda$

$\lambda$ controls how strongly the penalty is applied:
- $\lambda = 0$: no regularization (standard training).
- $\lambda$ too large: weights are driven near zero, the network underfits.
- $\lambda$ just right: balances the fit to training data against model simplicity.

$\lambda$ is a hyperparameter to tune, typically with cross-validation.

---

## Level 2 — Mathematics

### Penalized loss

Regularization adds a penalty $\Omega(\theta)$ to the loss:

$$\tilde{\mathcal{L}}(x, y, \theta) = \mathcal{L}(x, y, \theta) + \Omega(\theta)$$

where $\theta = \{W^{(l)}, b^{(l)}\}$ are all network parameters. In practice, **biases are typically not regularized** (they have far fewer parameters and their regularization has little effect).

---

### L2 Regularization (Ridge / Weight Decay)

$$\Omega_{L2}(W) = \frac{\lambda}{2} \sum_{l} \|W^{(l)}\|_F^2 = \frac{\lambda}{2} \sum_l \sum_{ij} \left(W^{(l)}_{ij}\right)^2$$

The factor $\frac{1}{2}$ simplifies the derivative.

**Gradient contribution**:

$$\frac{\partial \Omega_{L2}}{\partial W^{(l)}} = \lambda W^{(l)}$$

During the weight update, this adds a "decay" term:

$$W \leftarrow W - \alpha \left(\frac{\partial \mathcal{L}}{\partial W} + \lambda W\right) = W(1 - \alpha\lambda) - \alpha \frac{\partial \mathcal{L}}{\partial W}$$

The factor $(1 - \alpha\lambda)$ shrinks weights toward zero each step — hence the name **weight decay**.

**Bayesian interpretation**: L2 regularization corresponds to placing a **Gaussian prior** on the weights:

$$p(W) = \prod_{ij} \mathcal{N}(W_{ij}; 0, \lambda^{-1})$$

Maximum a posteriori (MAP) estimation with this prior is equivalent to minimizing the L2-penalized loss. The prior encodes the belief that "weights should be small."

**Reference**: Tikhonov, A. N. (1963). Solution of incorrectly formulated problems and the regularization method. *Soviet Mathematics Doklady*, 4. [L2 regularization as Tikhonov regularization.]

---

### L1 Regularization (Lasso)

$$\Omega_{L1}(W) = \lambda \sum_l \|W^{(l)}\|_1 = \lambda \sum_l \sum_{ij} |W^{(l)}_{ij}|$$

**Gradient contribution** (subgradient at 0):

$$\frac{\partial \Omega_{L1}}{\partial W^{(l)}_{ij}} = \lambda \cdot \text{sign}(W^{(l)}_{ij})$$

where $\text{sign}(0)$ is defined as 0 in the subgradient sense.

**Key property: sparsity**. The L1 penalty has a singularity (non-differentiability) at $W = 0$. This means gradient descent will drive exactly-small weights to exactly 0 (as opposed to L2, which shrinks but never reaches 0). The result is a **sparse weight matrix** — many connections become exactly zero, effectively performing feature selection.

Geometrically: the L1 ball (unit ball for the $\ell_1$ norm) has corners at the coordinate axes. Constrained optimization problems tend to have solutions at these corners where many weights are zero.

**Bayesian interpretation**: L1 corresponds to a **Laplace prior**:

$$p(W) \propto \exp\left(-\lambda \sum_{ij} |W_{ij}|\right) = \prod_{ij} \text{Laplace}(W_{ij}; 0, \lambda^{-1})$$

The Laplace distribution has heavier tails than a Gaussian and a sharper peak at 0, encouraging most weights to be exactly zero while allowing a few to be large.

**Reference**: Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *JRSS-B*, 58(1), 267–288.

---

### ElasticNet

$$\Omega_{\text{EN}}(W) = \lambda \left[ \rho \sum_l \|W^{(l)}\|_1 + \frac{1-\rho}{2} \sum_l \|W^{(l)}\|_F^2 \right]$$

where $\rho \in [0, 1]$ controls the L1–L2 mix:
- $\rho = 1$: pure L1 (Lasso)
- $\rho = 0$: pure L2 (Ridge)

**Gradient**:

$$\frac{\partial \Omega_{\text{EN}}}{\partial W_{ij}} = \lambda \left[\rho \cdot \text{sign}(W_{ij}) + (1-\rho) \cdot W_{ij}\right]$$

**Motivation**: Lasso has issues when predictors are correlated (it arbitrarily selects one and zeros the others). ElasticNet handles correlated features more gracefully — it groups and shrinks them together (via L2) while still producing some sparsity (via L1).

**Reference**: Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS-B*, 67(2), 301–320.

---

### Implementation in the library

In `trainer.rs`, regularization gradients are applied **after** the backpropagation gradients are accumulated and **before** the optimizer step. This ordering is mathematically equivalent to including the penalty in the loss, but separating them into two code paths:

1. Backprop computes $\partial \mathcal{L} / \partial W$ (the data-fitting gradient).
2. The regularization component adds $\partial \Omega / \partial W$ (the penalty gradient).
3. The optimizer applies the combined gradient.

The regularization is applied only to **weights** $W$, not to **biases** $b$ — a standard practice that avoids biasing the network's shift parameters.

---

### Bias-variance decomposition

Regularization can be understood through the **bias-variance trade-off**. For any estimator $\hat{f}$:

$$\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[\left(\hat{f}(x) - \mathbb{E}[\hat{f}(x)]\right)^2\right]}_{\text{Variance}} + \sigma^2_\varepsilon$$

- **High variance** (overfitting): the model is sensitive to the particular training set; different datasets would give very different models. Regularization reduces variance.
- **High bias** (underfitting): the model is systematically wrong regardless of the training set. Regularization increases bias slightly.

The right $\lambda$ trades a small increase in bias for a large reduction in variance, reducing overall test error.

**Reference**: Geman, S., Bienenstock, E., & Doursat, R. (1992). Neural Networks and the Bias/Variance Dilemma. *Neural Computation*, 4(1).
