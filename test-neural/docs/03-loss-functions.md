# 03 — Loss Functions

> **Navigation** ← [02 — Activations](02-activations.md) | [04 — Backpropagation →](04-backpropagation.md)

---

## Level 1 — Concepts

### What is a loss function?

A **loss function** (also called a **cost function** or **objective function**) measures how wrong the network's predictions are. Given the network's output $\hat{y}$ and the true target $y$, the loss $\mathcal{L}(\hat{y}, y)$ produces a single number: **0 when predictions are perfect, larger numbers when they are wrong**.

Training is the process of minimizing this number by adjusting the weights.

### Choosing the right loss

The right loss depends on your task:

| Task | Output activation | Loss function |
|------|------------------|--------------|
| Regression (predict a number) | Linear | MSE or MAE |
| Robust regression (outliers present) | Linear | Huber |
| Binary classification (yes/no) | Sigmoid | Binary Cross-Entropy |
| Multi-class classification | Softmax | Categorical Cross-Entropy |

A mismatch between the output activation and the loss function will cause poor training. For instance, using MSE with a Softmax output is theoretically valid but leads to slow convergence and poor gradient signals.

### Intuition for cross-entropy

Cross-entropy from information theory measures the *surprise* caused by using the predicted distribution $\hat{y}$ when the true distribution is $y$. If the network predicts probability 0.99 for the correct class, the surprise is small ($-\log 0.99 \approx 0.01$). If it predicts 0.01, the surprise is large ($-\log 0.01 \approx 4.6$). The goal is to minimize surprise — i.e., make the high-probability predictions align with the actual labels.

---

## Level 2 — Mathematics

### Mean Squared Error (MSE)

$$\mathcal{L}_{\text{MSE}}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

**Gradient** with respect to the prediction:

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)$$

**Probabilistic interpretation (Maximum Likelihood Estimation)**: MSE is the negative log-likelihood of a Gaussian noise model. Assume $y_i = f(x_i) + \varepsilon_i$ where $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$. Then:

$$\log p(y | x, w) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_i (y_i - \hat{y}_i)^2$$

Maximizing the log-likelihood is equivalent to minimizing $\sum_i (y_i - \hat{y}_i)^2$ — exactly MSE.

**Consequence**: MSE is optimal when residuals are Gaussian. When they are not (heavy tails, outliers), see Huber loss.

**Reference**: Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. §1.2.5.

---

### Mean Absolute Error (MAE)

$$\mathcal{L}_{\text{MAE}}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|$$

**Gradient**:

$$\frac{\partial \mathcal{L}_{\text{MAE}}}{\partial \hat{y}_i} = \frac{1}{n} \text{sign}(\hat{y}_i - y_i)$$

Note: the gradient is constant in magnitude regardless of how large the error is. This makes MAE **more robust to outliers** than MSE (which scales error quadratically), but it has a discontinuous gradient at $\hat{y}_i = y_i$, which can cause oscillation near the optimum.

**Probabilistic interpretation**: MAE corresponds to a Laplace noise model $p(\varepsilon) \propto e^{-|\varepsilon|/b}$.

---

### Huber Loss

$$\mathcal{L}_{\delta}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} h_\delta(\hat{y}_i - y_i)$$

where the Huber function $h_\delta$ with threshold $\delta = 1$ (as in this library) is:

$$h_\delta(d) = \begin{cases} \frac{1}{2} d^2 & |d| \leq \delta \\ \delta \left(|d| - \frac{\delta}{2}\right) & |d| > \delta \end{cases}$$

**Gradient**:

$$\frac{\partial h_\delta}{\partial d} = \begin{cases} d & |d| \leq \delta \\ \delta \cdot \text{sign}(d) & |d| > \delta \end{cases}$$

**Properties**: Huber loss is **quadratic near the origin** (like MSE, giving precise gradient signals for small errors) and **linear far from the origin** (like MAE, bounding the influence of outliers). It is everywhere differentiable, unlike MAE.

**Probabilistic interpretation**: Huber is an M-estimator. It arises from the robust statistical literature where the goal is to be efficient under Gaussian assumptions while being resistant to contamination.

**Reference**: Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73–101.

---

### Binary Cross-Entropy (BCE)

For binary classification with a single Sigmoid output $\hat{y} \in (0, 1)$ and target $y \in \{0, 1\}$:

$$\mathcal{L}_{\text{BCE}}(\hat{y}, y) = -\frac{1}{n}\sum_{i=1}^{n} \Bigl[ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) \Bigr]$$

Predictions are clamped to $[\varepsilon, 1-\varepsilon]$ with $\varepsilon = 10^{-15}$ to avoid $\ln 0 = -\infty$.

**Gradient** with respect to $\hat{y}_i$:

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \hat{y}_i} = \frac{1}{n}\left(\frac{1 - y_i}{1 - \hat{y}_i} - \frac{y_i}{\hat{y}_i}\right) = \frac{\hat{y}_i - y_i}{n \, \hat{y}_i (1 - \hat{y}_i)}$$

**Simplified gradient at the output layer**: when Sigmoid is the output activation and BCE is the loss, the chain rule gives:

$$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial z^{(L)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{(L)}} = \frac{\hat{y} - y}{n \hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) = \frac{\hat{y} - y}{n}$$

The $\hat{y}(1-\hat{y})$ cancels exactly. This is the **simplified delta** the library uses, avoiding numerical instability from computing the ratio.

**Probabilistic interpretation (MLE)**: BCE is the negative log-likelihood of a Bernoulli model: $y_i \sim \text{Bernoulli}(\hat{y}_i)$.

**Reference**: Bishop, C. M. (2006). §4.3.4. Good, I. J. (1952). Rational Decisions. *JRSS-B*, 14, 107–114 (the proper scoring rule perspective).

---

### Categorical Cross-Entropy (CCE)

For multi-class classification with Softmax output $\hat{y} \in (0,1)^K$ summing to 1, and one-hot target $y$:

$$\mathcal{L}_{\text{CCE}}(\hat{y}, y) = -\sum_{k=1}^{K} y_k \ln \hat{y}_k$$

When only one class is correct (one-hot $y$) this simplifies to $-\ln \hat{y}_{k^*}$ where $k^*$ is the true class.

**Gradient with respect to the Softmax pre-activations** $z$ (not $\hat{y}$):

$$\frac{\partial \mathcal{L}_{\text{CCE}}}{\partial z_k} = \hat{y}_k - y_k$$

This elegant result again comes from the cancellation between the CCE derivative and the Softmax Jacobian:

$$\frac{\partial \mathcal{L}}{\partial z_k} = \sum_j \frac{\partial \mathcal{L}}{\partial \hat{y}_j} \cdot \frac{\partial \hat{y}_j}{\partial z_k} = -\frac{y_k}{\hat{y}_k} \cdot \hat{y}_k(1 - \hat{y}_k) + \sum_{j \neq k} \frac{y_j}{\hat{y}_j} \cdot \hat{y}_j \hat{y}_k = \hat{y}_k - y_k$$

(using $\sum_j y_j = 1$).

The library exploits this simplification directly: `delta = predictions - targets`.

**Probabilistic interpretation**: CCE is the negative log-likelihood of a Categorical distribution: $y \sim \text{Categorical}(\hat{y})$.

**Relationship to KL divergence**:

$$\mathcal{L}_{\text{CCE}} = H(y, \hat{y}) = H(y) + D_{\text{KL}}(y \| \hat{y})$$

Since $H(y)$ is constant with respect to the network weights, minimizing CCE is equivalent to minimizing the KL divergence between the true and predicted distributions.

**Reference**: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. §6.2.1–6.2.2. Bishop, C. M. (2006). §4.3.4.
