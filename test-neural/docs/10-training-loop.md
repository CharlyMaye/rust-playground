# 10 — Training Loop, Mini-batches & Callbacks

> **Navigation** ← [09 — LR Schedules](09-lr-schedules.md) | [11 — Metrics →](11-metrics.md)

---

## Level 1 — Concepts

### The training loop

The training loop is the engine that drives learning. It repeats these steps many times:

1. **Shuffle** the dataset.
2. **Divide** it into mini-batches of size $B$.
3. For each mini-batch:
   a. Run the forward pass (compute predictions).
   b. Compute the loss.
   c. Run the backward pass (compute gradients).
   d. Apply the optimizer step (update weights).
4. Compute validation loss on the held-out validation set.
5. Call all callbacks (early stopping, checkpointing, LR scheduling, progress bar).

One full pass through the entire training set is called an **epoch**.

### Epochs and steps

- **Epoch**: one full pass over the training data.
- **Step** (or iteration): one optimizer update, using one mini-batch.

If you have 1000 training samples and a batch size of 32, one epoch contains $\lceil 1000/32 \rceil = 32$ steps.

Typical training runs last 10–300 epochs depending on the problem complexity and dataset size.

### Mini-batches

- **Batch size = 1** (online learning): very noisy gradients; can escape local minima but slow convergence.
- **Batch size = full dataset** (batch gradient descent): exact gradients; slow per epoch, memory-intensive.
- **Mini-batch** (e.g., 32–256 samples): the practical compromise — vectorised computation on GPUs/SIMD, reasonable gradient estimates, fast updates.

The library uses `Dataset::batches(batch_size)` to produce mini-batch iterators. The dataset is shuffled before each epoch to prevent the model from learning batch order artifacts.

### Validation

A separate **validation set** (not used for training) is used to monitor how well the model generalizes after each epoch. This is the signal for:
- **EarlyStopping**: stop training when validation loss stops improving.
- **ModelCheckpoint**: save the model whenever validation loss reaches a new best.
- **ReduceOnPlateau**: reduce LR when validation loss stagnates.

### Callbacks

Callbacks are hooks that run at specific points in the training loop. The library provides:

| Callback | When it runs | What it does |
|----------|-------------|--------------|
| `EarlyStopping` | End of each epoch | Stops training if val loss doesn't improve for `patience` epochs |
| `ModelCheckpoint` | End of each epoch | Saves model to disk when val loss improves |
| `LearningRateScheduler` | End of each epoch | Adjusts LR according to a schedule |
| `ProgressBar` | During training | Prints progress, loss, ETA |

---

## Level 2 — Mathematics

### Mini-batch SGD as a Monte Carlo gradient estimator

The true gradient of the loss over the full dataset of $N$ samples is:

$$\nabla_W \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \nabla_W \ell_i(W)$$

Computing this requires $N$ forward/backward passes per step — prohibitively expensive for large datasets.

The mini-batch estimate uses a random subset $\mathcal{B} \subset \{1, \ldots, N\}$ of size $B$:

$$\hat{g}_\mathcal{B} = \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_W \ell_i(W)$$

**Unbiasedness**: $\mathbb{E}[\hat{g}_\mathcal{B}] = \nabla_W \mathcal{L}$ (each sample is equally likely to appear in any batch when shuffling is uniform).

**Variance**: $\text{Var}[\hat{g}_\mathcal{B}] = \frac{1}{B} \text{Var}[\nabla_W \ell_i]$. Larger batches reduce variance $(\propto 1/B)$ but do not reduce bias (which is 0).

**The noise is beneficial**: gradient noise helps SGD escape sharp local minima. Sharp minima generalize poorly; flat minima generalize well. Mini-batch noise provides an implicit regularization effect that biases the optimizer toward flat minima.

**Reference**: Keskar, N. S., et al. (2017). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. *ICLR*. (Empirically shows large batches converge to sharper minima.)

---

### Dataset shuffling — Fisher-Yates algorithm

The library implements Fisher-Yates shuffle (`dataset.rs`):

```
for i from N-1 down to 1:
    j = random integer in [0, i]
    swap(data[i], data[j])
```

This produces a **uniformly random permutation** in $O(N)$ time. Each of the $N!$ possible orderings is equally likely.

**Why shuffle each epoch?** Without shuffling, the mini-batches are the same in every epoch. The optimizer could in principle learn to exploit batch ordering (e.g., if samples are sorted by label). Shuffling before each epoch breaks this and ensures each epoch sees a different partition into mini-batches.

**Reference**: Fisher, R. A., & Yates, F. (1938). *Statistical Tables for Biological, Agricultural and Medical Research*. (The original shuffle algorithm.) Knuth, D. E. (1969). *The Art of Computer Programming*, Vol. 2. (Modern formulation.)

---

### Early stopping as regularization

Early stopping halts training when the validation loss has not improved by `min_delta` for `patience` consecutive epochs. The best model state (at the epoch of minimum validation loss) is optionally restored.

Formally, early stopping at iteration $t^* = \arg\min_t \mathcal{L}_{\text{val}}(t)$ gives a model with effective L2 regularization strength approximately:

$$\lambda_{\text{eff}} \approx \frac{1}{\alpha \cdot t^*}$$

where $\alpha$ is the learning rate. Early stopping in a regime of small LR and large training time approaches zero regularization; stopping early is equivalent to strong regularization.

This equivalence (for gradient descent on quadratic losses) was shown formally by:

**Reference**: Yao, Y., Rosasco, L., & Caponnetto, A. (2007). On Early Stopping in Gradient Descent Learning. *Constructive Approximation*, 26(2), 289–315.

---

### ModelCheckpoint semantics

`ModelCheckpoint` monitors `val_loss` and saves the model when `new_val_loss < best_val_loss - min_delta`. The file extension determines the format:
- `.json` → human-readable JSON (via `serde_json`).
- `.bin` → compact binary (via `bincode`).

Saving at the best validation checkpoint and loading it at the end is equivalent to **early stopping with model restoration** — it combines the stopping criterion with restoring the optimal state found during training.

---

### ProgressBar and WASM compatibility

The `ProgressBar` callback reports epoch progress, current training/validation losses, and an ETA estimate. Because `std::time::Instant` is not available in WebAssembly (WASM targets lack a standard monotonic clock without JavaScript interop), the ETA feature is conditionally compiled:

```
#[cfg(not(target_arch = "wasm32"))]
let elapsed = start.elapsed();
```

On WASM, the progress bar still reports losses and epoch counts but omits timing information. This allows the same training code to run in both native and browser environments.
