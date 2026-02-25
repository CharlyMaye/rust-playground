# 00 — Introduction to Neural Networks

> **Navigation** ← [README](README.md) | [01 — Architecture →](01-architecture.md)

---

## Level 1 — Concepts

### What is a neural network?

A neural network is a mathematical function that maps an input (a vector of numbers) to an output (another vector of numbers). The word "neural" comes from a loose analogy with biological brains: the network is composed of many simple units called **neurons**, organized in **layers**, each layer transforming the signal produced by the previous one.

Despite the biological metaphor, you do not need to think about biology to understand or use neural networks. They are, fundamentally, **parameterized functions with a very large number of tunable knobs** (called **weights** and **biases**). Training a network means adjusting those knobs so that the function produces the right output for a given input.

### The central promise — Universal Approximation

The reason neural networks are so powerful is captured by a remarkable theoretical result: **a neural network with at least one hidden layer and a non-linear activation function can approximate, to arbitrary precision, any continuous function on a compact domain**, given enough neurons.

This means that if there exists any mathematical relationship between your inputs and outputs (and there almost always does in practice), a neural network can in principle learn it from data — without you having to specify the functional form.

### The workflow

1. **Define the architecture** — how many layers, how many neurons per layer, what activation functions.
2. **Initialize the weights** — set the initial values of the knobs (see [05 — Weight Initialization](05-weight-init.md)).
3. **Feed data through** (forward pass) — compute the network's output for a given input.
4. **Measure the error** (loss) — how far is the output from what we wanted? (see [03 — Loss Functions](03-loss-functions.md)).
5. **Compute gradients** (backward pass) — how does each weight contribute to the error? (see [04 — Backpropagation](04-backpropagation.md)).
6. **Update the weights** (optimizer step) — nudge each weight in the direction that reduces the error (see [08 — Optimizers](08-optimizers.md)).
7. **Repeat** steps 3–6 many times over the dataset.

This library (`cma-neural-network`) implements steps 1–7 for fully connected (dense) networks. The `cma-cnn` crate extends this to convolutional networks.

### What this library covers

The `cma-neural-network` crate provides:
- Dense (fully connected) layers of arbitrary depth and width.
- 15 activation functions (ReLU, Sigmoid, Tanh, GELU, Swish, Mish, …).
- 5 loss functions (MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy, Huber).
- 5 optimizers (SGD, Momentum, RMSprop, Adam, AdamW).
- 4 weight initialization strategies (Uniform, Xavier, He, LeCun).
- L1 / L2 / ElasticNet regularization and Dropout.
- A fluent builder API so architectures can be described in a few readable lines.

---

## Level 2 — Mathematics

### The McCulloch-Pitts neuron (1943)

The formal origins of artificial neurons date to McCulloch and Pitts (1943), who proposed a binary threshold unit:

$$y = \begin{cases} 1 & \text{if } \sum_i w_i x_i \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

where $x_i \in \{0,1\}$ are binary inputs, $w_i$ are real-valued weights, and $\theta$ is a threshold. This is the abstract ancestor of every neuron in this library.

**Reference**: McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *Bulletin of Mathematical Biophysics*, 5(4), 115–133.

### The Perceptron (Rosenblatt, 1958)

Rosenblatt replaced the fixed threshold with a trainable bias and introduced a learning rule. For a binary classification task, the perceptron update is:

$$w \leftarrow w + \eta (t - \hat{y}) \, x$$

where $\eta > 0$ is the learning rate, $t \in \{0, 1\}$ is the target, and $\hat{y}$ is the prediction.

**Perceptron Convergence Theorem**: If the training data is linearly separable, the perceptron algorithm converges to a solution in a finite number of steps.

The critical limitation — linear decision boundaries — was formalized by Minsky and Papert (1969), who showed the perceptron cannot learn XOR. This limitation is resolved by stacking layers (deep networks) with non-linear activations, which is exactly what this library provides.

**References**:
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386.
- Minsky, M., & Papert, S. (1969). *Perceptrons*. MIT Press.

### Universal Approximation Theorem (Cybenko 1989, Hornik 1991)

**Theorem (Cybenko, 1989)**: Let $\sigma$ be any continuous sigmoidal function. Then finite sums of the form

$$G(x) = \sum_{j=1}^{N} \alpha_j \, \sigma(w_j^T x + b_j)$$

are dense in $C([0,1]^n)$, the space of continuous functions on the unit $n$-cube, with the uniform norm.

In plain language: a single hidden layer network with a sigmoidal activation can approximate any continuous function on a bounded domain to any desired precision, by using sufficiently many hidden neurons.

**Theorem (Hornik, 1991)**: The result holds for *any* squashing function $\sigma$ (not just sigmoidal), and also for output-layer approximation metrics other than the uniform norm.

**Important caveat**: these theorems guarantee *existence* of such a network, not learnability. They say nothing about how to find the right weights, how much data is required, or how long training will take. This is why the rest of the chapters in this documentation matter.

**References**:
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
- Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257.

### Depth and the expressivity advantage

There is a separate line of theory about *depth* specifically. Depth does not just matter for the UAT (which requires only one hidden layer) — it matters for *efficiency*. A function that requires exponentially many neurons to represent with one hidden layer can often be represented with polynomially many neurons when organized into multiple layers.

**Reference**: Delalleau, O., & Bengio, Y. (2011). Shallow vs. Deep Sum-Product Networks. *NeurIPS* 24. Pascanu et al. (2013) provide further analysis for ReLU networks.

### The role of `f32` (single precision)

This library uses `type Float = f32`. The choice is deliberate:
- `f32` occupies 4 bytes vs 8 bytes for `f64` — halving memory bandwidth and storage.
- Modern hardware (GPUs, SIMD/AVX) processes `f32` at 2× the throughput of `f64`.
- Neural network training is **not** sensitive to rounding at `f32` precision: gradients are stochastic estimates anyway (mini-batch noise dominates floating-point noise).
- The WebAssembly target has no hardware `f64` acceleration.

This is consistent with the industry standard: PyTorch and TensorFlow default to `float32`.

**Reference**: Gupta, S., et al. (2015). Deep Learning with Limited Numerical Precision. *ICML*. Micikevicius et al. (2018). Mixed Precision Training. *ICLR* — justifies `f32` (and even `f16`/`bf16`) as sufficient.
