# Neural Network Library in Rust

A lightweight, educational neural network library written in pure Rust. Designed for learning, experimentation, and understanding how neural networks work under the hood.

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Table of Contents

- [What is a Neural Network?](#what-is-a-neural-network)
  - [Biological Inspiration](#biological-inspiration)
  - [Mathematical Model](#mathematical-model)
  - [How Learning Works](#how-learning-works)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Your First Network](#your-first-network)
  - [Available Examples](#available-examples)
- [Builder Pattern API](#builder-pattern-api)
  - [NetworkBuilder](#networkbuilder)
  - [TrainingBuilder](#trainingbuilder)
- [Optimizers](#optimizers)
  - [Available Optimizers](#available-optimizers)
  - [Optimizer Selection Guide](#optimizer-selection-guide)
- [Activation Functions](#activation-functions)
  - [Common Activations](#common-activations)
  - [Mathematical Formulas](#mathematical-formulas)
  - [Activation Selection Guide](#activation-selection-guide)
- [Loss Functions](#loss-functions)
  - [Available Loss Functions](#available-loss-functions)
  - [Loss Selection Guide](#loss-selection-guide)
- [Regularization](#regularization)
  - [Dropout](#dropout)
  - [L1 and L2 Regularization](#l1-and-l2-regularization)
- [Mini-Batch Training](#mini-batch-training)
  - [Dataset API](#dataset-api)
  - [Batch Size Selection](#batch-size-selection)
- [Callbacks](#callbacks)
  - [EarlyStopping](#earlystopping)
  - [ModelCheckpoint](#modelcheckpoint)
  - [LearningRateScheduler](#learningratescheduler)
  - [ProgressBar](#progressbar)
- [Metrics](#metrics)
  - [Available Metrics](#available-metrics)
  - [Confusion Matrix](#confusion-matrix)
- [Serialization](#serialization)
  - [JSON and Binary Formats](#json-and-binary-formats)
- [Compute Device](#compute-device)
  - [Available Devices](#available-devices)
  - [Device Selection in Training](#device-selection-in-training)
  - [Error Handling](#error-handling)
- [Reproducibility](#reproducibility)
  - [Setting a Seed](#setting-a-seed)
- [References](#references)

---

## What is a Neural Network?

### Biological Inspiration

Neural networks are computational models **inspired by the human brain**. Just as your brain contains billions of neurons connected by synapses, artificial neural networks consist of mathematical "neurons" connected by weighted connections.

**The biological neuron:**
```
Dendrites (inputs) → Cell Body (processing) → Axon (output)
```

**The artificial neuron:**
```
Inputs × Weights → Sum + Bias → Activation → Output
```

When you learn something new, your brain strengthens certain neural connections. Similarly, when we train a neural network, we adjust the **weights** of connections to improve predictions.

> **Further reading:** 
> - [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk) - Excellent visual explanations
> - Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"

### Mathematical Model

A neuron computes a **weighted sum** of its inputs, adds a **bias**, and applies an **activation function**:

$$y = f\left(\sum_{i=1}^{n} w_i \cdot x_i + b\right)$$

Where:
- $x_i$ = inputs
- $w_i$ = weights (learned parameters)
- $b$ = bias (learned parameter)
- $f$ = activation function (introduces non-linearity)
- $y$ = output

**Why activation functions?** Without them, a neural network would just be a linear function, unable to learn complex patterns. The activation function introduces **non-linearity**, allowing the network to approximate any function (Universal Approximation Theorem).

### How Learning Works

Neural networks learn through a process called **backpropagation** combined with **gradient descent**:

1. **Forward Pass**: Input flows through the network to produce a prediction
2. **Loss Calculation**: Compare prediction to the true value (how wrong are we?)
3. **Backward Pass**: Calculate how each weight contributed to the error
4. **Weight Update**: Adjust weights to reduce the error

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Input     │───▶│   Hidden    │───▶│   Output    │
│   Layer     │    │   Layer(s)  │    │   Layer     │
└─────────────┘    └─────────────┘    └─────────────┘
      ▲                                      │
      │            Backpropagation           │
      └──────────────────────────────────────┘
                  (adjust weights)
```

This process repeats thousands of times until the network makes accurate predictions.

> **Further reading:**
> - Rumelhart, D. E., Hinton, G. E., &amp; Williams, R. J. (1986). "Learning representations by back-propagating errors"
> - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Free online book by Michael Nielsen

---

## Quick Start

### Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
test-neural = { path = "." }
ndarray = "0.15"
```

Build and run:
```bash
# Build in release mode (faster)
cargo build --release

# Run examples
cargo run --release --example getting_started
cargo run --release --example serialization
cargo run --release --example metrics_demo
cargo run --release --example minibatch_demo
```

### Your First Network

The simplest way to create and train a neural network:

```rust
use test_neural::builder::NetworkBuilder;
use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use ndarray::array;

fn main() {
    // 1. Build the network
    let mut network = NetworkBuilder::new(2, 1)       // 2 inputs, 1 output
        .hidden_layer(8, Activation::Tanh)            // Hidden layer with 8 neurons
        .output_activation(Activation::Sigmoid)       // Sigmoid for binary output
        .loss(LossFunction::BinaryCrossEntropy)       // Binary classification
        .optimizer(OptimizerType::adam(0.01))         // Adam optimizer
        .build();

    // 2. Prepare XOR training data
    let inputs = vec![
        array![0.0, 0.0], array![0.0, 1.0],
        array![1.0, 0.0], array![1.0, 1.0],
    ];
    let targets = vec![
        array![0.0], array![1.0],
        array![1.0], array![0.0],
    ];

    // 3. Train the network
    for _ in 0..5000 {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            network.train(input, target);
        }
    }

    // 4. Test predictions
    for input in &inputs {
        let prediction = network.predict(input);
        println!("{:?} → {:.3}", input, prediction[0]);
    }
}
```

### Available Examples

| Example | Description | Command |
|---------|-------------|---------|
| `getting_started` | Complete walkthrough of all features | `cargo run --example getting_started` |
| `serialization` | Save/load models in JSON and binary | `cargo run --example serialization` |
| `metrics_demo` | Evaluation metrics (accuracy, F1, ROC) | `cargo run --example metrics_demo` |
| `minibatch_demo` | Mini-batch training performance | `cargo run --example minibatch_demo` |

---

## Builder Pattern API

This library uses the **Builder Pattern** for creating networks and configuring training. This provides a fluent, readable API that guides you through configuration.

### NetworkBuilder

`NetworkBuilder` creates the network architecture:

```rust
use test_neural::builder::NetworkBuilder;
use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;

let network = NetworkBuilder::new(784, 10)           // 784 inputs (e.g., MNIST), 10 outputs
    .hidden_layer(256, Activation::ReLU)             // First hidden layer
    .hidden_layer(128, Activation::ReLU)             // Second hidden layer
    .output_activation(Activation::Softmax)          // Multi-class output
    .loss(LossFunction::CategoricalCrossEntropy)     // Multi-class loss
    .optimizer(OptimizerType::adam(0.001))           // Adam optimizer
    .dropout(0.3)                                    // 30% dropout
    .l2(0.0001)                                      // L2 regularization
    .build();
```

**Available methods:**

| Method | Description |
|--------|-------------|
| `new(input_size, output_size)` | Create builder with input/output dimensions |
| `hidden_layer(size, activation)` | Add a hidden layer |
| `output_activation(activation)` | Set output layer activation |
| `loss(loss_function)` | Set loss function |
| `optimizer(optimizer)` | Set optimizer |
| `dropout(rate)` | Add dropout regularization (0.0-1.0) |
| `l2(lambda)` | Add L2 regularization (weight decay) |
| `l1(lambda)` | Add L1 regularization (sparsity) |
| `build()` | Create the network |

### TrainingBuilder

`TrainingBuilder` configures the training loop with callbacks:

```rust
use test_neural::builder::NetworkTrainer;
use test_neural::dataset::Dataset;
use test_neural::callbacks::{EarlyStopping, DeltaMode, ProgressBar, LearningRateScheduler, LRSchedule};

// Prepare datasets
let dataset = Dataset::from_vecs(inputs, targets);
let (train, val) = dataset.split(0.8);

// Train with full configuration
let history = network.trainer()
    .train_data(&train)
    .validation_data(&val)
    .epochs(100)
    .batch_size(32)
    .callback(Box::new(EarlyStopping::new(15, 0.001).mode(DeltaMode::Relative)))
    .callback(Box::new(ProgressBar::new(100)))
    .scheduler(LearningRateScheduler::new(
        LRSchedule::ReduceOnPlateau {
            patience: 10,
            factor: 0.5,
            min_delta: 0.0001
        }
    ))
    .fit();

// history contains (train_loss, val_loss) for each epoch
println!("Training completed in {} epochs", history.len());
```

---

## Optimizers

**What is an optimizer?** An optimizer determines *how* the network updates its weights based on the computed gradients. Different optimizers use different strategies to navigate the loss landscape efficiently.

Think of it like descending a mountain in fog:
- **SGD**: Take small steps directly downhill
- **Momentum**: Build up speed going downhill, coast through small uphills
- **Adam**: Adapt step size for each direction based on terrain

### Available Optimizers

#### SGD (Stochastic Gradient Descent)

The simplest optimizer. Updates weights proportionally to the gradient.

$$w_{t+1} = w_t - \eta \cdot \nabla L$$

```rust
let optimizer = OptimizerType::sgd(0.1);  // learning_rate = 0.1
```

| | |
|---|---|
| **Use case** | Simple problems, research/reproducibility |
| **Learning rate** | 0.01-0.5 |
| ✅ **Pros** | Simple, fast, reproducible |
| ❌ **Cons** | Slow convergence, requires LR tuning |

#### Momentum

Adds "momentum" to accelerate convergence and smooth out oscillations.

$$v_{t+1} = \beta \cdot v_t + \nabla L$$
$$w_{t+1} = w_t - \eta \cdot v_{t+1}$$

```rust
let optimizer = OptimizerType::momentum(0.1);  // beta = 0.9 by default
```

| | |
|---|---|
| **Use case** | Faster convergence than SGD |
| **Learning rate** | 0.01-0.1 |
| ✅ **Pros** | Faster than SGD, navigates valleys better |
| ❌ **Cons** | Requires beta tuning, can overshoot |

#### RMSprop

Adapts learning rate per parameter based on gradient history.

$$s_{t+1} = \beta \cdot s_t + (1-\beta) \cdot (\nabla L)^2$$
$$w_{t+1} = w_t - \frac{\eta}{\sqrt{s_{t+1} + \epsilon}} \cdot \nabla L$$

```rust
let optimizer = OptimizerType::rmsprop(0.01);
```

| | |
|---|---|
| **Use case** | RNNs, unstable gradients |
| **Learning rate** | 0.001-0.01 |
| ✅ **Pros** | Handles unstable gradients well, per-parameter LR |
| ❌ **Cons** | No momentum, can be slow on some problems |

#### Adam (Recommended ⭐)

Combines Momentum and RMSprop. The most popular optimizer for deep learning.

$$m_{t+1} = \beta_1 \cdot m_t + (1-\beta_1) \cdot \nabla L$$
$$v_{t+1} = \beta_2 \cdot v_t + (1-\beta_2) \cdot (\nabla L)^2$$
$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \cdot \hat{m}_{t+1}$$

```rust
let optimizer = OptimizerType::adam(0.001);  // Recommended default
```

| | |
|---|---|
| **Use case** | General deep learning (default choice) |
| **Learning rate** | 0.001 (most universal) |
| ✅ **Pros** | Combines momentum + RMSprop, 2-10x faster than SGD, bias correction, per-parameter LR |
| ❌ **Cons** | More memory (stores m and v), can generalize worse than SGD |

#### AdamW

Adam with decoupled weight decay. Better generalization than L2 regularization.

```rust
let optimizer = OptimizerType::adamw(0.001, 0.01);  // lr=0.001, weight_decay=0.01
```

| | |
|---|---|
| **Use case** | Preventing overfitting, large models |
| **Learning rate** | 0.001, weight_decay: 0.01-0.1 |
| ✅ **Pros** | Better regularization than L2, improved generalization |
| ❌ **Cons** | Extra hyperparameter (weight_decay) |

### Optimizer Selection Guide

| Use Case | Recommended | Learning Rate |
|----------|-------------|---------------|
| **General deep learning** | Adam | 0.001 |
| **Preventing overfitting** | AdamW | 0.001 (wd=0.01) |
| **Research/reproducibility** | SGD + Momentum | 0.01-0.1 |
| **Unstable gradients (RNN)** | RMSprop | 0.001 |
| **Quick prototyping** | Adam | 0.001 |

### Practical Tips

**Starting Learning Rates:**
- SGD: 0.01 - 0.1
- Momentum: 0.01 - 0.1  
- RMSprop: 0.001 - 0.01
- Adam: **0.001** (most universal)
- AdamW: 0.001

**If training doesn't converge:**
1. Reduce learning rate (÷10)
2. Try Adam if using SGD
3. Check weight initialization (Xavier for Sigmoid/Tanh, He for ReLU)

**For better results:**
- Adam is the best default choice
- AdamW if you observe overfitting
- Momentum + SGD for academic research
- RMSprop for RNN/LSTM

> **Reference:** Kingma, D. P., &amp; Ba, J. (2014). "Adam: A Method for Stochastic Optimization." *arXiv:1412.6980*

---

## Activation Functions

**What are activation functions?** They introduce non-linearity into the network, allowing it to learn complex patterns. Without them, a neural network would just be a linear transformation, no matter how many layers.

### Common Activations

#### Sigmoid

Squashes output to range (0, 1). Good for binary classification output.

$$f(x) = \frac{1}{1 + e^{-x}}$$

```rust
Activation::Sigmoid  // Output: (0, 1)
```

| | |
|---|---|
| **Use** | Binary classification output layer |
| ✅ **Pros** | Normalized output, interpretable as probability, well-defined gradient |
| ❌ **Cons** | Vanishing gradient for large/small values, not zero-centered, slow (`exp()`) |

#### Tanh

Squashes output to range (-1, 1). Zero-centered, often better than Sigmoid for hidden layers.

$$f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```rust
Activation::Tanh  // Output: (-1, 1)
```

| | |
|---|---|
| **Use** | Hidden layers, RNN/LSTM gates |
| ✅ **Pros** | Zero-centered output, stronger gradient than sigmoid |
| ❌ **Cons** | Vanishing gradient (less than sigmoid), slow (`exp()`) |

#### ReLU (Rectified Linear Unit)

The modern default for hidden layers. Fast and effective.

$$f(x) = \max(0, x)$$

```rust
Activation::ReLU  // Output: [0, ∞)
```

| | |
|---|---|
| **Use** | Hidden layers (default choice) |
| ✅ **Pros** | Very fast (simple comparison), no vanishing gradient for positive values, promotes sparsity |
| ❌ **Cons** | **Dying neurons** (gradient = 0 forever), not zero-centered |

#### Leaky ReLU

Fixes the "dying ReLU" problem by allowing small negative gradients.

$$f(x) = \begin{cases} x &amp; \text{if } x > 0 \\ 0.01x &amp; \text{if } x \leq 0 \end{cases}$$

```rust
Activation::LeakyReLU  // Output: (-∞, ∞)
```

| | |
|---|---|
| **Use** | Deep networks with dying neuron issues |
| ✅ **Pros** | Solves dying ReLU problem, fast, gradient always active |
| ❌ **Cons** | Inconsistent results across tasks, requires alpha hyperparameter |

#### Softmax

Converts outputs to probability distribution. Use for multi-class classification output.

$$f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

```rust
Activation::Softmax  // Output: (0, 1), sum = 1
```

| | |
|---|---|
| **Use** | Multi-class classification output layer |
| ✅ **Pros** | Clear probabilistic interpretation, standard for multi-class |
| ❌ **Cons** | Only for output layer, computationally more expensive |

### Mathematical Formulas

| Activation | Formula | Derivative | Range |
|------------|---------|------------|-------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | (0, 1) |
| Tanh | $\tanh(x)$ | $1 - \tanh^2(x)$ | (-1, 1) |
| ReLU | $\max(0, x)$ | $\begin{cases} 1 &amp; x > 0 \\ 0 &amp; x \leq 0 \end{cases}$ | [0, ∞) |
| Leaky ReLU | $\max(0.01x, x)$ | $\begin{cases} 1 &amp; x > 0 \\ 0.01 &amp; x \leq 0 \end{cases}$ | (-∞, ∞) |

### Activation Selection Guide

| Layer Type | Recommended | Why |
|------------|-------------|-----|
| **Hidden layers (default)** | ReLU | Fast, effective, standard |
| **Hidden (deep networks)** | Leaky ReLU, ELU | Avoids dying neurons |
| **Binary classification output** | Sigmoid | Output in (0, 1) |
| **Multi-class output** | Softmax | Probability distribution |
| **Regression output** | Linear (none) | Unbounded output |
| **RNN/LSTM gates** | Tanh, Sigmoid | Traditional choice |
| **Transformers (GPT, BERT)** | GELU | State-of-the-art for NLP |

### Activation Decision Tree

```
What layer are you configuring?
├─ OUTPUT Layer
│  ├─ Binary classification? → Sigmoid
│  ├─ Multi-class classification? → Softmax
│  ├─ Regression (continuous values)? → Linear (none)
│  └─ Regression (positive values)? → Softplus / ReLU
│
└─ HIDDEN Layer
   ├─ Speed constraint?
   │  ├─ Ultra-fast (embedded)? → Hard Sigmoid / Hard Tanh
   │  └─ Fast → ReLU, Leaky ReLU
   │
   ├─ Network type?
   │  ├─ Transformer / NLP? → GELU
   │  ├─ Deep CNN? → Mish
   │  ├─ RNN / LSTM? → Tanh
   │  └─ Feedforward? → See below
   │
   ├─ Network depth?
   │  ├─ Shallow (< 5 layers)? → ReLU
   │  ├─ Deep (> 10 layers)? → SELU, ELU
   │  └─ Very deep (> 50)? → SELU with LeCun init
   │
   └─ Dying neurons problem?
      ├─ Yes → Leaky ReLU, ELU
      └─ No → ReLU
```

---

## Loss Functions

**What is a loss function?** It measures how wrong the network's predictions are. The goal of training is to minimize this value. Think of it as a "score" of how badly the network is doing.

### Available Loss Functions

#### MSE (Mean Squared Error)

Best for regression tasks. Heavily penalizes large errors.

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

```rust
LossFunction::MSE
```

| | |
|---|---|
| **Use** | Regression (predicting continuous values) |
| ✅ **Pros** | Heavily penalizes large errors, differentiable everywhere, intuitive interpretation |
| ❌ **Cons** | Not optimal for classification, vanishing gradient with Sigmoid |

#### MAE (Mean Absolute Error)

More robust to outliers than MSE.

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

```rust
LossFunction::MAE
```

| | |
|---|---|
| **Use** | Regression with outliers in data |
| ✅ **Pros** | Robust to outliers, intuitive, treats all errors linearly |
| ❌ **Cons** | Constant gradients (slower convergence), not differentiable at zero |

#### Binary Cross-Entropy

Standard for binary classification with Sigmoid output.

$$\text{BCE} = -\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

```rust
LossFunction::BinaryCrossEntropy
```

| | |
|---|---|
| **Use** | Binary classification (yes/no, spam/not spam) |
| ✅ **Pros** | Probabilistic interpretation, stable gradient, fast convergence, standard for binary |
| ❌ **Cons** | Requires predictions in [0, 1], unstable if prediction = 0 or 1 |

#### Categorical Cross-Entropy

Standard for multi-class classification with Softmax output.

$$\text{CCE} = -\sum_{i} y_i \log(\hat{y}_i)$$

```rust
LossFunction::CategoricalCrossEntropy
```

| | |
|---|---|
| **Use** | Multi-class classification (cat/dog/bird) |
| ✅ **Pros** | Multi-class standard, clear probabilistic interpretation, works well with Softmax |
| ❌ **Cons** | Requires one-hot encoded targets |

#### Huber Loss

Combines MSE and MAE. Less sensitive to outliers.

$$L_\delta = \begin{cases} \frac{1}{2}(y-\hat{y})^2 &amp; |y-\hat{y}| \leq \delta \\ \delta|y-\hat{y}| - \frac{1}{2}\delta^2 &amp; \text{otherwise} \end{cases}$$

```rust
LossFunction::Huber
```

| | |
|---|---|
| **Use** | Regression with some outliers |
| ✅ **Pros** | Combines MSE (small errors) and MAE (large errors), less sensitive to outliers, differentiable |
| ❌ **Cons** | Requires delta hyperparameter |

### Loss Selection Guide

| Task | Output Activation | Loss Function |
|------|-------------------|---------------|
| **Regression** | Linear | MSE or Huber |
| **Binary classification** | Sigmoid | BinaryCrossEntropy |
| **Multi-class classification** | Softmax | CategoricalCrossEntropy |
| **Regression with outliers** | Linear | MAE or Huber |

---

## Regularization

**What is overfitting?** When a model performs great on training data but poorly on new data. It has "memorized" the training examples instead of learning general patterns.

**Solution:** Regularization techniques that prevent the model from becoming too complex.

### Dropout

Randomly disables neurons during training, forcing the network to not rely on any single neuron. It's like training an ensemble of smaller networks.

```rust
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(100, Activation::ReLU)
    .dropout(0.3)  // 30% of neurons disabled during training
    .build();

// Important: switch modes for training vs inference
network.train_mode();  // Dropout active
network.eval_mode();   // Dropout disabled (all neurons active)
```

**How it works:**
- **Training**: Each neuron has probability `p` of being "dropped" (output = 0)
- **Inference**: All neurons active, outputs scaled by `(1-p)`

| | |
|---|---|
| **Typical values** | 0.2-0.5 for hidden layers |
| ✅ **Pros** | Very effective against overfitting, equivalent to ensemble of models, no computational cost at inference |
| ❌ **Cons** | Increases training time, requires train/eval mode switching |

> **Reference:** Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR*

### L1 and L2 Regularization

Add penalty terms to the loss function to encourage smaller weights.

**L2 Regularization (Weight Decay):**

$$L_{total} = L_{original} + \frac{\lambda}{2}\sum w^2$$

Encourages small weights but rarely exactly zero.

```rust
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(50, Activation::ReLU)
    .l2(0.001)  // Lambda = 0.001
    .build();
```

| | |
|---|---|
| **Typical values** | λ = 0.0001-0.01 |
| ✅ **Pros** | Simple, stabilizes training, improves generalization |
| ❌ **Cons** | Weights rarely become exactly zero |

**L1 Regularization (Lasso):**

$$L_{total} = L_{original} + \lambda\sum |w|$$

Encourages sparsity (many weights become exactly zero).

```rust
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(50, Activation::ReLU)
    .l1(0.001)
    .build();
```

| | |
|---|---|
| **Typical values** | λ = 0.01-0.1 |
| ✅ **Pros** | More compact models (many weights = 0), built-in feature selection, better interpretability |
| ❌ **Cons** | Can be unstable, may remove useful features |

**Selection Guide:**

| Situation | Recommended | Parameters |
|-----------|-------------|------------|
| Small dataset | Dropout + L2 | dropout=0.3-0.5, λ=0.01 |
| Large network | Dropout | dropout=0.4-0.5 |
| Need sparse weights | L1 | λ=0.01-0.1 |
| General use | L2 | λ=0.0001-0.001 |

### Diagnosing Overfitting

**Signs of overfitting:**
- Training loss very low but validation loss high
- Perfect predictions on training set, poor on test set
- Very large weights in the network

**Solutions (in order of priority):**
1. **More data** (if possible)
2. **Dropout** (0.3-0.5) - Most effective
3. **L2 regularization** (0.001-0.01)
4. **Reduce network size**
5. **Early stopping**

**Tuning strategy:**
- Start without regularization
- If overfitting: add Dropout (0.3)
- If still overfitting: increase dropout (0.4-0.5) or add L2
- If underfitting: reduce regularization

---

## Mini-Batch Training

**What is mini-batch training?** Instead of updating weights after each sample (slow, noisy) or after the entire dataset (memory-intensive, less frequent updates), we update after small batches of samples. This balances speed and stability.

### Why Mini-Batch?

**❌ Problems with Single-Sample Training (pure SGD):**
- Very slow on large datasets
- Noisy gradients → unstable convergence
- Cannot use vectorization
- Too frequent weight updates

**✅ Advantages of Mini-Batch:**
- **2-3x faster** in practice
- More stable gradients (averaged over batch)
- Better CPU cache utilization
- Smoother convergence
- Enables parallelization

### Dataset API

```rust
use test_neural::dataset::Dataset;
use ndarray::array;

// Create dataset from ndarray
let inputs = vec![array![0.0, 0.0], array![0.0, 1.0], /* ... */];
let targets = vec![array![0.0], array![1.0], /* ... */];
let dataset = Dataset::new(inputs, targets);

// Or from Vec<Vec<f64>>
let dataset = Dataset::from_vecs(input_vecs, target_vecs);

// Split into train/test
let (train, test) = dataset.split(0.8);  // 80% train, 20% test

// Split into train/val/test
let (train, val, test) = dataset.split_three(0.7, 0.15);  // 70%/15%/15%

// Shuffle (important: do this before each epoch!)
let mut shuffleable = train.shuffleable();
shuffleable.shuffle();

// Iterate over batches
for (batch_inputs, batch_targets) in shuffleable.batches(32) {
    network.train_batch(&batch_inputs, &batch_targets);
}
```

### Batch Size Selection

| Dataset Size | Recommended Batch Size |
|--------------|------------------------|
| &lt; 1,000 samples | 16-32 |
| 1,000-10,000 | 32-64 |
| 10,000-100,000 | 64-128 |
| &gt; 100,000 | 128-256 |

**Tips:**
- Use powers of 2 (16, 32, 64, 128) for better CPU/GPU optimization
- Larger batches → faster training but may need higher learning rate
- Smaller batches → more noise, can help escape local minima
- **Always shuffle** before each epoch to prevent learning data order

### Adjusting Learning Rate for Batch Size

```rust
// Single-sample training
OptimizerType::adam(0.001)

// Mini-batch training (batch_size=32)
OptimizerType::adam(0.01)   // 10x higher

// Mini-batch training (batch_size=128)
OptimizerType::adam(0.03)   // 30x higher
```

**Rule of thumb:** Learning rate ≈ 0.001 × sqrt(batch_size)

### Mini-Batch Best Practices

✅ **Do:**
- Always `shuffle()` the dataset before each epoch
- Split into train/val/test to detect overfitting
- Start with batch_size=32 then experiment
- Increase learning rate for batch training
- Monitor validation loss (early stopping)

❌ **Don't:**
- Forget to shuffle → network learns the order!
- Use batch size of 1 on large dataset (too slow)
- Use same learning rate as single-sample
- Use batch size > 10% of dataset (loses SGD benefit)

---

## Callbacks

**What are callbacks?** Functions that execute automatically at specific points during training (start/end of epoch, etc.). They automate common tasks like early stopping and model checkpointing.

### Why Callbacks?

**❌ Problems without callbacks:**
- Verbose, repetitive training code
- Difficult to monitor progression
- No automatic saving of best model
- Risk of overtraining (overfitting) without monitoring
- Fixed learning rate = suboptimal convergence

**✅ With callbacks:**
- **EarlyStopping**: Automatically stops if overfitting
- **ModelCheckpoint**: Saves best model automatically
- **LearningRateScheduler**: Adapts LR dynamically
- **ProgressBar**: Real-time progress display
- Clean, maintainable, reusable code

### EarlyStopping

Stops training when validation loss stops improving, preventing overfitting and saving time.

```rust
use test_neural::callbacks::{EarlyStopping, DeltaMode};

// Absolute mode (default): stop if loss doesn't improve by at least min_delta
let early_stop = EarlyStopping::new(15, 0.0001);  // patience=15, min_delta=0.0001

// Relative mode: stop if loss doesn't improve by at least min_delta * best_loss
// Better for very small loss values
let early_stop = EarlyStopping::new(15, 0.001)   // 0.1% improvement required
    .mode(DeltaMode::Relative);
```

**Parameters:**
- `patience`: Number of epochs to wait without improvement before stopping
- `min_delta`: Minimum improvement required to count as "improvement"
- `mode`: `Absolute` (default) or `Relative` comparison

### ModelCheckpoint

Automatically saves the model when validation loss improves.

```rust
use test_neural::callbacks::ModelCheckpoint;

let checkpoint = ModelCheckpoint::new("examples/data/best_model.json", true);  // save_best_only=true
```

**Supported formats:**
- `.json` - Human-readable, larger files
- `.bin` - Binary format, 2-3x smaller

### LearningRateScheduler

Dynamically adjusts learning rate during training. Helps fine-tune convergence.

```rust
use test_neural::callbacks::{LearningRateScheduler, LRSchedule};

// Reduce LR on plateau (recommended) - adapts to training progress
let scheduler = LearningRateScheduler::new(
    LRSchedule::ReduceOnPlateau {
        patience: 10,     // Wait 10 epochs without improvement
        factor: 0.5,      // Multiply LR by 0.5
        min_delta: 0.0001
    }
);

// Step decay: reduce every N epochs (predictable schedule)
let scheduler = LearningRateScheduler::new(
    LRSchedule::StepLR {
        step_size: 30,    // Reduce every 30 epochs
        gamma: 0.1        // Multiply LR by 0.1
    }
);

// Exponential decay: smooth continuous reduction
let scheduler = LearningRateScheduler::new(
    LRSchedule::ExponentialLR { gamma: 0.95 }  // LR *= 0.95 each epoch
);
```

### ProgressBar

Displays real-time training progress with ETA.

```rust
use test_neural::callbacks::ProgressBar;

let progress = ProgressBar::new(100);  // 100 total epochs
```

**Output:**
```
🚀 Training started (100 epochs)
Epoch 10/100 [10.0%] - train_loss: 0.123456 - val_loss: 0.234567 - ETA: 45s
Epoch 20/100 [20.0%] - train_loss: 0.056789 - val_loss: 0.123456 - ETA: 36s
✅ Training completed in 50.23s
```

### Combining Callbacks

The real power comes from combining multiple callbacks:

```rust
let history = network.trainer()
    .train_data(&train)
    .validation_data(&val)
    .epochs(100)
    .batch_size(32)
    .callback(Box::new(EarlyStopping::new(15, 0.001).mode(DeltaMode::Relative)))
    .callback(Box::new(ProgressBar::new(100)))
    .scheduler(LearningRateScheduler::new(LRSchedule::ReduceOnPlateau {
        patience: 10, factor: 0.5, min_delta: 0.0001
    }))
    .fit();
```

### Callbacks Comparison

| Aspect | Without Callbacks | With Callbacks |
|--------|------------------|----------------|
| **Code** | Verbose, repetitive | Concise, reusable |
| **Monitoring** | Manual (print in loop) | Automatic (ProgressBar) |
| **Saving** | Manual (if best_loss...) | Automatic (ModelCheckpoint) |
| **Overfitting** | High risk | Prevented (EarlyStopping) |
| **Learning Rate** | Fixed, suboptimal | Adapted (LR Scheduler) |
| **Dev time** | Longer | Shorter |
| **Maintainability** | Difficult | Easy |

### Callback Selection Guide

| Situation | Recommended Callbacks |
|-----------|----------------------|
| **Quick prototyping** | ProgressBar |
| **Long training** | EarlyStopping + ProgressBar |
| **Production** | EarlyStopping + ModelCheckpoint + ReduceOnPlateau |
| **Fine-tuning** | ExponentialLR + ModelCheckpoint |
| **Small dataset** | EarlyStopping (patience=5) + Dropout |
| **Large dataset** | ReduceOnPlateau + ModelCheckpoint |
| **Optimal (recommended)** | **All combined!** |

---

## Metrics

**Why metrics?** Accuracy alone doesn't tell the whole story, especially with imbalanced datasets. Different metrics highlight different aspects of model performance.

### Available Metrics

```rust
use test_neural::metrics::{accuracy, binary_metrics, confusion_matrix_binary, auc_roc};

// Accuracy: percentage of correct predictions
let acc = accuracy(&predictions, &targets, 0.5);  // threshold = 0.5

// Complete binary classification metrics
let metrics = binary_metrics(&predictions, &targets, 0.5);
println!("{}", metrics.summary());
// Accuracy: 0.95 | Precision: 0.92 | Recall: 0.96 | F1: 0.94

// Individual metrics access
println!("Precision: {:.3}", metrics.precision);  // TP / (TP + FP)
println!("Recall: {:.3}", metrics.recall);        // TP / (TP + FN)
println!("F1 Score: {:.3}", metrics.f1_score);    // Harmonic mean

// ROC-AUC (threshold-independent performance measure)
let auc = auc_roc(&predictions, &targets);
println!("AUC: {:.4}", auc);  // 1.0 = perfect, 0.5 = random
```

**Metric definitions:**
- **Precision**: "When I predict positive, how often am I right?" = TP / (TP + FP)
- **Recall**: "Of all actual positives, how many did I find?" = TP / (TP + FN)
- **F1-Score**: Harmonic mean of Precision and Recall

### Confusion Matrix

```rust
use test_neural::metrics::{confusion_matrix_binary, format_confusion_matrix};

let matrix = confusion_matrix_binary(&predictions, &targets, 0.5);
println!("{}", format_confusion_matrix(&matrix, Some(&["Negative", "Positive"])));
```

Output:
```
Confusion Matrix:
                Predicted
            Negative  Positive
Actual  Negative   19        2
        Positive    1       24
```

**Metrics Selection:**

| Situation | Recommended Metric | Why |
|-----------|-------------------|-----|
| Balanced dataset | Accuracy | Simple and meaningful |
| Imbalanced dataset | F1-Score, AUC | Accuracy can be misleading |
| High cost of false positives | Precision | Don't want false alarms |
| High cost of false negatives | Recall | Don't want to miss cases |
| Model comparison | AUC | Threshold-independent |

---

## Serialization

Save and load trained models for later use or deployment.

### JSON and Binary Formats

```rust
use test_neural::io;

// Save to JSON (human-readable, good for debugging)
io::save_json(&network, "examples/data/model.json")?;

// Save to binary (compact, 2-3x smaller)
io::save_binary(&network, "examples/data/model.bin")?;

// Load from file
let loaded = io::load_json("examples/data/model.json")?;
let loaded = io::load_binary("examples/data/model.bin")?;

// Compare serialized sizes
let (json_size, bin_size) = io::get_serialized_size(&network);
println!("JSON: {} bytes, Binary: {} bytes", json_size, bin_size);
```

**When to use which:**
- **JSON**: Debugging, human inspection, version control diffs
- **Binary**: Production, storage efficiency, faster I/O

---

## Compute Device

**What is a compute device?** It determines *where* the training computations are executed. This abstraction separates the network architecture (the "what") from the execution backend (the "how"), following best practices from frameworks like PyTorch and TensorFlow.

### Available Devices

| Device | Status | Description |
|--------|--------|-------------|
| `ComputeDevice::Cpu` | ✅ Available | Standard CPU execution (default) |
| `ComputeDevice::Gpu` | 🚧 Planned | GPU acceleration (not yet implemented) |

```rust
use cma_neural_network::compute::ComputeDevice;

// Check device availability
assert!(ComputeDevice::Cpu.is_available());   // true
assert!(!ComputeDevice::Gpu.is_available());  // false (not yet implemented)

// Validate before use
ComputeDevice::Cpu.validate()?;  // Ok(())
ComputeDevice::Gpu.validate()?;  // Err(ComputeDeviceError::GpuNotAvailable)
```

### Device Selection in Training

You can specify the compute device when using the `TrainingBuilder`:

```rust
use cma_neural_network::builder::NetworkTrainer;
use cma_neural_network::compute::ComputeDevice;

// Method 1: Explicit device selection
let history = network.trainer()
    .device(ComputeDevice::Cpu)
    .train_data(&dataset)
    .epochs(100)
    .batch_size(32)
    .fit();

// Method 2: Shorthand methods
let history = network.trainer()
    .cpu()  // Same as .device(ComputeDevice::Cpu)
    .train_data(&dataset)
    .fit();

// GPU (when available)
let history = network.trainer()
    .gpu()  // Will error until GPU support is implemented
    .train_data(&dataset)
    .try_fit();  // Use try_fit() to handle errors gracefully
```

**Note:** CPU is the default device. If you don't specify a device, CPU will be used.

### Error Handling

When using GPU (or any device that might not be available), use `try_fit()` instead of `fit()` to handle errors gracefully:

```rust
use cma_neural_network::compute::ComputeDeviceError;

// Safe: returns Result
let result = network.trainer()
    .gpu()
    .train_data(&dataset)
    .try_fit();

match result {
    Ok(history) => println!("Training completed: {} epochs", history.len()),
    Err(ComputeDeviceError::GpuNotAvailable) => {
        println!("GPU not available, falling back to CPU");
        // Retry with CPU
        network.trainer()
            .cpu()
            .train_data(&dataset)
            .fit();
    }
}

// Or using fit() which will panic if device unavailable
// Only use this when you're sure the device is available
network.trainer()
    .cpu()  // CPU is always available
    .train_data(&dataset)
    .fit();  // Safe to use fit() here
```

### Architecture Overview

The compute device abstraction follows a clean separation of concerns:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Network      │     │    Trainer      │     │  ComputeDevice  │
│  (architecture) │────▶│   (training)    │────▶│   (execution)   │
│                 │     │                 │     │                 │
│ - layers        │     │ - backprop     │     │ - CPU           │
│ - activations   │     │ - gradients    │     │ - GPU (planned) │
│ - forward pass  │     │ - optimizer    │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Benefits of this architecture:**
- ✅ Network code stays clean (no `#[cfg(feature = "gpu")]` everywhere)
- ✅ Easy to add new backends (e.g., multi-threaded CPU with Rayon)
- ✅ API remains stable as new devices are added
- ✅ Runtime device selection (no recompilation needed)

### WebAssembly Compatibility

This library is designed to run in WebAssembly (WASM) for browser-based neural network training. The compute device abstraction is WASM-compatible:

- **CPU**: Fully supported in WASM
- **GPU**: WebGPU support is planned for future versions

```rust
// Works in both native Rust and WebAssembly
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(8, Activation::Tanh)
    .build();

// CPU training works everywhere
network.trainer()
    .cpu()
    .train_data(&dataset)
    .fit();
```

---

## Activation Functions Comparison Table

| **Function** | **Range** | **Speed** | **Main Use** | **Since** |
|--------------|-----------|-----------|--------------|-----------|
| Sigmoid | [0, 1] | Slow | Binary output | Classic |
| Tanh | [-1, 1] | Slow | Hidden layers | Classic |
| ReLU | [0, ∞) | **Very fast** | Hidden layers (default) | 2010 |
| Leaky ReLU | (-∞, ∞) | **Very fast** | Fix dying neurons | 2013 |
| ELU | (-α, ∞) | Medium | Deep networks | 2015 |
| SELU | (-λα, ∞) | Medium | FeedForward (no BN) | 2017 |
| Swish/SiLU | (-∞, ∞) | Medium | ReLU alternative | 2017 |
| GELU | (-∞, ∞) | Slow | **Transformers (GPT, BERT)** | 2016 |
| Softmax | [0, 1] (sum=1) | Medium | Multi-class output | Classic |

---

## Reproducibility

For reproducible experiments, you can set a random seed that controls dropout masks during training.

### Setting a Seed

```rust
use cma_neural_network::builder::NetworkBuilder;
use cma_neural_network::network::Activation;

let mut network = NetworkBuilder::new(2, 1)
    .hidden_layer(16, Activation::ReLU)
    .dropout(0.3)
    .build();

// Set seed for reproducible training
network.set_seed(42);

// Training will now be deterministic
for epoch in 0..100 {
    network.train_batch(&inputs, &targets);
}

// Clear seed to use system entropy again
network.clear_seed();
```

**Why reproducibility matters:**
- **Debugging**: Same results help isolate issues
- **Research**: Required for scientific reproducibility
- **Comparison**: Fair comparison between experiments

**What is controlled by the seed:**
- Dropout masks during training
- Order is still deterministic if you shuffle data with the same seed

**What is NOT controlled:**
- Weight initialization (set at network creation time)
- Data shuffling (use `Dataset::shuffle_with_seed()` if available)

---

## References

### Foundational Papers

1. **Backpropagation:** Rumelhart, D. E., Hinton, G. E., &amp; Williams, R. J. (1986). "Learning representations by back-propagating errors." *Nature*, 323(6088), 533-536.

2. **Adam Optimizer:** Kingma, D. P., &amp; Ba, J. (2014). "Adam: A Method for Stochastic Optimization." *arXiv:1412.6980*.

3. **Dropout:** Srivastava, N., et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR*, 15(1), 1929-1958.

4. **Batch Normalization:** Ioffe, S., &amp; Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training." *arXiv:1502.03167*.

5. **ReLU:** Nair, V., &amp; Hinton, G. E. (2010). "Rectified Linear Units Improve Restricted Boltzmann Machines." *ICML*.

6. **Universal Approximation:** Hornik, K., Stinchcombe, M., &amp; White, H. (1989). "Multilayer feedforward networks are universal approximators." *Neural Networks*, 2(5), 359-366.

### Learning Resources

- **[3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)** - Beautiful visual explanations (video series)
- **[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)** - Free online book by Michael Nielsen
- **[The Rust ML Book](https://rust-ml.github.io/book/)** - Machine learning in Rust
- **[ML Cheatsheet](https://ml-cheatsheet.readthedocs.io/)** - Quick reference for concepts
- **[ML Cheatsheet - Loss Functions](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)** - Detailed loss function reference
- **[ML Cheatsheet - Activation Functions](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)** - Detailed activation reference
- **[Deep Learning Book](https://www.deeplearningbook.org/)** - Comprehensive textbook by Goodfellow, Bengio, Courville
- **[Neural Networks from Scratch](https://nnfs.io/)** - Book with detailed mathematical explanations
- **[CS231n Stanford](https://cs231n.github.io/)** - Convolutional Neural Networks course notes

### Library Documentation

- **[ndarray Documentation](https://docs.rs/ndarray/)** - N-dimensional array library used in this project
- **[ndarray Crate](https://crates.io/crates/ndarray)** - Crates.io page
- **[serde Documentation](https://serde.rs/)** - Serialization framework for Rust
- **[bincode Documentation](https://docs.rs/bincode/)** - Binary serialization format
- **[Rust Book](https://doc.rust-lang.org/book/)** - Official Rust programming language book

### Additional Resources

- **[Distill.pub](https://distill.pub/)** - Clear explanations of ML concepts with interactive visualizations
- **[Papers With Code](https://paperswithcode.com/)** - ML papers with implementations
- **[Towards Data Science](https://towardsdatascience.com/)** - ML tutorials and articles
- **[Jay Alammar's Blog](https://jalammar.github.io/)** - Visual guides to transformers and attention

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*Built with ❤️ in Rust*
