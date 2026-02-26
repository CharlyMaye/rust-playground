# Usage Guides

Practical how-to documentation for all four crates. Each guide is self-contained and focuses on **how to write working Rust code**, not theory. For the underlying mathematics, see the [theory chapters](../README.md).

---

## Guides

| # | Guide | Crate | What you will build |
|---|-------|-------|---------------------|
| 01 | [Quickstart](01-quickstart.md) | `cma-neural-network` | XOR classifier in 20 lines |
| 02 | [Dense Network — Full API](02-dense-network.md) | `cma-neural-network` | Complete tour: datasets, optimizers, callbacks, metrics, serialization |
| 03 | [CNN Inference](03-cnn-inference.md) | `cma-cnn` | Build a convolutional feature extractor, inspect shapes, run a forward pass |
| 04 | [Autograd Training](04-autograd-training.md) | `cma-autograd` | Train a CNN end-to-end with automatic differentiation |
| 05 | [Pre-built Architectures](05-models.md) | `cma-models` | Instantiate LeNet-5, ResNet, VGG-16, AlexNet, EfficientNet-B0 |

---

## Runnable Examples

Each guide has a corresponding Rust example under `examples/` or `cma-models/examples/`:

```
# Core dense-network examples
cargo run --example getting_started      # Guide 01 + 02 highlights
cargo run --example metrics_demo         # Guide 02 — metrics section
cargo run --example serialization        # Guide 02 — IO section
cargo run --example minibatch_demo       # Guide 02 — batch training

# CNN / autograd
cargo run --example cnn_inference        # Guide 03
cargo run --example autograd_training    # Guide 04

# Pre-built models (run from cma-models crate)
cargo run --manifest-path cma-models/Cargo.toml --example lenet5_paper
cargo run --manifest-path cma-models/Cargo.toml --example resnet_paper
cargo run --manifest-path cma-models/Cargo.toml --example vgg_paper
cargo run --manifest-path cma-models/Cargo.toml --example alexnet_paper
cargo run --manifest-path cma-models/Cargo.toml --example efficientnet_paper
```

---

## Crate Dependency Overview

```
cma-neural-network   ←   cma-cnn   ←   cma-models
         ↑_________________↑
              cma-autograd
```

| Crate | Role | Training? |
|-------|------|-----------|
| `cma-neural-network` | Dense (fully-connected) networks | Yes — `TrainingBuilder` |
| `cma-cnn` | CNN layers + inference pipeline | No — forward pass only |
| `cma-autograd` | Automatic differentiation engine | Yes — `CnnTrainer` |
| `cma-models` | Pre-built architectures (uses `cma-cnn`) | No — feature extraction only |

> **Key architectural note**: `cma-cnn::Sequential` and `cma-autograd::Sequential` are two distinct, incompatible types. The former is for serializable inference; the latter is for gradient-tracked training. See [Guide 03](03-cnn-inference.md) and [Guide 04](04-autograd-training.md) for details.

---

## Common Imports

```rust
// Dense networks
use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use cma_neural_network::dataset::Dataset;
use cma_neural_network::callbacks::{EarlyStopping, DeltaMode, LRSchedule, LearningRateScheduler, ProgressBar};
use cma_neural_network::metrics::{accuracy, binary_metrics, auc_roc};
use cma_neural_network::io;

// CNN inference
use cma_cnn::{Sequential, Conv2D, BatchNorm2D, MaxPool2D, Flatten, ActivationLayer};
use cma_cnn::tensor::{Tensor4D, TensorShape};

// Autograd training
use cma_autograd::prelude::*;
use cma_autograd::builder::CnnBuilder;
use cma_autograd::optim::Adam;
use cma_autograd::loss::cross_entropy_loss;

// Pre-built models
use cma_models::lenet::{LeNet5, LeNet5Config};
use cma_models::resnet::ResNetBuilder;
use cma_models::vgg::{VGG16, VGGConfig};
use cma_models::alexnet::{AlexNet, AlexNetConfig};
use cma_models::efficientnet::{EfficientNetB0, EfficientNetConfig};
```

---

## `Float` type

All crates use `Float = f32`. Import it from any crate:

```rust
use cma_neural_network::Float;  // or cma_cnn::Float, cma_autograd::Float
```

Use `Float` for all numeric literals to stay compatible:

```rust
let lr: Float = 0.001;
let array: ndarray::Array1<Float> = ndarray::Array1::zeros(10);
```
