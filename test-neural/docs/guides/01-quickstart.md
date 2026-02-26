# Guide 01 — Quickstart

> **Navigation** [README](README.md) | [02 — Dense Network →](02-dense-network.md)
>
> **Theory**: [00 — Introduction](../00-introduction.md) | [01 — Architecture](../01-architecture.md)
>
> **Runnable**: `cargo run --example getting_started`

Solve the XOR problem — the canonical first test for any neural network library — in under 25 lines of Rust.

---

## `Cargo.toml`

```toml
[dependencies]
cma-neural-network = { path = "../cma-neural-network" }
ndarray = "0.16"
```

---

## Minimal XOR Example

```rust
use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::dataset::Dataset;
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use ndarray::array;

fn main() {
    // 1. Dataset
    let inputs  = vec![array![0.0f32, 0.0], array![0.0, 1.0],
                       array![1.0, 0.0], array![1.0, 1.0]];
    let targets = vec![array![0.0f32], array![1.0], array![1.0], array![0.0]];
    let mut dataset = Dataset::new(inputs.clone(), targets.clone());

    // 2. Build network
    let mut net = NetworkBuilder::new(2, 1)     // 2 inputs, 1 output
        .hidden_layer(8, Activation::Tanh)       // one hidden layer, 8 neurons
        .output_activation(Activation::Sigmoid)  // binary output ∈ (0, 1)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();

    net.set_seed(42); // reproducible

    // 3. Train
    net.trainer()
        .train_data(&mut dataset)
        .epochs(500)
        .batch_size(4)
        .verbose(1)
        .fit();

    // 4. Infer
    for (x, y) in inputs.iter().zip(targets.iter()) {
        let pred = net.predict(x);
        println!("[{:.0}, {:.0}] -> {:.3}  (expected {:.0})", x[0], x[1], pred[0], y[0]);
    }
}
```

Expected output:

```
[0, 0] -> 0.023  (expected 0)
[0, 1] -> 0.971  (expected 1)
[1, 0] -> 0.968  (expected 1)
[1, 1] -> 0.031  (expected 0)
```

---

## Step-by-step Breakdown

### 1. Dataset

`Dataset::new` takes `Vec<Array1<Float>>` inputs and targets.

```rust
let mut dataset = Dataset::new(inputs, targets);
// Shuffle before training (important if data is ordered):
dataset.shuffle();
// Split into train / validation:
let (mut train, val) = dataset.split(0.8);  // 80 % / 20 %
```

`split()` does **not** shuffle automatically — call `shuffle()` first if needed.

### 2. `NetworkBuilder`

```rust
let net = NetworkBuilder::new(input_size, output_size)
    .hidden_layer(neurons, Activation::ReLU)    // call multiple times for deep nets
    .output_activation(Activation::Sigmoid)     // default is Sigmoid
    .loss(LossFunction::BinaryCrossEntropy)     // default is BinaryCrossEntropy
    .optimizer(OptimizerType::adam(0.001))      // default is Adam(0.001)
    .build();                                   // panics if no hidden_layer() called
```

`.build()` panics when called with no hidden layers. A single `.hidden_layer()` is the minimum.

### 3. `TrainingBuilder`

`.trainer()` returns a fluent builder:

```rust
net.trainer()
    .train_data(&mut dataset)   // required
    .validation_data(&val)      // optional — enables val loss reporting
    .epochs(500)                // default: 100
    .batch_size(32)             // default: 32
    .verbose(1)                 // 0 = silent, 1 = per-epoch, 2 = verbose
    .fit();                     // consumes builder, returns Vec<(train_loss, Option<val_loss>)>
```

### 4. Inference

```rust
let output = net.predict(&input);  // always uses eval mode (dropout disabled)
```

`.predict()` automatically switches to eval mode — dropout is off, BatchNorm uses running statistics.

---

## Adding Callbacks

For anything beyond a toy problem, add at least `EarlyStopping` and a `ProgressBar`:

```rust
use cma_neural_network::callbacks::{EarlyStopping, DeltaMode, ProgressBar};

let epochs = 1000;
net.trainer()
    .train_data(&mut train)
    .validation_data(&val)
    .epochs(epochs)
    .batch_size(32)
    .callback(Box::new(EarlyStopping::new(20, 0.001).mode(DeltaMode::Relative)))
    .callback(Box::new(ProgressBar::new(epochs)))
    .fit();
```

`EarlyStopping::new(patience, min_delta)` — stops training when validation loss has not improved by at least `min_delta` (in relative terms when `.mode(DeltaMode::Relative)`) for `patience` consecutive epochs.

---

## What's Next

- **[Guide 02 — Dense Network Full API](02-dense-network.md)**: all optimizers, regularization, LR schedules, metrics, serialization.
- **[Guide 03 — CNN Inference](03-cnn-inference.md)**: build a Conv2D pipeline with `cma-cnn`.
- **[Guide 04 — Autograd Training](04-autograd-training.md)**: train a CNN end-to-end.
