# Guide 02 — Dense Network: Full API

> **Navigation** [← 01 Quickstart](01-quickstart.md) | [README](README.md) | [03 CNN Inference →](03-cnn-inference.md)
>
> **Theory**: [01 — Architecture](../01-architecture.md) | [02 — Activations](../02-activations.md) | [03 — Loss Functions](../03-loss-functions.md) | [06 — Regularization](../06-regularization.md) | [08 — Optimizers](../08-optimizers.md) | [09 — LR Schedules](../09-lr-schedules.md) | [11 — Metrics](../11-metrics.md)
>
> **Runnable**: `cargo run --example getting_started` | `cargo run --example metrics_demo` | `cargo run --example serialization` | `cargo run --example minibatch_demo`

---

## `Cargo.toml`

```toml
[dependencies]
cma-neural-network = { path = "../cma-neural-network" }
ndarray = "0.17"
```

---

## 1. Network Construction

### `NetworkBuilder` — all options

```rust
use cma_neural_network::builder::NetworkBuilder;
use cma_neural_network::network::{Activation, LossFunction, WeightInit};
use cma_neural_network::optimizer::OptimizerType;

let net = NetworkBuilder::new(input_size, output_size)
    // --- Layers (call multiple times to add hidden layers in order) ---
    .hidden_layer(64, Activation::ReLU)
    .hidden_layer(32, Activation::ReLU)
    .output_activation(Activation::Sigmoid)  // default: Sigmoid

    // --- Loss ---
    .loss(LossFunction::BinaryCrossEntropy)  // default: BinaryCrossEntropy

    // --- Optimizer ---
    .optimizer(OptimizerType::adam(0.001))   // default: Adam(0.001)

    // --- Weight initialization (auto-selected by default based on activation) ---
    .weight_init(WeightInit::He)

    // --- Regularization (pick one, or none) ---
    .dropout(0.3)                            // 30% dropout on all hidden layers
    .l2(0.001)                               // L2 weight decay
    // .l1(0.001)                            // L1 sparsity
    // .elastic_net(0.5, 0.001)              // ElasticNet (l1_ratio=0.5)

    .build();  // panics if no hidden_layer() was called
```

### Activation functions

```rust
use cma_neural_network::network::Activation;

Activation::Sigmoid      // σ(z) = 1/(1+e^{-z})    — binary output, gates
Activation::Tanh         // tanh(z)                 — hidden layers, XOR
Activation::ReLU         // max(0, z)               — deep networks (default hidden)
Activation::LeakyReLU    // max(0.01z, z)           — avoids dead neurons
Activation::ELU          // z > 0 ? z : α(e^z − 1) — smooth negative region
Activation::SELU         // self-normalising variant of ELU
Activation::Swish        // z·σ(z)                  — EfficientNet family
Activation::GELU         // z·Φ(z)                  — Transformer family
Activation::Mish         // z·tanh(softplus(z))
Activation::Softmax      // multi-class output (pair with CategoricalCrossEntropy)
Activation::Linear       // identity — regression output
```

### Loss functions

```rust
LossFunction::MSE                      // regression
LossFunction::MAE                      // robust regression
LossFunction::BinaryCrossEntropy       // binary classification (Sigmoid output)
LossFunction::CategoricalCrossEntropy  // multi-class (Softmax output)
LossFunction::Huber                    // robust regression — less sensitive to outliers
```

> **Important**: always pair `Softmax` output with `CategoricalCrossEntropy`. The backward pass uses the fused derivative `output − target`; calling the derivative of Softmax alone will cause a panic.

### Optimizers

```rust
use cma_neural_network::optimizer::OptimizerType;

OptimizerType::sgd(0.1)                    // vanilla SGD
OptimizerType::momentum(0.01)              // SGD + momentum (β=0.9)
OptimizerType::rmsprop(0.001)              // RMSprop (β=0.9, ε=1e-8)
OptimizerType::adam(0.001)                 // Adam (β₁=0.9, β₂=0.999, ε=1e-8)
OptimizerType::adamw(0.001, 0.01)          // AdamW — Adam + decoupled weight decay
```

### Weight initialisation

| Init | When to use |
|------|-------------|
| `WeightInit::He` | ReLU, LeakyReLU, ELU (default for these) |
| `WeightInit::Xavier` | Tanh, Sigmoid (default for these) |
| `WeightInit::LeCun` | SELU |
| `WeightInit::Uniform` | Manual override |

`WeightInit::for_activation(activation)` returns the recommended init for any activation.

---

## 2. Dataset

```rust
use cma_neural_network::dataset::Dataset;
use ndarray::Array1;

// Construction
let dataset = Dataset::new(inputs: Vec<Array1<Float>>, targets: Vec<Array1<Float>>);

// Info
dataset.len();           // number of samples
dataset.is_empty();
dataset.inputs();        // &[Array1<Float>]
dataset.targets();       // &[Array1<Float>]

// Shuffle in-place (Fisher-Yates)
dataset.shuffle();

// Split  — does NOT shuffle automatically
let (mut train, val) = dataset.split(0.8);              // 80/20
let (train, val, test) = dataset.split_three(0.7, 0.15); // 70/15/15

// Manual batch iteration
for (batch_inputs, batch_targets) in train.batches(32) {
    // batch_inputs:  &[Array1<Float>]
    // batch_targets: &[Array1<Float>]
    net.train_batch(batch_inputs, batch_targets);
}
```

> **Tip**: always call `dataset.shuffle()` before `split()` when data is ordered (e.g., class-sorted CSV files).

---

## 3. Training

### High-level `TrainingBuilder`

```rust
use cma_neural_network::builder::NetworkTrainer;

let history = net.trainer()
    .train_data(&mut train)         // required; takes &mut Dataset for shuffling
    .validation_data(&val)          // optional; enables val_loss in history
    .epochs(200)                    // default: 100
    .batch_size(32)                 // default: 32
    .eval_every(5)                  // compute val loss every N epochs (default: 1)
    .max_grad_norm(5.0)             // gradient clipping — prevents exploding gradients
    .verbose(1)                     // 0=silent, 1=normal, 2=verbose
    .callback(Box::new(...))        // see §4 — repeatable
    .scheduler(...)                 // LR scheduler callback
    .fit();                         // returns Vec<(Float, Option<Float>)>

// Use try_fit() to handle GPU unavailability gracefully:
match net.trainer().train_data(&mut train).epochs(100).try_fit() {
    Ok(history) => { /* ... */ }
    Err(e)      => eprintln!("Training failed: {}", e),
}
```

`history` is a `Vec<(train_loss, Option<val_loss>)>` — one entry per epoch.

### Low-level training loop

For custom needs (e.g., manual LR warmup, per-step logging):

```rust
net.train_mode();   // enable dropout

for epoch in 0..epochs {
    train.shuffle();
    for (batch_x, batch_y) in train.batches(32) {
        net.train_batch(batch_x, batch_y);   // forward + backward + update
    }
    // Per-sample alternative:
    // net.train(&input, &target);
}

net.eval_mode();    // disable dropout for inference
let loss = net.evaluate(val.inputs(), val.targets());
```

---

## 4. Callbacks

All callbacks implement the `Callback` trait and are passed to `.callback(Box::new(...))`.

### `EarlyStopping`

```rust
use cma_neural_network::callbacks::{EarlyStopping, DeltaMode};

let es = EarlyStopping::new(10, 0.0001)   // patience=10, min_delta=0.0001
    .mode(DeltaMode::Absolute);   // or DeltaMode::Relative (percentage improvement)

// After training:
es.stopped()      // true if training was stopped early
es.best_epoch()   // epoch with best validation loss
es.best_loss()    // best validation loss seen
```

`DeltaMode::Relative` treats `min_delta` as a fraction: `0.001` means 0.1% improvement required.

### `ModelCheckpoint`

```rust
use cma_neural_network::callbacks::ModelCheckpoint;

// Saves best model automatically. Format inferred from extension:
//   .json  → JSON (human-readable)
//   other  → binary (compact)
let checkpoint = ModelCheckpoint::new("checkpoints/best.json", true);
```

### `LearningRateScheduler`

```rust
use cma_neural_network::callbacks::{LRSchedule, LearningRateScheduler};

// Step decay: multiply LR by gamma every step_size epochs
LRSchedule::StepLR { step_size: 50, gamma: 0.5 }

// Reduce on plateau: halve LR when val loss stops improving
LRSchedule::ReduceOnPlateau { patience: 10, factor: 0.5, min_delta: 0.0001 }

// Exponential decay
LRSchedule::ExponentialLR { gamma: 0.99 }

// Cosine annealing: LR follows a cosine curve down to eta_min
LRSchedule::CosineAnnealing { t_max: 100, eta_min: 1e-6 }

// Linear warmup, then apply another schedule
LRSchedule::Warmup {
    warmup_epochs: 10,
    after: Box::new(LRSchedule::CosineAnnealing { t_max: 90, eta_min: 1e-6 })
}

// One-cycle policy (Smith 2018) — requires knowing total epochs
LRSchedule::OneCycle {
    max_lr: 0.01,
    div_factor: 25.0,   // initial_lr = max_lr / div_factor
    final_div: 1e4,     // final_lr   = initial_lr / final_div
    pct_start: 0.3,     // fraction of epochs in warmup phase
}

// Usage:
let scheduler = LearningRateScheduler::new(schedule).with_epochs(200); // required for OneCycle
net.trainer().scheduler(scheduler).fit();
```

### `ProgressBar`

```rust
use cma_neural_network::callbacks::ProgressBar;

let bar = ProgressBar::new(total_epochs)
    .set_verbose(true);  // print loss next to progress
```

---

## 5. Metrics

All metric functions are free functions in `cma_neural_network::metrics`.

### Binary classification

```rust
use cma_neural_network::metrics::{accuracy, binary_metrics, confusion_matrix_binary,
                                   format_confusion_matrix, roc_curve, auc_roc};

let preds: Vec<Array1<Float>> = inputs.iter().map(|x| net.predict(x)).collect();

// Accuracy
let acc: Float = accuracy(&preds, &targets, 0.5);

// Full precision / recall / F1
let m = binary_metrics(&preds, &targets, 0.5);
println!("{}", m.summary());
// Fields: m.accuracy, m.precision, m.recall, m.f1_score,
//         m.true_positives, m.false_positives, m.true_negatives, m.false_negatives

// Confusion matrix
let cm = confusion_matrix_binary(&preds, &targets, 0.5);
println!("{}", format_confusion_matrix(&cm, Some(&["Neg", "Pos"])));

// ROC curve → (fpr, tpr, thresholds)
let (fpr, tpr, thresholds) = roc_curve(&preds, &targets, 100);

// AUC-ROC
let auc: Float = auc_roc(&preds, &targets);  // 1.0 = perfect, 0.5 = random
```

### Multi-class

```rust
use cma_neural_network::metrics::{confusion_matrix_multiclass, format_confusion_matrix};

let cm = confusion_matrix_multiclass(&preds, &targets, 10);
println!("{}", format_confusion_matrix(&cm, Some(&["0","1","2","3","4","5","6","7","8","9"])));
```

---

## 6. Serialization

```rust
use cma_neural_network::io;

// Save
io::save_json(&net, "checkpoints/model.json")?;    // human-readable
io::save_binary(&net, "checkpoints/model.bin")?;   // compact (~2.5x smaller)

// Load
let loaded = io::load_json("checkpoints/model.json")?;
let loaded = io::load_binary("checkpoints/model.bin")?;

// Size report
let (json_bytes, bin_bytes) = io::get_serialized_size(&net);
println!("JSON: {} B  |  Binary: {} B  |  ratio: {:.2}x",
    json_bytes, bin_bytes, json_bytes as f64 / bin_bytes as f64);
```

`io::Result<T>` wraps `IoError` which implements `Display`. Use `?` for propagation or `match` for recovery.

`ModelCheckpoint` callback handles saving automatically during training and infers the format from the file extension.

---

## 7. Network Introspection

```rust
println!("{}", net.architecture_string());  // "2 → [64, 32] → 1"
net.input_size()   // usize
net.output_size()  // usize
net.num_layers()   // counting hidden + output layers
net.get_layers_info()  // Vec<(&Array2<Float>, &Array1<Float>, &str)> — (W, b, activation_name)
```

---

## 8. Reproducibility

```rust
net.set_seed(42);   // seeds weight init + dropout masks
net.clear_seed();   // revert to non-deterministic
net.seed()          // Option<u64>
```

Seeds affect both weight initialisation (at build time, if set before `.build()` is called — pass to builder) and dropout mask sampling during training.

---

## Full Working Example

```rust
use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::callbacks::{
    DeltaMode, EarlyStopping, LRSchedule, LearningRateScheduler, ProgressBar,
};
use cma_neural_network::dataset::Dataset;
use cma_neural_network::io;
use cma_neural_network::metrics::{accuracy, binary_metrics, auc_roc};
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use ndarray::array;

fn main() {
    // Build dataset (XOR, repeated for a realistic sample count)
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for _ in 0..200 {
        inputs.extend_from_slice(&[
            array![0.0f32, 0.0], array![0.0, 1.0],
            array![1.0, 0.0],    array![1.0, 1.0],
        ]);
        targets.extend_from_slice(&[
            array![0.0f32], array![1.0], array![1.0], array![0.0],
        ]);
    }
    let mut dataset = Dataset::new(inputs, targets);
    dataset.shuffle();
    let (mut train, val) = dataset.split(0.8);

    // Build model
    let mut net = NetworkBuilder::new(2, 1)
        .hidden_layer(16, Activation::ReLU)
        .hidden_layer(8, Activation::ReLU)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adamw(0.01, 0.001))
        .dropout(0.1)
        .build();
    net.set_seed(42);

    // Train
    let epochs = 500;
    let history = net.trainer()
        .train_data(&mut train)
        .validation_data(&val)
        .epochs(epochs)
        .batch_size(32)
        .callback(Box::new(EarlyStopping::new(30, 0.001).mode(DeltaMode::Relative)))
        .callback(Box::new(ProgressBar::new(epochs)))
        .scheduler(LearningRateScheduler::new(
            LRSchedule::ReduceOnPlateau { patience: 15, factor: 0.5, min_delta: 1e-4 }
        ))
        .fit();

    let (last_train, last_val) = history.last().unwrap();
    println!("Epochs run: {} | Train loss: {:.4} | Val loss: {:.4}",
        history.len(), last_train, last_val.unwrap_or(0.0));

    // Evaluate
    let test_inputs  = vec![array![0.0f32, 0.0], array![0.0, 1.0],
                             array![1.0, 0.0],    array![1.0, 1.0]];
    let test_targets = vec![array![0.0f32], array![1.0], array![1.0], array![0.0]];
    let preds: Vec<_> = test_inputs.iter().map(|x| net.predict(x)).collect();

    let acc = accuracy(&preds, &test_targets, 0.5);
    let m   = binary_metrics(&preds, &test_targets, 0.5);
    let auc = auc_roc(&preds, &test_targets);
    println!("Accuracy: {:.1}%  |  F1: {:.3}  |  AUC: {:.3}", acc * 100.0, m.f1_score, auc);

    // Save
    io::save_json(&net, "examples/data/xor_model.json").expect("save failed");
    let (j, b) = io::get_serialized_size(&net);
    println!("Saved: JSON {} B  |  binary {} B", j, b);

    // Load and verify
    let loaded = io::load_json("examples/data/xor_model.json").expect("load failed");
    let loaded_preds: Vec<_> = test_inputs.iter().map(|x| loaded.predict(x)).collect();
    let loaded_acc = accuracy(&loaded_preds, &test_targets, 0.5);
    println!("Loaded accuracy: {:.1}%", loaded_acc * 100.0);
}
```
