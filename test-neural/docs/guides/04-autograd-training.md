# Guide 04 — CNN Training with `cma-autograd`

> **Navigation** [← 03 CNN Inference](03-cnn-inference.md) | [README](README.md) | [05 Models →](05-models.md)
>
> **Theory**: [17 — Autograd Engine](../17-autograd.md) | [04 — Backpropagation](../04-backpropagation.md)
>
> **Runnable**: `cargo run --example autograd_training`

`cma-autograd` provides a dynamic computation graph (define-by-run) for training CNNs end-to-end.  
It is the **only crate in this library that can compute gradients through convolutional layers**.

---

## `Cargo.toml`

```toml
[dependencies]
cma-autograd = { path = "../cma-autograd" }
ndarray = "0.17"
```

---

## 1. Tensors and Gradient Tracking

### Creating `Tensor`

```rust
use cma_autograd::prelude::*;   // Tensor, Parameter, no_grad, etc.
use cma_autograd::Float;

let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], false);
let w = Tensor::randn(&[3, 4], true);   // requires_grad=true — will accumulate gradients
let b = Tensor::zeros(&[4], true);

let y = Tensor::zeros(&[2, 3], false);
let s = Tensor::scalar(1.0_f32, false);

// Shape info
w.shape();   // Vec<usize>  ← [3, 4]
w.ndim();    // 2
w.numel();   // 12
```

All arithmetic operators (`+`, `-`, `*`, `/`) build the dynamic computation graph automatically when either operand has `requires_grad = true`.

### Forward pass and backward

```rust
let z = &x * &w;       // tracked operation
let loss = z.sum();    // scalar

loss.backward();       // reverse-mode autodiff: accumulates gradients in all leaf tensors

println!("{:?}", w.grad());   // Option<ArrayD<Float>>
```

`backward()` must be called on a **scalar** tensor. If your loss is not scalar, call `.sum()` or `.mean()` first.

### Gradient accumulation and zeroing

```rust
// Gradients ACCUMULATE across backward() calls — always zero before each step:
w.zero_grad();

// Or zero all parameters at once via the optimizer (preferred):
optimizer.zero_grad();
```

### Disabling gradient tracking

```rust
use cma_autograd::engine::no_grad;

// During inference — no computation graph, lower memory:
let output = no_grad(|| model.forward(&input));

// RAII guard style:
use cma_autograd::engine::NoGradGuard;
let _guard = NoGradGuard::new();   // grad disabled until _guard drops
let output = model.forward(&input);
```

---

## 2. Low-level Operations

```rust
let a = Tensor::randn(&[4], true);
let b = Tensor::randn(&[4], true);

// Elementwise arithmetic
let c = &a + &b;
let c = &a - &b;
let c = &a * &b;
let c = &a / &b;

// Reduction
let s = a.sum();     // scalar Tensor
let m = a.mean();    // scalar Tensor

// Elementwise functions
let r = a.relu();
let e = a.exp();
let l = a.log();    // natural log — ensure a > 0
let p = a.powf(2.0);

// Matrix ops (used internally by Linear)
// matmul via ndarray dot is used inside Linear's forward
```

---

## 3. Modules: `Linear` and `Conv2D`

### `Linear`

```rust
use cma_autograd::module::Linear;

let layer = Linear::new(in_features: 128, out_features: 64);   // Xavier init, bias = 0
let layer = Linear::without_bias(128, 64);

let output = layer.forward(&input);   // input: [batch, in_features]

// Retrieve parameters for optimizer
let params: Vec<&Parameter> = layer.parameters();
layer.zero_grad();
```

### `Conv2D` (autograd version)

```rust
use cma_autograd::module::Conv2D;

let conv = Conv2D::new(3, 32, 3, 1, 1);   // (in_channels, out_channels, kernel_size, stride, padding) — He init
let conv = Conv2D::without_bias(3, 32, 3, 1, 1);

let output = conv.forward(&input);   // input must be a 4-D tensor [N,C,H,W]
```

---

## 4. Building a Network with `CnnBuilder`

`CnnBuilder` is the recommended way to compose layers into a `Sequential` model.

```rust
use cma_autograd::builder::CnnBuilder;

let model = CnnBuilder::new()
    // Individual layers
    .conv2d(in_ch, out_ch, kernel, stride, padding)
    .conv2d_no_bias(in_ch, out_ch, kernel, stride, padding)
    .batch_norm(num_channels)
    .relu()
    .maxpool(kernel_size, stride)
    .avgpool(kernel_size, stride)
    .global_avg_pool()
    .flatten()
    .linear(in_features, out_features)
    .dropout(p)

    // Composite blocks (shortcuts)
    .conv_relu(in_ch, out_ch, kernel, stride, padding)        // Conv → ReLU
    .conv_bn_relu(in_ch, out_ch, kernel, stride, padding)     // Conv → BN → ReLU
    .build();
```

### Architecture presets

```rust
// LeNet-5 adapted for 28×28 input (MNIST-compatible)
let model = CnnBuilder::lenet5(10);

// AlexNet simplified for smaller inputs
let model = CnnBuilder::alexnet_mnist(10);

// VGG-style for smaller inputs
let model = CnnBuilder::vgg_mnist(10);

// Plain deep CNN (NOT a true ResNet — no skip connections)
let model = CnnBuilder::resnet_mnist(10);
```

> **Warning**: `CnnBuilder::resnet_mnist` creates a deep plain network **without residual skip connections**, despite its name. For a real ResNet with skip connections, use `cma-models::ResNetBuilder` (Guide 05).

### Model summary and parameters

```rust
model.summary();                             // prints layer table to stdout
println!("{}", model.num_parameters());      // total trainable parameters
let params: Vec<&Parameter> = model.parameters();
```

---

## 5. Optimizers

```rust
use cma_autograd::optim::{SGD, Adam};
use cma_autograd::module::Module;

// Collect all parameters
let params: Vec<Parameter> = model.parameters().into_iter().cloned().collect();

// SGD
let mut opt = SGD::new(params.clone(), lr: 0.01);
let mut opt = SGD::with_momentum(params.clone(), lr: 0.01, momentum: 0.9);

// Adam (most common choice)
let mut opt = Adam::new(params, lr: 0.001)
    .beta1(0.9)
    .beta2(0.999)
    .epsilon(1e-8)
    .weight_decay(0.01);   // → AdamW behaviour

// Training step
optimizer.zero_grad();    // clear accumulated gradients
loss.backward();          // compute gradients
optimizer.step();         // update parameters

// Dynamic LR adjustment
optimizer.set_lr(0.0001);
println!("LR: {}", optimizer.lr());
```

---

## 6. Loss Functions

```rust
use cma_autograd::loss::{mse_loss, cross_entropy_loss, binary_cross_entropy_loss};

// MSE — regression
let loss = mse_loss(&prediction, &target);   // both: [batch, out]

// Cross-entropy — multi-class
// prediction: raw logits [batch, classes] (do NOT apply softmax before)
// target: one-hot     [batch, classes]
let loss = cross_entropy_loss(&logits, &targets_one_hot);

// Binary cross-entropy — binary classification
// prediction: must be in (0, 1), i.e. after sigmoid
let loss = binary_cross_entropy_loss(&sigmoid_output, &target);
```

---

## 7. Training with `CnnTrainer`

`CnnTrainer` is a fluent training loop builder that mirrors `TrainingBuilder` from `cma-neural-network`:

```rust
let metrics = model
    .trainer(&mut optimizer)
    .train_data(&train_inputs, &train_targets)      // &[ArrayD<Float>]
    .validation_data(&val_inputs, &val_targets)     // optional
    .loss_fn(cross_entropy_loss)                    // default: cross_entropy_loss
    .epochs(20)                                     // default: 10
    .batch_size(32)                                 // default: 32
    .verbose(true)
    .early_stopping(patience: 5)                    // 0 = disabled (default)
    .fit();

// metrics: Vec<EpochMetrics>
for m in &metrics {
    println!("train_loss={:.4}  val_loss={:.4?}  train_acc={:.4?}",
        m.train_loss, m.val_loss, m.train_accuracy);
}
```

**Key difference from `cma-neural-network::TrainingBuilder`**: `CnnTrainer` takes `&[ArrayD<Float>]`, not a `Dataset`. Format your data as `ndarray::ArrayD<Float>` slices before calling `.train_data()`.

---

## 8. Manual Training Loop

For full control (custom LR scheduling, per-batch logging, gradient clipping):

```rust
use cma_autograd::engine::no_grad;
use cma_autograd::loss::cross_entropy_loss;

model.train();  // enable dropout and BN train mode

for epoch in 0..epochs {
    // Shuffle indices manually
    // ...

    for batch in batches {
        let input   = Tensor::from_vec(batch_data,    &[batch_size, c, h, w], false);
        let targets = Tensor::from_vec(batch_targets, &[batch_size, num_classes], false);

        // Forward
        let logits = model.forward(&input);
        let loss   = cross_entropy_loss(&logits, &targets);

        // Backward
        optimizer.zero_grad();
        loss.backward();

        // (Optional) gradient clipping
        // clip gradients manually by iterating parameters

        // Update
        optimizer.step();
    }

    // Validation
    model.eval();
    let val_loss = no_grad(|| {
        let logits = model.forward(&val_input);
        cross_entropy_loss(&logits, &val_targets).item()
    });
    println!("Epoch {} | val_loss: {:.4}", epoch, val_loss);
    model.train();
}
```

---

## 9. The `Sequential` API

```rust
// Composition
model.push(layer);           // takes ownership; mut ref required
let model = model.add(layer); // builder style, consumes self

// Training mode management
model.train();   // all layers go to train mode
model.eval();    // all layers go to eval mode

// Accessing layers
model.len();
model.layers();   // &[Box<dyn Layer>]
```

---

## Full Working Example

```rust
use cma_autograd::builder::CnnBuilder;
use cma_autograd::loss::cross_entropy_loss;
use cma_autograd::module::Module;
use cma_autograd::optim::Adam;
use cma_autograd::prelude::*;
use ndarray::{ArrayD, IxDyn};

fn main() {
    // Build LeNet-5 for 28×28 MNIST (10 classes)
    let mut model = CnnBuilder::lenet5(10);
    model.summary();

    // Synthetic dataset: 128 random "images" of shape [1, 28, 28]
    let n = 128;
    let mut rng_inputs  = Vec::new();
    let mut rng_targets = Vec::new();

    for i in 0..n {
        // Flat image: 1*28*28 = 784 values
        let data: Vec<Float> = (0..784).map(|_| rand::random::<Float>()).collect();
        rng_inputs.push(ArrayD::from_shape_vec(IxDyn(&[1, 28, 28]), data).unwrap());

        // One-hot target (class = i % 10)
        let mut one_hot = vec![0.0f32; 10];
        one_hot[i % 10] = 1.0;
        rng_targets.push(ArrayD::from_shape_vec(IxDyn(&[10]), one_hot).unwrap());
    }

    let split = (n as f32 * 0.8) as usize;
    let (train_x, val_x) = rng_inputs.split_at(split);
    let (train_y, val_y) = rng_targets.split_at(split);

    // Optimizer
    let params: Vec<Parameter> = model.parameters().into_iter().cloned().collect();
    let mut opt = Adam::new(params, 0.001);

    // Train via CnnTrainer
    let history = model
        .trainer(&mut opt)
        .train_data(train_x, train_y)
        .validation_data(val_x, val_y)
        .loss_fn(cross_entropy_loss)
        .epochs(5)
        .batch_size(16)
        .verbose(true)
        .fit();

    println!("Final train loss: {:.4}", history.last().unwrap().train_loss);
}
```

---

## Comparison: `cma-autograd::Sequential` vs `cma-cnn::Sequential`

| Aspect | `cma-autograd::Sequential` | `cma-cnn::Sequential` |
|--------|---------------------------|----------------------|
| Gradients | Yes — dynamic graph | No |
| Training | Yes — `CnnTrainer` | No |
| Serialisation | No (use `cma-cnn` or custom) | Yes — serde |
| `BatchNorm2D` | Gradient-tracked γ, β | BN forward only |
| Suitable for | End-to-end CNN training | Inference, serialised deployment |
| Layer input type | `Tensor` (autograd) | `Tensor4D` (raw) |
