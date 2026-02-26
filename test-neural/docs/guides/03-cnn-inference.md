# Guide 03 — CNN Inference with `cma-cnn`

> **Navigation** [← 02 Dense Network](02-dense-network.md) | [README](README.md) | [04 Autograd Training →](04-autograd-training.md)
>
> **Theory**: [12 — CNN Basics](../12-cnn-basics.md) | [13 — Conv2D](../13-conv2d.md) | [14 — Pooling](../14-pooling.md) | [15 — BatchNorm](../15-batchnorm.md) | [16 — Depthwise Conv](../16-depthwise-conv.md)
>
> **Runnable**: `cargo run --example cnn_inference`

`cma-cnn` provides a serializable, inference-only CNN pipeline built on immutable forward passes.  
**It has no backpropagation.** To train a CNN end-to-end, use `cma-autograd` (Guide 04).

---

## `Cargo.toml`

```toml
[dependencies]
cma-cnn = { path = "../cma-cnn" }
ndarray = "0.17"
```

`cma-cnn` re-exports `Float`, `Activation`, `LossFunction`, `Network`, `NetworkBuilder`, and `OptimizerType` from `cma-neural-network`, so you usually only need `cma-cnn` as a dependency.

---

## 1. Tensors and Shapes

`cma-cnn` uses **NCHW** layout: `[batch, channels, height, width]`.

```rust
use cma_cnn::tensor::{Tensor4D, TensorShape};

// Shape descriptor
let shape = TensorShape::new(1, 3, 32, 32);
println!("{}", shape);  // "[1, 3, 32, 32]"

// Shape arithmetic (for manual sanity checks)
let after_conv  = shape.after_conv(16, 3, 1, 1);  // (out_channels, kernel_size, stride, padding)
let after_pool  = after_conv.after_pool(2, 2);     // (pool_size, stride)
let after_gap   = after_pool.after_global_pool();  // → [batch, channels, 1, 1]

// Build tensors
let zeros   = Tensor4D::zeros(shape);
let ones    = Tensor4D::ones(shape);
let random  = Tensor4D::random(shape);                // uniform ∈ [−1, 1]
let from_arr = Tensor4D::from_array(array4d_data);    // from ndarray::Array4<Float>

// Access
let arr: &ndarray::Array4<Float> = tensor.data();
let shape: TensorShape = tensor.shape();
tensor.reset_to_zero();   // &mut, in-place
```

---

## 2. Individual Layers

### `Conv2D`

```rust
use cma_cnn::Conv2D;

// new(in_channels, out_channels, kernel_size, stride, padding)
let conv = Conv2D::new(3, 64, 3, 1, 1);         // standard 3×3, same padding
let conv = Conv2D::same_padding(3, 64, 3);       // shortcut: stride=1, padding=kernel/2
let conv = Conv2D::new(3, 64, 3, 1, 0).without_bias(); // no bias (use before BatchNorm)
```

He initialisation is applied automatically.

### `DepthwiseConv2D`

```rust
use cma_cnn::DepthwiseConv2D;

// Per-channel convolution — does NOT mix channels (see Guide 16 — theory)
let dw = DepthwiseConv2D::new(32, 3, 1, 1);  // (channels, kernel_size, stride, padding)
let dw = DepthwiseConv2D::new(32, 3, 1, 1).without_bias();
```

### Pooling

```rust
use cma_cnn::{MaxPool2D, AvgPool2D, GlobalAvgPool2D};

let maxpool = MaxPool2D::new(2, 2);  // (pool_size, stride)
let avgpool = AvgPool2D::new(2, 2);
let gap     = GlobalAvgPool2D::new();   // reduces H×W → 1×1 per channel
```

### `BatchNorm2D`

```rust
use cma_cnn::BatchNorm2D;

let mut bn = BatchNorm2D::new(64);  // num_features
bn.train_mode();   // uses batch statistics, updates running mean/var
bn.eval_mode();    // uses frozen running statistics (for final inference)
```

**Always call `eval_mode()` before inference.** Running statistics are only reliable after sufficient training.

### Activation

```rust
use cma_cnn::ActivationLayer;

let relu    = ActivationLayer::relu();
let sigmoid = ActivationLayer::sigmoid();
let tanh    = ActivationLayer::tanh();
let custom  = ActivationLayer::new(Activation::Swish);
```

### `Dropout2D` and `Flatten`

```rust
use cma_cnn::{Dropout2D, Flatten};

let drop = Dropout2D::new(0.5);  // spatial dropout — drops entire channels; rate=0.5
let flat = Flatten::new();             // [N, C, H, W] → [N, C*H*W, 1, 1]
```

---

## 3. Building a `Sequential` Pipeline

`cma_cnn::Sequential` is the primary composition type. It supports a **fluent builder** interface:

```rust
use cma_cnn::{Sequential, Conv2D, BatchNorm2D, MaxPool2D, Flatten, ActivationLayer};
use cma_cnn::tensor::TensorShape;

let model = Sequential::named("LeNet-like")
    // Block 1
    .add_conv2d(Conv2D::new(1, 32, 3, 1, 1))   // 1→32 channels, 3×3 same
    .add_batchnorm(BatchNorm2D::new(32))
    .add_activation(ActivationLayer::relu())
    .add_maxpool(MaxPool2D::new(2, 2))            // ÷2 in H, W

    // Block 2
    .add_conv2d(Conv2D::new(32, 64, 3, 1, 1))
    .add_batchnorm(BatchNorm2D::new(64))
    .add_activation(ActivationLayer::relu())
    .add_maxpool(MaxPool2D::new(2, 2))

    // Flatten for FC head
    .add_flatten();

// Shortcut methods (conv + relu in one call):
let model = Sequential::new()
    .add_conv_relu(1, 32, 3, 1, 1)       // Conv → ReLU
    .add_conv_bn_relu(32, 64, 3, 1, 1)   // Conv → BN → ReLU
    .add_maxpool(MaxPool2D::new(2, 2))
    .add_global_avgpool()                // use instead of Flatten + FC when possible
    .add_flatten();
```

### Summary and parameter count

```rust
let input_shape = TensorShape::new(1, 1, 28, 28);  // one MNIST image
model.summary(input_shape);     // Keras-style table to stdout
println!("Parameters: {}", model.num_parameters());
println!("Output shape: {}", model.output_shape(input_shape));
```

---

## 4. Running a Forward Pass

```rust
let input = Tensor4D::random(TensorShape::new(1, 1, 28, 28));  // one random image
let output = model.forward(&input);                             // borrows, returns new Tensor4D
let output = model.forward_owned(input);                        // consumes, avoids a clone

println!("{}", output.shape());   // e.g. "[1, 64, 7, 7]" after two 2×2 max-pools on 28×28
println!("{:?}", output.data());
```

### Batch forward pass

```rust
// From a slice of Tensor4D
let outputs: Vec<Tensor4D> = model.forward_all(inputs.iter());

// With a callback per sample (useful for streaming / progress reporting)
model.forward_batches(inputs.iter(), |i, output| {
    println!("sample {}: shape {}", i, output.shape());
});
```

### Debug: intermediate feature maps

```rust
let steps = model.forward_with_intermediates(&input);
for (layer_name, config_str, feature_map) in &steps {
    println!("{} ({}): {}", layer_name, config_str, feature_map.shape());
}
```

---

## 5. Train / Eval modes

```rust
model.train_mode();   // BatchNorm2D updates running stats; Dropout2D is active
model.eval_mode();    // BatchNorm2D uses frozen stats; Dropout2D is a no-op

// Always switch to eval before inference:
model.eval_mode();
let output = model.forward(&input);
```

---

## 6. Serialization

`Sequential` implements `serde::Serialize` and `serde::Deserialize` via `serde_json` or `bincode`:

```rust
// JSON
let json = serde_json::to_string_pretty(&model)?;
let restored: Sequential = serde_json::from_str(&json)?;

// Binary
let bytes = bincode::serialize(&model)?;
let restored: Sequential = bincode::deserialize(&bytes)?;
```

---

## 7. Wiring CNN Features to a Dense Head

`cma-cnn` has no end-to-end training. The standard pattern: extract features with the CNN, then classify with a `cma-neural-network::Network`:

```rust
// 1. Forward CNN (eval mode — frozen BN)
model.eval_mode();
let feat_tensor = model.forward(&input);      // shape: [1, C, 1, 1] after GlobalAvgPool
let flat_data   = feat_tensor.data();         // &Array4<Float>

// 2. Flatten to 1-D for the dense head
use ndarray::Array1;
let flat: Array1<Float> = flat_data.clone().into_shape_with_order(flat_data.len()).unwrap().into_dimensionality().unwrap();

// 3. Run dense head (trained separately with cma-neural-network)
let logits = dense_head.predict(&flat);
```

For a **fully** end-to-end trainable pipeline, use `cma-autograd::CnnBuilder` (see Guide 04).

---

## Full Working Example (CNN feature extractor)

```rust
use cma_cnn::{Sequential, Conv2D, BatchNorm2D, MaxPool2D, GlobalAvgPool2D, Flatten, ActivationLayer};
use cma_cnn::tensor::{Tensor4D, TensorShape};

fn main() {
    // Build: small conv net for 28×28 grayscale images → 64-d feature vector
    let mut model = Sequential::named("SmallCNN")
        .add_conv_bn_relu(1, 32, 3, 1, 1)   // 28×28×32
        .add_maxpool(MaxPool2D::new(2, 2))   // 14×14×32
        .add_conv_bn_relu(32, 64, 3, 1, 1)  // 14×14×64
        .add_global_avgpool()                // 1×1×64
        .add_flatten();                      // 64

    let input_shape = TensorShape::new(1, 1, 28, 28);
    model.summary(input_shape);
    println!("Parameters: {}", model.num_parameters());

    // Run inference
    model.eval_mode();
    let img = Tensor4D::random(input_shape);
    let features = model.forward(&img);
    println!("Feature vector shape: {}", features.shape());  // [1, 64, 1, 1]
}
```

---

## Key Limitations of `cma-cnn`

| Limitation | Workaround |
|---|---|
| No backpropagation | Use `cma-autograd` for training |
| `BatchNorm2D` gradients not tracked | Use `cma-autograd::layers::BatchNorm2D` for training |
| No integrated classification head | Wire `Flatten` output into `cma-neural-network::Network` |
| GPU not supported | CPU only (f32 SIMD via ndarray) |
