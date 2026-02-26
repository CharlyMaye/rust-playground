# Guide 05 — Pre-built Architectures: `cma-models`

> **Navigation** [← 04 Autograd Training](04-autograd-training.md) | [README](README.md)
>
> **Theory**: [18 — LeNet-5](../18-lenet.md) | [19 — ResNet](../19-resnet.md) | [20 — VGG](../20-vgg.md) | [21 — AlexNet](../21-alexnet.md) | [22 — EfficientNet](../22-efficientnet.md)
>
> **Runnable examples** (run from workspace root):
> ```
> cargo run --manifest-path cma-models/Cargo.toml --example lenet5_paper
> cargo run --manifest-path cma-models/Cargo.toml --example resnet_paper
> cargo run --manifest-path cma-models/Cargo.toml --example vgg_paper
> cargo run --manifest-path cma-models/Cargo.toml --example alexnet_paper
> cargo run --manifest-path cma-models/Cargo.toml --example efficientnet_paper
> ```

---

## `Cargo.toml`

```toml
[dependencies]
cma-models = { path = "../cma-models" }
cma-cnn    = { path = "../cma-cnn" }   # for Tensor4D / TensorShape
```

---

## Important: All `forward()` Calls Return Features, Not Logits

Every architecture in `cma-models` outputs a **flat feature vector** (as a `Tensor4D`), not class-probability logits. You must wire the output to a separate classifier head.

```
cma-models model           →  Tensor4D features   →  FC head  →  logits / probabilities
(cma-cnn::Sequential)         [N, features, 1, 1]    (Network)
```

The FC head is typically built with `cma-neural-network::NetworkBuilder`.

---

## 1. LeNet-5

**Paper**: LeCun et al. (1998) — [Theory](../18-lenet.md)  
**Parameters**: ~33 K (MNIST config) to ~44 K (original config)  
**Input**: grayscale images (1 channel), 28×28 (mnist) or 32×32 (original)

```rust
use cma_models::lenet::{LeNet5, LeNet5Config};
use cma_cnn::tensor::{Tensor4D, TensorShape};

// Preset configs
let config = LeNet5Config::mnist();     // 28×28, Tanh, no BN, 10 classes
let config = LeNet5Config::original();  // 32×32, Tanh, no BN — faithful to paper
let config = LeNet5Config::modern();    // 28×28, ReLU + BatchNorm

// Build
let lenet = LeNet5::new(10);                    // uses mnist() config
let lenet = LeNet5::with_config(LeNet5Config::modern());

lenet.summary();
println!("Parameters: {}", lenet.num_parameters());

// Forward pass (outputs 120-dimensional feature vector)
lenet.conv_layers.eval_mode();                              // always before inference
let input  = Tensor4D::random(TensorShape::new(1, 1, 28, 28));
let output = lenet.forward(&input);                         // [1, 120, 1, 1]
println!("Output shape: {}", output.shape());

// Wire to FC head:
// LeNet → [N, 120] → FC(84) → FC(10) → Softmax
```

### Full LeNet-5 inference pipeline

```rust
use cma_models::lenet::{LeNet5, LeNet5Config};
use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use cma_cnn::tensor::{Tensor4D, TensorShape};
use ndarray::Array1;

let mut lenet = LeNet5::new(10);
lenet.conv_layers.eval_mode();

// FC classifier: 120 features → 84 → 10 classes
let fc_head = NetworkBuilder::new(120, 10)
    .hidden_layer(84, Activation::Tanh)
    .output_activation(Activation::Softmax)
    .loss(LossFunction::CategoricalCrossEntropy)
    .optimizer(OptimizerType::adam(0.001))
    .build();

// Inference
let img     = Tensor4D::random(TensorShape::new(1, 1, 28, 28));
let feat    = lenet.forward(&img);                        // [1, 120, 1, 1]
let flat: Array1<Float> = {
    let arr = feat.data();
    arr.clone().into_shape_with_order(120).unwrap().into_dimensionality().unwrap()
};
let logits  = fc_head.predict(&flat);                     // [10] class probabilities
```

---

## 2. ResNet (Residual Network)

**Paper**: He et al. (2016) — [Theory](../19-resnet.md)  
**Parameters**: ~87 K (MNIST preset) to ~11 M (ImageNet)

```rust
use cma_models::resnet::{ResNet, ResNetBuilder};

// Preset builders
let builder = ResNetBuilder::mnist();     // 1ch, 28×28, [16,32,64] channels, [2,2,2] blocks
let builder = ResNetBuilder::cifar();     // 3ch, 32×32, [16,32,64] channels, [2,2,2] blocks
let builder = ResNetBuilder::imagenet();  // 3ch, 224×224, [64,128,256,512], [2,2,2,2]

// Custom configuration
let builder = ResNetBuilder::new()
    .input_channels(1)
    .input_size(28)
    .channels(&[16, 32, 64])      // channels per stage
    .blocks(&[2, 2, 2])           // BasicBlocks per stage
    .stem_channels(16)            // stem conv output channels
    .stem_pooling(false);         // disable stem max-pool for small inputs (MNIST/CIFAR)

let mut resnet = builder.build();

resnet.summary();
println!("Parameters:       {}", resnet.num_parameters());
println!("Output features:  {}", resnet.output_features()); // for sizing the FC head

// Forward pass
resnet.eval_mode();
let input = Tensor4D::random(TensorShape::new(1, 1, 28, 28));
let feat  = resnet.forward(&input);   // [1, output_features, 1, 1] — GlobalAvgPool applied
println!("Feature shape: {}", feat.shape());

// FC head: output_features → num_classes
let fc = NetworkBuilder::new(resnet.output_features(), 10)
    .hidden_layer(256, Activation::ReLU)
    .output_activation(Activation::Softmax)
    .loss(LossFunction::CategoricalCrossEntropy)
    .optimizer(OptimizerType::adam(0.001))
    .build();
```

### `ResidualBlock` directly

```rust
use cma_models::resnet::ResidualBlock;

// new(in_channels, out_channels, stride)
//   stride=1 → same spatial size (no downsampling)
//   stride=2 → halve H and W, projection shortcut added automatically
let block = ResidualBlock::new(64, 128, 2);  // downsample block
println!("Block parameters: {}", block.num_parameters());
```

---

## 3. VGG

**Paper**: Simonyan & Zisserman (2015) — [Theory](../20-vgg.md)  
**Parameters**: VGG-16 ≈ 138 M | VGG-19 ≈ 144 M | CIFAR-10 variant ≪ 10 M

```rust
use cma_models::vgg::{VGG16, VGG19, VGGConfig};

// Preset configs
let config = VGGConfig::vgg16();    // 224×224, 3ch, 13 conv + 3 FC
let config = VGGConfig::vgg19();    // 224×224, 3ch, 16 conv + 3 FC
let config = VGGConfig::vgg11();    // 224×224, 3ch, 8 conv
let config = VGGConfig::cifar10();  // 32×32, 3ch, 8 conv — viable on CPU

// Build with default config
let mut vgg16 = VGG16::new(1000);
let mut vgg19 = VGG19::new(1000);

// Build with custom config
let small = VGG16::with_config(VGGConfig::cifar10().clone());

vgg16.summary();
println!("VGG-16 parameters: {}", vgg16.num_parameters());   // ~138 M

// Forward pass (features, NOT logits — needs FC head)
vgg16.features.eval_mode();
let input = Tensor4D::random(TensorShape::new(1, 3, 224, 224));
let feat  = vgg16.forward(&input);
println!("Feature shape: {}", feat.shape());   // [1, flat_features, 1, 1]
```

> **Memory note**: Running VGG-16 on a full 224×224 image requires ~500 MB for the forward pass on CPU. For experiments, use `VGGConfig::cifar10()` with 32×32 images.

---

## 4. AlexNet

**Paper**: Krizhevsky et al. (2012) — [Theory](../21-alexnet.md)  
**Parameters**: ~62 M (ImageNet) | ~6 M (CIFAR-10) | ~1 M (mini / small inputs)

```rust
use cma_models::alexnet::{AlexNet, AlexNetConfig};

// Preset configs
let config = AlexNetConfig::imagenet();           // 227×227, 3ch, BN, dropout=0.5
let config = AlexNetConfig::cifar10();            // 32×32, 3ch
let config = AlexNetConfig::small(10); // 64×64 — intermediate

// Build
let mut alex = AlexNet::new(1000);
let mut alex = AlexNet::with_config(AlexNetConfig::cifar10());

alex.summary();
println!("AlexNet parameters: {}", alex.num_parameters());

// Forward pass
alex.features.eval_mode();
let input   = Tensor4D::random(TensorShape::new(1, 3, 32, 32));  // CIFAR-10 size
let features = alex.forward(&input);
println!("Feature shape: {}", features.shape());
```

Build variant is selected automatically based on `config.input_size`:
- ≥ 200 pixels → full architecture (5 conv stages)
- ≥ 64 pixels → medium architecture
- < 64 pixels → mini architecture

---

## 5. EfficientNet-B0

**Paper**: Tan & Le (2019) — [Theory](../22-efficientnet.md)  
**Parameters**: ~5.3 M (B0) — Pareto-optimal on accuracy / FLOPs

```rust
use cma_models::efficientnet::{EfficientNetB0, EfficientNetConfig, MBConvBlock};

// Preset configs
let config = EfficientNetConfig::b0();      // 224×224, width=1.0, depth=1.0
let config = EfficientNetConfig::b1();      // 240×240, depth=1.1
let config = EfficientNetConfig::b2();      // 260×260, width=1.1, depth=1.2
let config = EfficientNetConfig::cifar10(); // 32×32, width=0.5, depth=0.5

// Build
let mut eff = EfficientNetB0::new(10);
let mut eff = EfficientNetB0::with_config(EfficientNetConfig::cifar10());

eff.summary();
println!("EfficientNet-B0 parameters: {}", eff.num_parameters());   // ~5.3 M for full B0

// Forward pass
let input = Tensor4D::random(TensorShape::new(1, 3, 32, 32));
let feat  = eff.forward(&input);
println!("Feature shape: {}", feat.shape());
```

### Custom `MBConvBlock`

```rust
// Build a single MBConv6 block (expand_ratio=6, SE enabled)
let block = MBConvBlock::builder(16, 24)  // (in_channels, out_channels)
    .expand_ratio(6)
    .kernel_size(3)
    .stride(2)                    // stride=2 → no skip connection
    .with_squeeze_excitation()    // enable Squeeze-and-Excitation
    .build();

println!("Block parameters: {}", block.num_parameters());
```

---

## 6. Custom `VGGConfig`

The `blocks` field in `VGGConfig` controls the number of conv layers and channels per stage:

```rust
let config = VGGConfig {
    num_classes: 10,
    input_size: 32,
    in_channels: 3,
    use_batch_norm: true,
    blocks: vec![
        (2, 32),   // block 1: 2 conv layers, 32 channels, then MaxPool
        (2, 64),   // block 2
        (3, 128),  // block 3
    ],
};
let model = VGG16::with_config(config);
```

---

## 7. Parameter and Shape Summary

| Architecture | Config | Parameters | Input | Feature dim |
|---|---|---|---|---|
| LeNet-5 | `mnist()` | ~33 K | 28×28×1 | 120 |
| LeNet-5 | `original()` | ~44 K | 32×32×1 | 120 |
| ResNet | `mnist()` | ~87 K | 28×28×1 | 64 |
| ResNet | `cifar()` | ~270 K | 32×32×3 | 64 |
| ResNet | `imagenet()` | ~11 M | 224×224×3 | 512 |
| VGG-11 | `vgg11()` | ~133 M | 224×224×3 | (see notes) |
| VGG-16 | `vgg16()` | ~138 M | 224×224×3 | (see notes) |
| VGG-16 | `cifar10()` | ≪ 10 M | 32×32×3 | (see notes) |
| AlexNet | `cifar10()` | ~6 M | 32×32×3 | (post-flatten) |
| AlexNet | `imagenet()` | ~62 M | 227×227×3 | (post-flatten) |
| EfficientNet-B0 | `b0()` | ~5.3 M | 224×224×3 | 1280 |
| EfficientNet-B0 | `cifar10()` | < 1 M | 32×32×3 | (scaled) |

Use `model.output_features()` (ResNet) or inspect `model.forward(&dummy).shape()` to determine the feature dimension before building the FC head.
