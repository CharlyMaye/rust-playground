# test-neural — Deep Learning in Rust, from Scratch to the Browser

A **self-contained deep learning ecosystem** written in pure Rust, designed to be understood layer by layer — from the mathematics of a single neuron up to convolutional architectures running live in the browser via WebAssembly.

> This is a learning & exploration project. The goal is not to replace PyTorch. It is to *understand* what PyTorch does, by building it.

---

## Motivation

Most deep learning practitioners use frameworks as black boxes. This project takes the opposite approach: every piece — the forward pass, backpropagation, each optimizer update rule, the convolution kernel — is written from scratch in Rust, with accompanying documentation that explains **why** each design decision was made, at both an intuitive and a mathematical level.

Secondary goals:
- Demonstrate that Rust is a viable language for numerical computing and ML research code.
- Produce models that compile to **WebAssembly** and run in any modern browser with no server needed.
- Provide a readable, well-documented codebase that can serve as a reference for learning.

---

## Workspace Structure

This is a Cargo workspace. Each crate has a single, focused responsibility.

```
test-neural/
├── cma-neural-network/   # Dense layers, activations, optimizers, training loop
├── cma-cnn/              # Convolutional layers (Conv2D, MaxPool, BatchNorm)
├── cma-autograd/         # Automatic differentiation engine (dynamic graph)
├── cma-models/           # Ready-to-use architectures (LeNet, VGG, ResNet, …)
│
├── neural-wasm/          # WebAssembly bindings for each model
│   ├── shared/           #   Shared types exposed to JavaScript
│   ├── xor/              #   XOR binary classifier
│   ├── iris/             #   Iris multi-class classifier
│   ├── mnist/            #   MNIST digit recognition (Dense)
│   ├── mnist-lenet/      #   MNIST with LeNet-5
│   ├── mnist-alexnet/    #   MNIST with AlexNet
│   ├── mnist-vgg/        #   MNIST with VGG
│   └── mnist-resnet/     #   MNIST with ResNet
│
├── ai-web-app/           # Angular front-end that loads the WASM modules
├── www/                  # Static HTML demos
├── docs/                 # Theory documentation (22 chapters)
├── examples/             # Standalone Rust usage examples
└── src/                  # Workspace binary entry-point
```

---

## Crate Dependency Graph

```
cma-models
    └── depends on cma-cnn
            └── depends on cma-neural-network
                        ↑
              cma-autograd also depends on cma-neural-network & cma-cnn
```

| Crate | Status | Responsibility |
|---|---|---|
| `cma-neural-network` | ✅ Active | Dense layers, 15+ activations, 5 optimizers, regularization, metrics |
| `cma-cnn` | 🚧 In progress | Conv2D, MaxPool2D, BatchNorm2D, Depthwise Conv |
| `cma-autograd` | 🚧 In progress | Dynamic computation graph, automatic gradient computation |
| `cma-models` | 🚧 In progress | LeNet-5, AlexNet, VGG, ResNet, EfficientNet |

---

## What Each Crate Provides

### `cma-neural-network` — Foundations
The base layer. Every other crate builds on this.

- Fully-connected (Dense) layers of arbitrary depth and width
- 15 activation functions: ReLU, Sigmoid, Tanh, GELU, Swish, Mish, LeakyReLU, ELU, SELU, Softmax, …
- 5 loss functions: MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy, Huber
- 5 optimizers: SGD, Momentum, RMSprop, Adam, AdamW
- Weight initialization strategies: Uniform, Xavier/Glorot, He/Kaiming, LeCun
- Regularization: L1, L2, ElasticNet, Dropout
- Learning rate schedules: StepLR, ReduceOnPlateau, CosineAnnealing, Warmup, OneCycle
- Evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC
- Fluent builder API, callbacks, early stopping, model serialization (JSON)

### `cma-cnn` — Convolutional Layers
Adds spatial processing on top of the dense foundation.

- 2D convolution with im2col + GEMM implementation
- MaxPool2D, AvgPool2D, GlobalAvgPool2D
- Batch Normalization (train and eval modes, running statistics)
- Depthwise separable convolutions (MobileNet-style)

### `cma-autograd` — Automatic Differentiation
A dynamic computation graph engine similar to PyTorch's `autograd`.

- Every tensor operation records itself in a graph
- `backward()` traverses the graph in reverse topological order
- Gradients are accumulated on leaf tensors
- Enables arbitrary model architectures without manually deriving backprop

### `cma-models` — Classic Architectures
Pre-built, paper-faithful implementations of landmark CNN architectures.

| Architecture | Paper | Year |
|---|---|---|
| LeNet-5 | LeCun et al. | 1998 |
| AlexNet | Krizhevsky et al. | 2012 |
| VGG-16 | Simonyan & Zisserman | 2014 |
| ResNet-18/34 | He et al. | 2015 |
| EfficientNet | Tan & Le | 2019 |

---

## WebAssembly & Browser Demos

The `neural-wasm/` directory compiles each model to `.wasm` using `wasm-bindgen`. The built packages are served by the Angular app in `ai-web-app/` or the static pages in `www/`.

**Build all WASM modules:**
```bash
cd neural-wasm
./build_all.sh
```

**Run the static demos:**
```bash
cd www
python3 -m http.server 8080
# open http://localhost:8080
```

---

## Quickstart (Rust)

```rust
use cma_neural_network::NetworkBuilder;

let mut model = NetworkBuilder::new()
    .input(4)
    .dense(16, "relu")
    .dense(8, "relu")
    .dense(3, "softmax")
    .loss("categorical_crossentropy")
    .optimizer("adam", 0.001)
    .build();

model.fit(&x_train, &y_train, epochs: 50, batch_size: 32);
```

See the [`examples/`](examples/) directory for complete runnable programs:
- [getting_started.rs](examples/getting_started.rs) — minimal end-to-end example
- [minibatch_demo.rs](examples/minibatch_demo.rs) — mini-batch training
- [metrics_demo.rs](examples/metrics_demo.rs) — evaluation metrics
- [serialization.rs](examples/serialization.rs) — save and load a model
- [autograd_training.rs](examples/autograd_training.rs) — training via autograd
- [cnn_inference.rs](examples/cnn_inference.rs) — running a CNN model

---

## Documentation

The [`docs/`](docs/) directory contains 22 chapters of theory, each written at two levels simultaneously:

- **Level 1 — Concepts**: intuitive explanations, no prerequisites beyond high-school math.
- **Level 2 — Mathematics**: formal definitions, derivations, and academic references.

| Phase | Chapters | Topic |
|---|---|---|
| 1 — Dense Networks | 00–11 | Architecture, activations, loss, backprop, optimizers, training loop, metrics |
| 2 — CNNs | 12–16 | Convolution, pooling, batch norm, depthwise separable |
| 3 — Autograd | 17 | Dynamic graphs, topological sort, gradient accumulation |
| 4 — Architectures | 18–22 | LeNet, ResNet, VGG, AlexNet, EfficientNet |

Start with [docs/00-introduction.md](docs/00-introduction.md).

There are also focused implementation guides in [`docs/guides/`](docs/guides/):
- [01 — Quickstart](docs/guides/01-quickstart.md)
- [02 — Building a Dense Network](docs/guides/02-dense-network.md)
- [03 — CNN Inference](docs/guides/03-cnn-inference.md)
- [04 — Training with Autograd](docs/guides/04-autograd-training.md)
- [05 — Using Pre-built Models](docs/guides/05-models.md)

---

## Design Philosophy

**Correctness over performance.** This is not a production inference engine. When a choice must be made between a fast but opaque implementation and a slower but readable one, readability wins. The goal is understanding.

**No unsafe, no magic.** Almost no `unsafe` Rust. No C FFI to BLAS (at least for now). Everything is inspectable.

**Incremental complexity.** The crate dependency graph mirrors the learning path: you can understand `cma-neural-network` fully before needing to look at `cma-cnn`, and so on.

---

## Requirements

- Rust 1.70+ (`rustup update stable`)
- For WASM builds: `wasm-pack` (`cargo install wasm-pack`)
- For the Angular app: Node.js 18+ and `npm`

---

## License

MIT
