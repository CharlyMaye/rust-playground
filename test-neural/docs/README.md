# Neural Network Documentation

This documentation covers the theoretical foundations of neural networks as implemented in this library (`cma-neural-network`, `cma-cnn`, `cma-autograd`, `cma-models`). It is written for two audiences simultaneously, each section is explicitly marked:

- **Level 1 — Concepts** : intuitive explanations, analogies, birds-eye view. No prerequisites beyond high-school algebra.
- **Level 2 — Mathematics** : formal definitions, derivations, proofs, and academic references. Assumes familiarity with linear algebra, multivariate calculus, and basic probability.

---

## Phase 1 — Dense Networks (`cma-neural-network`)

| Chapter | Topic |
|---------|-------|
| [00 — Introduction](00-introduction.md) | What is a neural network? Universal approximation. |
| [01 — Architecture](01-architecture.md) | Neurons, layers, the forward pass, `NetworkBuilder`. |
| [02 — Activation Functions](02-activations.md) | All 15 activations: intuition, formulas, derivatives. |
| [03 — Loss Functions](03-loss-functions.md) | MSE, MAE, BCE, CCE, Huber — why and when. |
| [04 — Backpropagation](04-backpropagation.md) | Chain rule, delta rule, gradient flow. |
| [05 — Weight Initialization](05-weight-init.md) | Uniform, Xavier, He, LeCun — stopping collapse. |
| [06 — Regularization](06-regularization.md) | L1, L2, ElasticNet — fighting overfitting. |
| [07 — Dropout](07-dropout.md) | Stochastic regularization and ensemble interpretation. |
| [08 — Optimizers](08-optimizers.md) | SGD, Momentum, RMSprop, Adam, AdamW. |
| [09 — Learning Rate Schedules](09-lr-schedules.md) | StepLR, Plateau, Cosine, Warmup, OneCycle. |
| [10 — Training Loop](10-training-loop.md) | Epochs, mini-batches, callbacks, early stopping. |
| [11 — Metrics & Evaluation](11-metrics.md) | Accuracy, F1, confusion matrix, ROC, AUC. |

---

## Phase 2 — Convolutional Networks (`cma-cnn`)

| Chapter | Topic |
|---------|-------|
| [12 — CNN Basics](12-cnn-basics.md) | Local connectivity, weight sharing, NCHW layout, receptive field. |
| [13 — Conv2D](13-conv2d.md) | 2D convolution, im2col+GEMM, padding, He initialization. |
| [14 — Pooling Layers](14-pooling.md) | MaxPool, AvgPool, GlobalAvgPool — spatial reduction. |
| [15 — Batch Normalization](15-batchnorm.md) | Internal covariate shift, γ/β, running stats, train/eval modes. |
| [16 — Depthwise Convolution](16-depthwise-conv.md) | Channel-wise filters, MobileNet factorization, parameter savings. |

## Phase 3 — Automatic Differentiation (`cma-autograd`)

| Chapter | Topic |
|---------|-------|
| [17 — Autograd Engine](17-autograd.md) | Dynamic computation graph, topological sort, GradFn design, all ops. |

## Phase 4 — Architectures (`cma-models`)

| Chapter | Topic |
|---------|-------|
| [18 — LeNet-5](18-lenet.md) | LeCun 1998 — first end-to-end deep learning success. |
| [19 — ResNet](19-resnet.md) | He 2015 — residual connections, solving vanishing gradients at depth. |
| [20 — VGG](20-vgg.md) | Simonyan 2014 — uniform 3×3 convolutions, depth vs receptive field. |
| [21 — AlexNet](21-alexnet.md) | Krizhevsky 2012 — deep learning on ImageNet, ReLU, Dropout. |
| [22 — EfficientNet](22-efficientnet.md) | Tan & Le 2019 — MBConv, Squeeze-and-Excitation, compound scaling. |

---

## Key Global Conventions in the Library

| Convention | Detail |
|------------|--------|
| `Float` | Type alias for `f32`. Single precision is the industry standard for deep learning (halves memory over `f64`, hardware-accelerated on GPUs/SIMD). |
| `Dim` | Type alias for `u32`, used for serialized tensor dimensions (portable across x86-64 and WASM32). |
| WASM compatibility | All features work in WebAssembly except `parallel` (which requires Rayon threads). |
| Serialization | Networks serialize to JSON (human-readable) or Bincode (compact binary). |

---

## Primary References

| Reference | Notes |
|-----------|-------|
| Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. [deeplearningbook.org](https://www.deeplearningbook.org) | The standard textbook. Chapters 6–8 cover feedforward networks. |
| Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. | Rigorous Bayesian treatment of neural networks. Chapters 4–5 are especially relevant. |
| Nielsen, M. (2015). *Neural Networks and Deep Learning*. [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com) | Free online book, very accessible Level 1 reading. |
| LeCun, Y., Bottou, L., Orr, G., Müller, K. (1998). *Efficient BackProp*. In Neural Networks: Tricks of the Trade. | Practical recommendations still relevant today. |
| Bengio, Y. (2012). *Practical Recommendations for Gradient-Based Training of Deep Networks*. arXiv:1206.5533. | Concise practical guide. |
