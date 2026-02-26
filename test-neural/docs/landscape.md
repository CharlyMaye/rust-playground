# Deep Learning Landscape

An exhaustive map of deep learning approaches — as a reference for what exists, what is implemented here, and where to go next.

For datasets to use with each family, see [datasets.md](datasets.md).

---

## Already Implemented

```
cma-models      LeNet · ResNet · VGG · AlexNet · EfficientNet
cma-autograd    dynamic graph, gradient engine
cma-cnn         Conv2D · Pooling · BatchNorm · DepthwiseConv
cma-neural-network   Dense · Activations · Optimizers
```

---

## Sequence & Temporal Models

| Family | Key Architectures | Use Cases |
|---|---|---|
| **RNN** | Vanilla RNN, LSTM, GRU, BiRNN | Time series, text, audio |
| **Temporal CNN** | WaveNet, TCN (Temporal Convolutional Network) | Audio generation, forecasting |
| **State Space Models** | S4, Mamba, Hyena | Long sequences (alternative to Transformers) |

---

## Attention & Transformers

| Family | Key Architectures | Use Cases |
|---|---|---|
| **Encoder-only** | BERT, RoBERTa, DistilBERT | Classification, NER, embeddings |
| **Decoder-only** | GPT-2/3/4, LLaMA, Mistral | Text generation, LLMs |
| **Encoder-Decoder** | T5, BART, original Transformer | Translation, summarization, seq2seq |
| **Vision Transformer** | ViT, DeiT, Swin Transformer | Image classification (patch-based) |
| **Multimodal** | CLIP, Flamingo, LLaVA | Image + text understanding |
| **Cross-attention variants** | Perceiver, Perceiver IO | Any modality, irregular inputs |

---

## Generative Models

| Family | Key Architectures | Use Cases |
|---|---|---|
| **Autoencoders** | AE, Sparse AE, Denoising AE | Compression, anomaly detection |
| **Variational** | VAE, β-VAE, VQ-VAE | Latent space generation |
| **Adversarial** | GAN, DCGAN, StyleGAN, CycleGAN | Image synthesis, domain transfer |
| **Normalizing Flows** | RealNVP, Glow, FFJORD | Exact likelihood, density estimation |
| **Diffusion** | DDPM, DDIM, Stable Diffusion, DiT | Image/audio/video generation |
| **Autoregressive** | PixelCNN, WaveNet, AR Transformers | Sequential generation |
| **Energy-Based** | EBM, Contrastive Divergence | Density modeling |

---

## Graph & Relational

| Family | Key Architectures | Use Cases |
|---|---|---|
| **GNN** | GCN, GraphSAGE, GAT | Node/graph classification |
| **Message-passing** | MPNN, DimeNet | Molecular property prediction |
| **Graph Transformers** | Graphormer, GPS | Molecules, knowledge graphs |

---

## Self-Supervised & Representation Learning

| Family | Key Architectures | Use Cases |
|---|---|---|
| **Contrastive** | SimCLR, MoCo, NNCLR | Pre-training without labels |
| **Self-distillation** | BYOL, DINO, DINOv2 | Vision pre-training |
| **Masked modeling** | MAE (images), BERT (text), AudioMAE | Reconstruction-based pre-training |

---

## Geometric & Physics-Informed

| Family | Key Architectures | Use Cases |
|---|---|---|
| **Equivariant** | E(n)-GNN, SE(3)-Transformer | 3D molecules, robotics |
| **Neural ODEs** | Neural ODE, Latent ODE | Continuous-time dynamics |
| **PINNs** | Physics-Informed Neural Networks | Scientific computing, PDEs |

---

## Reinforcement Learning

| Family | Key Architectures | Use Cases |
|---|---|---|
| **Value-based** | DQN, Double DQN, Dueling DQN | Discrete action spaces |
| **Policy gradient** | REINFORCE, PPO, A3C | Continuous control |
| **Actor-Critic** | SAC, TD3, DDPG | Robotics, games |
| **Model-based** | World Models, DreamerV3 | Planning, sample efficiency |
| **Transformer-based RL** | Decision Transformer, Gato | Offline RL, multi-task |

---

## Efficient & Compressed Architectures

| Family | Key Architectures | Notes |
|---|---|---|
| **Depthwise-separable** | MobileNet v1/v2/v3 | Mobile/edge inference |
| **Inverted residuals** | MobileNetV2, ShuffleNet | Low-parameter CNNs |
| **Neural Architecture Search** | NASNet, EfficientNet (automated) | AutoML approach |
| **Knowledge Distillation** | Teacher-student | Compress large → small |
| **Quantization** | INT8, QAT | Inference acceleration |
| **Pruning** | Magnitude, structured | Sparse networks |
| **LoRA / Adapters** | LoRA, QLoRA, Adapters | Fine-tuning large models cheaply |

---

## Emerging / Research Frontiers

| Name | What it is |
|---|---|
| **Mixture of Experts (MoE)** | Sparse activation — only K of N expert sub-networks run per token (GPT-4, Mixtral) |
| **Hyper-networks** | A network that generates weights for another network |
| **Neural Radiance Fields (NeRF)** | MLP that encodes a 3D scene for novel view synthesis |
| **Implicit Neural Representations** | SIREN, NeRF — signals encoded as weights of an MLP |
| **Kolmogorov-Arnold Networks (KAN)** | Learnable activation functions on edges, not nodes (2024) |
| **Liquid Neural Networks** | Time-continuous RNNs with adaptive dynamics |
| **Test-Time Compute / Reasoning** | Chain-of-thought, tree-of-thought, o1-style reasoning loops |

---

## Placement in This Ecosystem

```
Transformers / LLMs / Diffusion / GNNs / RL …  ← not yet
────────────────────────────────────────────────
cma-models      LeNet · ResNet · VGG · AlexNet · EfficientNet
cma-autograd    dynamic graph, gradient engine
cma-cnn         Conv2D · Pooling · BatchNorm · DepthwiseConv
cma-neural-network   Dense · Activations · Optimizers
```

### Natural progression

| Step | What to add | Why |
|---|---|---|
| 1 | **RNN / LSTM / GRU** | Adds a time dimension; no new tensor shape beyond 2D; natural after dense nets |
| 2 | **Self-attention / Transformer block** | Pure matrix operations; buildable directly on `cma-autograd` |
| 3 | **ViT** | Transformer applied to image patches; bridges CNN and Transformer worlds |
| 4 | **VAE** | First generative model; probabilistic latent space on top of dense layers |
| 5 | **Diffusion (DDPM)** | Iterative denoising; needs UNet (CNN + skip connections already in codebase) |
| 6 | **GNN** | Requires a new graph tensor primitive; most isolated from current work |
