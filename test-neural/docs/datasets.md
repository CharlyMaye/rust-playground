# Datasets

Reference for finding training data, organised by what is already implemented and what comes next.

---

## General-Purpose Hubs

These platforms cover every domain below. Start here when looking for something specific.

| Platform | URL | Notes |
|---|---|---|
| **Hugging Face Datasets** | [huggingface.co/datasets](https://huggingface.co/datasets) | Largest searchable hub; one-line download API |
| **Kaggle** | [kaggle.com/datasets](https://www.kaggle.com/datasets) | Competitions + community datasets |
| **UCI ML Repository** | [archive.ics.uci.edu](https://archive.ics.uci.edu/) | Classic tabular/ML benchmarks |
| **OpenML** | [openml.org](https://www.openml.org/) | Versioned, API-accessible |
| **Papers With Code** | [paperswithcode.com/datasets](https://paperswithcode.com/datasets) | Datasets linked to their benchmark papers |
| **Google Dataset Search** | [datasetsearch.research.google.com](https://datasetsearch.research.google.com/) | Search engine over public datasets |
| **AWS Open Data** | [registry.opendata.aws](https://registry.opendata.aws/) | Large-scale datasets freely hosted on S3 |

---

## Already Implemented — Dense Networks (`cma-neural-network`)

These datasets work with the current codebase as-is. No new architecture needed.

### Tabular & Classification

| Dataset | URL | Task | Size |
|---|---|---|---|
| **XOR** ✅ *(in repo)* | Generated in code | Binary logic function | 4 samples |
| **Iris** ✅ *(in repo)* | [UCI](https://archive.ics.uci.edu/dataset/53/iris) | 3-class flower classification | 150 rows, 4 features |
| **Breast Cancer Wisconsin** | [UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | Binary classification | 569 rows, 30 features |
| **Wine Quality** | [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality) | Regression / classification | 6 500 rows, 11 features |
| **Adult Income** | [UCI](https://archive.ics.uci.edu/dataset/2/adult) | Binary income classification | 48 000 rows, 14 features |
| **Titanic** | [Kaggle](https://www.kaggle.com/c/titanic) | Binary survival prediction | 891 rows, mixed features |
| **California Housing** | [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) | Regression | 20 000 rows, 8 features |
| **MNIST** ✅ *(in repo)* | [Yann LeCun](http://yann.lecun.com/exdb/mnist/) | 10-class digit classification | 70 000 images, 28×28 |

---

## Already Implemented — CNNs (`cma-cnn` / `cma-models`)

Standard benchmarks for the architectures in the repo (LeNet, AlexNet, VGG, ResNet, EfficientNet).

| Dataset | URL | Task | Resolution | Notes |
|---|---|---|---|---|
| **MNIST** ✅ *(in repo)* | [Yann LeCun](http://yann.lecun.com/exdb/mnist/) | 10-class digits | 28×28 grayscale | LeNet-5 original target |
| **Fashion-MNIST** | [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) | 10-class clothing | 28×28 grayscale | Drop-in MNIST replacement, harder |
| **CIFAR-10 / CIFAR-100** | [Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) | 10 or 100 classes | 32×32 RGB | Standard CNN benchmark; fits on CPU |
| **SVHN** | [Stanford](http://ufldl.stanford.edu/housenumbers/) | Street digit recognition | 32×32 RGB | Harder than MNIST, real-world |
| **STL-10** | [Stanford](https://cs.stanford.edu/~acoates/stl10/) | 10-class | 96×96 RGB | Few labels + 100 k unlabeled images |
| **Tiny ImageNet** | [Stanford CS231n](http://cs231n.stanford.edu/tiny-imagenet-200.zip) | 200-class | 64×64 RGB | Subset of ImageNet; viable on CPU |
| **ImageNet (ILSVRC)** | [image-net.org](https://www.image-net.org/download.php) | 1 000-class | 224×224 RGB | AlexNet / VGG / ResNet target; registration required |
| **COCO** | [cocodataset.org](https://cocodataset.org/#download) | Detection, segmentation | Various | Large (20 GB+) |
| **Oxford Flowers 102** | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | 102-class fine-grained | 224×224 RGB | Transfer learning benchmark |
| **CUB-200 Birds** | [Caltech](https://www.vision.caltech.edu/datasets/cub_200_2011/) | 200-class fine-grained | Various | Fine-grained recognition |

---

## Next Step — Sequence & RNN / LSTM

| Dataset | URL | Task | Notes |
|---|---|---|---|
| **IMDb** | [Hugging Face](https://huggingface.co/datasets/imdb) | Sentiment classification | Good first text dataset |
| **SST-2** | [Hugging Face](https://huggingface.co/datasets/stanfordnlp/sst2) | Sentence sentiment | Short sentences, fast to train |
| **Penn Treebank** | [Hugging Face](https://huggingface.co/datasets/ptb-text-only) | Language modeling | Classic RNN benchmark |
| **WikiText-2 / WikiText-103** | [Hugging Face](https://huggingface.co/datasets/wikitext) | Language modeling | Cleaner than PTB; two sizes |
| **UCR Time Series Archive** | [timeseriesclassification.com](https://www.timeseriesclassification.com/) | 128 TS classification tasks | Best starting point for time series |
| **ETT (Electricity Transformer)** | [GitHub](https://github.com/zhouhaoyi/ETDataset) | Long-term forecasting | Standard forecasting benchmark |
| **M5 Forecasting** | [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy) | Retail sales forecasting | Hierarchical, real-world |
| **PhysioNet / MIMIC-III** | [physionet.org](https://physionet.org/content/mimiciii/1.4/) | Medical time series | Requires registration; clinical data |
| **LibriSpeech** | [openslr.org](https://www.openslr.org/12) | Speech recognition | 1 000 h of English audio |

---

## Next Step — Transformers & NLP

| Dataset | URL | Task | Notes |
|---|---|---|---|
| **GLUE / SuperGLUE** | [gluebenchmark.com](https://gluebenchmark.com/) | NLP benchmark suite | Classification, NLI, QA |
| **SQuAD 2.0** | [rajpurkar.github.io](https://rajpurkar.github.io/SQuAD-explorer/) | Reading comprehension / QA | Classic extractive QA |
| **WMT** | [statmt.org](https://statmt.org/wmt24/) | Machine translation | EN↔FR, EN↔DE, and more |
| **Common Crawl / C4** | [Hugging Face](https://huggingface.co/datasets/allenai/c4) | Pre-training corpus | Large; for training from scratch |
| **LAION-5B** | [laion.ai](https://laion.ai/blog/laion-5b/) | Image-text pairs | Multimodal / CLIP pre-training |

---

## Next Step — Generative Models (VAE, GAN, Diffusion)

| Dataset | URL | Task | Notes |
|---|---|---|---|
| **MNIST / Fashion-MNIST** | *(links above)* | Image generation | Easiest starting point for VAE |
| **CelebA** | [mmlab.ie.cuhk.edu.hk](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Face generation / editing | 200 k faces, 40 attributes |
| **FFHQ** | [GitHub / NVIDIA](https://github.com/NVlabs/ffhq-dataset) | High-quality face generation | 70 k faces at 1024×1024 (StyleGAN target) |
| **LSUN Bedrooms** | [GitHub](https://github.com/fyu/lsun) | Scene generation | DCGAN original benchmark |
| **LJSpeech** | [keithito.com](https://keithito.com/LJ-Speech-Dataset/) | Audio synthesis / TTS | 24 h of single-speaker English |
| **MVTec AD** | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Industrial anomaly detection | Autoencoder-based anomaly scoring |

---

## Next Step — Graph Neural Networks

| Dataset | URL | Task | Notes |
|---|---|---|---|
| **Cora / Citeseer / PubMed** | [GitHub (PyG)](https://github.com/kimiyoung/planetoid) | Citation graph node classification | Classic GNN benchmark; small and fast |
| **Open Graph Benchmark (OGB)** | [ogb.stanford.edu](https://ogb.stanford.edu/) | Nodes, links, graphs | Unified, leaderboard-tracked |
| **QM9** | [Hugging Face](https://huggingface.co/datasets/graphs-datasets/QM9) | Molecular property prediction | 130 k small molecules, 19 properties |
| **ZINC** | [Hugging Face](https://huggingface.co/datasets/graphs-datasets/ZINC) | Molecular graph regression | 250 k drug-like molecules |
| **Wikidata** | [wikidata.org](https://www.wikidata.org/wiki/Wikidata:Database_download) | Knowledge graph completion | Very large; use subsets |

---

## Next Step — Self-Supervised Learning

These reuse CNN datasets (no labels used during pre-training).

| Dataset | URL | Notes |
|---|---|---|
| **STL-10** | [Stanford](https://cs.stanford.edu/~acoates/stl10/) | Specifically designed for SSL — 100 k unlabeled images |
| **CIFAR-10 (unlabeled)** | [Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) | Good small-scale SSL target |
| **ImageNet (unlabeled)** | [image-net.org](https://www.image-net.org/download.php) | Standard MAE / DINO pre-training target |
| **Common Crawl** | [commoncrawl.org](https://commoncrawl.org/) | Massive unlabeled web text for BERT-style pre-training |

---

## Next Step — Reinforcement Learning

RL uses interactive environments rather than static datasets.

| Platform | URL | Notes |
|---|---|---|
| **Gymnasium (OpenAI Gym)** | [gymnasium.farama.org](https://gymnasium.farama.org/) | Standard RL API: CartPole, LunarLander, Atari, MuJoCo |
| **ALE (Atari)** | [GitHub](https://github.com/mgbellemare/Arcade-Learning-Environment) | 57 Atari 2600 games — DQN original benchmark |
| **MuJoCo** | [mujoco.org](https://mujoco.org/) | Continuous control physics simulator (free since 2022) |
| **D4RL** | [GitHub](https://github.com/Farama-Foundation/D4RL) | Offline RL: pre-recorded expert trajectories |
| **ProcGen** | [GitHub](https://github.com/openai/procgen) | 16 procedurally generated environments for generalisation |

---

## Next Step — Physics & Geometry

| Dataset | URL | Task |
|---|---|---|
| **QM9** | [Hugging Face](https://huggingface.co/datasets/graphs-datasets/QM9) | 3D molecular geometry + quantum properties |
| **MD17** | [sgdml.org](http://www.sgdml.org/#datasets) | Molecular dynamics trajectories |
| **Burgers / Navier-Stokes** | [GitHub (FNO)](https://github.com/neuraloperator/neuraloperator) | PDE benchmark for PINNs and neural operators |
| **ShapeNet** | [shapenet.org](https://shapenet.org/) | 3D object meshes for NeRF / equivariant nets |
| **NeRF synthetic** | [GitHub (NeRF)](https://github.com/bmild/nerf) | 3D scene reconstruction |
