# cma-neural-network

**Bibliothèque de fondations pour réseaux de neurones en Rust**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

⚠️ **Projet en développement actif** - Non publié sur crates.io

---

## 📦 Écosystème cma-neural

Cette bibliothèque est la **fondation** d'un écosystème modulaire pour le Deep Learning en Rust :

```
┌─────────────────────────────────────┐
│  cma-models                         │  Architectures prêtes à l'emploi
│  LeNet-5, ResNet, EfficientNet     │  (Papers 1998-2019)
└──────────────┬──────────────────────┘
               │ dépend de
┌──────────────▼──────────────────────┐
│  cma-cnn                            │  Couches convolutionnelles
│  Conv2D, MaxPool, BatchNorm         │  (Images, Computer Vision)
└──────────────┬──────────────────────┘
               │ dépend de
┌──────────────▼──────────────────────┐
│  cma-neural-network  ← VOUS ÊTES ICI│  Fondations neuronales
│  Dense, Activations, Optimiseurs    │  (Données tabulaires, MNIST)
└─────────────────────────────────────┘
```

| Crate | Statut | Description |
|-------|--------|-------------|
| **cma-neural-network** | ✅ **Actif** | Couches Dense, 15+ activations, 5 optimiseurs, callbacks |
| **cma-cnn** | 🚧 Planned | Couches convolutionnelles (Conv2D, MaxPool2D, BatchNorm2D) |
| **cma-models** | 🚧 Planned | Architectures historiques (LeNet, AlexNet, VGG, ResNet, EfficientNet) |

> 💡 **Vous cherchez CNN pour images ?** Cette bibliothèque contient uniquement les réseaux **Fully-Connected** (Dense). Pour les convolutions, voir la roadmap dans [`../ANALYSE_ARCHITECTURE_IMAGE.md`](../ANALYSE_ARCHITECTURE_IMAGE.md).

---

## 🎯 Scope et Limitations

### ✅ Ce que fait cette bibliothèque

- **Réseaux Fully-Connected (Dense)** avec backpropagation
- **15+ fonctions d'activation** (Sigmoid, Tanh, ReLU, LeakyReLU, GELU, Mish, Swish, etc.)
- **5 optimiseurs modernes** (SGD, Momentum, RMSprop, Adam, AdamW)
- **Régularisation** (L1, L2, Dropout, Elastic Net)
- **Callbacks avancés** (EarlyStopping, ModelCheckpoint, LRScheduler, ProgressBar)
- **Mini-batch training** avec Dataset API
- **Métriques complètes** (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC)
- **Sérialisation** (JSON, binaire)
- **WebAssembly compatible** (MNIST dans navigateur)

### ❌ Ce qu'elle ne fait PAS (encore)

- Couches convolutionnelles (CNN)
- Images haute résolution (> 32×32)
- Détection d'objets, segmentation
- Support GPU
- Batch Normalization
- Connexions résiduelles (ResNet-style)

**Pour les CNN**, voir la roadmap complète dans [`ANALYSE_ARCHITECTURE_IMAGE.md`](../ANALYSE_ARCHITECTURE_IMAGE.md) (850 lignes, 42 références académiques).

### 📊 Cas d'Usage

| ✅ Recommandé | ❌ Non Adapté |
|--------------|--------------|
| MNIST (28×28 grayscale) | ImageNet (224×224 RGB) |
| Iris dataset (classification florale) | Détection d'objets (YOLO, Faster R-CNN) |
| Classification binaire/multi-classe | Segmentation sémantique (U-Net) |
| Données tabulaires | Traitement vidéo |
| XOR, spirales (démonstration) | Images médicales haute résolution |
| Prototypage rapide | Production computer vision |

---

## 🚀 Quick Start

### Installation

```toml
[dependencies]
cma-neural-network = { path = "../cma-neural-network" }
ndarray = "0.15"
```

### Exemple Minimal (XOR)

```rust
use cma_neural_network::builder::NetworkBuilder;
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use ndarray::array;

fn main() {
    // 1. Construire le réseau
    let mut network = NetworkBuilder::new(2, 1)           // 2 inputs, 1 output
        .hidden_layer(8, Activation::Tanh)                // Couche cachée
        .output_activation(Activation::Sigmoid)           // Sortie [0, 1]
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.001))
        .build();

    // 2. Données d'entraînement (XOR)
    let inputs = vec![
        array![0.0, 0.0], array![0.0, 1.0],
        array![1.0, 0.0], array![1.0, 1.0],
    ];
    let targets = vec![
        array![0.0], array![1.0],
        array![1.0], array![0.0],
    ];

    // 3. Entraîner
    for _ in 0..10000 {
        for (input, target) in inputs.iter().zip(&targets) {
            network.train(input, target);
        }
    }

    // 4. Prédire
    for (input, target) in inputs.iter().zip(&targets) {
        let pred = network.predict(input);
        println!("Input: {:?} → Pred: {:.3} (Target: {})", 
                 input, pred[0], target[0]);
    }
}
```

**Sortie attendue:**
```
Input: [0.0, 0.0] → Pred: 0.021 (Target: 0)
Input: [0.0, 1.0] → Pred: 0.978 (Target: 1)
Input: [1.0, 0.0] → Pred: 0.981 (Target: 1)
Input: [1.0, 1.0] → Pred: 0.019 (Target: 0)
```

### Exemple MNIST

Voir [`../neural-wasm/mnist/src/train_mnist.rs`](../neural-wasm/mnist/src/train_mnist.rs) pour un exemple complet :
- Architecture : 784 → [128, 64] → 10
- Normalisation z-score
- Early stopping
- Accuracy : 95-98%
- Déploiement WebAssembly

---

## 📚 Documentation

| Document | Contenu | Lignes |
|----------|---------|--------|
| **[README_FULL.md](README_FULL.md)** | Guide complet utilisateur | 1326 |
| **[ANALYSE_ARCHITECTURE_IMAGE.md](../ANALYSE_ARCHITECTURE_IMAGE.md)** | Vision architecturale & roadmap CNN | 1100 |
| **[TODO.md](TODO.md)** | Améliorations prévues | 250 |
| **[METRICS_GUIDE.md](METRICS_GUIDE.md)** | Guide détaillé métriques | 551 |
| **[BUILDER_PATTERN.md](BUILDER_PATTERN.md)** | Design pattern expliqué | - |
| **[examples/](examples/)** | Code exécutable | 4 exemples |

### Navigation Rapide

**Je veux...**
- **Apprendre les bases** → [`README_FULL.md`](README_FULL.md) sections "What is a Neural Network"
- **API complète** → [`README_FULL.md`](README_FULL.md) (NetworkBuilder, Callbacks, Metrics)
- **Comprendre l'architecture** → [`ANALYSE_ARCHITECTURE_IMAGE.md`](../ANALYSE_ARCHITECTURE_IMAGE.md) section 1
- **Roadmap CNN** → [`ANALYSE_ARCHITECTURE_IMAGE.md`](../ANALYSE_ARCHITECTURE_IMAGE.md) sections 4-5
- **Code pratique** → [`examples/getting_started.rs`](examples/getting_started.rs)
- **MNIST complet** → [`../neural-wasm/mnist/`](../neural-wasm/mnist/)

---

## 🔧 Fonctionnalités Détaillées

### Optimiseurs

| Optimiseur | Learning Rate | Usage |
|------------|---------------|-------|
| **Adam** ⭐ | 0.001 | **Défaut recommandé** - Convergence rapide |
| **AdamW** | 0.001 | Meilleure généralisation, prévient overfitting |
| **RMSprop** | 0.001-0.01 | RNN, gradients instables |
| **Momentum** | 0.01-0.1 | Accélération convergence |
| **SGD** | 0.01-0.5 | Baseline, recherche académique |

### Activations

| Activation | Output | Usage |
|------------|--------|-------|
| **ReLU** | [0, ∞) | Couches cachées (défaut) |
| **GELU** | (-∞, ∞) | Transformers, NLP moderne |
| **Mish** | (-∞, ∞) | État de l'art 2019 |
| **Sigmoid** | (0, 1) | Classification binaire output |
| **Softmax** | (0, 1), Σ=1 | Classification multi-classe output |
| **Tanh** | (-1, 1) | RNN, réseaux peu profonds |

### Loss Functions

| Loss | Tâche |
|------|-------|
| **BinaryCrossEntropy** | Classification binaire (Sigmoid output) |
| **CategoricalCrossEntropy** | Multi-classe (Softmax output) |
| **MSE** | Régression |
| **MAE** | Régression robuste (outliers) |
| **Huber** | Régression (compromis MSE/MAE) |

### Callbacks

| Callback | Fonction |
|----------|----------|
| **EarlyStopping** | Arrêt automatique si overfitting |
| **ModelCheckpoint** | Sauvegarde meilleur modèle |
| **LearningRateScheduler** | Adaptation dynamique du LR |
| **ProgressBar** | Affichage progression temps réel |

---

## 🗺️ Roadmap

### ✅ Phase Actuelle : Fondations Stables (v0.3)

- [x] Dense layers avec backpropagation correcte
- [x] 15+ activations modernes (GELU, Mish, Swish)
- [x] 5 optimiseurs (Adam, AdamW, RMSprop, Momentum, SGD)
- [x] Régularisation (L1, L2, Dropout, Elastic Net)
- [x] Callbacks avancés (EarlyStopping, LRScheduler)
- [x] Mini-batch training avec Dataset API
- [x] Métriques complètes (Accuracy, F1, ROC-AUC)
- [x] Sérialisation JSON/binaire
- [x] WebAssembly compatible

### 🚧 Phase 1 : Fondations CNN (1-2 mois)

Création de **`cma-cnn`** :
- [ ] Tenseur 4D : `[batch, channels, height, width]`
- [ ] Conv2D avec im2col (forward + backward)
- [ ] MaxPool2D, AvgPool2D
- [ ] BatchNorm2D
- [ ] Trait `Layer` unifié
- [ ] `Sequential` container
- [ ] **Validation** : LeNet-5 sur MNIST atteint 99%+

### 🚧 Phase 2 : Architectures Historiques (2 mois)

Création de **`cma-models`** :
- [ ] **LeNet-5** (1998) - Yann LeCun
- [ ] **AlexNet** (2012) - Krizhevsky & Hinton
- [ ] **VGG-16** (2014) - Simonyan & Zisserman
- [ ] **ResNet-50** (2015) - Kaiming He (skip connections)
- [ ] **EfficientNet-B0** (2019) - Tan & Le (compound scaling)

### 🎯 Phase 3 : Production Ready (2-3 mois)

- [ ] Data augmentation (rotation, flip, crop, color jitter)
- [ ] Transfer learning (chargement PyTorch models)
- [ ] Support GPU (wgpu)
- [ ] ONNX export
- [ ] Quantization INT8
- [ ] Model Zoo pré-entraîné

**Détails complets** : [`ANALYSE_ARCHITECTURE_IMAGE.md`](../ANALYSE_ARCHITECTURE_IMAGE.md) section 5 (roadmap en 7 phases).

---

## 🤝 Contributing

Ce projet suit une roadmap structurée avec milestones clairement définis.

**Avant de contribuer** :
1. Lire [`ANALYSE_ARCHITECTURE_IMAGE.md`](../ANALYSE_ARCHITECTURE_IMAGE.md) section 0.11 (architecture modulaire)
2. Consulter [`TODO.md`](TODO.md) pour les tâches prioritaires
3. Vérifier les milestones en cours

**Contributions bienvenues sur** :
- ✅ **Optimisations performance** (voir `ANALYSE_PERFORMANCE_V2.md`)
- ✅ **Tests unitaires** (coverage >90%)
- ✅ **Documentation et exemples**
- ✅ **Benchmarks** (comparaisons avec PyTorch)
- ✅ **Bug fixes**

**Architecture à respecter** :
- `cma-neural-network` : fondations uniquement (pas de CNN ici)
- `cma-cnn` : toutes les couches spatiales
- `cma-models` : compositions d'architectures

---

## 🎓 Références Académiques

### Papers Implémentés

- **Rumelhart, Hinton & Williams (1986)** : "Learning representations by back-propagating errors" - Backpropagation
- **Kingma & Ba (2015)** : "Adam: A Method for Stochastic Optimization" - Adam optimizer
- **Loshchilov & Hutter (2019)** : "Decoupled Weight Decay Regularization" - AdamW
- **Srivastava et al. (2014)** : "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- **Glorot & Bengio (2010)** : "Understanding the difficulty of training deep feedforward neural networks" - Xavier init
- **He et al. (2015)** : "Delving Deep into Rectifiers" - He initialization

### Papers Roadmap (CNN)

- **LeCun et al. (1998)** : "Gradient-Based Learning Applied to Document Recognition" - LeNet-5, convolutions
- **Krizhevsky et al. (2012)** : "ImageNet Classification with Deep Convolutional Neural Networks" - AlexNet
- **Simonyan & Zisserman (2014)** : "Very Deep Convolutional Networks for Large-Scale Image Recognition" - VGG
- **He et al. (2015)** : "Deep Residual Learning for Image Recognition" - ResNet, skip connections
- **Ioffe & Szegedy (2015)** : "Batch Normalization: Accelerating Deep Network Training" - BatchNorm
- **Tan & Le (2019)** : "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

**Liste complète** : [`ANALYSE_ARCHITECTURE_IMAGE.md`](../ANALYSE_ARCHITECTURE_IMAGE.md) section 7 (42 références).

---

## 📖 Ressources Externes

### Cours Recommandés

- **Stanford CS231n** : Convolutional Neural Networks for Visual Recognition  
  http://cs231n.stanford.edu/
  
- **3Blue1Brown** : Neural Networks (vidéos pédagogiques)  
  https://www.youtube.com/watch?v=aircAruvnKk
  
- **Neural Networks and Deep Learning** (Michael Nielsen)  
  http://neuralnetworksanddeeplearning.com/

### Frameworks Rust

- **burn** : Framework complet, très prometteur  
  https://github.com/tracel-ai/burn
  
- **candle** : Par Hugging Face, léger et rapide  
  https://github.com/huggingface/candle
  
- **tch-rs** : Bindings PyTorch pour Rust  
  https://github.com/LaurentMazare/tch-rs

### Datasets

- **MNIST** : Handwritten digits (28×28, 10 classes, 70k images)  
  http://yann.lecun.com/exdb/mnist/
  
- **Fashion-MNIST** : Vêtements (drop-in replacement MNIST)  
  https://github.com/zalandoresearch/fashion-mnist
  
- **CIFAR-10** : Images couleur (32×32, 10 classes, 60k)  
  https://www.cs.toronto.edu/~kriz/cifar.html

---

## 📊 Benchmarks

### MNIST (28×28 grayscale, 10 classes)

| Architecture | Params | Accuracy | Type |
|--------------|--------|----------|------|
| **FC (actuel)** | 109k | 95-98% | cma-neural-network |
| LeNet-5 (cible) | 60k | 99.2% | cma-cnn (Phase 1) |
| VGG-Mini (cible) | 50k | 99.5%+ | cma-cnn (Phase 2) |
| ResNet-18 (cible) | 11M | 99.7%+ | cma-models (Phase 2) |

### Temps d'Entraînement (estimés)

| Modèle | CPU (single-thread) | GPU (future) |
|--------|---------------------|--------------|
| FC MNIST | ~5 min | ~30 sec |
| LeNet-5 | ~10 min | ~1 min |
| ResNet-18 (CIFAR-10) | ~3 heures | ~15 min |

---

## ⚖️ License

MIT License - Voir [LICENSE](LICENSE) pour détails.

---

## 🙏 Remerciements

- **Yann LeCun** : Convolutions, LeNet-5, pionnier du Deep Learning
- **Geoffrey Hinton** : Backpropagation, dropout, mentor de la génération
- **Yoshua Bengio** : Fondations théoriques
- **Diederik Kingma & Jimmy Ba** : Adam optimizer
- Communauté Rust pour ndarray, serde, rayon

---

**📞 Contact & Questions**

Pour questions sur l'architecture ou contributions :
- Lire d'abord [`ANALYSE_ARCHITECTURE_IMAGE.md`](../ANALYSE_ARCHITECTURE_IMAGE.md) (très détaillé)
- Consulter [`TODO.md`](TODO.md) pour tâches en cours
- Créer une issue GitHub avec contexte

**🚀 Happy Neural Networking!**
