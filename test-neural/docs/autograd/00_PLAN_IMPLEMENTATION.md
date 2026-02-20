# 🧠 Implémentation Autograd pour cma-neural-network

> **Objectif** : Ajouter le support des graphes dynamiques (autograd) pour permettre l'entraînement end-to-end des CNN.

---

## 📋 Table des Matières

1. [Contexte et Problème](#1-contexte-et-problème)
2. [Analyse de l'Existant](#2-analyse-de-lexistant)
3. [Approches Possibles](#3-approches-possibles)
4. [Architecture Cible](#4-architecture-cible)
5. [Spécifications Techniques](#5-spécifications-techniques)
6. [Plan d'Implémentation](#6-plan-dimplémentation)
7. [Formules Mathématiques](#7-formules-mathématiques)
8. [Tests et Validation](#8-tests-et-validation)
9. [Références](#9-références)

---

## 1. Contexte et Problème

### 1.1 Problème Identifié

Les logs d'entraînement révèlent que **les CNN ne convergent pas** :

```
AlexNet - Epoch 1:  loss 0.59 → Epoch 14: loss 2.05 (DIVERGENCE !)
LeNet   - Convergence anormalement lente
MNIST   - Overfitting sévère
```

**Cause racine** : Les poids des couches CNN ne sont **jamais mis à jour** !

```rust
// Code actuel (train_alexnet.rs)
let train_features = extract_cnn_features(&cnn, train.inputs());  // CNN utilisé en forward only
// Les poids CNN restent à leur valeur d'initialisation aléatoire !
```

### 1.2 Pourquoi le Backward Manque ?

| Composant | Forward | Backward | Statut |
|-----------|---------|----------|--------|
| `cma-neural-network` (Dense) | ✅ | ✅ | Complet |
| `cma-cnn` (Conv, Pool, BN) | ✅ | ❌ | **Forward seulement** |

Le trait `Layer` dans `cma-cnn` ne définit que `forward()` :

```rust
pub trait Layer: Send + Sync {
    fn forward(&self, input: &Tensor4D) -> Tensor4D;  // ✅ Existe
    // fn backward(...) -> ...;                       // ❌ MANQUE !
}
```

### 1.3 Besoin de Graphes Dynamiques

L'utilisateur a exprimé le besoin de supporter des architectures avec :
- Séquences de longueur variable (RNN, Transformers)
- Chemins conditionnels
- Boucles de taille variable

Cela nécessite un système **autograd** avec graphes dynamiques, similaire à PyTorch.

---

## 2. Analyse de l'Existant

### 2.1 Structure des Crates

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE ACTUELLE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  cma-models/          Architectures pré-définies               │
│  ├── lenet.rs         (LeNet-5)                                │
│  ├── alexnet.rs       (AlexNet-Mini)                           │
│  ├── vgg.rs           (VGG)                                    │
│  └── resnet.rs        (ResNet)                                 │
│       │                                                        │
│       ▼                                                        │
│  cma-cnn/             Couches CNN                              │
│  ├── layers.rs        Conv2D, MaxPool2D, BatchNorm2D, etc.    │
│  ├── ops.rs           im2col, col2im, convolutions            │
│  ├── sequential.rs    Container séquentiel                    │
│  └── tensor.rs        Tensor4D (NCHW)                         │
│       │                                                        │
│       ▼                                                        │
│  cma-neural-network/  Couches Dense + Training                 │
│  ├── network.rs       Network, Activation, LossFunction       │
│  ├── trainer.rs       Backpropagation (Dense only)            │
│  ├── optimizer.rs     Adam, SGD, RMSprop, etc.                │
│  └── builder.rs       API fluent                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 État Détaillé de cma-cnn

#### Tensor4D (`tensor.rs`)

```rust
pub struct Tensor4D {
    data: Array4<Float>,  // Données uniquement, pas de gradient
}
```

**Méthodes disponibles** :
- `zeros()`, `ones()`, `random()` - Création
- `shape()` - Dimensions
- `data()`, `data_mut()` - Accès
- `map()`, `flatten()` - Transformations

**Ce qui manque** :
- `grad: Option<Array4<Float>>` - Gradient accumulé
- `requires_grad: bool` - Tracking activé ?
- `grad_fn` - Fonction pour calculer le gradient

#### Couches (`layers.rs`)

| Couche | Paramètres | Forward | Backward Possible |
|--------|------------|---------|-------------------|
| `Conv2D` | weights, bias | ✅ via im2col | ✅ via col2im |
| `MaxPool2D` | - | ✅ + indices | ✅ indices stockés |
| `AvgPool2D` | - | ✅ | ✅ trivial |
| `GlobalAvgPool2D` | - | ✅ | ✅ trivial |
| `BatchNorm2D` | gamma, beta | ✅ | ⚠️ complexe |
| `Dropout2D` | - | ✅ | ✅ masque |
| `Flatten` | - | ✅ | ✅ reshape |
| `ActivationLayer` | - | ✅ | ✅ derivative existe |

#### Opérations (`ops.rs`)

**Disponibles** :
- ✅ `im2col_single()` - Transformation pour convolution efficace
- ✅ `col2im()` - **Inverse de im2col (clé pour backward !)**
- ✅ `conv2d_im2col()` - Convolution optimisée
- ✅ `maxpool2d()` - **Retourne les indices !**
- ✅ `avgpool2d()`, `global_avgpool2d()`

**Ce qui manque** :
- `conv2d_backward_input()` - Gradient par rapport à l'input
- `conv2d_backward_weight()` - Gradient par rapport aux poids
- `maxpool2d_backward()` - Utilise les indices
- `batchnorm2d_backward()` - Complexe

### 2.3 État Détaillé de cma-neural-network

#### Activation (`network.rs`)

```rust
impl Activation {
    pub fn apply(&self, x: &Array1<Float>) -> Array1<Float>;
    pub fn derivative_from_preactivation(&self, z: &Array1<Float>) -> Array1<Float>;  // ✅
}
```

**Toutes les activations ont leur dérivée** : Sigmoid, Tanh, ReLU, LeakyReLU, ELU, SELU, Swish, GELU, Mish, Softplus, Softsign, HardSigmoid, HardTanh, Softmax, Linear.

#### LossFunction (`network.rs`)

```rust
impl LossFunction {
    pub fn compute(&self, predictions: &Array1<Float>, targets: &Array1<Float>) -> Float;
    pub fn derivative(&self, predictions: &Array1<Float>, targets: &Array1<Float>) -> Array1<Float>;  // ✅
}
```

**Losses disponibles** : MSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy, Huber.

#### Trainer (`trainer.rs`)

```rust
impl Trainer {
    fn compute_deltas(...) -> Vec<Array1<Float>>;  // Backprop pour Dense
    fn apply_gradients_single(...);                 // Update poids
    fn apply_gradients_batch(...);                  // Update poids (batch)
}
```

**Pattern actuel (Dense)** :
1. Forward pass → stocke `pre_activations`, `activations`, `dropout_masks`
2. Compute deltas → chaîne les dérivées couche par couche
3. Apply gradients → utilise l'optimizer

#### Optimizer (`optimizer.rs`)

```rust
pub enum OptimizerType {
    SGD { learning_rate },
    Momentum { learning_rate, beta },
    RMSprop { learning_rate, beta, epsilon },
    Adam { learning_rate, beta1, beta2, epsilon },
    AdamW { learning_rate, beta1, beta2, epsilon, weight_decay },
}

impl OptimizerState2D {
    pub fn step(&mut self, weights: &mut Array2<Float>, gradient: &Array2<Float>, optimizer: &OptimizerType);
}
```

**Réutilisable pour l'autograd** : Il suffit de généraliser pour `Array4<Float>`.

---

## 3. Approches Possibles

### 3.1 Approche 1 : Modifier le trait `Layer` existant

```rust
pub trait Layer: Send + Sync {
    fn forward(&self, input: &Tensor4D) -> Tensor4D;
    fn backward(&self, grad_output: &Tensor4D, cache: &LayerCache) -> BackwardResult;
}
```

| ✅ Avantages | ❌ Inconvénients |
|-------------|-----------------|
| Pattern cohérent | Breaking change |
| Chaque couche autonome | Nécessite LayerCache |
| Facile à tester | Modifie l'existant |

### 3.2 Approche 2 : Nouveau trait `TrainableLayer`

```rust
pub trait Layer { fn forward(...); }  // Inchangé

pub trait TrainableLayer: Layer {
    type Cache;
    type Gradients;
    fn forward_with_cache(&self, input: &Tensor4D) -> (Tensor4D, Self::Cache);
    fn backward(&self, grad_output: &Tensor4D, cache: &Self::Cache) -> BackwardResult;
}
```

| ✅ Avantages | ❌ Inconvénients |
|-------------|-----------------|
| Backward compatible | Deux traits |
| Séparation claire | Plus verbeux |
| Layers non-trainables simples | |

### 3.3 Approche 3 : Enum (style Activation/LossFunction)

```rust
pub enum ConvOperation {
    Conv2D { weights, bias, stride, padding },
    MaxPool2D { kernel_size, stride },
    // ...
}

impl ConvOperation {
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D;
    pub fn backward(&self, grad: &Tensor4D, cache: &OpCache) -> BackwardResult;
}
```

| ✅ Avantages | ❌ Inconvénients |
|-------------|-----------------|
| Cohérent avec existant | Enum fermé |
| Pattern matching | Moins extensible |

### 3.4 Approche 4 : Autograd complet (PyTorch-style)

```rust
pub struct Tensor {
    data: ArrayD<Float>,
    grad: Option<ArrayD<Float>>,
    requires_grad: bool,
    grad_fn: Option<Arc<dyn GradFunction>>,
}

pub trait GradFunction: Send + Sync {
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
    fn saved_tensors(&self) -> &[Tensor];
}
```

| ✅ Avantages | ❌ Inconvénients |
|-------------|-----------------|
| Graphes dynamiques | Plus complexe |
| Standard industrie | Overhead mémoire |
| Maximum flexibilité | Plus de travail |

### 3.5 Décision : Approche 4 (Autograd)

**Raison** : L'utilisateur a besoin de graphes dynamiques pour RNN/Transformers.

---

## 4. Architecture Cible

### 4.1 Nouveau Crate : `cma-autograd`

**Plan initial** :
```
cma-autograd/
├── Cargo.toml
└── src/
    ├── lib.rs              # Exports publics
    ├── tensor.rs           # Tensor avec gradient tracking
    ├── grad_fn.rs          # Trait GradFunction
    ├── engine.rs           # Backward engine (parcours du graphe)
    ├── variable.rs         # Wrapper pour paramètres entraînables
    └── ops/
        ├── mod.rs
        ├── basic.rs        # Add, Sub, Mul, Div, MatMul
        ├── conv.rs         # Conv2D forward/backward
        ├── pool.rs         # MaxPool, AvgPool, GlobalAvgPool
        ├── norm.rs         # BatchNorm
        ├── activation.rs   # ReLU, Sigmoid, etc.
        └── reshape.rs      # Flatten, View, Transpose
```

**✅ Structure réelle implémentée** (architecture simplifiée — modules plats) :
```
cma-autograd/
├── Cargo.toml
├── src/
│   ├── lib.rs          # Exports publics + prelude
│   ├── tensor.rs       # Tensor (Arc<TensorInner>, RwLock pour grad)
│   ├── grad_fn.rs      # Trait GradFn + 17 backward implémentations
│   ├── engine.rs       # Backward engine (tri topologique) + no_grad
│   ├── ops.rs          # Toutes les opérations (arith, matmul, activations, reshape)
│   ├── module.rs       # Parameter (Arc<UnsafeCell>), Module/TrainableLayer traits, Linear, Conv2D
│   ├── layers.rs       # Couches stateless: ReLU, Sigmoid, Tanh, Flatten, MaxPool2D, Dropout, Softmax
│   ├── optim.rs        # SGD (+ momentum), Adam/AdamW
│   └── loss.rs         # mse_loss, cross_entropy_loss, binary_cross_entropy_loss
└── tests/
    └── gradient_check.rs  # 39 tests (numerical gradient checking, integration, XOR e2e)
```

**Décisions architecturales clés** :
- `Variable` remplacé par `Parameter` (wrapper `Arc<UnsafeCell<Tensor>>` pour ownership partagé optimizer↔module)
- Ops regroupées dans un seul `ops.rs` (pas de sous-dossier — plus simple, ~340 lignes)
- `GradFn` utilise `ArrayD<Float>` au lieu de `Tensor` pour backward (évite cycles de références)
- Thread-safety : `Arc<TensorInner>` + `RwLock` pour grad (pas `RefCell` — `Sync` requis pour `GradFn: Send + Sync`)

### 4.2 Intégration avec l'Existant

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE CIBLE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  cma-autograd (NOUVEAU)                                            │
│  ├── Tensor (avec grad tracking)                                   │
│  ├── GradFunction (trait)                                          │
│  ├── backward engine                                               │
│  │                                                                 │
│  │    Dépendances:                                                 │
│  │    ├── cma-cnn/ops.rs (im2col, col2im, maxpool indices)        │
│  │    └── cma-neural-network/optimizer.rs (Adam, SGD...)          │
│  │                                                                 │
│  └── Variable (paramètres avec grad)                              │
│                                                                     │
│  L'ancien code reste compatible (inférence)                        │
│  Nouveau code utilise autograd (training)                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 API Cible

```rust
use cma_autograd::{Tensor, Variable, conv2d, relu, max_pool2d};

// Création de paramètres (requires_grad = true)
let weight = Variable::randn(&[64, 1, 3, 3], true);
let bias = Variable::zeros(&[64], true);

// Forward (construit le graphe dynamiquement)
let x = Tensor::from_data(input_data, false);  // Input, pas de grad
let y = conv2d(&x, &weight, &bias, 1, 1);      // Enregistre Conv2DBackward
let y = relu(&y);                               // Enregistre ReLUBackward
let y = max_pool2d(&y, 2, 2);                  // Enregistre MaxPool2DBackward

// Compute loss
let loss = cross_entropy(&y, &target);

// Backward (parcourt le graphe, calcule les gradients)
loss.backward();

// Update (utilise les gradients calculés)
optimizer.step(&[&weight, &bias]);
optimizer.zero_grad();
```

---

## 5. Spécifications Techniques

### 5.1 Tensor avec Gradient

```rust
/// Tensor avec support autograd
#[derive(Clone)]
pub struct Tensor {
    /// Données du tenseur (N-dimensionnel)
    data: Arc<ArrayD<Float>>,
    
    /// Gradient accumulé (même shape que data)
    grad: RefCell<Option<ArrayD<Float>>>,
    
    /// Ce tenseur nécessite-t-il un gradient ?
    requires_grad: bool,
    
    /// Fonction pour calculer le gradient (None si leaf)
    grad_fn: Option<Arc<dyn GradFunction>>,
    
    /// Est-ce un tenseur créé par l'utilisateur (vs calculé) ?
    is_leaf: bool,
    
    /// Conserver le gradient après backward() ?
    retain_grad: bool,
}
```

**Méthodes clés** :

```rust
impl Tensor {
    // Création
    pub fn from_data(data: ArrayD<Float>, requires_grad: bool) -> Self;
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self;
    pub fn randn(shape: &[usize], requires_grad: bool) -> Self;
    
    // Accès
    pub fn data(&self) -> &ArrayD<Float>;
    pub fn grad(&self) -> Option<ArrayD<Float>>;
    pub fn shape(&self) -> &[usize];
    
    // Gradient
    pub fn requires_grad(&self) -> bool;
    pub fn set_requires_grad(&mut self, requires_grad: bool);
    pub fn backward(&self);  // Lance la backpropagation
    pub fn zero_grad(&mut self);  // Remet le gradient à zéro
    
    // Détachement
    pub fn detach(&self) -> Self;  // Crée une copie sans grad_fn
}
```

### 5.2 Trait GradFunction

```rust
/// Fonction de gradient pour le backward pass
pub trait GradFunction: Send + Sync + std::fmt::Debug {
    /// Calcule les gradients par rapport aux inputs
    /// 
    /// # Arguments
    /// * `grad_output` - Gradient venant de la couche suivante (∂L/∂output)
    /// 
    /// # Returns
    /// Vecteur de gradients pour chaque input (∂L/∂input_i)
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;
    
    /// Nom de l'opération (pour debug)
    fn name(&self) -> &'static str;
}
```

### 5.3 Structures de Cache

```rust
/// Cache pour Conv2D backward
pub struct Conv2DBackward {
    input: Tensor,           // Sauvé pour grad_weight
    weight: Tensor,          // Sauvé pour grad_input
    stride: usize,
    padding: usize,
    // im2col_result optionnel pour éviter recalcul
}

/// Cache pour MaxPool2D backward
pub struct MaxPool2DBackward {
    input_shape: Vec<usize>,
    indices: Array4<usize>,  // Positions des max
    pool_size: usize,
    stride: usize,
}

/// Cache pour BatchNorm2D backward
pub struct BatchNorm2DBackward {
    input: Tensor,
    gamma: Tensor,
    normalized: ArrayD<Float>,  // (x - mean) / std
    std_inv: ArrayD<Float>,     // 1 / sqrt(var + eps)
    batch_size: usize,
}
```

### 5.4 OptimizerState Généralisé

```rust
/// État de l'optimiseur pour tenseur N-dimensionnel
pub struct OptimizerStateND {
    pub m: Option<ArrayD<Float>>,  // Premier moment (momentum)
    pub v: Option<ArrayD<Float>>,  // Second moment (variance)
    pub t: usize,                  // Nombre d'itérations
}

impl OptimizerStateND {
    pub fn step(
        &mut self,
        param: &mut Tensor,
        gradient: &ArrayD<Float>,
        optimizer: &OptimizerType,
    );
}
```

---

## 6. Plan d'Implémentation

### Phase 1 : Structure de Base ✅ TERMINÉE

| Fichier | Description | Statut |
|---------|-------------|--------|
| `tensor.rs` (445 lignes) | Tensor `Arc<TensorInner>` + `RwLock` pour grad | ✅ |
| `grad_fn.rs` (555 lignes) | Trait `GradFn` + 17 backward impls | ✅ |
| `engine.rs` (201 lignes) | Backward engine + `no_grad`/`NoGradGuard` | ✅ |
| `module.rs` (482 lignes) | `Parameter` (`Arc<UnsafeCell>`) + `Module`/`TrainableLayer` traits | ✅ |

**Livrables** :
- [x] `Tensor::new()`, `Tensor::from_vec()`, `Tensor::zeros()`, `Tensor::randn()`, `Tensor::scalar()`
- [x] `Tensor::backward()` avec parcours topologique (DFS post-order → reverse)
- [x] `Tensor::from_op()` pour attacher `GradFn` au graphe dynamique
- [x] `Parameter` avec ownership partagé (`Arc<UnsafeCell>`) — clones partagent l'état
- [x] Tests unitaires (6 unit tests dans tensor.rs + engine.rs)

### Phase 2 : Opérations de Base ✅ TERMINÉE

| Fichier | Opérations | Statut |
|---------|------------|--------|
| `ops.rs` (336 lignes) | add, sub, mul, mul_scalar, neg, matmul, transpose, sum, sum_axis, mean, powf, log, exp, relu, sigmoid, tanh_act, reshape | ✅ |
| `layers.rs` (731 lignes) | ReLU, Sigmoid, Tanh, Flatten, MaxPool2D, Dropout, Softmax, BatchNorm2D, AvgPool2D, GlobalAvgPool2D | ✅ |
| `loss.rs` (155 lignes) | mse_loss, cross_entropy_loss (CrossEntropyBackward GradFn), binary_cross_entropy_loss | ✅ |
| `optim.rs` (231 lignes) | SGD (+momentum), Adam/AdamW | ✅ |

**Livrables** :
- [x] 17 opérations avec backward (`GradFn`) : Add, Sub, Mul, MulScalar, Neg, MatMul, Sum, SumAxis, Mean, Powf, Log, Exp, ReLU, Sigmoid, Tanh, Reshape, Transpose
- [x] Helper `unbroadcast()` pour réduire gradients après broadcasting
- [x] Tests de gradient numérique (39 tests dans `gradient_check.rs`)
- [x] Linear layer (forward : `input @ W^T + b`, fully tracked)
- [x] XOR end-to-end training test (converge < 0.05 loss en 500 epochs avec Adam)

### Phase 3 : Opérations CNN ✅ TERMINÉE

| Opération | Forward | Backward | Statut |
|-----------|---------|----------|--------|
| Conv2D | ✅ im2col+matmul | ✅ `Conv2DBackward` (col2im, grad_weight 4D, grad_bias) | ✅ Complet |
| MaxPool2D | ✅ avec argmax | ✅ `MaxPool2DBackward` (scatter vers argmax positions) | ✅ Complet |
| AvgPool2D | ✅ | ✅ `AvgPool2DBackward` (gradient uniforme 1/k²) | ✅ Complet |
| GlobalAvgPool | ✅ | ✅ `GlobalAvgPool2DBackward` (gradient uniforme 1/HW) | ✅ Complet |
| BatchNorm2D | ✅ train/eval | ✅ `BatchNorm2DBackward` (grad_input, grad_gamma, grad_beta) | ✅ Complet |

**Corrections effectuées** :
- `Conv2D::forward()` réécrit : forward complet (im2col→matmul→permute→bias) avec `Conv2DBackward` GradFn attaché via `Tensor::from_op()`
- `MaxPool2D` réécrit : stocke indices argmax (`max_idx_h`/`max_idx_w`), attache `MaxPool2DBackward` GradFn
- Weight gradient reshape : `grad_w` correctement reshaped de [C_out, C_in*kH*kW] → [C_out, C_in, kH, kW] via `from_shape_vec()`
- `BatchNorm2D` avec running stats EMA, train/eval mode, backward complet

**Livrables** :
- [x] `Conv2D` forward entièrement tracké (im2col→matmul→permute→bias — tout via `GradFn`)
- [x] `Conv2DBackward` avec col2im helper, weight gradient 4D reshape
- [x] `MaxPool2DBackward` GradFn stockant les indices argmax
- [x] `AvgPool2D` + `AvgPool2DBackward` (gradient uniforme)
- [x] `GlobalAvgPool2D` + `GlobalAvgPool2DBackward`
- [x] `BatchNorm2D` layer avec running stats + backward + train/eval mode
- [x] 27 tests CNN (gradient checks, forward shapes, numerical validation, mini-CNN pipelines, training convergence)

### Phase 4 : Intégration ✅ TERMINÉE

| Tâche | Statut | Notes |
|-------|--------|-------|
| Optimizer pour Tensor | ✅ | SGD (+momentum) et Adam/AdamW implémentés dans `optim.rs` |
| CNN training convergence | ✅ | `test_cnn_training_loss_decreases` — Conv2D+Linear avec Adam converge |
| CNN+BN pipeline | ✅ | `test_cnn_with_batchnorm` — Conv→BN→ReLU→Pool→FC backward complet |
| Sequential autograd | ✅ | `sequential.rs` — `Layer` trait + `Sequential` conteneur (`Vec<Box<dyn Layer>>`) |
| Training loop helper | ✅ | `train()` function — mini-batch, shuffled, validation, early stopping |

**Corrections effectuées** :
- `cross_entropy_loss` réécrit : ancienne version créait un `Tensor::new()` déconnecté du graphe autograd. Nouvelle version calcule softmax+NLL sur arrays bruts et attache `CrossEntropyBackward` GradFn (∂L/∂logits = (softmax − targets) / batch)
- `Layer` trait retiré du prelude pour éviter ambiguïté avec `Module::forward()` — utilisé uniquement en interne par `Sequential`

**Livrables** :
- [x] `Optimizer::step()` pour `Parameter` (via `update_data` + `set_data`)
- [x] `Optimizer::zero_grad()` pour reset des gradients
- [x] `Sequential` autograd-aware (conteneur de `Box<dyn Layer>`) avec builder pattern, forward, parameters, train/eval mode, summary
- [x] `Layer` trait unifié — 11 impl (Linear, Conv2D, BatchNorm2D, ReLU, Sigmoid, Tanh, Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, Softmax, Dropout)
- [x] `train()` helper — mini-batch training avec shuffled indices, accuracy tracking, validation phase (en `no_grad`), early stopping
- [x] `TrainerConfig` + `EpochMetrics` structs
- [x] 18 tests d'intégration Sequential (MLP, CNN, LeNet-5, XOR convergence, mini-CNN classification, early stopping)

### Phase 5 : Tests et Documentation ✅ COMPLÈTE

| Type | Statut | Détails |
|------|--------|--------|
| Unit tests | ✅ 6 tests | tensor creation, grad accumulation, no_grad context |
| Gradient check | ✅ 20 tests | Toutes les ops de base vérifiées vs finite differences (ε=1e-4, tol=2e-2) |
| Chain rule | ✅ 5 tests | Diamond graph, tensor reuse, compound ops (sigmoid+MSE) |
| Modules | ✅ 5 tests | Linear forward/backward/params, SGD step, Adam step |
| Loss functions | ✅ 3 tests | MSE value+backward, BCE value |
| End-to-end | ✅ 1 test | XOR training (Linear+ReLU+Linear, Adam, 500 epochs → loss < 0.05) |
| Layers | ✅ 2 tests | ReLU layer, Flatten layer |
| CNN forward | ✅ 7 tests | Conv2D shapes, MaxPool2D values, AvgPool2D, GlobalAvgPool2D, BatchNorm2D |
| CNN backward | ✅ 11 tests | Conv2D numerical grad, weight grad, MaxPool2D scatter, AvgPool/BN grads |
| CNN pipelines | ✅ 8 tests | Conv+ReLU, mini-CNN (MaxPool/AvgPool/GlobalAvg/BN), training convergence |
| CNN end-to-end | ✅ 1 test | CNN training loss decreases (Conv2D+Flatten+Linear, Adam, 20 epochs) |
| Sequential | ✅ 18 tests | MLP/CNN forward, backward, zero_grad, train/eval, LeNet-5 (shapes/batch/params/backward), XOR convergence, mini-CNN classification, early stopping |
| **Total** | **90 tests passent** | 6 unit + 39 base grad + 27 CNN grad + 18 sequential |
| Benchmarks | ❌ | À faire après intégration finale |

---

## 7. Formules Mathématiques

### 7.1 Convolution 2D

**Forward** :
$$Y = X * W + b$$

Où $*$ est la convolution (implémentée via im2col + GEMM).

**Backward** :

Gradient par rapport à l'input :
$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} *_{full} W^{rot180}$$

Équivalent à une convolution avec padding "full" et kernel retourné.

Gradient par rapport aux poids :
$$\frac{\partial L}{\partial W} = X *_{valid} \frac{\partial L}{\partial Y}$$

Gradient par rapport au biais :
$$\frac{\partial L}{\partial b} = \sum_{batch, h, w} \frac{\partial L}{\partial Y}$$

### 7.2 Max Pooling

**Forward** :
$$Y[b,c,h,w] = \max_{i,j \in window} X[b,c,h \cdot s + i, w \cdot s + j]$$

Stocke les indices $(i^*, j^*)$ des max.

**Backward** :
$$\frac{\partial L}{\partial X}[b,c,h',w'] = \begin{cases} 
\frac{\partial L}{\partial Y}[b,c,h,w] & \text{si } (h',w') = \text{argmax} \\
0 & \text{sinon}
\end{cases}$$

### 7.3 Average Pooling

**Forward** :
$$Y[b,c,h,w] = \frac{1}{k^2} \sum_{i,j \in window} X[b,c,h \cdot s + i, w \cdot s + j]$$

**Backward** :
$$\frac{\partial L}{\partial X}[b,c,h',w'] = \frac{1}{k^2} \frac{\partial L}{\partial Y}[b,c,h,w]$$

Le gradient est distribué uniformément.

### 7.4 Batch Normalization

**Forward** :
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

**Backward** (formules simplifiées) :

$$\frac{\partial L}{\partial \gamma} = \sum \frac{\partial L}{\partial y} \cdot \hat{x}$$

$$\frac{\partial L}{\partial \beta} = \sum \frac{\partial L}{\partial y}$$

$$\frac{\partial L}{\partial x} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left( \frac{\partial L}{\partial y} - \frac{1}{N}\sum \frac{\partial L}{\partial y} - \frac{\hat{x}}{N} \sum \frac{\partial L}{\partial y} \cdot \hat{x} \right)$$

### 7.5 Activations (exemples)

**ReLU** :
$$f(x) = \max(0, x)$$
$$f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

**Sigmoid** :
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Softmax** (avec Cross-Entropy) :
$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

(La dérivée combinée se simplifie)

---

## 8. Tests et Validation

### 8.1 Gradient Checking

Vérifie numériquement que les gradients analytiques sont corrects :

```rust
fn gradient_check(f: impl Fn(&Tensor) -> Tensor, x: &Tensor, eps: Float) -> bool {
    // Gradient analytique
    let y = f(x);
    y.backward();
    let analytical = x.grad().unwrap();
    
    // Gradient numérique
    let mut numerical = ArrayD::zeros(x.shape());
    for i in 0..x.data().len() {
        let mut x_plus = x.data().clone();
        let mut x_minus = x.data().clone();
        x_plus.as_slice_mut().unwrap()[i] += eps;
        x_minus.as_slice_mut().unwrap()[i] -= eps;
        
        let y_plus = f(&Tensor::from_data(x_plus, false)).data().sum();
        let y_minus = f(&Tensor::from_data(x_minus, false)).data().sum();
        
        numerical.as_slice_mut().unwrap()[i] = (y_plus - y_minus) / (2.0 * eps);
    }
    
    // Compare
    let diff = (&analytical - &numerical).mapv(|x| x.abs()).sum();
    diff < eps * 10.0
}
```

### 8.2 Tests par Opération

| Opération | Test Forward | Test Backward |
|-----------|--------------|---------------|
| Add | ✅ y = a + b | ✅ grad_a = grad_y, grad_b = grad_y |
| Mul | ✅ y = a * b | ✅ grad_a = grad_y * b, grad_b = grad_y * a |
| MatMul | ✅ vs numpy | ✅ gradient check |
| Conv2D | ✅ vs naive impl | ✅ gradient check |
| MaxPool | ✅ vs naive impl | ✅ indices correctement utilisés |
| BatchNorm | ✅ running stats | ✅ gamma, beta, input gradients |

### 8.3 Tests d'Intégration

```rust
#[test]
fn test_lenet_training() {
    let model = LeNetAutograd::new();
    let (train_data, test_data) = load_mnist();
    
    let initial_loss = evaluate(&model, &test_data);
    train(&model, &train_data, epochs: 5);
    let final_loss = evaluate(&model, &test_data);
    
    assert!(final_loss < initial_loss * 0.5);  // Au moins 50% de réduction
}
```

---

## 9. Références

### 9.1 Papers

- **Backpropagation** : Rumelhart, Hinton & Williams (1986) - "Learning representations by back-propagating errors"
- **Convolutions** : LeCun et al. (1998) - "Gradient-Based Learning Applied to Document Recognition"
- **Batch Normalization** : Ioffe & Szegedy (2015) - "Batch Normalization: Accelerating Deep Network Training"
- **Adam Optimizer** : Kingma & Ba (2015) - "Adam: A Method for Stochastic Optimization"

### 9.2 Implémentations de Référence

- **PyTorch** : [pytorch/pytorch](https://github.com/pytorch/pytorch) - Autograd complet
- **tinygrad** : [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) - Implémentation minimaliste
- **micrograd** : [karpathy/micrograd](https://github.com/karpathy/micrograd) - Autograd éducatif
- **Caffe** : Forward/Backward explicites par layer

### 9.3 Documentation

- [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [CS231n Backpropagation](https://cs231n.github.io/optimization-2/)
- [Convolution Arithmetic](https://github.com/vdumoulin/conv_arithmetic)

---

## 📊 Résumé des Estimations

### Estimations initiales vs réalité

| Phase | Estimé | Réel | Fichiers | Lignes réelles |
|-------|--------|------|----------|----------------|
| Phase 1 : Structure | 2-3 jours | ✅ ~1 jour | 4 fichiers | 1683 lignes |
| Phase 2 : Ops de base | 2-3 jours | ✅ ~1 jour | 4 fichiers | 1022 lignes |
| Phase 3 : Ops CNN | 4-5 jours | ✅ ~0.5 jour | 3 fichiers | ~700 lignes ajoutées |
| Phase 4 : Intégration | 2-3 jours | ✅ ~0.5 jour | 1 fichier (+fix loss.rs) | 561 lignes (sequential.rs) |
| Phase 5 : Tests | 2-3 jours | ✅ ~0.5 jour | 3 fichiers | ~2200 lignes |
| **Total actuel** | — | **~3.5 jours** | **13 fichiers** | **~6300 lignes** |

> **Note** : L'architecture plus simple (modules plats, pas de sous-dossier ops/) a permis
> une implémentation plus rapide que prévu pour les phases 1-2. Les optimizers et losses
> ont été intégrés directement dans la Phase 2 au lieu d'une Phase 4 séparée.
> Phase 4 (Sequential + training helper) réalisée en ~0.5 jour grâce au trait `Layer`
> unifié et au fix critique de `cross_entropy_loss` (graphe autograd déconnecté).

---

## ✅ Checklist de Progression

### Structure de Base
- [x] `Tensor` avec gradient tracking (`Arc<TensorInner>`, `RwLock` pour grad)
- [x] `GradFn` trait (`Send + Sync + Debug`, backward retourne `Vec<ArrayD<Float>>`)
- [x] Backward engine (tri topologique DFS, accumulation dans feuilles)
- [x] `Parameter` pour paramètres (`Arc<UnsafeCell<Tensor>>` — ownership partagé)
- [x] `Module` / `TrainableLayer` traits
- [x] `no_grad()` / `NoGradGuard` (thread-local `GRAD_ENABLED`)

### Opérations de Base
- [x] Add, Sub, Mul, MulScalar, Neg (avec `unbroadcast` pour broadcasting)
- [x] MatMul (2D, gradient via `grad @ B^T` et `A^T @ grad`)
- [x] Sum, SumAxis, Mean
- [x] Reshape, Flatten, Transpose
- [x] Powf, Log, Exp
- [x] Operator overloading (`&Tensor + &Tensor`, `&Tensor * Float`, etc.)

### Activations
- [x] ReLU (backward: mask `x > 0`)
- [x] Sigmoid (backward: `σ(1−σ)`)
- [x] Tanh (backward: `1 − tanh²`)
- [x] Softmax (forward-only, non-tracked — utilisé dans cross_entropy à la place)

### Couches Trainables
- [x] `Linear` (y = x @ W^T + b, He init)
- [x] `Conv2D` (im2col + matmul — ✅ graphe autograd complet via `Conv2DBackward`)
- [x] `BatchNorm2D` (normalisation + gamma/beta — running stats EMA, train/eval mode)

### Couches Stateless
- [x] ReLU, Sigmoid, Tanh (wrappers de ops)
- [x] Flatten (reshape [batch, ...] → [batch, flat])
- [x] MaxPool2D (✅ forward + backward avec indices argmax)
- [x] AvgPool2D (forward + backward, gradient uniforme 1/k²)
- [x] GlobalAvgPool2D (forward + backward, gradient uniforme 1/HW)
- [x] Dropout (train/eval mode, scaling `1/(1-p)`)

### Loss Functions
- [x] MSE (autograd-tracked: `mean((pred - target)²)`)
- [x] Cross-Entropy (log-softmax + NLL, numériquement stable)
- [x] Binary Cross-Entropy (clamping ε=1e-7)

### Optimizers
- [x] SGD (+momentum optionnel)
- [x] Adam/AdamW (bias correction, weight decay)
- [x] `Optimizer` trait (`step()`, `zero_grad()`)

### Opérations CNN
- [x] Conv2D backward complet (`Conv2DBackward` avec col2im, weight grad 4D reshape)
- [x] MaxPool2D backward (`MaxPool2DBackward` avec indices argmax)
- [x] AvgPool2D forward + backward (gradient uniforme 1/k²)
- [x] GlobalAvgPool forward + backward (gradient uniforme 1/HW)
- [x] BatchNorm2D forward + backward (running stats EMA, train/eval, gamma/beta gradients)

### Intégration
- [x] Optimizer adapté pour `Parameter` (via `update_data`/`set_data`)
- [x] Sequential autograd-aware (`sequential.rs` — `Layer` trait + `Sequential` struct, 11 impl)
- [x] Training loop helper (`train()` — mini-batch, shuffled, validation, early stopping)

### Validation
- [x] Gradient checking pour toutes les ops de base (39 tests, finite differences)
- [x] Gradient checking pour ops CNN (27 tests, numerical + pipeline)
- [x] Test XOR convergence (end-to-end, Linear+ReLU+Linear)
- [x] Test CNN convergence (Conv2D+Flatten+Linear, Adam, loss decreases)
- [x] Test mini-CNN pipelines (Conv→ReLU→Pool→Flatten→Linear, Conv+BN+ReLU+Pool+FC)
- [x] Test Sequential (18 tests: MLP, CNN, LeNet-5 shapes/backward, XOR convergence, mini-CNN classification, early stopping)
- [ ] Test LeNet convergence sur MNIST réel (données non incluses dans les tests)
- [ ] Benchmark performance

---

*Document créé le 9 février 2026*
*Dernière mise à jour : 9 février 2026 — Phases 1-4 terminées, 90 tests passent, 13 fichiers, ~6300 lignes. Phase 4 : Sequential container + train() helper + fix cross_entropy_loss*
