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

### Phase 1 : Structure de Base (2-3 jours)

| Fichier | Description | Priorité |
|---------|-------------|----------|
| `tensor.rs` | Tensor avec grad tracking | 🔴 Haute |
| `grad_fn.rs` | Trait GradFunction | 🔴 Haute |
| `engine.rs` | Backward engine | 🔴 Haute |
| `variable.rs` | Wrapper pour paramètres | 🟡 Moyenne |

**Livrables** :
- [ ] `Tensor::new()`, `Tensor::from_data()`
- [ ] `Tensor::backward()` avec parcours topologique
- [ ] Tests unitaires pour le graphe simple

### Phase 2 : Opérations de Base (2-3 jours)

| Fichier | Opérations | Priorité |
|---------|------------|----------|
| `ops/basic.rs` | Add, Sub, Mul, Div, MatMul, Sum | 🔴 Haute |
| `ops/reshape.rs` | View, Flatten, Transpose | 🔴 Haute |
| `ops/activation.rs` | ReLU, Sigmoid, Tanh, Softmax | 🟡 Moyenne |

**Livrables** :
- [ ] Toutes les ops arithmétiques avec backward
- [ ] Tests de gradient numérique (gradient checking)

### Phase 3 : Opérations CNN (4-5 jours)

| Fichier | Opérations | Complexité |
|---------|------------|------------|
| `ops/conv.rs` | Conv2D | ⭐⭐⭐⭐ |
| `ops/pool.rs` | MaxPool2D, AvgPool2D, GlobalAvgPool | ⭐⭐ |
| `ops/norm.rs` | BatchNorm2D | ⭐⭐⭐ |

**Livrables** :
- [ ] `conv2d()` avec backward complet
- [ ] `max_pool2d()` avec backward (utilise indices)
- [ ] `batch_norm2d()` avec backward
- [ ] Tests vs implémentation PyTorch

### Phase 4 : Intégration (2-3 jours)

| Tâche | Description |
|-------|-------------|
| Optimizer | Adapter pour ArrayD |
| Sequential | Créer version autograd-aware |
| Training loop | Helper pour training standard |

**Livrables** :
- [ ] `Optimizer::step()` pour Tensor
- [ ] `AutogradSequential` ou adaptation
- [ ] Exemple MNIST end-to-end

### Phase 5 : Tests et Documentation (2-3 jours)

| Type | Description |
|------|-------------|
| Unit tests | Chaque opération |
| Gradient check | Vérification numérique |
| Integration | LeNet, AlexNet complets |
| Benchmarks | Comparaison avec/sans autograd |

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

| Phase | Durée | Fichiers | Lignes de code |
|-------|-------|----------|----------------|
| Phase 1 : Structure | 2-3 jours | 4 | ~600 |
| Phase 2 : Ops de base | 2-3 jours | 3 | ~800 |
| Phase 3 : Ops CNN | 4-5 jours | 3 | ~1200 |
| Phase 4 : Intégration | 2-3 jours | 3 | ~500 |
| Phase 5 : Tests | 2-3 jours | 5 | ~800 |
| **Total** | **12-17 jours** | **~18** | **~3900** |

---

## ✅ Checklist de Progression

### Structure de Base
- [ ] `Tensor` avec gradient tracking
- [ ] `GradFunction` trait
- [ ] Backward engine (parcours topologique)
- [ ] `Variable` pour paramètres

### Opérations de Base
- [ ] Add, Sub, Mul, Div
- [ ] MatMul
- [ ] Sum, Mean
- [ ] View, Flatten, Transpose

### Opérations CNN
- [ ] Conv2D forward + backward
- [ ] MaxPool2D forward + backward
- [ ] AvgPool2D forward + backward
- [ ] BatchNorm2D forward + backward
- [ ] Toutes les activations

### Intégration
- [ ] Optimizer adapté pour Tensor
- [ ] Sequential autograd-aware
- [ ] Training loop helper

### Validation
- [ ] Gradient checking pour toutes les ops
- [ ] Test LeNet convergence
- [ ] Test AlexNet convergence
- [ ] Benchmark performance

---

*Document généré le 9 février 2026*
*À mettre à jour au fur et à mesure de l'implémentation*
