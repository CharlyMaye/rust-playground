# TODO - Améliorations du Réseau de Neurones

## ✅ Complété

- [x] Implémentation des fonctions d'activation configurables (15 fonctions)
- [x] Documentation détaillée de toutes les activations
- [x] Implémentation des fonctions de perte (5 loss functions)
- [x] Méthode `predict()` pour l'inférence
- [x] Méthode `predict_with_confidence()` pour estimation d'incertitude
- [x] Méthode `evaluate()` pour calculer la loss sans update
- [x] Documentation complète dans readme.md
- [x] **Architecture multi-couches** avec `Network::new_deep()`
- [x] Backpropagation généralisée pour N couches
- [x] Tests sur XOR avec réseaux profonds (2 et 3 couches)
- [x] **Initialisation des poids** (Xavier, He, LeCun) avec sélection automatique
- [x] **Sérialisation** (save/load) avec module I/O externalisé
- [x] **Métriques d'évaluation** (accuracy, precision, recall, F1, confusion matrix, ROC/AUC)

### Résultats Architecture Multi-Couches

✅ **Fonctionne parfaitement :**
- Réseau simple : 2 → [5] → 1 (1 couche cachée)
- Réseau profond : 2 → [5, 3] → 1 (2 couches cachées)

⚠️ **Problème identifié :**
- Réseau très profond : 2 → [8, 5, 3] → 1 (3 couches) → Ne converge pas (vanishing gradients)
- Reste bloqué à 0.496 (prédiction = 0.5 partout)
- **Solution nécessaire :** Meilleure initialisation des poids + activation ReLU

---

## ✅ Sérialisation (Complétée)

### Module I/O Externalisé

La gestion de fichiers est complètement **externe au réseau de neurones**, comme demandé.

#### Structure

```rust
// src/io.rs - Module séparé pour la persistance
pub fn save_json(network: &Network, path: &str) -> Result<(), IoError>
pub fn load_json(path: &str) -> Result<Network, IoError>
pub fn save_binary(network: &Network, path: &str) -> Result<(), IoError>
pub fn load_binary(path: &str) -> Result<Network, IoError>
pub fn get_serialized_size(network: &Network) -> (usize, usize)  // (json, bincode)
```

#### Formats Supportés

1. **JSON** (`save_json`, `load_json`)
   - ✅ Human-readable
   - ✅ Editable manuellement
   - ✅ Compatible multi-plateformes
   - ⚠️ Plus volumineux (~660 bytes pour XOR)

2. **Binary** (`save_binary`, `load_binary`)
   - ✅ Compact (280 bytes pour XOR)
   - ✅ Compression ~2.35x vs JSON
   - ✅ Performant
   - ⚠️ Non-lisible

#### Résultats Tests XOR

```
Training: loss 0.0001 en 10000 epochs
JSON: 659 bytes
Binary: 280 bytes
Ratio: 2.35x compression

Loaded predictions:
[0,0] -> 0.000 ✓
[0,1] -> 1.000 ✓
[1,0] -> 1.000 ✓
[1,1] -> 0.000 ✓
```

#### Avantages Architecture

✅ **Séparation des responsabilités :**
- `Network` : logique d'apprentissage
- `io` module : persistance et I/O
- Pas de méthodes `save()`/`load()` dans `Network`

✅ **Flexibilité :**
- Plusieurs formats disponibles (JSON, bincode)
- Facile d'ajouter d'autres formats (YAML, MessagePack...)
- Pas de couplage fort

✅ **Testable :**
- Tests unitaires dans le module `io`
- Mock du système de fichiers possible
- Validation indépendante

---

## ✅ Métriques d'Évaluation (Complétées)

### Module `metrics.rs` - Évaluation Externalisée

- [x] **Accuracy** - Pourcentage de prédictions correctes (binaire + multi-classes)
- [x] **Binary Metrics** - Precision, Recall, F1-Score, TP/FP/TN/FN
- [x] **Confusion Matrix** - 2x2 (binaire) et NxN (multi-classes)
- [x] **ROC Curve & AUC** - Courbe ROC et aire sous la courbe

**Résultats Tests (XOR):**
```
Perfect: Accuracy=100%, Precision=1.0, Recall=1.0, F1=1.0
Imperfect: Accuracy=75%, Precision=1.0, Recall=0.5, F1=0.667
```

**Architecture:**
- Module séparé `metrics.rs` (indépendant de Network)
- Tests unitaires complets  
- Support binaire et multi-classes
- Exemple: `cargo run --example metrics_demo`

---

## Prochaines Priorités


### 1. **Optimiseurs Avancés (Adam, RMSprop)** 🚀
Convergence plus rapide et stable

- [ ] **Enum `Optimizer`**
  ```rust
  pub enum Optimizer {
      SGD { learning_rate: f64 },
      Momentum { learning_rate: f64, beta: f64 },
      RMSprop { learning_rate: f64, beta: f64, epsilon: f64 },
      Adam { learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64 },
      AdamW { learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64 },
  }
  ```

- [ ] **Adam Optimizer** (Priorité #1)
  - Adapte le learning rate par paramètre
  - Converge 2-10x plus vite que SGD
  - État : `m` (momentum) et `v` (variance) par poids
  - Standard moderne pour deep learning

- [ ] **RMSprop**
  - Adaptatif comme Adam mais plus simple
  - Bon pour RNN et problèmes non-stationnaires

- [ ] **Momentum**
  - Accélère SGD dans les bonnes directions
  - Réduit les oscillations

- [ ] **Learning Rate Scheduling**
  ```rust
  pub enum LRSchedule {
      Constant(f64),
      StepDecay { initial: f64, drop: f64, epochs_drop: usize },
      ExponentialDecay { initial: f64, decay_rate: f64 },
      CosineAnnealing { initial: f64, min_lr: f64, period: usize },
  }
  ```

---

### 2. **Régularisation** 🛡️
Éviter l'overfitting et améliorer la généralisation

- [ ] **Dropout**
  ```rust
  pub struct Layer {
      weights: Array2<f64>,
      biases: Array1<f64>,
      activation: Activation,
      dropout_rate: Option<f64>,  // Nouveau
  }
  ```
  - Désactive aléatoirement p% des neurones
  - Mode training vs inference
  - Recommandé : 0.2-0.5 pour couches cachées

- [ ] **L2 Regularization (Weight Decay)**
  ```rust
  loss = loss + lambda * weights.mapv(|w| w * w).sum()
  gradient = gradient + lambda * weights
  ```
  - Pénalise les poids trop grands
  - Typique : lambda = 0.0001 - 0.01

- [ ] **L1 Regularization**
  - Encourage la sparsité (poids à zéro)
  - Sélection automatique de features

- [ ] **Early Stopping**
  - Arrête l'entraînement si val_loss n'améliore plus
  - Paramètre `patience` (nombre d'epochs sans amélioration)

- [ ] **Batch Normalization**
  ```rust
  pub struct BatchNorm {
      gamma: Array1<f64>,      // Scale (learnable)
      beta: Array1<f64>,       // Shift (learnable)
      running_mean: Array1<f64>,
      running_var: Array1<f64>,
      momentum: f64,
      epsilon: f64,
  }
  ```
  - Normalise les activations par batch
  - Accélère convergence
  - Réduit vanishing gradients

---

### 4. **Mini-Batch Training** 📦
Scalabilité sur gros datasets

- [ ] **Dataset Struct**
  ```rust
  pub struct Dataset {
      inputs: Vec<Array1<f64>>,
      targets: Vec<Array1<f64>>,
  }
  
  impl Dataset {
      pub fn shuffle(&mut self);
      pub fn split(&self, ratios: (f64, f64, f64)) 
          -> (Dataset, Dataset, Dataset);  // train, val, test
      pub fn batches(&self, batch_size: usize) -> BatchIterator;
  }
  ```

- [ ] **Méthode `train_batch()`**
  ```rust
  pub fn train_batch(&mut self, 
                     batch: &[(Array1<f64>, Array1<f64>)], 
                     optimizer: &Optimizer)
  ```
  - Accumule gradients sur le batch
  - Update poids une fois par batch
  - 10-100x plus rapide que SGD pur

- [ ] **Stratégies de Batching**
  - Batch size typique : 16, 32, 64, 128
  - Trade-off : vitesse vs qualité du gradient
  - Plus petit batch = plus de bruit (peut aider la généralisation)

- [ ] **Shuffling**
  - Mélanger les données avant chaque epoch
  - Évite l'apprentissage de l'ordre

---

### 4. **Callbacks et Contrôle de l'Entraînement** 🎛️
Monitoring et automation

- [ ] **Trait `Callback`**
  ```rust
  pub trait Callback {
      fn on_epoch_begin(&mut self, epoch: usize);
      fn on_epoch_end(&mut self, epoch: usize, metrics: &Metrics);
      fn on_train_begin(&mut self);
      fn on_train_end(&mut self);
      fn should_stop(&self) -> bool;
  }
  ```

- [ ] **EarlyStopping Callback**
  ```rust
  pub struct EarlyStopping {
      patience: usize,
      best_loss: f64,
      wait: usize,
      restore_best_weights: bool,
  }
  ```
  - Arrête si val_loss ne s'améliore pas
  - Restaure les meilleurs poids

- [ ] **ModelCheckpoint Callback**
  ```rust
  pub struct ModelCheckpoint {
      filepath: String,
      save_best_only: bool,
      monitor: String,  // "loss" ou "val_loss"
  }
  ```
  - Sauvegarde automatique du meilleur modèle
  - Évite de perdre le progrès

- [ ] **LearningRateScheduler Callback**
  - Ajuste le learning rate pendant l'entraînement
  - Warmup, decay, cyclic LR

- [ ] **ProgressBar et Logging**
  - Affichage temps réel : epoch, loss, metrics
  - Estimation du temps restant
  - Logging dans fichier CSV/JSON

---

### 6. **Architecture et Validation** 🏗️

- [ ] **Méthode `fit()` Complète**
  ```rust
  pub fn fit(&mut self,
             train_data: &Dataset,
             validation_data: Option<&Dataset>,
             epochs: usize,
             batch_size: usize,
             optimizer: Optimizer,
             callbacks: Vec<Box<dyn Callback>>) -> History
  ```
  - Interface unifiée pour l'entraînement
  - Validation automatique à chaque epoch
  - Retourne historique (loss, metrics par epoch)

- [ ] **Cross-Validation**
  ```rust
  pub fn cross_validate(network_builder: impl Fn() -> Network,
                        dataset: &Dataset,
                        k_folds: usize) -> Vec<f64>
  ```
  - K-fold cross-validation
  - Évaluation robuste sur petits datasets

- [ ] **Grid Search / Random Search**
  - Recherche automatique d'hyperparamètres
  - Learning rate, architecture, dropout rate, etc.

---

### 7. **Datasets et Benchmarks** 📊

- [ ] **Chargeurs de Datasets Standard**
  ```rust
  pub fn load_mnist() -> (Dataset, Dataset)
  pub fn load_iris() -> Dataset
  pub fn load_wine() -> Dataset
  ```
  - MNIST : 28x28 images de chiffres
  - Iris : classification de fleurs (classique)
  - Wine : classification de vins

- [ ] **Data Augmentation**
  - Rotation, flip, noise pour images
  - Augmente artificiellement le dataset
  - Améliore généralisation

- [ ] **Normalisation**
  ```rust
  pub fn normalize(&mut self, method: NormMethod)
  
  pub enum NormMethod {
      MinMax,           // [0, 1]
      StandardScore,    // mean=0, std=1
      MaxAbs,           // [-1, 1]
  }
  ```

---

### 8. **Visualisation et Debug** 🔍

- [ ] **Visualisation des Poids**
  ```rust
  pub fn visualize_weights(&self, layer: usize) -> Array2<f64>
  ```
  - Comprendre ce que le réseau a appris

- [ ] **Activation Maps**
  - Voir quels neurones s'activent pour une entrée donnée

- [ ] **Gradient Flow Analysis**
  - Détecter vanishing/exploding gradients
  - Norms des gradients par couche

- [ ] **Learning Curves**
  - Plot train_loss vs val_loss
  - Détecter overfitting/underfitting

---

### 9. **Performance et Optimisation** ⚡

- [ ] **Parallelisation**
  - Utiliser `rayon` pour paralléliser batch processing
  - Multi-threading pour forward/backward pass

- [ ] **SIMD Optimizations**
  - Vectorisation avec instructions CPU modernes
  - ndarray supporte déjà partiellement

- [ ] **GPU Support** (Long terme)
  - Intégration avec `wgpu` ou `cudarc`
  - 10-100x speedup sur gros réseaux

- [ ] **Quantization**
  - Réduire précision (f32 → f16, int8)
  - Inference plus rapide, moins de mémoire

---

### 10. **Architecture Avancées** 🧠

#### Convolutional Neural Networks (CNN)
- [ ] **Conv2D Layer**
  ```rust
  pub struct Conv2D {
      filters: Array4<f64>,  // [num_filters, channels, height, width]
      stride: (usize, usize),
      padding: Padding,
  }
  ```
- [ ] **MaxPool2D / AvgPool2D**
- [ ] **Flatten Layer**
- [ ] Example : LeNet-5, ResNet basique

#### Recurrent Neural Networks (RNN)
- [ ] **LSTM Cell**
  ```rust
  pub struct LSTM {
      input_size: usize,
      hidden_size: usize,
      // Gates : forget, input, output
  }
  ```
- [ ] **GRU Cell** (version simplifiée de LSTM)
- [ ] **Bidirectional RNN**
- [ ] Example : classification de séquences

#### Attention Mechanisms
- [ ] **Multi-Head Attention**
- [ ] **Transformer Block** (très long terme)

---

## 🎯 Roadmap Recommandée

### Phase 1 : Métriques et Optimisation (1-2 semaines)
1. ✅ Sérialisation (FAIT)
2. **Méthode `accuracy()`** ← Commencer ici
3. Adam optimizer
4. Mini-batch training basique

### Phase 2 : Régularisation (1 semaine)
1. Dropout
2. L2 regularization
3. Early stopping
4. Dataset struct avec split/shuffle

### Phase 3 : Production Ready (1-2 semaines)
1. Callbacks (EarlyStopping, ModelCheckpoint)
2. Méthode `fit()` unifiée
3. Cross-validation
4. Chargeurs de datasets (MNIST, Iris)

### Phase 4 : Architectures Avancées (Long terme)
1. CNN layers
2. RNN/LSTM
3. GPU support

---



## ✅ Initialisation des Poids (Complétée)

**Problème résolu !** L'initialisation Xavier/He permet maintenant aux réseaux profonds de converger.

#### Résultats avec XOR

**Avant (Uniform -1..1) :**
- ✅ Réseau simple : 2 → [5] → 1 (converge)
- ✅ Réseau 2 couches : 2 → [5, 3] → 1 (converge)  
- ❌ Réseau 3 couches : 2 → [8, 5, 3] → 1 (ne converge PAS)

**Après (Xavier/He automatique) :**
- ✅ Réseau simple : 2 → [5] → 1 (converge)
- ✅ Réseau 2 couches : 2 → [5, 3] → 1 (converge)
- ✅ Réseau 3 couches : 2 → [8, 5, 3] → 1 (**converge maintenant !** avec lr=0.3, 100k epochs)

#### Implémentation

- [x] Enum `WeightInit { Uniform, Xavier, He, LeCun }`
- [x] Méthode automatique `WeightInit::for_activation()` 
- [x] Méthode `new_deep_with_init()` pour contrôle manuel
- [x] Distribution gaussienne via Box-Muller transform
- [x] Biases initialisés à zéro (recommandé)
- [x] Tests sur XOR avec réseaux profonds

#### Mapping Implémenté

| **Activation** | **Initialisation Auto** |
|----------------|-------------------------|
| Sigmoid, Tanh, Softsign, HardSigmoid, HardTanh, Softmax | Xavier |
| ReLU, LeakyReLU, ELU, GELU, Swish, Mish, Softplus | He |
| SELU | LeCun |
| Linear | Xavier |

---

## 🔄 Priorités Suivantes

| XOR, problèmes simples | 1-2 couches cachées | Suffisant |
| MNIST (chiffres) | 2-3 couches | Patterns simples |
| Images (CIFAR, ImageNet) | 10-50+ couches | Hiérarchie complexe |
| Traitement du langage | 12-96+ couches (Transformers) | Relations longue distance |
| Jeux (AlphaGo) | 40+ couches | Stratégie complexe |

---

### 2. Initialisation des Poids

**Problème actuel :** Initialisation uniforme `random_range(-1.0..1.0)` ne prend pas en compte :
- La taille de la couche
- Le type d'activation utilisé
- Risque de gradients qui disparaissent/explosent

#### Méthodes d'Initialisation

##### Uniform (Actuelle)
```rust
weight = rng.random_range(-1.0..1.0)
```
✅ Simple, fonctionne pour réseaux peu profonds  
❌ Pas adaptée aux réseaux profonds

##### Xavier/Glorot Initialization

Pour Tanh et Sigmoid :
```rust
let std = (2.0 / (input_size + output_size) as f64).sqrt();
let weight = rng.sample::<f64, _>(StandardNormal) * std;
```
✅ Maintient la variance constante à travers les couches  
✅ Idéal pour activations symétriques (Tanh, Softsign)

##### He Initialization

Pour ReLU et variantes :
```rust
let std = (2.0 / input_size as f64).sqrt();
let weight = rng.sample::<f64, _>(StandardNormal) * std;
```
✅ Compense pour les neurones "morts" de ReLU  
✅ Standard moderne pour réseaux profonds

##### LeCun Initialization

Pour SELU :
```rust
let std = (1.0 / input_size as f64).sqrt();
let weight = rng.sample::<f64, _>(StandardNormal) * std;
```

#### Implémentation

- [x] Enum `WeightInit { Uniform, Xavier, He, LeCun }`
- [x] Adapter l'initialisation selon l'activation choisie
- [ ] Initialisation automatique basée sur l'activation
- [ ] Benchmark comparatif des méthodes

#### Mapping Recommandé

| **Activation** | **Initialisation Recommandée** |
|----------------|-------------------------------|
| Sigmoid, Tanh | Xavier/Glorot |
| ReLU, LeakyReLU, ELU | He |
| SELU | LeCun |
| GELU, Swish, Mish | He (expérimental) |
| Softmax | Xavier |

---

### 3. Optimiseurs Avancés

#### Actuellement : SGD Simple

```rust
weight = weight - learning_rate * gradient
```

#### Adam Optimizer

```rust
m = beta1 * m + (1 - beta1) * gradient          // Momentum
v = beta2 * v + (1 - beta2) * gradient^2        // RMSprop
m_hat = m / (1 - beta1^t)                        // Bias correction
v_hat = v / (1 - beta2^t)
weight = weight - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

**Avantages :**
- Adapte le learning rate par paramètre
- Converge plus vite
- Plus stable

- [ ] Implémenter enum `Optimizer { SGD, Momentum, RMSprop, Adam, AdamW }`
- [ ] Ajouter états optimizer dans Network (m, v pour Adam)
- [ ] Modifier train() pour utiliser l'optimizer choisi

---

### 4. Régularisation

#### Dropout

Désactive aléatoirement p% des neurones pendant l'entraînement :

```rust
if training {
    let mask: Array1<f64> = Array1::from_shape_fn(size, |_| {
        if rng.gen::<f64>() > dropout_rate { 1.0 / (1.0 - dropout_rate) } else { 0.0 }
    });
    hidden = hidden * mask;
}
```

- [ ] Ajouter paramètre `dropout_rate` par couche
- [ ] Mode training/inference distinct
- [ ] Désactiver dropout en inference

#### L1/L2 Regularization

Pénalise les poids trop grands :

```rust
// L2 (Weight Decay)
loss = loss + lambda * (weights^2).sum()
gradient = gradient + lambda * weights

// L1 (Lasso)
loss = loss + lambda * |weights|.sum()
gradient = gradient + lambda * sign(weights)
```

- [ ] Ajouter paramètre `l2_lambda` dans Network
- [ ] Modifier calcul des gradients

---

### 5. Batch Normalization

Normalise les activations de chaque couche :

```rust
// Training
mean = batch.mean()
var = batch.var()
x_normalized = (x - mean) / sqrt(var + epsilon)
output = gamma * x_normalized + beta  // Paramètres apprenables

// Inference
output = gamma * (x - running_mean) / sqrt(running_var + epsilon) + beta
```

**Avantages :**
- Accélère la convergence
- Réduit le problème des gradients qui disparaissent
- Permet d'utiliser des learning rates plus élevés

- [ ] Implémenter struct `BatchNorm`
- [ ] Maintenir running_mean et running_var
- [ ] Mode training/inference

---

### 6. Sérialisation (Sauvegarder/Charger le Modèle)

```rust
// Sauvegarder
network.save("model.bin")?;
network.save_json("model.json")?;

// Charger
let network = Network::load("model.bin")?;
```

- [ ] Implémenter `serde::Serialize` et `Deserialize`
- [ ] Méthodes `save()` et `load()`
- [ ] Support formats : binaire (bincode), JSON, ONNX

---

### 7. Métriques d'Évaluation

```rust
// Accuracy
let accuracy = network.accuracy(&test_inputs, &test_targets);

// Precision, Recall, F1
let (precision, recall, f1) = network.metrics(&test_inputs, &test_targets);

// Confusion Matrix
let confusion = network.confusion_matrix(&test_inputs, &test_targets);
```

- [ ] Méthode `accuracy()`
- [ ] Méthode `precision_recall_f1()`
- [ ] Méthode `confusion_matrix()`
- [ ] Courbes ROC/AUC pour classification

---

### 8. Dataset Helpers

```rust
// Train/validation/test split
let (train, val, test) = dataset.split(0.7, 0.15, 0.15);

// Mini-batches
for batch in train.batches(batch_size=32) {
    network.train_batch(batch.inputs, batch.targets, lr);
}

// Data augmentation (images)
let augmented = dataset.augment(rotation=15, flip=true);
```

- [ ] Struct `Dataset` avec split
- [ ] Iterator pour mini-batches
- [ ] Shuffle avant chaque epoch
- [ ] Data augmentation basique

---

### 9. Callbacks et Logging

```rust
let callbacks = vec![
    EarlyStopping::new(patience=10),
    ModelCheckpoint::new("best_model.bin"),
    LearningRateScheduler::new(|epoch| 0.1 * 0.95_f64.powi(epoch)),
];

network.fit(
    &train_inputs, &train_targets,
    epochs=100,
    validation_data=(&val_inputs, &val_targets),
    callbacks=callbacks
);
```

- [ ] Trait `Callback`
- [ ] `EarlyStopping`
- [ ] `ModelCheckpoint`
- [ ] `LearningRateScheduler`
- [ ] `TensorBoard` logging (optionnel)

---

### 10. Architectures Spécialisées

#### CNN (Convolutional Neural Networks)

Pour images :
```rust
let cnn = CNN::new()
    .add(Conv2D::new(32, kernel_size=3))
    .add(MaxPool2D::new(pool_size=2))
    .add(Conv2D::new(64, kernel_size=3))
    .add(Flatten::new())
    .add(Dense::new(128))
    .add(Dense::new(10));
```

- [ ] Couches convolutionnelles (`Conv2D`)
- [ ] Pooling layers (`MaxPool2D`, `AvgPool2D`)
- [ ] Padding et stride

#### RNN (Recurrent Neural Networks)

Pour séquences :
```rust
let rnn = RNN::new()
    .add(LSTM::new(128))
    .add(Dense::new(10));
```

- [ ] LSTM cells
- [ ] GRU cells
- [ ] Bidirectional RNN

---

## 📚 Références

### Livres
- **"Deep Learning"** by Goodfellow, Bengio, Courville
- **"Neural Networks from Scratch"** by Harrison Kinsley
- **"Hands-On Machine Learning"** by Aurélien Géron

### Papers
- **Dropout:** Srivastava et al., 2014
- **Batch Normalization:** Ioffe & Szegedy, 2015
- **Adam:** Kingma & Ba, 2015
- **ResNet:** He et al., 2015
- **Transformers:** Vaswani et al., 2017

### Ressources Rust
- [burn](https://github.com/tracel-ai/burn) - Deep learning framework en Rust
- [candle](https://github.com/huggingface/candle) - ML framework Hugging Face
- [tch-rs](https://github.com/LaurentMazare/tch-rs) - Rust bindings pour PyTorch

---

## 💡 Notes Techniques

### Design Patterns Rust
- **Builder Pattern** : Pour construction flexible des réseaux
- **Type Safety** : Utiliser types phantom pour valider architecture à compile-time
- **Zero-Cost Abstractions** : Pas de runtime overhead pour les abstractions
- **Error Handling** : `Result<T, E>` partout, jamais de panic en production

### Best Practices
- Tests unitaires pour chaque feature
- Benchmarks avec `criterion`
- Documentation avec exemples exécutables (`cargo test --doc`)
- CI/CD avec GitHub Actions

### Performance Tips
- `ndarray` avec BLAS (OpenBLAS, MKL) pour algebra linéaire
- Profile avec `perf`, `flamegraph`
- Éviter allocations inutiles dans boucles d'entraînement
- `cargo build --release` donne 10-100x speedup vs debug

---

## 📚 Références Techniques

### Papers Fondamentaux
- **Dropout:** Srivastava et al., 2014 - "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- **Batch Normalization:** Ioffe & Szegedy, 2015 - "Batch Normalization: Accelerating Deep Network Training"
- **Adam:** Kingma & Ba, 2015 - "Adam: A Method for Stochastic Optimization"
- **Xavier Init:** Glorot & Bengio, 2010 - "Understanding the difficulty of training deep feedforward neural networks"
- **He Init:** He et al., 2015 - "Delving Deep into Rectifiers"

### Frameworks Rust ML/DL
- **burn** - Framework complet, très prometteur
- **candle** - Par Hugging Face, léger et rapide
- **tch-rs** - Bindings PyTorch pour Rust
- **linfa** - Scikit-learn-like pour Rust

### Datasets
- **MNIST** : 60k images de chiffres manuscrits
- **CIFAR-10** : 60k images 32x32 (10 classes)
- **Iris** : 150 samples, classification florale
- **Boston Housing** : Régression de prix
