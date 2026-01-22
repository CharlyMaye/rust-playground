# TODO - Améliorations du Réseau de Neurones

## ✅ Récemment Accompli

### Corrections Mathématiques (v0.2)
- [x] **Séparation pré-activation / post-activation**
  - Nouvelle méthode `derivative_from_preactivation(z)` pour calculs corrects
  - Les dérivées GELU, Mish, Swish, SELU, ELU, Softplus sont maintenant mathématiquement exactes
  - Structure `ForwardResult` stocke z et a séparément

- [x] **Dropout complet en backward**
  - Les masques dropout sont stockés et réappliqués au gradient
  - Inverted dropout correctement implémenté

- [x] **Reproductibilité**
  - Méthode `set_seed(u64)` pour entraînement déterministe
  - RNG stocké dans Network pour éviter recréations répétées

- [x] **Sécurité Softmax**
  - `debug_assert!` ajouté pour prévenir usage incorrect de la dérivée générique

---

## 🎯 Prochaines Priorités

### 1. **Datasets et Benchmarks** 📊

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

### 2. **Cross-Validation** 🔄

- [ ] **K-Fold Cross-Validation**
  ```rust
  pub fn cross_validate(
      dataset: &Dataset, 
      k: usize, 
      network_builder: impl Fn() -> Network
  ) -> Vec<f64>
  ```
  - Divise le dataset en k folds
  - Entraîne k fois (chaque fold sert une fois de validation)
  - Retourne les k scores
  - Moyenne pour score final

- [ ] **Stratified K-Fold**
  - Préserve la distribution des classes dans chaque fold
  - Essentiel pour datasets déséquilibrés

---

### 3. **Hyperparameter Search** 🔍

- [ ] **Grid Search**
  ```rust
  pub struct GridSearch {
      learning_rates: Vec<f64>,
      hidden_sizes: Vec<Vec<usize>>,
      dropout_rates: Vec<f64>,
  }
  ```
  - Teste toutes les combinaisons
  - Lent mais exhaustif

- [ ] **Random Search**
  - Échantillonne aléatoirement l'espace des hyperparamètres
  - Plus efficace que Grid Search en haute dimension

---

### 4. **Visualisation et Debug** 🔍

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

### 5. **Performance et Optimisation** ⚡

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

### 6. **Architecture Avancées** 🧠

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
