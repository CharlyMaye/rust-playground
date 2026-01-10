# TODO - Améliorations du Réseau de Neurones

## ✅ Complété

- [x] Implémentation des fonctions d'activation configurables (15 fonctions)
- [x] Documentation détaillée de toutes les activations
- [x] Implémentation des fonctions de perte (5 loss functions)
- [x] Méthode `predict()` pour l'inférence
- [x] Méthode `predict_with_confidence()` pour estimation d'incertitude
- [x] Méthode `evaluate()` pour calculer la loss sans update
- [x] Documentation complète dans readme.md

---

## 🔄 Prochaines Étapes

### 1. Architecture Multi-Couches (Deep Learning)

#### Changements Fondamentaux

**Actuellement (1 couche cachée) :**
```
Input → Hidden Layer → Output
  2   →      5       →    1
```

**Avec plusieurs couches (Deep Neural Network) :**
```
Input → Hidden1 → Hidden2 → Hidden3 → Output
  2   →    10    →    8    →    5    →   1
```

#### Capacité d'Apprentissage

**1 couche cachée :**
- ✅ Peut approximer n'importe quelle fonction continue (théorème d'approximation universelle)
- ❌ Besoin de BEAUCOUP de neurones pour des fonctions complexes
- ❌ Apprend des features "plates" (pas hiérarchiques)

**Plusieurs couches (Deep Learning) :**
- ✅ Apprend des **représentations hiérarchiques**
- ✅ Chaque couche apprend des abstractions plus complexes
- ✅ Moins de neurones nécessaires au total

**Exemple (vision) :**
```
Couche 1: Détecte bords, coins
Couche 2: Détecte formes simples (cercles, carrés)
Couche 3: Détecte parties d'objets (yeux, roues)
Couche 4: Détecte objets complets (visage, voiture)
```

#### Structure de Données Nécessaire

**Actuellement :**
```rust
pub struct Network {
    weights1: Array2<f64>,  // 1 matrice
    biases1: Array1<f64>,   // 1 vecteur
    weights2: Array2<f64>,  // 1 matrice
    biases2: Array1<f64>,   // 1 vecteur
}
```

**Avec plusieurs couches :**
```rust
pub struct Network {
    layers: Vec<Layer>,  // Liste de couches
}

struct Layer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    activation: Activation,
}
```

#### Forward Pass Multi-Couches

```rust
pub fn forward(&self, input: &Array1<f64>) -> Vec<Array1<f64>> {
    let mut activations = vec![input.clone()];
    
    // Pour chaque couche
    for layer in &self.layers {
        let z = layer.weights.dot(activations.last().unwrap()) + &layer.biases;
        let a = layer.activation.apply(&z);
        activations.push(a);
    }
    
    activations  // Retourne toutes les activations (besoin pour backprop)
}
```

#### Backpropagation Multi-Couches

```rust
// Partir de la fin et remonter
let mut deltas = Vec::new();

// Couche de sortie
let mut delta = target - &activations.last().unwrap();
deltas.push(delta);

// Remonter couche par couche (de la fin vers le début)
for i in (1..self.layers.len()).rev() {
    let errors = self.layers[i].weights.t().dot(&delta);
    delta = &errors * &self.layers[i].activation.derivative(&activations[i]);
    deltas.push(delta);
}

// Mettre à jour tous les poids
for (i, delta) in deltas.iter().enumerate() {
    let layer_idx = self.layers.len() - 1 - i;
    // Update weights[layer_idx] et biases[layer_idx]
}
```

#### API Proposée

```rust
// Réseau simple (existant)
let network = Network::new(2, 5, 1, 
    Activation::Tanh, 
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy);

// Réseau profond (nouveau)
let network = Network::new_deep(
    2,                        // Input size
    vec![10, 8, 5],          // Hidden layers: 3 couches de 10, 8, 5 neurones
    1,                        // Output size
    vec![Activation::ReLU, Activation::ReLU, Activation::ReLU],  // Hidden activations
    Activation::Sigmoid,      // Output activation
    LossFunction::BinaryCrossEntropy
);
```

#### Nouveaux Problèmes à Gérer

##### A. Vanishing/Exploding Gradients

Avec beaucoup de couches, les gradients peuvent :
- **Disparaître** (vanishing) : devenir trop petits → les premières couches n'apprennent plus
- **Exploser** (exploding) : devenir trop grands → poids qui divergent

**Solutions :**
- [ ] Meilleure initialisation des poids (Xavier, He)
- [ ] Batch Normalization
- [ ] Skip connections (ResNet)
- [ ] Gradient clipping
- [ ] Préférer ReLU/GELU au lieu de Sigmoid

##### B. Surapprentissage (Overfitting)

Plus de couches = plus de paramètres = risque de surapprentissage

**Solutions :**
- [ ] Dropout (désactiver aléatoirement des neurones)
- [ ] Régularisation L1/L2
- [ ] Early stopping
- [ ] Augmentation de données

##### C. Performance

- [ ] Optimisation GPU (intégration CUDA ou ROCm)
- [ ] Optimiseurs avancés (Adam, RMSprop, AdamW)
- [ ] Mini-batch training
- [ ] Parallélisation

#### Quand Utiliser Plus de Couches ?

| **Problème** | **Couches Recommandées** | **Pourquoi** |
|--------------|-------------------------|--------------|
| XOR, problèmes simples | 1-2 couches cachées | Suffisant |
| MNIST (chiffres) | 2-3 couches | Patterns simples |
| Images (CIFAR, ImageNet) | 10-50+ couches | Hiérarchie complexe |
| Traitement du langage | 12-96+ couches (Transformers) | Relations longue distance |
| Jeux (AlphaGo) | 40+ couches | Stratégie complexe |

---

### 2. Initialisation des Poids

#### Xavier/Glorot Initialization

Pour Tanh et Sigmoid :
```rust
let std = (2.0 / (input_size + output_size) as f64).sqrt();
let weight = rng.sample::<f64, _>(StandardNormal) * std;
```

#### He Initialization

Pour ReLU et variantes :
```rust
let std = (2.0 / input_size as f64).sqrt();
let weight = rng.sample::<f64, _>(StandardNormal) * std;
```

- [ ] Implémenter enum `WeightInit { Xavier, He, Uniform, Normal }`
- [ ] Adapter l'initialisation selon l'activation choisie

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

## 🎯 Priorités

### Court Terme (Améliorer l'Existant)
1. [ ] Meilleure initialisation des poids
2. [ ] Optimiseur Adam
3. [ ] Sérialisation (save/load)
4. [ ] Métriques (accuracy, F1)

### Moyen Terme (Deep Learning)
1. [ ] Architecture multi-couches
2. [ ] Dropout
3. [ ] Batch Normalization
4. [ ] Mini-batch training

### Long Terme (Avancé)
1. [ ] CNN pour images
2. [ ] RNN/LSTM pour séquences
3. [ ] Optimisation GPU
4. [ ] Architectures modernes (Transformers)

---

## 💡 Notes

- XOR fonctionne parfaitement avec 1 couche cachée - pas besoin de plus pour ce problème
- Commencer par améliorer l'architecture simple avant d'ajouter des couches
- Focus sur la qualité du code et la documentation
- Tester chaque feature indépendamment
