# Réseau de Neurones en Rust

## Quick Start

### Compilation et Exécution

```bash
# Compiler le projet
cargo build --release

# Exécuter le programme principal
cargo run --release

# Exécuter les exemples
cargo run --release --example xor_tests       # Tests de fonctions de perte et réseaux profonds
cargo run --release --example serialization   # Démonstration save/load de modèles
cargo run --release --example metrics_demo    # Démonstration des métriques d'évaluation
```

### Exemples Disponibles

1. **`xor_tests`** - Tests complets du réseau
   - Teste toutes les fonctions de perte (MSE, MAE, BCE, Huber)
   - Teste différentes combinaisons d'activations
   - Teste les réseaux profonds multi-couches
   - Validation complète sur le problème XOR

2. **`serialization`** - Persistance des modèles
   - Entraîne un réseau sur XOR
   - Sauvegarde en JSON (human-readable) et binaire (compact)
   - Charge et vérifie les prédictions
   - Compare les tailles de fichiers

3. **`metrics_demo`** - Évaluation de performance
   - Entraîne un réseau sur XOR
   - Calcule accuracy, precision, recall, F1-score
   - Affiche la matrice de confusion
   - Compare différents seuils de classification
   - Calcule ROC-AUC

### Utilisation Basique

```rust
use test_neural::network::{Network, Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::io;
use test_neural::metrics::accuracy;
use ndarray::array;

// Créer un réseau simple avec optimiseur Adam
let mut network = Network::new(
    2,                              // 2 entrées
    5,                              // 5 neurones cachés
    1,                              // 1 sortie
    Activation::Tanh,               // Activation couche cachée
    Activation::Sigmoid,            // Activation sortie
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.01)       // Optimiseur Adam, lr=0.01
);

// Entraîner (learning rate est dans l'optimiseur)
let input = array![0.0, 1.0];
let target = array![1.0];
network.train(&input, &target);

// Prédire
let prediction = network.predict(&input);

// Évaluer
let predictions = vec![network.predict(&array![0.0, 1.0])];
let targets = vec![array![1.0]];
let acc = accuracy(&predictions, &targets, 0.5);
println!("Accuracy: {:.2}%", acc * 100.0);

// Sauvegarder
io::save_json(&network, "model.json").unwrap();

// Charger
let loaded = io::load_json("model.json").unwrap();
```

---

## Optimiseurs

Le module `optimizer` fournit 5 algorithmes d'optimisation modernes pour l'entraînement des réseaux.

### Optimiseurs Disponibles

#### 1. **SGD** - Stochastic Gradient Descent (Simple)
```rust
use test_neural::optimizer::OptimizerType;

let optimizer = OptimizerType::sgd(0.1);
```
- **Utilisation** : Basique, pour débuter ou tester
- **Learning rate** : Typiquement 0.01 - 0.5
- **Avantages** : Simple, rapide, reproductible
- **Inconvénients** : Convergence lente, nécessite tuning du LR

#### 2. **Momentum** - SGD avec momentum
```rust
let optimizer = OptimizerType::momentum(0.1);  // beta=0.9 par défaut
```
- **Utilisation** : Accélère la convergence
- **Learning rate** : Typiquement 0.01 - 0.1
- **Avantages** : Plus rapide que SGD, navigue mieux les vallées
- **Beta** : 0.9 (défaut) accumule 90% du gradient précédent

#### 3. **RMSprop** - Root Mean Square Propagation
```rust
let optimizer = OptimizerType::rmsprop(0.01);  // beta=0.9, epsilon=1e-8
```
- **Utilisation** : Adapte le learning rate par paramètre
- **Learning rate** : Typiquement 0.001 - 0.01
- **Avantages** : Gère bien les gradients instables
- **Idéal pour** : RNN, problèmes avec gradients variables

#### 4. **Adam** - Adaptive Moment Estimation (Recommandé ⭐)
```rust
let optimizer = OptimizerType::adam(0.001);  // beta1=0.9, beta2=0.999, epsilon=1e-8
```
- **Utilisation** : **Standard moderne pour la plupart des cas**
- **Learning rate** : Typiquement 0.001 (3e-4 à 1e-3)
- **Avantages** : 
  - Combine momentum + RMSprop
  - Convergence 2-10x plus rapide que SGD
  - Adapte le LR par paramètre
  - Correction de biais au début
- **Idéal pour** : Deep learning en général, par défaut

#### 5. **AdamW** - Adam avec Weight Decay découplé
```rust
let optimizer = OptimizerType::adamw(0.001, 0.01);  // lr=0.001, weight_decay=0.01
```
- **Utilisation** : Améliore la généralisation
- **Learning rate** : Typiquement 0.001
- **Weight decay** : Typiquement 0.01 - 0.1
- **Avantages** : Meilleure régularisation que L2 classique
- **Idéal pour** : Grands modèles, prévenir l'overfitting

### Comparaison de Performance

```bash
cargo run --example optimizer_comparison --release
```

Résultats sur XOR (2000 epochs) :
| Optimiseur | Loss finale | Vitesse | Remarques |
|-----------|-------------|---------|-----------|
| SGD (lr=0.5) | 0.000471 | 🐢 Lent | Nécessite LR élevé |
| Momentum (lr=0.1) | 0.000138 | 🏃 Rapide | 3x plus rapide que SGD |
| RMSprop (lr=0.01) | ~0.000000 | 🚀 Très rapide | Excellente convergence |
| Adam (lr=0.01) | 0.000207 | 🚀 Très rapide | **Meilleur compromis** |
| AdamW (lr=0.01) | 0.001215 | 🚀 Rapide | Meilleure généralisation |

### Personnalisation des Paramètres

```rust
use test_neural::optimizer::OptimizerType;

// Momentum personnalisé
let momentum = OptimizerType::Momentum { 
    learning_rate: 0.05, 
    beta: 0.95  // Plus de momentum
};

// Adam personnalisé
let adam = OptimizerType::Adam {
    learning_rate: 0.0005,
    beta1: 0.9,      // Momentum
    beta2: 0.999,    // Variance
    epsilon: 1e-8    // Stabilité numérique
};

// AdamW personnalisé
let adamw = OptimizerType::AdamW {
    learning_rate: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    weight_decay: 0.05  // Plus de régularisation
};

let network = Network::new(2, 5, 1, 
    Activation::ReLU, 
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    adam
);
```

### Guide de Sélection

| Cas d'Usage | Optimiseur Recommandé | Raison |
|-------------|----------------------|--------|
| **Premier essai / Prototype** | Adam (lr=0.001) | Fonctionne dans 90% des cas |
| **Petit dataset** | AdamW (wd=0.01) | Évite l'overfitting |
| **Grand dataset** | Adam ou SGD + Momentum | SGD scale mieux |
| **Recherche / Benchmark** | SGD avec schedule | Reproductibilité |
| **Gradients instables** | RMSprop | Adapte le LR |
| **Besoin de vitesse** | Adam | Convergence la plus rapide |

### Conseils Pratiques

**Learning Rates de Départ :**
- SGD : 0.01 - 0.1
- Momentum : 0.01 - 0.1  
- RMSprop : 0.001 - 0.01
- Adam : **0.001** (le plus universel)
- AdamW : 0.001

**Si l'entraînement ne converge pas :**
1. Réduire le learning rate (÷10)
2. Essayer Adam si vous utilisiez SGD
3. Vérifier l'initialisation des poids (Xavier pour Sigmoid/Tanh, He pour ReLU)

**Pour de meilleurs résultats :**
- Adam est le meilleur choix par défaut
- AdamW si vous observez de l'overfitting
- Momentum + SGD pour la recherche académique
- RMSprop pour les RNN/LSTM

---

## Régularisation

La régularisation permet de **prévenir l'overfitting** en pénalisant les modèles trop complexes qui "mémorisent" les données d'entraînement au lieu de généraliser.

### Qu'est-ce que l'Overfitting ?

**Overfitting** = Le modèle performe très bien sur les données d'entraînement mais mal sur de nouvelles données.

**Signes d'overfitting :**
- Loss d'entraînement très faible mais loss de validation élevée
- Prédictions parfaites sur le training set, mauvaises sur le test set
- Poids très grands dans le réseau

**Solution : Régularisation** 🛡️

### Types de Régularisation

#### 1. **Dropout** - Désactivation Aléatoire de Neurones

```rust
use test_neural::network::{Network, Activation, LossFunction};
use test_neural::optimizer::OptimizerType;

let network = Network::new(
    2, 20, 1,
    Activation::ReLU,
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.001)
).with_dropout(0.3);  // 30% des neurones désactivés pendant training
```

**Comment ça marche :**
- **Training** : Désactive aléatoirement 30% des neurones (rate=0.3)
- **Inference** : Tous les neurones actifs (mise à l'échelle automatique)
- Force le réseau à ne pas dépendre d'un seul neurone

**Quand l'utiliser :**
- Dataset petit (risque d'overfitting élevé)
- Réseaux profonds ou larges
- Typiquement : **0.2 - 0.5** pour couches cachées

**Avantages :**
- Très efficace contre l'overfitting
- Équivalent à entraîner un ensemble de modèles
- Pas de coût computationnel en inference

#### 2. **L2 Regularization (Weight Decay)** - Pénalise les Grands Poids

```rust
let network = Network::new(
    2, 20, 1,
    Activation::ReLU,
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.001)
).with_l2(0.01);  // Lambda = 0.01
```

**Comment ça marche :**
- Ajoute une pénalité proportionnelle au carré des poids : `loss += 0.5 * lambda * Σ(w²)`
- Pousse les poids vers zéro (mais jamais exactement zéro)
- Favorise des solutions plus "lisses" et simples

**Quand l'utiliser :**
- **Par défaut** pour la plupart des modèles
- Lambda typique : **0.0001 - 0.01**
- Plus lambda est grand, plus la régularisation est forte

**Avantages :**
- Simple et efficace
- Stabilise l'entraînement
- Améliore la généralisation

#### 3. **L1 Regularization (Lasso)** - Encourage la Sparsité

```rust
let network = Network::new(
    2, 50, 1,
    Activation::ReLU,
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.001)
).with_l1(0.01);  // Lambda = 0.01
```

**Comment ça marche :**
- Ajoute une pénalité proportionnelle à la valeur absolue des poids : `loss += lambda * Σ|w|`
- Pousse de nombreux poids **exactement à zéro**
- Sélection automatique de features

**Quand l'utiliser :**
- Besoin de **sparsité** (poids à zéro)
- Sélection de features automatique
- Interprétabilité du modèle

**Avantages :**
- Modèles plus compacts (beaucoup de poids à 0)
- Feature selection intégrée
- Meilleure interprétabilité

#### 4. **Elastic Net** - Combine L1 et L2

```rust
let network = Network::new(
    2, 50, 1,
    Activation::ReLU,
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.001)
).with_elastic_net(0.5, 0.01);  // 50% L1, 50% L2
```

**Comment ça marche :**
- Combine les avantages de L1 et L2
- `l1_ratio` contrôle la balance (0.0 = pur L2, 1.0 = pur L1)

**Quand l'utiliser :**
- Quand vous voulez sparsité ET stabilité
- Features corrélées

### Modes Training vs Eval

**Important** : Le dropout doit être désactivé lors de l'inférence !

```rust
// Training
network.train_mode();  // Active le dropout
for epoch in 0..1000 {
    for (input, target) in train_data {
        network.train(&input, &target);
    }
}

// Evaluation/Inference
network.eval_mode();  // Désactive le dropout
let predictions = test_data.iter()
    .map(|input| network.predict(input))
    .collect();
```

### Combiner Plusieurs Régularisations

```rust
// Dropout + L2 (approche recommandée)
let network = Network::new(
    2, 100, 1,
    Activation::ReLU,
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.001)
)
.with_dropout(0.2)   // Dropout léger
.with_l2(0.005);     // L2 modéré

// Entraînement
network.train_mode();
// ... training loop ...

// Inference
network.eval_mode();
let prediction = network.predict(&input);
```

### Guide de Sélection

| Situation | Régularisation Recommandée | Paramètres |
|-----------|---------------------------|------------|
| **Dataset petit (<1000 exemples)** | Dropout + L2 | dropout=0.3-0.5, λ=0.01 |
| **Dataset moyen (1k-100k)** | L2 ou Dropout léger | dropout=0.2, λ=0.001-0.01 |
| **Dataset grand (>100k)** | L2 faible | λ=0.0001-0.001 |
| **Réseau très large** | Dropout fort | dropout=0.4-0.5 |
| **Besoin de sparsité** | L1 | λ=0.01-0.1 |
| **Features corrélées** | Elastic Net | l1_ratio=0.5, λ=0.01 |

### Conseils Pratiques

**Diagnostic de l'overfitting :**
1. Split vos données : train (70%), validation (15%), test (15%)
2. Surveillez train_loss vs val_loss
3. Si val_loss monte pendant que train_loss baisse → **Overfitting !**

**Solutions par ordre de priorité :**
1. **Plus de données** (si possible)
2. **Dropout** (0.3-0.5) - Le plus efficace
3. **L2 regularization** (0.001-0.01)
4. **Réduire la taille du réseau**
5. **Early stopping**

**Tuning des hyperparamètres :**
- Commencer sans régularisation
- Si overfitting : ajouter Dropout (0.3)
- Si encore overfitting : augmenter dropout (0.4-0.5) ou ajouter L2
- Si underfitting : réduire la régularisation

### Exemple Complet

```rust
use test_neural::network::{Network, Activation, LossFunction};
use test_neural::optimizer::OptimizerType;

// Créer un réseau avec régularisation
let mut network = Network::new(
    784,    // MNIST input size
    128,    // Hidden layer (large)
    10,     // 10 classes
    Activation::ReLU,
    Activation::Softmax,
    LossFunction::CategoricalCrossEntropy,
    OptimizerType::adam(0.001)
)
.with_dropout(0.3)   // Prevent overfitting
.with_l2(0.001);     // Weight decay

// Training mode
network.train_mode();
for epoch in 0..epochs {
    for (input, target) in train_data.iter() {
        network.train(input, target);
    }
    
    // Validation (en mode eval)
    network.eval_mode();
    let val_loss = network.evaluate(&val_inputs, &val_targets);
    println!("Epoch {}: val_loss = {:.4}", epoch, val_loss);
    network.train_mode();  // Retour en mode training
}

// Final evaluation
network.eval_mode();
let test_accuracy = accuracy(&test_predictions, &test_targets, 0.5);
println!("Test Accuracy: {:.2}%", test_accuracy * 100.0);
```

### Démo

```bash
cargo run --example regularization_demo --release
```

Résultats sur XOR avec réseau surdimensionné (2 → [20] → 1) :
| Méthode | Loss Finale | Convergence | Généralisation |
|---------|-------------|-------------|----------------|
| Sans régularisation | 0.000000 | Très rapide | Risque d'overfitting |
| Dropout (0.3) | 0.000001 | Stable | Excellente |
| L2 (0.01) | 0.135389 | Lente | Très bonne |
| L1 (0.01) | Variable | Instable | Bonne avec sparsité |
| Combiné | 0.00001 | **Optimale** | **Meilleure** |

**Conclusion** : Sur les petits datasets, **Dropout + L2** offre le meilleur compromis.

---

## Mini-Batch Training

Le **mini-batch training** consiste à entraîner le réseau sur des groupes d'exemples (batches) au lieu d'un seul exemple à la fois. C'est une technique essentielle pour l'entraînement efficace sur de grands datasets.

### Pourquoi Mini-Batch ?

**❌ Problèmes du Single-Sample Training (SGD pur):**
- Très lent sur grands datasets
- Gradients bruités → convergence instable
- Impossible d'utiliser la vectorisation
- Mise à jour trop fréquente des poids

**✅ Avantages du Mini-Batch:**
- **2-3x plus rapide** en pratique
- Gradients plus stables (moyenne sur le batch)
- Meilleure utilisation du cache CPU
- Convergence plus smooth
- Permet la parallélisation

### Utilisation du Module `dataset`

```rust
use test_neural::network::{Network, Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::dataset::Dataset;
use ndarray::array;

// 1. Créer le dataset
let inputs = vec![
    array![0.0, 0.0], array![0.0, 1.0],
    array![1.0, 0.0], array![1.0, 1.0],
];
let targets = vec![
    array![0.0], array![1.0],
    array![1.0], array![0.0],
];

let dataset = Dataset::new(inputs, targets);

// 2. Split train/validation/test
let (train, val, test) = dataset.split_three(0.7, 0.15, 0.15);
// Résultat: 70% train, 15% validation, 15% test

// 3. Créer le réseau (learning rate plus élevé pour batch training)
let mut network = Network::new(
    2, 8, 1,
    Activation::Tanh,
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.01)  // 10x plus que single-sample (0.001)
);

// 4. Entraîner avec mini-batches
let batch_size = 32;
let epochs = 100;

for epoch in 0..epochs {
    // IMPORTANT: Shuffle avant chaque epoch !
    train.shuffle();
    
    // Itérer sur les batches
    for (batch_inputs, batch_targets) in train.batches(batch_size) {
        network.train_batch(&batch_inputs, &batch_targets);
    }
    
    // Évaluation périodique
    if epoch % 10 == 0 {
        let train_loss = network.evaluate(train.inputs(), train.targets());
        let val_loss = network.evaluate(val.inputs(), val.targets());
        println!("Epoch {}: train={:.4}, val={:.4}", epoch, train_loss, val_loss);
    }
}

// 5. Test final
let test_loss = network.evaluate(test.inputs(), test.targets());
println!("Test loss: {:.6}", test_loss);
```

### API du Module Dataset

#### **`Dataset::new(inputs, targets)`**
Crée un dataset à partir de vecteurs d'inputs et targets.

```rust
let dataset = Dataset::new(inputs, targets);
println!("Dataset size: {}", dataset.len());
```

#### **`dataset.shuffle()`**
Mélange aléatoirement l'ordre des exemples.

```rust
dataset.shuffle();  // À appeler avant chaque epoch !
```

⚠️ **IMPORTANT** : Toujours shuffle entre les epochs pour éviter que le réseau apprenne l'ordre des exemples.

#### **`dataset.split(ratio)`**
Split en train/test.

```rust
let (train, test) = dataset.split(0.8);  // 80% train, 20% test
```

#### **`dataset.split_three(train_ratio, val_ratio)`**
Split en train/validation/test.

```rust
let (train, val, test) = dataset.split_three(0.7, 0.15);
// 70% train, 15% val, 15% test (le reste)
```

#### **`dataset.batches(batch_size)`**
Retourne un iterator sur les batches.

```rust
for (batch_inputs, batch_targets) in dataset.batches(32) {
    network.train_batch(&batch_inputs, &batch_targets);
}
```

### Comparaison: Single-Sample vs Mini-Batch

Sur un dataset de 1000 exemples (50 epochs):

| Méthode | Batch Size | Temps | Loss Finale | Speedup |
|---------|------------|-------|-------------|---------|
| Single-sample | 1 | 0.10s | 0.000000 | 1.0x (baseline) |
| Mini-batch | 32 | **0.05s** | 0.001794 | **2.1x** ⚡ |
| Mini-batch | 64 | 0.05s | 0.006283 | 2.1x |
| Mini-batch | 128 | 0.05s | 0.015565 | 2.2x |

**Résultats** (exemple minibatch_demo.rs):
- **batch_size=32** offre le meilleur compromis vitesse/qualité
- Plus le batch est grand, plus c'est rapide mais convergence légèrement moins bonne
- Ajuster le learning rate : ×10 pour batch training vs single-sample

### Guide de Sélection du Batch Size

| Taille Dataset | Batch Size Recommandé | Raison |
|----------------|----------------------|--------|
| < 1000 exemples | 16-32 | Dataset petit, petits batches suffisent |
| 1000-10k exemples | 32-64 | Compromis optimal |
| 10k-100k exemples | 64-128 | Meilleure vectorisation |
| > 100k exemples | 128-256 | Maximiser la vitesse |

**Règles générales:**
- Batch size **trop petit** (< 16) : trop lent, gradients bruités
- Batch size **trop grand** (> 256) : convergence plus difficile, demande plus de mémoire
- **Puissance de 2** (16, 32, 64, 128) : optimisé pour le CPU
- Toujours **augmenter le learning rate** proportionnellement au batch size

### Ajuster le Learning Rate

```rust
// Single-sample training
OptimizerType::adam(0.001)

// Mini-batch training (batch_size=32)
OptimizerType::adam(0.01)   // 10x plus élevé

// Mini-batch training (batch_size=128)
OptimizerType::adam(0.03)   // 30x plus élevé
```

**Règle empirique** : Learning rate ≈ 0.001 × sqrt(batch_size)

### Conseils Pratiques

✅ **À faire:**
- Toujours `shuffle()` le dataset avant chaque epoch
- Split en train/val/test pour détecter l'overfitting
- Commencer avec batch_size=32 puis expérimenter
- Augmenter le learning rate pour le batch training
- Surveiller la loss de validation (early stopping)

❌ **À éviter:**
- Oublier de shuffle → le réseau apprend l'ordre !
- Batch size de 1 sur grand dataset (trop lent)
- Utiliser le même learning rate que single-sample
- Batch size > 10% du dataset (perd le bénéfice SGD)

### Démo

```bash
cargo run --example minibatch_demo --release
```

Résultats sur dataset XOR élargi (1000 exemples, 50 epochs):
```
📈 Temps d'entraînement:
  • Single-sample:  0.10s
  • Mini-batch (32): 0.05s (2.1x speedup) ⚡

🎯 Loss finale (test):
  • Single-sample:  0.000000
  • Mini-batch (32): 0.001794  (excellent compromis)
```

**Conclusion** : Le mini-batch training est **essentiel** pour datasets > 1000 exemples. Batch size 32 offre le meilleur compromis.

---

## Callbacks - Automatisation de l'Entraînement

Les **callbacks** sont des fonctions qui s'exécutent automatiquement à différents moments de l'entraînement (début/fin epoch, début/fin training). Ils permettent d'**automatiser** et d'**optimiser** l'entraînement sans modifier la boucle principale.

### Pourquoi les Callbacks ?

**❌ Problèmes sans callbacks:**
- Code d'entraînement répétitif et verbeux
- Difficile de surveiller la progression
- Pas de sauvegarde automatique du meilleur modèle
- Surentraînement (overfitting) si on ne surveille pas
- Learning rate fixe = convergence sous-optimale

**✅ Avec callbacks:**
- **EarlyStopping** : Arrête automatiquement si overfitting
- **ModelCheckpoint** : Sauvegarde le meilleur modèle
- **LearningRateScheduler** : Adapte le LR dynamiquement
- **ProgressBar** : Affiche la progression en temps réel
- Code propre, maintenable, réutilisable

### Callbacks Disponibles

#### 1. **EarlyStopping** - Arrêt Précoce

Surveille la validation loss et **arrête l'entraînement** après `patience` epochs sans amélioration.

```rust
use test_neural::callbacks::EarlyStopping;

let mut early_stop = EarlyStopping::new(
    10,      // patience: attendre 10 epochs sans amélioration
    0.0001   // min_delta: amélioration minimale requise
);

// Dans la boucle d'entraînement
let mut callbacks: Vec<Box<dyn Callback>> = vec![
    Box::new(early_stop),
];

network.fit(&train, Some(&val), 100, 32, &mut callbacks);
```

**Fonctionnement:**
- Compare val_loss à chaque epoch
- Si amélioration < min_delta pendant `patience` epochs → **arrête**
- Évite l'overfitting automatiquement
- Économise du temps de calcul

**Quand utiliser:**
- Toujours ! Surtout sur petits datasets
- patience=10-20 pour datasets moyens
- patience=5-10 pour petits datasets
- min_delta=0.0001 typique

#### 2. **ModelCheckpoint** - Sauvegarde Automatique

Sauvegarde automatiquement le modèle quand la validation loss **s'améliore**.

```rust
use test_neural::callbacks::ModelCheckpoint;

let mut checkpoint = ModelCheckpoint::new(
    "best_model.json",  // Chemin du fichier
    true                // save_best_only: sauvegarder uniquement si amélioration
);

let mut callbacks: Vec<Box<dyn Callback>> = vec![
    Box::new(checkpoint),
];

network.fit(&train, Some(&val), 100, 32, &mut callbacks);

// Après l'entraînement, charger le meilleur modèle
let best_network = test_neural::io::load_json("best_model.json").unwrap();
```

**Fonctionnement:**
- Compare val_loss à chaque epoch
- Si amélioration → sauvegarde automatique (JSON ou binary)
- Vous récupérez le meilleur modèle même si l'entraînement overfitte ensuite

**Formats supportés:**
- `.json` → JSON (human-readable)
- `.bin` → Binary (compact, 2-3x plus petit)

**Quand utiliser:**
- Entraînements longs (> 50 epochs)
- Quand la loss peut fluctuer
- Pour garder le meilleur modèle automatiquement

#### 3. **LearningRateScheduler** - Ajustement Dynamique du LR

Ajuste automatiquement le learning rate pendant l'entraînement. Trois stratégies disponibles.

##### **StepLR** - Réduction à Intervalles Réguliers

```rust
use test_neural::callbacks::{LearningRateScheduler, LRSchedule};

let mut scheduler = LearningRateScheduler::new(
    LRSchedule::StepLR {
        step_size: 10,  // Réduire tous les 10 epochs
        gamma: 0.5      // Diviser LR par 2
    }
);

network.fit_with_scheduler(&train, Some(&val), 50, 32, &mut scheduler, &mut callbacks);
```

**Fonctionnement:**
- Tous les `step_size` epochs: `LR = LR × gamma`
- Exemple: LR=0.1 → 0.05 → 0.025 → 0.0125...
- Simple, prévisible

**Quand utiliser:**
- Convergence instable avec LR fixe
- Vous connaissez approximativement la durée de l'entraînement
- step_size=10-20 typique

##### **ReduceOnPlateau** - Réduction Intelligente

```rust
let mut scheduler = LearningRateScheduler::new(
    LRSchedule::ReduceOnPlateau {
        patience: 5,      // Attendre 5 epochs sans amélioration
        factor: 0.5,      // Diviser LR par 2
        min_delta: 0.0001 // Amélioration minimale
    }
);

network.fit_with_scheduler(&train, Some(&val), 50, 32, &mut scheduler, &mut callbacks);
```

**Fonctionnement:**
- Surveille la validation loss
- Si stagnation pendant `patience` epochs → `LR = LR × factor`
- S'adapte automatiquement à la convergence

**Quand utiliser:**
- **Recommandé dans la plupart des cas**
- Convergence adaptative, intelligente
- Ne nécessite pas de connaître la durée d'entraînement
- patience=5-10 typique

##### **ExponentialLR** - Décroissance Exponentielle

```rust
let mut scheduler = LearningRateScheduler::new(
    LRSchedule::ExponentialLR {
        gamma: 0.95  // Multiplier LR par 0.95 chaque epoch
    }
);

network.fit_with_scheduler(&train, Some(&val), 50, 32, &mut scheduler, &mut callbacks);
```

**Fonctionnement:**
- Chaque epoch: `LR = LR × gamma`
- Décroissance smooth et continue
- LR diminue exponentiellement

**Quand utiliser:**
- Fine-tuning avec décroissance lente
- gamma=0.95-0.99 typique
- Convergence très smooth

#### 4. **ProgressBar** - Affichage de Progression

Affiche la progression en temps réel avec ETA (temps restant estimé).

```rust
use test_neural::callbacks::ProgressBar;

let mut progress = ProgressBar::new(100);  // 100 epochs total

let mut callbacks: Vec<Box<dyn Callback>> = vec![
    Box::new(progress),
];

network.fit(&train, Some(&val), 100, 32, &mut callbacks);
```

**Affichage:**
```
🚀 Début de l'entraînement (100 epochs)
Epoch 10/100 [10.0%] - train_loss: 0.123456 - val_loss: 0.234567 - ETA: 45s
Epoch 20/100 [20.0%] - train_loss: 0.056789 - val_loss: 0.123456 - ETA: 36s
...
✅ Entraînement terminé en 50.23s
```

**Quand utiliser:**
- Entraînements longs (> 20 epochs)
- Pour suivre la progression visuellement
- Estimée du temps restant utile

### Combiner Plusieurs Callbacks

La vraie puissance vient de la **combinaison** de callbacks :

```rust
use test_neural::network::{Network, Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::dataset::Dataset;
use test_neural::callbacks::{
    EarlyStopping, ModelCheckpoint, LearningRateScheduler,
    ProgressBar, LRSchedule, Callback
};

// 1. Créer le réseau
let mut network = Network::new(
    2, 8, 1,
    Activation::Tanh,
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.01)
);

// 2. Préparer les données
let dataset = Dataset::new(inputs, targets);
let (train, val) = dataset.split(0.8);

// 3. Configurer les callbacks
let mut scheduler = LearningRateScheduler::new(
    LRSchedule::ReduceOnPlateau {
        patience: 5,
        factor: 0.5,
        min_delta: 0.0001
    }
);

let mut callbacks: Vec<Box<dyn Callback>> = vec![
    Box::new(EarlyStopping::new(15, 0.00001)),
    Box::new(ModelCheckpoint::new("best_model.json", true)),
    Box::new(ProgressBar::new(100)),
];

// 4. Entraîner avec tout automatisé !
let history = network.fit_with_scheduler(
    &train,
    Some(&val),
    100,        // max epochs
    32,         // batch size
    &mut scheduler,
    &mut callbacks
);

// 5. Résultat
println!("Entraînement terminé en {} epochs", history.len());
println!("Meilleur modèle sauvegardé automatiquement dans best_model.json");
```

**Résultat:**
- ✅ Progression affichée en temps réel
- ✅ Learning rate adapté automatiquement quand stagnation
- ✅ Arrêt automatique si overfitting
- ✅ Meilleur modèle sauvegardé automatiquement
- ✅ Code propre, maintenable, professionnel

### API Complète

#### **Méthodes d'Entraînement avec Callbacks**

```rust
// Avec callbacks standard (pas de LR scheduler)
pub fn fit(
    &mut self,
    train_dataset: &Dataset,
    val_dataset: Option<&Dataset>,
    epochs: usize,
    batch_size: usize,
    callbacks: &mut Vec<Box<dyn Callback>>,
) -> Vec<(f64, Option<f64>)>  // Retourne history (train_loss, val_loss)

// Avec LR scheduler
pub fn fit_with_scheduler(
    &mut self,
    train_dataset: &Dataset,
    val_dataset: Option<&Dataset>,
    epochs: usize,
    batch_size: usize,
    scheduler: &mut LearningRateScheduler,
    callbacks: &mut Vec<Box<dyn Callback>>,
) -> Vec<(f64, Option<f64>)>
```

#### **Trait Callback** - Créer Vos Propres Callbacks

```rust
pub trait Callback {
    fn on_train_begin(&mut self, network: &Network) {}
    fn on_train_end(&mut self, network: &Network) {}
    fn on_epoch_begin(&mut self, epoch: usize, network: &Network) {}
    fn on_epoch_end(&mut self, epoch: usize, network: &Network, 
                     train_loss: f64, val_loss: Option<f64>) -> bool {
        true  // Return false to stop training
    }
}
```

**Exemple - Callback Personnalisé:**

```rust
use test_neural::callbacks::Callback;
use test_neural::network::Network;

struct LossLogger {
    losses: Vec<f64>,
}

impl Callback for LossLogger {
    fn on_epoch_end(&mut self, epoch: usize, _network: &Network, 
                     _train_loss: f64, val_loss: Option<f64>) -> bool {
        if let Some(loss) = val_loss {
            self.losses.push(loss);
            println!("Epoch {}: val_loss = {:.6}", epoch, loss);
        }
        true  // Continue training
    }
}
```

### Comparaison: Avec vs Sans Callbacks

| Aspect | Sans Callbacks | Avec Callbacks |
|--------|---------------|----------------|
| **Code** | Verbeux, répétitif | Concis, réutilisable |
| **Monitoring** | Manuel (print dans boucle) | Automatique (ProgressBar) |
| **Sauvegarde** | Manuelle (if best_loss...) | Automatique (ModelCheckpoint) |
| **Overfitting** | Risque élevé | Prévenu (EarlyStopping) |
| **Learning Rate** | Fixe, sous-optimal | Adapté (LR Scheduler) |
| **Temps dev** | Plus long | Plus court |
| **Maintenabilité** | Difficile | Facile |
| **Professionnalisme** | Amateur | Production-ready |

### Guide de Sélection

| Situation | Callbacks Recommandés |
|-----------|----------------------|
| **Prototypage rapide** | ProgressBar |
| **Entraînement long** | EarlyStopping + ProgressBar |
| **Production** | EarlyStopping + ModelCheckpoint + ReduceOnPlateau |
| **Fine-tuning** | ExponentialLR + ModelCheckpoint |
| **Petit dataset** | EarlyStopping (patience=5) + Dropout |
| **Grand dataset** | ReduceOnPlateau + ModelCheckpoint |
| **Optimal (recommandé)** | **Tous combinés !** |

### Conseils Pratiques

✅ **À faire:**
- Toujours utiliser **EarlyStopping** (évite overfitting)
- **ModelCheckpoint** pour entraînements > 20 epochs
- **ReduceOnPlateau** = meilleur scheduler dans la plupart des cas
- Combiner plusieurs callbacks pour résultat optimal
- Ajuster `patience` selon la taille du dataset

❌ **À éviter:**
- Entraînement sans validation dataset (impossible d'utiliser callbacks intelligemment)
- patience trop faible (< 5) → arrêt prématuré
- Oublier save_best_only=true dans ModelCheckpoint
- Ne pas vérifier que val_dataset est fourni

### Démo

```bash
cargo run --example callbacks_demo --release
```

**Résultats** (dataset XOR 1000 exemples, 100 epochs max):

| Configuration | Epochs | Loss Finale | Notes |
|--------------|--------|-------------|-------|
| Baseline (sans callbacks) | 100 | 0.000291 | Overfitting possible |
| EarlyStopping | 90 | 0.000349 | Arrêt automatique ✓ |
| ModelCheckpoint | 50 | 0.001442 | Meilleur modèle sauvegardé ✓ |
| StepLR | 50 | 0.000166 | LR réduit 3× |
| ReduceOnPlateau | 50 | 0.001441 | LR adapté intelligemment ✓ |
| ExponentialLR | 50 | 0.000685 | Décroissance smooth |
| **Combinaison optimale** | **90** | **0.000079** | **Meilleur résultat** ⚡ |

**Observation**: La combinaison **EarlyStopping + ModelCheckpoint + ReduceOnPlateau + ProgressBar** donne les meilleurs résultats avec automatisation complète.

**Conclusion** : Les callbacks transforment l'entraînement de réseaux neuronaux. Ils sont **essentiels** pour un code production-ready, évitent l'overfitting, et optimisent automatiquement la convergence.

---

## Métriques d'Évaluation

Le module `metrics` fournit des outils complets pour évaluer la performance de vos modèles.

### Métriques Disponibles

#### 1. **`accuracy()`** - Exactitude
```rust
use test_neural::metrics::accuracy;

let acc = accuracy(&predictions, &targets, 0.5);
println!("Accuracy: {:.2}%", acc * 100.0);
```
- **Binaire** : seuil personnalisable (défaut 0.5)
- **Multi-classes** : argmax automatique
- Simple, rapide, intuitif
- Retourne le pourcentage de prédictions correctes

#### 2. **`binary_metrics()`** - Métriques Complètes pour Classification Binaire
```rust
use test_neural::metrics::binary_metrics;

let metrics = binary_metrics(&predictions, &targets, 0.5);
println!("{}", metrics.summary());
// Accuracy: 0.9500 | Precision: 0.9231 | Recall: 0.9600 | F1: 0.9412
// TP: 24 | FP: 2 | TN: 19 | FN: 1
```

**Métriques retournées :**
- **Precision** : `TP / (TP + FP)` - "Quand je prédis positif, à quelle fréquence ai-je raison?"
- **Recall** : `TP / (TP + FN)` - "Je capture quel % de tous les positifs réels?"
- **F1-Score** : Moyenne harmonique de Precision et Recall
- **TP, FP, TN, FN** : True/False Positives/Negatives

#### 3. **`confusion_matrix_binary()` & `confusion_matrix_multiclass()`** - Matrice de Confusion
```rust
use test_neural::metrics::{confusion_matrix_binary, format_confusion_matrix};

let matrix = confusion_matrix_binary(&predictions, &targets, 0.5);
println!("{}", format_confusion_matrix(&matrix, Some(&["Neg", "Pos"])));
```

```
Confusion Matrix:
                Predicted
              Neg      Pos 
Actual   Neg   19        2
         Pos    1       24
```

- **Binaire** : Matrice 2x2
- **Multi-classes** : Matrice NxN
- Helper `format_confusion_matrix()` pour affichage lisible
- Visualise précisément les types d'erreurs

#### 4. **`roc_curve()` & `auc_roc()`** - Analyse ROC
```rust
use test_neural::metrics::{roc_curve, auc_roc};

// Courbe ROC complète
let (fpr, tpr, thresholds) = roc_curve(&predictions, &targets, 100);

// AUC (Area Under Curve)
let auc = auc_roc(&predictions, &targets);
println!("AUC: {:.4}", auc);
// AUC: 0.9850 (1.0 = parfait, 0.5 = aléatoire)
```

- **Courbe ROC** : FPR vs TPR à différents seuils
- **AUC** : 1.0 = prédictions parfaites, 0.5 = performance aléatoire
- **Indépendant du seuil** : Évalue la performance globale
- Idéal pour comparer différents modèles

### Quand Utiliser Quelle Métrique ?

| Situation | Métrique Recommandée | Raison |
|-----------|---------------------|--------|
| **Dataset équilibré** | Accuracy | Simple et intuitif |
| **Dataset déséquilibré** | F1-Score, Recall | Évite les fausses bonnes performances |
| **Coût FP élevé** (ex: spam) | Precision | Ne pas bloquer vrais emails |
| **Coût FN élevé** (ex: médical) | Recall | Ne pas manquer de malades |
| **Comparaison de modèles** | AUC | Indépendant du seuil |
| **Analyse détaillée** | Confusion Matrix | Voir précisément les erreurs |

### Exemple Complet

```rust
use test_neural::network::{Network, Activation, LossFunction};
use test_neural::metrics::{accuracy, binary_metrics, confusion_matrix_binary};
use ndarray::array;

// Entraîner le réseau
let mut network = Network::new(2, 5, 1, 
    Activation::Tanh, 
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy
);

// ... entraînement ...

// Obtenir les prédictions
let predictions: Vec<_> = test_inputs.iter()
    .map(|input| network.predict(input))
    .collect();

// Évaluer avec différentes métriques
let acc = accuracy(&predictions, &test_targets, 0.5);
let metrics = binary_metrics(&predictions, &test_targets, 0.5);
let matrix = confusion_matrix_binary(&predictions, &test_targets, 0.5);

println!("Accuracy: {:.2}%", acc * 100.0);
println!("{}", metrics.summary());
println!("{}", format_confusion_matrix(&matrix, Some(&["Neg", "Pos"])));
```

Pour plus de détails, consultez [METRICS_GUIDE.md](METRICS_GUIDE.md) qui contient :
- Guide complet de toutes les métriques
- Cas d'usage par domaine (médical, finance, vision, NLP)
- Métriques avancées à implémenter
- Bonnes pratiques et pièges à éviter

---

## Concepts Clés

### 1. Architecture (Couches/Neurones)

La **structure** de ton réseau : nombre de couches et nombre de neurones par couche.

- Dans ton code : `Network::new(2, 3, 1)` = 2 entrées → 3 neurones cachés → 1 sortie
- Plus de neurones/couches = plus de capacité d'apprentissage, mais risque de **surapprentissage**

### 2. Fonctions d'Activation (sigmoid, ReLU, tanh...)

Fonction qui **transforme** la sortie d'un neurone.

**Actuellement utilisée : Sigmoid**
- Formule : `1 / (1 + e^-x)`
- Sortie : entre `[0, 1]`

**Alternatives :**
- **ReLU** : `max(0, x)` → plus rapide, standard moderne
- **tanh** : `tanh(x)` → sortie entre `[-1, 1]`
- **Leaky ReLU**, **ELU**, etc.

---

## Fonctions d'Activation Détaillées

### Sigmoid (Logistic)
**Formule :** $f(x) = \frac{1}{1 + e^{-x}}$

**Dérivée :** $f'(x) = f(x) \cdot (1 - f(x))$

**Propriétés :**
- Sortie : `[0, 1]`
- Lisse et différentiable partout
- Interprétable comme une probabilité

**Avantages :**
- ✅ Sortie normalisée (bonne pour la couche de sortie en classification binaire)
- ✅ Gradient bien défini

**Inconvénients :**
- ❌ **Problème du gradient qui disparaît** (vanishing gradient) pour grandes/petites valeurs
- ❌ Sortie non centrée sur zéro
- ❌ Coûteux en calcul (`exp()`)

**Implémentation Rust :**
```rust
fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
    x * &(1.0 - x)
}
```

---

### ReLU (Rectified Linear Unit)
**Formule :** $f(x) = \max(0, x)$

**Dérivée :** $f'(x) = \begin{cases} 1 & \text{si } x > 0 \\ 0 & \text{si } x \leq 0 \end{cases}$

**Propriétés :**
- Sortie : `[0, +∞)`
- Linéaire pour valeurs positives, zéro sinon
- **Standard moderne pour les couches cachées**

**Avantages :**
- ✅ **Très rapide** (simple comparaison et multiplication)
- ✅ Pas de gradient qui disparaît pour valeurs positives
- ✅ Favorise la sparsité (certains neurones s'éteignent)
- ✅ Convergence plus rapide que sigmoid/tanh

**Inconvénients :**
- ❌ **Problème des neurones morts** : si gradient = 0, le neurone ne s'active plus jamais
- ❌ Sortie non centrée sur zéro
- ❌ Non différentiable en x = 0 (en pratique, on prend 0 ou 1)

**Implémentation Rust :**
```rust
fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x.max(0.0))
}

fn relu_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}
```

---

### Leaky ReLU
**Formule :** $f(x) = \begin{cases} x & \text{si } x > 0 \\ \alpha x & \text{si } x \leq 0 \end{cases}$ (typiquement $\alpha = 0.01$)

**Dérivée :** $f'(x) = \begin{cases} 1 & \text{si } x > 0 \\ \alpha & \text{si } x \leq 0 \end{cases}$

**Propriétés :**
- Sortie : `(-∞, +∞)`
- Petite pente pour valeurs négatives

**Avantages :**
- ✅ Résout le problème des neurones morts de ReLU
- ✅ Rapide comme ReLU
- ✅ Garde un gradient pour valeurs négatives

**Inconvénients :**
- ❌ Résultats incohérents selon les tâches
- ❌ Nécessite un hyperparamètre (alpha)

**Implémentation Rust :**
```rust
fn leaky_relu(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|x| if x > 0.0 { x } else { alpha * x })
}

fn leaky_relu_derivative(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|x| if x > 0.0 { 1.0 } else { alpha })
}
```

---

### Tanh (Tangente Hyperbolique)
**Formule :** $f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

**Dérivée :** $f'(x) = 1 - f(x)^2$

**Propriétés :**
- Sortie : `[-1, 1]`
- **Centrée sur zéro** (contrairement à sigmoid)
- Version "étendue" de sigmoid

**Avantages :**
- ✅ Sortie centrée → convergence plus rapide que sigmoid
- ✅ Gradient plus fort que sigmoid
- ✅ Bon pour les couches cachées

**Inconvénients :**
- ❌ Problème du gradient qui disparaît (moins que sigmoid)
- ❌ Coûteux en calcul

**Implémentation Rust :**
```rust
fn tanh(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x.tanh())
}

fn tanh_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| 1.0 - x.powi(2))
}
```

---

### ELU (Exponential Linear Unit)
**Formule :** $f(x) = \begin{cases} x & \text{si } x > 0 \\ \alpha(e^x - 1) & \text{si } x \leq 0 \end{cases}$ (typiquement $\alpha = 1.0$)

**Dérivée :** $f'(x) = \begin{cases} 1 & \text{si } x > 0 \\ f(x) + \alpha & \text{si } x \leq 0 \end{cases}$

**Propriétés :**
- Sortie : `(-α, +∞)`
- Lisse partout

**Avantages :**
- ✅ Moyenne des activations proche de zéro
- ✅ Pas de neurones morts
- ✅ Gradient non-nul partout

**Inconvénients :**
- ❌ Coûteux (`exp()`)
- ❌ Légèrement plus lent que ReLU

**Implémentation Rust :**
```rust
fn elu(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
}

fn elu_derivative(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|x| if x > 0.0 { 1.0 } else { alpha * x.exp() })
}
```

---

### Softmax (pour classification multi-classes)
**Formule :** $f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$

**Propriétés :**
- Sortie : `[0, 1]` pour chaque neurone, somme = 1
- Convertit logits en probabilités
- **Uniquement pour la couche de sortie**

**Avantages :**
- ✅ Interprétation probabiliste claire
- ✅ Standard pour classification multi-classes

**Implémentation Rust :**
```rust
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|x| (x - max).exp());
    let sum = exp_x.sum();
    exp_x / sum
}
```

---

## Fonctions d'Activation Avancées

### PReLU (Parametric ReLU)
**Formule :** $f(x) = \begin{cases} x & \text{si } x > 0 \\ \alpha x & \text{si } x \leq 0 \end{cases}$ où $\alpha$ est **appris** pendant l'entraînement

**Dérivée :** $f'(x) = \begin{cases} 1 & \text{si } x > 0 \\ \alpha & \text{si } x \leq 0 \end{cases}$

**Avantages :**
- ✅ Alpha adaptatif par neurone
- ✅ Plus flexible que Leaky ReLU

**Inconvénients :**
- ❌ Plus de paramètres à entraîner
- ❌ Risque de surapprentissage

**Implémentation Rust :**
```rust
fn prelu(x: &Array1<f64>, alpha: &Array1<f64>) -> Array1<f64> {
    x.iter().zip(alpha.iter())
        .map(|(&x, &a)| if x > 0.0 { x } else { a * x })
        .collect()
}
```

---

### GELU (Gaussian Error Linear Unit)
**Formule :** $f(x) = x \cdot \Phi(x)$ où $\Phi$ est la fonction de distribution cumulative gaussienne

**Approximation :** $f(x) \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$

**Propriétés :**
- Lisse et non-monotone
- **Utilisé dans BERT, GPT**

**Avantages :**
- ✅ Performance SOTA sur transformers
- ✅ Lisse partout
- ✅ Probabilistiquement motivé

**Inconvénients :**
- ❌ Coûteux en calcul

**Implémentation Rust :**
```rust
fn gelu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| {
        0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() 
            * (x + 0.044715 * x.powi(3))).tanh())
    })
}
```

---

### Swish / SiLU (Sigmoid Linear Unit)
**Formule :** $f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$

**Dérivée :** $f'(x) = f(x) + \sigma(x)(1 - f(x))$

**Propriétés :**
- Lisse, non-monotone
- **Découvert par Google via recherche automatique**

**Avantages :**
- ✅ Meilleure performance que ReLU sur certaines tâches
- ✅ Lisse partout

**Inconvénients :**
- ❌ Plus coûteux que ReLU

**Implémentation Rust :**
```rust
fn swish(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x / (1.0 + (-x).exp()))
}

fn swish_derivative(x: &Array1<f64>) -> Array1<f64> {
    let sigmoid = x.mapv(|x| 1.0 / (1.0 + (-x).exp()));
    let swish = x * &sigmoid;
    &swish + &sigmoid * &(1.0 - &swish)
}
```

---

### Mish
**Formule :** $f(x) = x \cdot \tanh(\ln(1 + e^x)) = x \cdot \tanh(\text{softplus}(x))$

**Propriétés :**
- Lisse, non-monotone
- **Alternatives récente à Swish**

**Avantages :**
- ✅ Meilleure régularisation que ReLU/Swish
- ✅ Gradient non-nul pour valeurs négatives

**Inconvénients :**
- ❌ Très coûteux en calcul

**Implémentation Rust :**
```rust
fn mish(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x * ((1.0 + x.exp()).ln()).tanh())
}
```

---

### SELU (Scaled ELU)
**Formule :** $f(x) = \lambda \begin{cases} x & \text{si } x > 0 \\ \alpha(e^x - 1) & \text{si } x \leq 0 \end{cases}$

**Constantes :** $\lambda \approx 1.0507$, $\alpha \approx 1.6733$

**Propriétés :**
- Auto-normalisant (préserve moyenne=0, variance=1)
- **Conçu pour FeedForward Networks**

**Avantages :**
- ✅ Pas besoin de Batch Normalization
- ✅ Convergence plus rapide

**Inconvénients :**
- ❌ Sensible à l'initialisation (utiliser LeCun)
- ❌ Fonctionne mal avec Dropout

**Implémentation Rust :**
```rust
fn selu(x: &Array1<f64>) -> Array1<f64> {
    let lambda = 1.0507;
    let alpha = 1.6733;
    x.mapv(|x| {
        lambda * if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
    })
}
```

---

### Softplus
**Formule :** $f(x) = \ln(1 + e^x)$

**Dérivée :** $f'(x) = \frac{1}{1 + e^{-x}} = \sigma(x)$ (sigmoid!)

**Propriétés :**
- Version lisse de ReLU
- Toujours positif

**Avantages :**
- ✅ Différentiable partout
- ✅ Pas de neurones morts

**Inconvénients :**
- ❌ Coûteux (`exp`, `log`)
- ❌ Gradient qui disparaît pour grandes valeurs négatives

**Implémentation Rust :**
```rust
fn softplus(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| (1.0 + x.exp()).ln())
}

fn softplus_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| 1.0 / (1.0 + (-x).exp())) // sigmoid
}
```

---

### Softsign
**Formule :** $f(x) = \frac{x}{1 + |x|}$

**Dérivée :** $f'(x) = \frac{1}{(1 + |x|)^2}$

**Propriétés :**
- Sortie : `(-1, 1)`
- Alternative à tanh

**Avantages :**
- ✅ Plus rapide que tanh (pas d'exponentielle)
- ✅ Gradient décroît plus lentement

**Inconvénients :**
- ❌ Rarement utilisé en pratique

**Implémentation Rust :**
```rust
fn softsign(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x / (1.0 + x.abs()))
}

fn softsign_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| 1.0 / (1.0 + x.abs()).powi(2))
}
```

---

### Hard Sigmoid
**Formule :** $f(x) = \max(0, \min(1, 0.2x + 0.5))$

**Propriétés :**
- Approximation linéaire par morceaux de sigmoid
- Très rapide

**Avantages :**
- ✅ Calcul extrêmement rapide (pas d'exponentielle)
- ✅ Utile pour les appareils embarqués

**Implémentation Rust :**
```rust
fn hard_sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| (0.2 * x + 0.5).max(0.0).min(1.0))
}
```

---

### Hard Tanh
**Formule :** $f(x) = \max(-1, \min(1, x))$

**Propriétés :**
- Approximation linéaire par morceaux de tanh
- Sortie : `[-1, 1]`

**Avantages :**
- ✅ Très rapide

**Implémentation Rust :**
```rust
fn hard_tanh(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x.max(-1.0).min(1.0))
}
```

---

## Tableau Comparatif Complet

| **Fonction** | **Plage** | **Vitesse** | **Usage Principal** | **Depuis** |
|--------------|-----------|-------------|---------------------|------------|
| Sigmoid | [0, 1] | Lent | Sortie binaire | Classique |
| Tanh | [-1, 1] | Lent | Couches cachées | Classique |
| ReLU | [0, ∞) | **Très rapide** | Couches cachées (défaut) | 2010 |
| Leaky ReLU | (-∞, ∞) | **Très rapide** | Fix neurones morts | 2013 |
| PReLU | (-∞, ∞) | Rapide | Amélioration LeakyReLU | 2015 |
| ELU | (-α, ∞) | Moyen | Réseaux profonds | 2015 |
| SELU | (-λα, ∞) | Moyen | FeedForward (sans BN) | 2017 |
| Swish/SiLU | (-∞, ∞) | Moyen | Alternative ReLU | 2017 |
| GELU | (-∞, ∞) | Lent | **Transformers (GPT, BERT)** | 2016 |
| Mish | (-∞, ∞) | Lent | Vision profonde | 2019 |
| Softmax | [0, 1] (somme=1) | Moyen | Sortie multi-classe | Classique |
| Softplus | (0, ∞) | Lent | ReLU lisse | Classique |
| Hard Sigmoid | [0, 1] | **Très rapide** | Embarqué | Mobile |
| Hard Tanh | [-1, 1] | **Très rapide** | Embarqué | Mobile |

---

## Guide de Sélection

### Par Cas d'Usage

| **Cas d'Usage** | **Fonction Recommandée** | **Raison** |
|-----------------|--------------------------|------------|
| **Couches cachées (défaut 2024)** | **ReLU** | Rapide, efficace, standard industriel |
| Couches cachées (si neurones morts) | **Leaky ReLU** ou **ELU** | Gradient toujours actif |
| Couches cachées (réseaux profonds) | **SELU** ou **ELU** | Auto-normalisation, évite gradient qui disparaît |
| Couches cachées (recherche de performance) | **Swish** ou **Mish** | Performance SOTA sur certaines tâches |
| **Transformers / NLP (GPT, BERT)** | **GELU** | Standard pour attention mechanisms |
| **Vision par ordinateur (CNN)** | **ReLU** ou **Mish** | Rapide pour CNN, Mish pour profonds |
| Réseaux récurrents (RNN, LSTM) | **Tanh** | Standard historique pour gates |
| **Sortie classification binaire** | **Sigmoid** | Sortie [0,1] = probabilité |
| **Sortie classification multi-classes** | **Softmax** | Distribution de probabilités (somme=1) |
| **Sortie régression** | **Linéaire** (aucune) | Valeurs continues illimitées |
| Sortie régression (valeurs positives) | **Softplus** ou **ReLU** | Force sortie ≥ 0 |
| **Appareils embarqués / Mobile** | **Hard Sigmoid** / **Hard Tanh** | Pas d'exponentielle, ultra-rapide |
| Recherche / Expérimentation | **PReLU** | Alpha adaptatif par neurone |

### Par Priorité

#### 🏆 **Si tu veux la meilleure performance (sans contrainte)** :
1. **Couches cachées** : GELU, Swish, Mish
2. **Sortie** : Softmax (multi-classe), Sigmoid (binaire)

#### ⚡ **Si tu veux la rapidité (contrainte temps réel)** :
1. **Couches cachées** : ReLU, Leaky ReLU
2. **Embarqué** : Hard Sigmoid, Hard Tanh

#### 🎯 **Si tu veux la stabilité (réseaux très profonds)** :
1. **Couches cachées** : SELU (avec initialisation LeCun), ELU
2. **Éviter** : Sigmoid, Tanh (gradient qui disparaît)

#### 🔧 **Si tu débutes / prototype rapide** :
1. **Défaut recommandé** : ReLU partout sauf sortie
2. **Sortie** : Sigmoid (binaire), Softmax (multi-classe)

### Par Type de Réseau

| **Architecture** | **Couches Cachées** | **Sortie** |
|------------------|---------------------|------------|
| **Feedforward simple** | ReLU | Sigmoid / Softmax |
| **Feedforward profond** | SELU, ELU | Sigmoid / Softmax |
| **CNN (Computer Vision)** | ReLU, Mish | Softmax |
| **RNN / LSTM** | Tanh | Sigmoid / Softmax |
| **Transformer** | GELU | Softmax |
| **GAN (Générateur)** | ReLU, Leaky ReLU | Tanh |
| **GAN (Discriminateur)** | Leaky ReLU | Sigmoid |
| **Autoencoder** | ReLU | Sigmoid (binaire), Linéaire (continu) |
| **Reinforcement Learning** | ReLU, ELU | Linéaire, Softmax |

### Arbre de Décision

```
Quelle est ta couche ?
├─ Couche de SORTIE
│  ├─ Classification binaire ? → Sigmoid
│  ├─ Classification multi-classes ? → Softmax
│  ├─ Régression (valeurs continues) ? → Linéaire (aucune activation)
│  └─ Régression (valeurs positives) ? → Softplus / ReLU
│
└─ Couche CACHÉE
   ├─ Contrainte de VITESSE ?
   │  ├─ Ultra-rapide (embarqué) ? → Hard Sigmoid / Hard Tanh
   │  └─ Rapide → ReLU, Leaky ReLU
   │
   ├─ Type de RÉSEAU ?
   │  ├─ Transformer / NLP ? → GELU
   │  ├─ CNN profond ? → Mish
   │  ├─ RNN / LSTM ? → Tanh
   │  └─ Feedforward ? → Voir ci-dessous
   │
   ├─ Profondeur du RÉSEAU ?
   │  ├─ Peu de couches (< 5) ? → ReLU
   │  ├─ Profond (> 10 couches) ? → SELU, ELU
   │  └─ Très profond (> 50) ? → SELU avec LeCun init
   │
   ├─ Problème de NEURONES MORTS (gradient = 0) ?
   │  ├─ Oui → Leaky ReLU, PReLU, ELU
   │  └─ Non → ReLU
   │
   └─ Recherche de PERFORMANCE maximale ?
      ├─ Oui (GPU puissant) → Swish, Mish, GELU
      └─ Non → ReLU (défaut)
```

### Recommandations par Année

| **Époque** | **Standard** | **Contexte** |
|------------|--------------|--------------|
| 1990-2010 | Sigmoid, Tanh | Réseaux peu profonds |
| 2010-2015 | ReLU | Révolution deep learning |
| 2015-2017 | Leaky ReLU, ELU, PReLU | Amélioration ReLU |
| 2017-2019 | Swish, SELU | Auto-recherche Google |
| 2019-2024 | **GELU** (transformers), **Mish** (vision) | SOTA actuel |
| 2024+ | **GELU** (défaut NLP), **ReLU** (défaut vision) | Standard industriel |

### Combinaisons Éprouvées

**Classification d'images (CNN) :**
```rust
// Couches conv : ReLU ou Mish
// Couches fully-connected : ReLU
// Sortie : Softmax
```

**Modèle de langage (Transformer) :**
```rust
// Attention + FFN : GELU
// Sortie : Softmax
```

**Réseau profond (> 20 couches) :**
```rust
// Toutes couches cachées : SELU
// Initialisation : LeCun normal
// PAS de Batch Normalization
// Sortie : Sigmoid / Softmax
```

**Prototype rapide :**
```rust
// Couches cachées : ReLU
// Sortie : Sigmoid (binaire) ou Softmax (multi-classe)
```

### 3. Learning Rate (Taux d'apprentissage)

**Vitesse d'apprentissage** : à quel point modifier les poids à chaque étape.

- Actuellement : `0.1`
- **Trop petit** → apprentissage lent
- **Trop grand** → instabilité, ne converge pas
- **Typique** : `0.001` à `0.1`

---

## Fonctions de Perte (Loss Functions)

### Concept de Base

La **loss function** (fonction de perte/coût) mesure **à quel point le réseau se trompe** dans ses prédictions.

**Objectif :** Minimiser l'erreur entre la prédiction et la valeur réelle.

```
Loss = Différence(Prédiction, Valeur_Réelle)
```

Plus la loss est **petite** → meilleure prédiction  
Plus la loss est **grande** → pire prédiction

### Cycle d'apprentissage

```
1. Forward pass → Prédiction
2. Calcul de la Loss → Mesurer l'erreur
3. Backpropagation → Calculer les gradients
4. Update des poids → Réduire la Loss
```

---

### 1. MSE (Mean Squared Error)

**Formule :** $\text{MSE} = \frac{1}{n}\sum(y - \hat{y})^2$

**Usage :** Régression (prédire des valeurs continues)

**Avantages :**
- ✅ Pénalise fortement les grandes erreurs
- ✅ Différentiable partout
- ✅ Interprétation intuitive

**Inconvénients :**
- ❌ Pas optimal pour classification
- ❌ Gradient qui disparaît avec Sigmoid

**Exemple :**
```
Prédiction: 2.5, Réel: 3.0
Loss = (3.0 - 2.5)² = 0.25
```

**Implémentation Rust :**
```rust
fn mse(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let diff = predictions - targets;
    (&diff * &diff).sum() / predictions.len() as f64
}
```

---

### 2. MAE (Mean Absolute Error)

**Formule :** $\text{MAE} = \frac{1}{n}\sum|y - \hat{y}|$

**Usage :** Régression (moins sensible aux outliers)

**Avantages :**
- ✅ Robuste aux outliers
- ✅ Interprétation intuitive
- ✅ Toutes les erreurs traitées linéairement

**Inconvénients :**
- ❌ Gradients constants (convergence plus lente)
- ❌ Non différentiable en zéro

**Exemple :**
```
Prédiction: 2.5, Réel: 3.0
Loss = |3.0 - 2.5| = 0.5
```

**Implémentation Rust :**
```rust
fn mae(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    (predictions - targets).mapv(|x| x.abs()).sum() / predictions.len() as f64
}
```

---

### 3. Binary Cross-Entropy (Log Loss)

**Formule :** $\text{BCE} = -\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$

**Usage :** Classification binaire (avec Sigmoid)

**Avantages :**
- ✅ Interprétation probabiliste
- ✅ Gradient plus stable que MSE pour classification
- ✅ Convergence plus rapide
- ✅ Standard pour classification binaire

**Inconvénients :**
- ❌ Nécessite prédictions dans [0, 1]
- ❌ Instable si prédiction = 0 ou 1 (log(0))

**Exemple :**
```
Prédiction: 0.9, Réel: 1 (classe positive)
Loss = -[1×log(0.9) + 0×log(0.1)] = 0.105  // Bonne prédiction

Prédiction: 0.1, Réel: 1 (classe positive)
Loss = -[1×log(0.1) + 0×log(0.9)] = 2.303  // Grosse erreur!
```

**Implémentation Rust :**
```rust
fn binary_cross_entropy(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let epsilon = 1e-15; // Éviter log(0)
    let mut sum = 0.0;
    for (p, t) in predictions.iter().zip(targets.iter()) {
        let p_clamped = p.max(epsilon).min(1.0 - epsilon);
        sum += -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln());
    }
    sum / predictions.len() as f64
}
```

---

### 4. Categorical Cross-Entropy

**Formule :** $\text{CCE} = -\sum y_i \log(\hat{y}_i)$

**Usage :** Classification multi-classes (avec Softmax)

**Avantages :**
- ✅ Standard pour multi-classes
- ✅ Interprétation probabiliste claire
- ✅ Gradient bien adapté avec Softmax

**Exemple :**
```
Classes: [Chat, Chien, Oiseau]
Réel:    [1,    0,     0]      // C'est un chat
Prédit:  [0.7,  0.2,   0.1]
Loss = -(1×log(0.7) + 0×log(0.2) + 0×log(0.1)) = 0.357
```

**Implémentation Rust :**
```rust
fn categorical_cross_entropy(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let epsilon = 1e-15;
    -targets.iter()
        .zip(predictions.iter())
        .map(|(t, p)| t * (p.max(epsilon)).ln())
        .sum::<f64>()
}
```

---

### 5. Huber Loss

**Formule :** 
$$\text{Huber} = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{si } |y - \hat{y}| \leq \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{sinon} \end{cases}$$

**Usage :** Régression robuste aux outliers

**Avantages :**
- ✅ Combine MSE (petites erreurs) et MAE (grandes erreurs)
- ✅ Moins sensible aux outliers que MSE
- ✅ Différentiable partout

**Paramètre :** $\delta$ (typiquement 1.0) = seuil entre comportement MSE et MAE

**Implémentation Rust :**
```rust
fn huber_loss(predictions: &Array1<f64>, targets: &Array1<f64>, delta: f64) -> f64 {
    let diff = predictions - targets;
    let mut sum = 0.0;
    for &d in diff.iter() {
        let abs_d = d.abs();
        if abs_d <= delta {
            sum += 0.5 * d * d;  // MSE pour petites erreurs
        } else {
            sum += delta * (abs_d - 0.5 * delta);  // MAE pour grandes erreurs
        }
    }
    sum / predictions.len() as f64
}
```

---

### Guide de Sélection des Loss Functions

| **Tâche** | **Activation Sortie** | **Loss Function Recommandée** | **Pourquoi** |
|-----------|----------------------|-------------------------------|--------------|
| Régression | Linear | **MSE** | Standard, pénalise grandes erreurs |
| Régression robuste | Linear | **MAE** ou **Huber** | Résiste aux outliers |
| Classification binaire | Sigmoid | **Binary Cross-Entropy** | Interprétation probabiliste |
| Classification multi-classes | Softmax | **Categorical Cross-Entropy** | Standard multi-classes |
| Détection d'objets | Variable | IoU Loss, Focal Loss | Adapté aux boîtes |
| Segmentation | Softmax | Dice Loss, Focal Loss | Adapté aux pixels |

---

### Comparaison MSE vs Binary Cross-Entropy (XOR)

**Problème :** Classification binaire avec Sigmoid

#### MSE pour classification
- ❌ Gradient qui disparaît quand proche de 0 ou 1
- ❌ Pas d'interprétation probabiliste
- ❌ Convergence plus lente

#### Binary Cross-Entropy pour classification
- ✅ Gradient plus stable
- ✅ Converge plus vite
- ✅ Interprétation comme probabilité
- ✅ Meilleur choix pour XOR

**Résultats typiques (50k epochs, lr=0.5) :**
```
MSE:  Final loss: 0.0000 ✓
BCE:  Final loss: 0.0000 ✓
MAE:  Final loss: 0.2500 (nécessite lr=0.2, epochs=150k)
```

---

### Visualisation de la Convergence

```
Haute Loss ━━━━━━━━━┓
                    ┃    Début
                    ┃      ↓
                    ┃      •
                    ┃     ╱
                    ┃    ╱
                    ┃   ╱     Training
                    ┃  ╱      ↓
                    ┃ ╱
Basse Loss ━━━━━━━━━┃╱________• Convergence
                    └────────────────→
                         Epochs
```

**Objectif du training :** Descendre cette courbe le plus vite possible en ajustant les poids.

---

## Documentation Recommandée

1. **[3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)** (YouTube) : visualisations excellentes
2. **[The Rust ML Book](https://rust-ml.github.io/book/)** : apprentissage automatique en Rust
3. **[ndarray docs](https://docs.rs/ndarray/latest/ndarray/)** : documentation de la bibliothèque
4. **Neural Networks from Scratch** (livre) : explications mathématiques détaillées
5. **[ML Cheatsheet - Loss Functions](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)** : référence complète

## Expérimentation

### Architecture
```rust
Network::new(2, 5, 1)   // 5 neurones cachés
Network::new(2, 10, 1)  // 10 neurones cachés
```

### Learning Rate
```rust
let learning_rate = 0.01;  // Plus lent
let learning_rate = 0.5;   // Plus rapide
let learning_rate = 1.0;   // Très rapide (attention à la stabilité)
```

### Fonction d'Activation et Loss
```rust
// Classification binaire (XOR)
Network::new(2, 5, 1, 
    Activation::Tanh,           // Couche cachée
    Activation::Sigmoid,        // Sortie
    LossFunction::BinaryCrossEntropy)

// Régression
Network::new(4, 10, 1,
    Activation::ReLU,           // Couche cachée
    Activation::Linear,         // Sortie
    LossFunction::MSE)

// Multi-classes
Network::new(784, 128, 10,
    Activation::GELU,           // Couche cachée
    Activation::Softmax,        // Sortie
    LossFunction::CategoricalCrossEntropy)
```
