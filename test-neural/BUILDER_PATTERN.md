# 🏗️ Builder Pattern Guide

Le Builder Pattern est l'API recommandée pour construire et entraîner des réseaux de neurones avec `test-neural`. Il offre une interface fluide et intuitive qui remplace les multiples méthodes de construction traditionnelles.

## Table des matières

- [Pourquoi le Builder Pattern ?](#pourquoi-le-builder-pattern-)
- [NetworkBuilder](#networkbuilder)
- [TrainingBuilder](#trainingbuilder)
- [Exemples complets](#exemples-complets)
- [Comparaison avec l'API traditionnelle](#comparaison-avec-lapi-traditionnelle)

---

## Pourquoi le Builder Pattern ?

### Problèmes résolus

❌ **Avant** - Prolifération de méthodes:
- `Network::new()` - réseau simple (1 couche cachée)
- `Network::new_deep()` - réseau profond (init auto)
- `Network::new_deep_with_init()` - réseau profond (init manuelle)
- `Network::fit()` - entraînement avec callbacks
- `Network::fit_with_scheduler()` - entraînement avec scheduler
- Gestion manuelle de `Vec<Box<dyn Callback>>`
- Confusion sur quelle méthode utiliser

✅ **Après** - Une seule manière:
- `NetworkBuilder` - construction intuitive par chaînage
- `.trainer()` - entraînement unifié
- Plus besoin de gérer les Vec manuellement
- API auto-documentée

---

## NetworkBuilder

### Construction simple

```rust
use test_neural::builder::NetworkBuilder;
use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;

let network = NetworkBuilder::new(2, 1)  // input_size, output_size
    .hidden_layer(8, Activation::Tanh)   // 1 couche cachée
    .build();
```

### Réseau profond

```rust
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(16, Activation::ReLU)
    .hidden_layer(8, Activation::ReLU)
    .hidden_layer(4, Activation::Tanh)
    .build();
```

### Configuration complète

```rust
let network = NetworkBuilder::new(input_size, output_size)
    // Couches cachées
    .hidden_layer(64, Activation::ReLU)
    .hidden_layer(32, Activation::ReLU)
    .hidden_layer(16, Activation::Tanh)
    
    // Sortie
    .output_activation(Activation::Sigmoid)  // défaut: Sigmoid
    .loss(LossFunction::BinaryCrossEntropy)  // défaut: BCE
    
    // Optimizer
    .optimizer(OptimizerType::adam(0.001))   // défaut: Adam(0.001)
    
    // Régularisation
    .dropout(0.3)           // appliqué aux couches cachées
    .l2(0.01)               // L2 regularization
    
    // Initialisation optionnelle
    .weight_init(WeightInit::He)  // sinon: auto selon activation
    
    .build();
```

### Options de régularisation

```rust
// L1 (Lasso) - encourage la sparsité
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(8, Activation::Tanh)
    .l1(0.001)
    .build();

// L2 (Ridge) - pénalise les grands poids
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(8, Activation::Tanh)
    .l2(0.01)
    .build();

// Elastic Net - combine L1 et L2
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(8, Activation::Tanh)
    .elastic_net(0.5, 0.01)  // l1_ratio=0.5, lambda=0.01
    .build();

// Dropout + L2 (recommandé)
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(16, Activation::ReLU)
    .hidden_layer(8, Activation::ReLU)
    .dropout(0.3)
    .l2(0.001)
    .build();
```

### Valeurs par défaut

Si vous n'spécifiez pas certaines options, les valeurs par défaut sont:
- `output_activation`: `Activation::Sigmoid`
- `loss`: `LossFunction::BinaryCrossEntropy`
- `optimizer`: `OptimizerType::adam(0.001)`
- `weight_init`: Auto-détection selon l'activation
- `dropout`: Aucun
- `regularization`: Aucune

---

## TrainingBuilder

### Entraînement simple

```rust
use test_neural::builder::NetworkTrainer;  // Trait pour .trainer()

let history = network.trainer()
    .train_data(&train_dataset)
    .epochs(100)
    .fit();
```

### Avec validation

```rust
let history = network.trainer()
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    .epochs(100)
    .batch_size(32)
    .fit();
```

### Avec callbacks

```rust
use test_neural::callbacks::{EarlyStopping, ModelCheckpoint, ProgressBar};

let history = network.trainer()
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    .epochs(200)
    .batch_size(32)
    .callback(Box::new(EarlyStopping::new(10, 0.0001)))
    .callback(Box::new(ModelCheckpoint::new("best_model.json", true)))
    .callback(Box::new(ProgressBar::new(200)))
    .fit();
```

### Avec Learning Rate Scheduler

```rust
use test_neural::callbacks::{LearningRateScheduler, LRSchedule};

// StepLR: réduit le LR tous les N epochs
let history = network.trainer()
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    .epochs(100)
    .batch_size(32)
    .scheduler(LearningRateScheduler::new(
        LRSchedule::StepLR { 
            step_size: 30,  // tous les 30 epochs
            gamma: 0.1      // multiplier par 0.1
        }
    ))
    .fit();

// ReduceOnPlateau: réduit le LR quand loss stagne (recommandé!)
let history = network.trainer()
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    .epochs(100)
    .batch_size(32)
    .scheduler(LearningRateScheduler::new(
        LRSchedule::ReduceOnPlateau { 
            patience: 10,          // attendre 10 epochs
            factor: 0.5,           // diviser par 2
            min_delta: 0.0001     // amélioration minimale
        }
    ))
    .fit();

// ExponentialLR: décroissance exponentielle
let history = network.trainer()
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    .epochs(100)
    .batch_size(32)
    .scheduler(LearningRateScheduler::new(
        LRSchedule::ExponentialLR { gamma: 0.95 }
    ))
    .fit();
```

### Configuration complète (tout combiné)

```rust
let history = network.trainer()
    // Données
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    
    // Hyperparamètres
    .epochs(200)
    .batch_size(32)
    
    // Learning rate scheduling
    .scheduler(LearningRateScheduler::new(
        LRSchedule::ReduceOnPlateau { 
            patience: 10, 
            factor: 0.5, 
            min_delta: 0.0001 
        }
    ))
    
    // Callbacks (dans l'ordre d'exécution)
    .callback(Box::new(ProgressBar::new(200)))
    .callback(Box::new(ModelCheckpoint::new("best_model.json", true)))
    .callback(Box::new(EarlyStopping::new(20, 0.00001)))
    
    .fit();

// history contient (train_loss, val_loss) pour chaque epoch
println!("Loss finale: {:.6}", history.last().unwrap().1.unwrap());
```

---

## Exemples complets

### Exemple 1: Classification binaire (XOR)

```rust
use test_neural::builder::{NetworkBuilder, NetworkTrainer};
use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::dataset::Dataset;
use test_neural::callbacks::{EarlyStopping, ModelCheckpoint};
use ndarray::array;

fn main() {
    // Données XOR
    let inputs = vec![
        array![0.0, 0.0],
        array![0.0, 1.0],
        array![1.0, 0.0],
        array![1.0, 1.0],
    ];
    
    let targets = vec![
        array![0.0],
        array![1.0],
        array![1.0],
        array![0.0],
    ];
    
    let dataset = Dataset::new(inputs.clone(), targets.clone());
    let (train, val) = dataset.split(0.75);
    
    // Construction du réseau
    let mut network = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    
    // Entraînement
    let history = network.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(1000)
        .batch_size(2)
        .callback(Box::new(EarlyStopping::new(50, 0.0001)))
        .callback(Box::new(ModelCheckpoint::new("best_xor.json", true)))
        .fit();
    
    // Prédictions
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let prediction = network.predict(input);
        println!("[{:.0}, {:.0}] → {:.3} (attendu {:.0})", 
            input[0], input[1], prediction[0], target[0]);
    }
}
```

### Exemple 2: Réseau profond avec régularisation

```rust
use test_neural::builder::{NetworkBuilder, NetworkTrainer};
use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::callbacks::{LearningRateScheduler, LRSchedule, ProgressBar};

fn main() {
    let mut network = NetworkBuilder::new(784, 10)  // MNIST-like
        .hidden_layer(128, Activation::ReLU)
        .hidden_layer(64, Activation::ReLU)
        .hidden_layer(32, Activation::ReLU)
        .output_activation(Activation::Softmax)
        .loss(LossFunction::CategoricalCrossEntropy)
        .optimizer(OptimizerType::adam(0.001))
        .dropout(0.3)
        .l2(0.0001)
        .build();
    
    let history = network.trainer()
        .train_data(&train_dataset)
        .validation_data(&val_dataset)
        .epochs(100)
        .batch_size(64)
        .scheduler(LearningRateScheduler::new(
            LRSchedule::ReduceOnPlateau { 
                patience: 5, 
                factor: 0.5, 
                min_delta: 0.001 
            }
        ))
        .callback(Box::new(ProgressBar::new(100)))
        .fit();
    
    println!("Entraînement terminé: {} epochs", history.len());
}
```

---

## Comparaison avec l'API traditionnelle

### Construction

**Traditionnelle**:
```rust
// Simple
let network = Network::new(
    2, 8, 1,
    Activation::Tanh,
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.01)
);

// Profond
let network = Network::new_deep(
    2,
    vec![16, 8, 4],           // Vec<usize>
    1,
    vec![Activation::ReLU, Activation::ReLU, Activation::Tanh],  // Vec<Activation>
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.001)
);
```

**Builder**:
```rust
// Simple
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(8, Activation::Tanh)
    .build();

// Profond
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(16, Activation::ReLU)
    .hidden_layer(8, Activation::ReLU)
    .hidden_layer(4, Activation::Tanh)
    .build();
```

### Entraînement

**Traditionnelle**:
```rust
// Sans scheduler
let mut callbacks: Vec<Box<dyn Callback>> = vec![
    Box::new(EarlyStopping::new(10, 0.0001)),
];
let history = network.fit(&train, Some(&val), 100, 32, &mut callbacks);

// Avec scheduler
let mut scheduler = LearningRateScheduler::new(...);
let mut callbacks: Vec<Box<dyn Callback>> = vec![...];
let history = network.fit_with_scheduler(
    &train, Some(&val), 100, 32, &mut scheduler, &mut callbacks
);
```

**Builder**:
```rust
// Tout unifié
let history = network.trainer()
    .train_data(&train)
    .validation_data(&val)
    .epochs(100)
    .batch_size(32)
    .callback(Box::new(EarlyStopping::new(10, 0.0001)))
    .scheduler(LearningRateScheduler::new(...))
    .fit();
```

---

## Conclusion

Le Builder Pattern offre:

✅ **API intuitive** - Code auto-documenté  
✅ **Moins d'erreurs** - Plus de Vec à gérer  
✅ **Flexibilité** - Combinez n'importe quelles options  
✅ **Unification** - Une seule manière de faire  
✅ **Évolutivité** - Facile d'ajouter de nouvelles options  
✅ **Backward compatible** - L'ancienne API reste disponible

🚀 **Commencez ici**: `cargo run --example builder_showcase`
