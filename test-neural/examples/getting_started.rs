//! Getting Started - Exemple complet de la bibliothèque
//!
//! Cet exemple montre toutes les fonctionnalités principales:
//! - Construction de réseaux avec le Builder Pattern
//! - Différents optimiseurs (SGD, Adam, etc.)
//! - Régularisation (Dropout, L2)
//! - Callbacks (EarlyStopping, ModelCheckpoint, LR Scheduler)
//! - Évaluation avec métriques

use test_neural::builder::{NetworkBuilder, NetworkTrainer};
use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::dataset::Dataset;
use test_neural::callbacks::{EarlyStopping, ModelCheckpoint, LearningRateScheduler, LRSchedule, ProgressBar};
use test_neural::metrics::{accuracy, binary_metrics};
use ndarray::array;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Test Neural - Getting Started Guide                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════════════
    // 1. PRÉPARATION DES DONNÉES
    // ═══════════════════════════════════════════════════════════════════════
    println!("📦 1. Préparation des données (XOR problem)\n");
    
    // Créer un dataset XOR étendu pour l'entraînement
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for _ in 0..100 {
        inputs.push(array![0.0, 0.0]); targets.push(array![0.0]);
        inputs.push(array![0.0, 1.0]); targets.push(array![1.0]);
        inputs.push(array![1.0, 0.0]); targets.push(array![1.0]);
        inputs.push(array![1.0, 1.0]); targets.push(array![0.0]);
    }
    
    let dataset = Dataset::new(inputs, targets);
    let (train, val) = dataset.split(0.8);
    
    println!("   Train: {} exemples | Validation: {} exemples\n", train.len(), val.len());

    // ═══════════════════════════════════════════════════════════════════════
    // 2. CONSTRUCTION D'UN RÉSEAU SIMPLE
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 2. Construction d'un réseau avec le Builder Pattern\n");
    
    let network = NetworkBuilder::new(2, 1)          // 2 entrées, 1 sortie
        .hidden_layer(8, Activation::Tanh)           // Couche cachée
        .output_activation(Activation::Sigmoid)      // Sortie binaire
        .loss(LossFunction::BinaryCrossEntropy)      // Classification binaire
        .optimizer(OptimizerType::adam(0.01))        // Adam optimizer
        .build();
    
    println!("   ✓ Réseau créé: 2 → [8] → 1");
    println!("   ✓ Activation: Tanh → Sigmoid");
    println!("   ✓ Optimizer: Adam (lr=0.01)\n");
    drop(network);

    // ═══════════════════════════════════════════════════════════════════════
    // 3. RÉSEAU AVEC RÉGULARISATION
    // ═══════════════════════════════════════════════════════════════════════
    println!("🛡️  3. Réseau avec régularisation (Dropout + L2)\n");
    
    let network_reg = NetworkBuilder::new(2, 1)
        .hidden_layer(16, Activation::ReLU)
        .hidden_layer(8, Activation::ReLU)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.001))
        .dropout(0.2)    // 20% des neurones désactivés pendant training
        .l2(0.001)       // Régularisation L2 (weight decay)
        .build();
    
    println!("   ✓ Architecture: 2 → [16, 8] → 1");
    println!("   ✓ Dropout: 0.2 (prévient l'overfitting)");
    println!("   ✓ L2: 0.001 (pénalise les grands poids)\n");
    drop(network_reg);

    // ═══════════════════════════════════════════════════════════════════════
    // 4. COMPARAISON D'OPTIMISEURS
    // ═══════════════════════════════════════════════════════════════════════
    println!("⚡ 4. Comparaison rapide des optimiseurs\n");
    
    let optimizers = vec![
        ("SGD",      OptimizerType::sgd(0.5)),
        ("Momentum", OptimizerType::momentum(0.1)),
        ("Adam",     OptimizerType::adam(0.01)),
    ];
    
    let test_inputs = vec![
        array![0.0, 0.0], array![0.0, 1.0],
        array![1.0, 0.0], array![1.0, 1.0],
    ];
    let test_targets = vec![
        array![0.0], array![1.0], array![1.0], array![0.0],
    ];
    
    for (name, optimizer) in optimizers {
        let mut net = NetworkBuilder::new(2, 1)
            .hidden_layer(8, Activation::Tanh)
            .output_activation(Activation::Sigmoid)
            .loss(LossFunction::BinaryCrossEntropy)
            .optimizer(optimizer)
            .build();
        
        // Entraînement rapide
        for _ in 0..1000 {
            for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
                net.train(input, target);
            }
        }
        
        let loss = net.evaluate(&test_inputs, &test_targets);
        println!("   {:<10} → Loss finale: {:.6}", name, loss);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // 5. ENTRAÎNEMENT AVEC CALLBACKS
    // ═══════════════════════════════════════════════════════════════════════
    println!("📊 5. Entraînement avec callbacks\n");
    
    let mut network = NetworkBuilder::new(2, 1)
        .hidden_layer(10, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.05))
        .build();
    
    println!("   Configuration:");
    println!("   • EarlyStopping (patience=15)");
    println!("   • ModelCheckpoint (sauvegarde le meilleur)");
    println!("   • LR Scheduler (ReduceOnPlateau)\n");
    
    let history = network.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(100)
        .batch_size(32)
        .callback(Box::new(EarlyStopping::new(15, 0.00001)))
        .callback(Box::new(ModelCheckpoint::new("best_model.json", true)))
        .callback(Box::new(ProgressBar::new(100).set_verbose(false)))
        .scheduler(LearningRateScheduler::new(
            LRSchedule::ReduceOnPlateau { 
                patience: 10, 
                factor: 0.5, 
                min_delta: 0.0001 
            }
        ))
        .fit();
    
    println!("\n   ✓ Entraînement terminé en {} epochs", history.len());
    if let Some((train_loss, val_loss)) = history.last() {
        println!("   ✓ Loss finale - Train: {:.6} | Val: {:.6}", 
            train_loss, val_loss.unwrap_or(0.0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 6. ÉVALUATION ET MÉTRIQUES
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n📈 6. Évaluation et métriques\n");
    
    network.eval_mode();  // Désactive le dropout pour l'inférence
    
    let predictions: Vec<_> = test_inputs.iter()
        .map(|input| network.predict(input))
        .collect();
    
    println!("   Prédictions:");
    for (input, (pred, target)) in test_inputs.iter()
        .zip(predictions.iter().zip(test_targets.iter())) 
    {
        let correct = (pred[0].round() - target[0]).abs() < 0.1;
        println!("   [{:.0}, {:.0}] → {:.3} (attendu {:.0}) {}", 
            input[0], input[1], pred[0], target[0],
            if correct { "✓" } else { "✗" });
    }
    
    let acc = accuracy(&predictions, &test_targets, 0.5);
    let metrics = binary_metrics(&predictions, &test_targets, 0.5);
    
    println!("\n   Métriques:");
    println!("   • Accuracy:  {:.1}%", acc * 100.0);
    println!("   • Precision: {:.3}", metrics.precision);
    println!("   • Recall:    {:.3}", metrics.recall);
    println!("   • F1-Score:  {:.3}", metrics.f1_score);

    // ═══════════════════════════════════════════════════════════════════════
    // RÉSUMÉ
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                        RÉSUMÉ                                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ • NetworkBuilder::new(input, output)                         ║");
    println!("║     .hidden_layer(size, activation)                          ║");
    println!("║     .optimizer(OptimizerType::adam(lr))                      ║");
    println!("║     .dropout(rate).l2(lambda)                                ║");
    println!("║     .build()                                                 ║");
    println!("║                                                              ║");
    println!("║ • network.trainer()                                          ║");
    println!("║     .train_data(&dataset)                                    ║");
    println!("║     .epochs(100).batch_size(32)                              ║");
    println!("║     .callback(Box::new(...))                                 ║");
    println!("║     .fit()                                                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    
    println!("📚 Autres exemples:");
    println!("   cargo run --example serialization   - Save/Load modèles");
    println!("   cargo run --example minibatch_demo  - Mini-batch training");
    println!("   cargo run --example metrics_demo    - Métriques détaillées\n");
    
    // Cleanup
    std::fs::remove_file("best_model.json").ok();
}
