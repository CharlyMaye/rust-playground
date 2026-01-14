/// Demonstration of Callbacks
/// 
/// This example shows:
/// - EarlyStopping: Prevents overfitting by stopping when validation loss plateaus
/// - ModelCheckpoint: Automatically saves the best model
/// - LearningRateScheduler: Adjusts LR dynamically (StepLR, ReduceOnPlateau, ExponentialLR)
/// - ProgressBar: Shows training progress in real-time
/// - Combining multiple callbacks

use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::dataset::Dataset;
use test_neural::builder::{NetworkBuilder, NetworkTrainer};
use test_neural::callbacks::{
    EarlyStopping, ModelCheckpoint, LearningRateScheduler, ProgressBar, 
    LRSchedule
};
use ndarray::array;

fn main() {
    println!("=== Démonstration des Callbacks ===\n");
    
    // Create dataset
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for _ in 0..250 {
        inputs.push(array![0.0, 0.0]);
        targets.push(array![0.0]);
        inputs.push(array![0.0, 1.0]);
        targets.push(array![1.0]);
        inputs.push(array![1.0, 0.0]);
        targets.push(array![1.0]);
        inputs.push(array![1.0, 1.0]);
        targets.push(array![0.0]);
    }
    
    let dataset = Dataset::new(inputs, targets);
    let (train, val) = dataset.split(0.8);
    
    println!("Dataset: {} train, {} val\n", train.len(), val.len());
    
    // ===== 1. Sans callbacks (baseline) =====
    println!("--- 1. Sans Callbacks (Baseline) ---");
    let mut network_baseline = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    
    let history = network_baseline.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(100)
        .batch_size(32)
        .fit();
    
    let final_loss = history.last().unwrap().1.unwrap();
    println!("✓ Loss finale: {:.6}\n", final_loss);
    
    // ===== 2. Avec EarlyStopping =====
    println!("--- 2. Avec EarlyStopping (patience=10) ---");
    let mut network_early = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    
    let history = network_early.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(100)
        .batch_size(32)
        .callback(Box::new(EarlyStopping::new(10, 0.0001)))
        .fit();
    
    println!("✓ Arrêté après {} epochs", history.len());
    println!("✓ Loss finale: {:.6}\n", history.last().unwrap().1.unwrap());
    
    // ===== 3. Avec ModelCheckpoint =====
    println!("--- 3. Avec ModelCheckpoint ---");
    let mut network_checkpoint = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    
    let history = network_checkpoint.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(50)
        .batch_size(32)
        .callback(Box::new(ModelCheckpoint::new("best_xor_model.json", true)))
        .fit();
    
    println!("✓ Entraînement terminé ({} epochs)", history.len());
    println!("✓ Meilleur modèle sauvegardé dans best_xor_model.json\n");
    
    // ===== 4. Avec LearningRateScheduler (StepLR) =====
    println!("--- 4. Avec LearningRateScheduler (StepLR) ---");
    let mut network_step = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.1))
        .build();
    
    let scheduler = LearningRateScheduler::new(
        LRSchedule::StepLR { step_size: 15, gamma: 0.5 }
    );
    
    let history = network_step.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(50)
        .batch_size(32)
        .scheduler(scheduler)
        .fit();
    
    println!("✓ Loss finale: {:.6}", history.last().unwrap().1.unwrap());
    
    // ===== 5. Avec LearningRateScheduler (ReduceOnPlateau) =====
    println!("--- 5. Avec LearningRateScheduler (ReduceOnPlateau) ---");
    let mut network_plateau = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    
    let scheduler = LearningRateScheduler::new(
        LRSchedule::ReduceOnPlateau { 
            patience: 5, 
            factor: 0.5, 
            min_delta: 0.0001 
        }
    );
    
    let history = network_plateau.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(50)
        .batch_size(32)
        .scheduler(scheduler)
        .fit();
    
    println!("✓ Loss finale: {:.6}\n", history.last().unwrap().1.unwrap());
    
    // ===== 6. Avec LearningRateScheduler (ExponentialLR) =====
    println!("--- 6. Avec LearningRateScheduler (ExponentialLR) ---");
    let mut network_exp = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.05))
        .build();
    
    let scheduler = LearningRateScheduler::new(
        LRSchedule::ExponentialLR { gamma: 0.95 }
    );
    
    let history = network_exp.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(50)
        .batch_size(32)
        .scheduler(scheduler)
        .fit();
    
    println!("✓ Loss finale: {:.6}\n", history.last().unwrap().1.unwrap());
    
    // ===== 7. Combinaison: EarlyStopping + ModelCheckpoint + ProgressBar =====
    println!("--- 7. Combinaison Optimale: EarlyStopping + ModelCheckpoint + ProgressBar ---");
    let mut network_combined = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    
    let history = network_combined.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(100)
        .batch_size(32)
        .callback(Box::new(EarlyStopping::new(15, 0.0001)))
        .callback(Box::new(ModelCheckpoint::new("best_combined_model.json", true)))
        .callback(Box::new(ProgressBar::new(100)))
        .fit();
    
    println!("\n✓ Entraînement terminé ({} epochs)", history.len());
    println!("✓ Loss finale: {:.6}\n", history.last().unwrap().1.unwrap());
    
    // ===== 8. Combinaison ultime: Tous les callbacks =====
    println!("--- 8. Combinaison Ultime: LR Scheduler + All Callbacks ---");
    let mut network_ultimate = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.05))
        .build();
    
    let scheduler = LearningRateScheduler::new(
        LRSchedule::ReduceOnPlateau { 
            patience: 10, 
            factor: 0.5, 
            min_delta: 0.0001 
        }
    );
    
    let history = network_ultimate.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(100)
        .batch_size(32)
        .scheduler(scheduler)
        .callback(Box::new(EarlyStopping::new(20, 0.00001)))
        .callback(Box::new(ModelCheckpoint::new("best_ultimate_model.json", true)))
        .callback(Box::new(ProgressBar::new(100)))
        .fit();
    
    println!("\n✓ Entraînement terminé ({} epochs)", history.len());
    println!("✓ Loss finale: {:.6}\n", history.last().unwrap().1.unwrap());
    
    // ===== Summary =====
    println!("=== Résumé ===\n");
    
    println!("📊 Comparaison des approches:");
    println!("  1. Baseline (100 epochs):              loss = {:.6}", final_loss);
    println!("  2. EarlyStopping:                      {} epochs", history.len());
    println!("  3. ModelCheckpoint:                    Meilleur modèle sauvegardé");
    println!("  4. StepLR:                             LR réduit progressivement");
    println!("  5. ReduceOnPlateau:                    LR adapté automatiquement");
    println!("  6. ExponentialLR:                      Décroissance exponentielle");
    println!("  7. Combinaison (Early+Checkpoint+Bar): Automatisation complète");
    println!("  8. Ultime (LR+Early+Checkpoint+Bar):   Meilleure convergence\n");
    
    println!("💡 Recommandations:");
    println!("  • Toujours utiliser EarlyStopping pour éviter l'overfitting");
    println!("  • ModelCheckpoint sauvegarde le meilleur modèle automatiquement");
    println!("  • ReduceOnPlateau adapte le LR intelligemment");
    println!("  • ProgressBar améliore l'expérience utilisateur");
    println!("  • Combiner plusieurs callbacks = meilleur résultat !");
    
    // Cleanup
    std::fs::remove_file("best_xor_model.json").ok();
    std::fs::remove_file("best_combined_model.json").ok();
    std::fs::remove_file("best_ultimate_model.json").ok();
}
