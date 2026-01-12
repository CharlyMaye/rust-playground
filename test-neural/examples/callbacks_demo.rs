/// Demonstration of Callbacks
/// 
/// This example shows:
/// - EarlyStopping: Prevents overfitting by stopping when validation loss plateaus
/// - ModelCheckpoint: Automatically saves the best model
/// - LearningRateScheduler: Adjusts LR dynamically (StepLR, ReduceOnPlateau, ExponentialLR)
/// - ProgressBar: Shows training progress in real-time
/// - Combining multiple callbacks

use test_neural::network::{Network, Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::dataset::Dataset;
use test_neural::callbacks::{
    EarlyStopping, ModelCheckpoint, LearningRateScheduler, ProgressBar, 
    LRSchedule, Callback
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
    let mut network_baseline = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.01)
    );
    
    let mut train_data = train.clone();
    for epoch in 0..100 {
        train_data.shuffle();
        for (batch_inputs, batch_targets) in train_data.batches(32) {
            network_baseline.train_batch(&batch_inputs, &batch_targets);
        }
        
        if (epoch + 1) % 20 == 0 {
            let val_loss = network_baseline.evaluate(val.inputs(), val.targets());
            println!("Epoch {}: val_loss = {:.6}", epoch + 1, val_loss);
        }
    }
    
    let final_loss = network_baseline.evaluate(val.inputs(), val.targets());
    println!("✓ Loss finale: {:.6}\n", final_loss);
    
    // ===== 2. Avec EarlyStopping =====
    println!("--- 2. Avec EarlyStopping (patience=10) ---");
    let mut network_early = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.01)
    );
    
    let mut callbacks: Vec<Box<dyn Callback>> = vec![
        Box::new(EarlyStopping::new(10, 0.0001)),
    ];
    
    let history = network_early.fit(&train, Some(&val), 100, 32, &mut callbacks);
    
    println!("✓ Arrêté après {} epochs", history.len());
    println!("✓ Loss finale: {:.6}\n", history.last().unwrap().1.unwrap());
    
    // ===== 3. Avec ModelCheckpoint =====
    println!("--- 3. Avec ModelCheckpoint ---");
    let mut network_checkpoint = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.01)
    );
    
    let mut callbacks: Vec<Box<dyn Callback>> = vec![
        Box::new(ModelCheckpoint::new("best_xor_model.json", true)),
    ];
    
    let history = network_checkpoint.fit(&train, Some(&val), 50, 32, &mut callbacks);
    
    println!("✓ Entraînement terminé ({} epochs)", history.len());
    println!("✓ Meilleur modèle sauvegardé dans best_xor_model.json\n");
    
    // ===== 4. Avec LearningRateScheduler (StepLR) =====
    println!("--- 4. Avec LearningRateScheduler (StepLR) ---");
    let mut network_step = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.1)  // LR plus élevé au départ
    );
    
    let mut scheduler = LearningRateScheduler::new(
        LRSchedule::StepLR { step_size: 15, gamma: 0.5 }
    );
    
    let mut callbacks: Vec<Box<dyn Callback>> = vec![];
    
    let history = network_step.fit_with_scheduler(&train, Some(&val), 50, 32, &mut scheduler, &mut callbacks);
    
    println!("✓ Loss finale: {:.6}", history.last().unwrap().1.unwrap());
    println!("✓ LR final: {:.6}\n", scheduler.current_lr());
    
    // ===== 5. Avec LearningRateScheduler (ReduceOnPlateau) =====
    println!("--- 5. Avec LearningRateScheduler (ReduceOnPlateau) ---");
    let mut network_plateau = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.01)
    );
    
    let mut scheduler = LearningRateScheduler::new(
        LRSchedule::ReduceOnPlateau { 
            patience: 5, 
            factor: 0.5, 
            min_delta: 0.0001 
        }
    );
    
    let mut callbacks: Vec<Box<dyn Callback>> = vec![];
    
    let history = network_plateau.fit_with_scheduler(&train, Some(&val), 50, 32, &mut scheduler, &mut callbacks);
    
    println!("✓ Loss finale: {:.6}", history.last().unwrap().1.unwrap());
    println!("✓ LR final: {:.6}\n", scheduler.current_lr());
    
    // ===== 6. Avec LearningRateScheduler (ExponentialLR) =====
    println!("--- 6. Avec LearningRateScheduler (ExponentialLR) ---");
    let mut network_exp = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.05)  // LR plus élevé au départ
    );
    
    let mut scheduler = LearningRateScheduler::new(
        LRSchedule::ExponentialLR { gamma: 0.95 }
    );
    
    let mut callbacks: Vec<Box<dyn Callback>> = vec![];
    
    let history = network_exp.fit_with_scheduler(&train, Some(&val), 50, 32, &mut scheduler, &mut callbacks);
    
    println!("✓ Loss finale: {:.6}", history.last().unwrap().1.unwrap());
    println!("✓ LR final: {:.6}\n", scheduler.current_lr());
    
    // ===== 7. Combinaison: EarlyStopping + ModelCheckpoint + ProgressBar =====
    println!("--- 7. Combinaison Optimale: EarlyStopping + ModelCheckpoint + ProgressBar ---");
    let mut network_combined = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.01)
    );
    
    let mut callbacks: Vec<Box<dyn Callback>> = vec![
        Box::new(EarlyStopping::new(15, 0.0001)),
        Box::new(ModelCheckpoint::new("best_combined_model.json", true)),
        Box::new(ProgressBar::new(100)),
    ];
    
    let history = network_combined.fit(&train, Some(&val), 100, 32, &mut callbacks);
    
    println!("\n✓ Entraînement terminé ({} epochs)", history.len());
    println!("✓ Loss finale: {:.6}\n", history.last().unwrap().1.unwrap());
    
    // ===== 8. Combinaison ultime: Tous les callbacks =====
    println!("--- 8. Combinaison Ultime: LR Scheduler + All Callbacks ---");
    let mut network_ultimate = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.05)  // LR élevé car géré par scheduler
    );
    
    let mut scheduler = LearningRateScheduler::new(
        LRSchedule::ReduceOnPlateau { 
            patience: 10, 
            factor: 0.5, 
            min_delta: 0.0001 
        }
    );
    
    let mut callbacks: Vec<Box<dyn Callback>> = vec![
        Box::new(EarlyStopping::new(20, 0.00001)),
        Box::new(ModelCheckpoint::new("best_ultimate_model.json", true)),
        Box::new(ProgressBar::new(100)),
    ];
    
    let history = network_ultimate.fit_with_scheduler(&train, Some(&val), 100, 32, &mut scheduler, &mut callbacks);
    
    println!("\n✓ Entraînement terminé ({} epochs)", history.len());
    println!("✓ Loss finale: {:.6}", history.last().unwrap().1.unwrap());
    println!("✓ LR final: {:.6}\n", scheduler.current_lr());
    
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
