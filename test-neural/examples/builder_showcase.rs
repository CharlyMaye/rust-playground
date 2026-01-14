/// Showcase du Builder Pattern - API simplifiée
///
/// Cet exemple montre comment le Builder Pattern simplifie l'utilisation
/// de la bibliothèque en comparaison avec l'API traditionnelle.

use test_neural::builder::{NetworkBuilder, NetworkTrainer};
use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::dataset::Dataset;
use test_neural::callbacks::{EarlyStopping, ModelCheckpoint, LearningRateScheduler, LRSchedule};
use ndarray::array;

fn main() {
    println!("=== Builder Pattern Showcase ===\n");
    
    // Préparer les données XOR
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
    
    println!("📦 Dataset: {} train, {} val\n", train.len(), val.len());
    
    // ========== EXEMPLE 1: Construction simple ==========
    println!("--- 1. Construction simple d'un réseau ---\n");
    
    let _network = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    
    println!("✓ Réseau créé: 2 → [8] → 1");
    println!("  • Activation cachée: Tanh");
    println!("  • Activation sortie: Sigmoid");
    println!("  • Loss: Binary Cross-Entropy");
    println!("  • Optimizer: Adam (lr=0.01)\n");
    
    // ========== EXEMPLE 2: Réseau profond avec régularisation ==========
    println!("--- 2. Réseau profond avec dropout et L2 ---\n");
    
    let _deep_network = NetworkBuilder::new(2, 1)
        .hidden_layer(16, Activation::ReLU)
        .hidden_layer(8, Activation::ReLU)
        .hidden_layer(4, Activation::ReLU)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.001))
        .dropout(0.2)
        .l2(0.001)
        .build();
    
    println!("✓ Réseau profond: 2 → [16, 8, 4] → 1");
    println!("  • Dropout: 0.2");
    println!("  • L2 regularization: 0.001\n");
    
    // ========== EXEMPLE 3: Entraînement avec callbacks ==========
    println!("--- 3. Entraînement avec callbacks complets ---\n");
    
    let mut trained_network = NetworkBuilder::new(2, 1)
        .hidden_layer(10, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.05))
        .build();
    
    let history = trained_network.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(200)
        .batch_size(2)
        .callback(Box::new(EarlyStopping::new(20, 0.00001)))
        .callback(Box::new(ModelCheckpoint::new("best_showcase.json", true)))
        .scheduler(LearningRateScheduler::new(
            LRSchedule::ReduceOnPlateau { 
                patience: 10, 
                factor: 0.5, 
                min_delta: 0.0001 
            }
        ))
        .fit();
    
    println!("✓ Entraînement terminé:");
    println!("  • Epochs: {}", history.len());
    println!("  • Loss finale: {:.6}", history.last().unwrap().1.unwrap());
    println!("  • Meilleur modèle sauvegardé\n");
    
    // ========== EXEMPLE 4: Predictions ==========
    println!("--- 4. Prédictions finales ---\n");
    
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let prediction = trained_network.predict(input);
        let binary_pred = if prediction[0] > 0.5 { 1.0 } else { 0.0 };
        let correct = (binary_pred - target[0]).abs() < 0.1;
        
        println!("  [{:.0}, {:.0}] → {:.3} (attendu {:.0}) {}", 
            input[0], input[1], prediction[0], target[0],
            if correct { "✓" } else { "✗" });
    }
    
    println!("\n=== Avantages du Builder Pattern ===\n");
    
    println!("✨ Construction intuitive:");
    println!("   NetworkBuilder::new(input, output)");
    println!("     .hidden_layer(size, activation)");
    println!("     .dropout(rate)");
    println!("     .l2(lambda)");
    println!("     .build()");
    
    println!("\n⚡ Entraînement fluide:");
    println!("   network.trainer()");
    println!("     .train_data(&dataset)");
    println!("     .epochs(100)");
    println!("     .callback(...)");
    println!("     .scheduler(...)");
    println!("     .fit()");
    
    println!("\n💡 Plus besoin de:");
    println!("   • Vec<usize> pour hidden_sizes");
    println!("   • Vec<Activation> pour hidden_activations");
    println!("   • new() vs new_deep() vs new_deep_with_init()");
    println!("   • fit() vs fit_with_scheduler()");
    println!("   • Gérer manuellement Vec<Box<dyn Callback>>");
    
    println!("\n🎯 Une seule manière de faire = code plus clair!");
    
    // Cleanup
    std::fs::remove_file("best_showcase.json").ok();
}
