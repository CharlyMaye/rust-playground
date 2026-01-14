/// Démo de la régularisation : Dropout, L1 et L2
/// 
/// Montre comment la régularisation aide à prévenir l'overfitting
/// en pénalisant les modèles trop complexes.

use test_neural::builder::NetworkBuilder;
use test_neural::network::{Network, Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use ndarray::array;

fn main() {
    println!("=== Démonstration de la Régularisation ===\n");

    // Dataset XOR (simple)
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

    let epochs = 5000;
    let hidden_size = 20;  // Réseau surdimensionné pour montrer l'overfitting

    println!("Configuration:");
    println!("  • Architecture: 2 → [{}] → 1 (réseau surdimensionné)", hidden_size);
    println!("  • Dataset: XOR (4 exemples seulement)");
    println!("  • Epochs: {}\n", epochs);

    // Test 1: Sans régularisation
    println!("--- 1. Sans Régularisation (Baseline) ---");
    let mut network_baseline = NetworkBuilder::new(2, 1)
        .hidden_layer(hidden_size, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();

    train_and_evaluate(&mut network_baseline, "Baseline", &inputs, &targets, epochs);

    // Test 2: Avec Dropout
    println!("\n--- 2. Avec Dropout (rate=0.3) ---");
    let mut network_dropout = NetworkBuilder::new(2, 1)
        .hidden_layer(hidden_size, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .dropout(0.3)  // 30% des neurones désactivés
        .build();

    train_and_evaluate(&mut network_dropout, "Dropout", &inputs, &targets, epochs);

    // Test 3: Avec L2 (Weight Decay)
    println!("\n--- 3. Avec L2 Regularization (lambda=0.01) ---");
    let mut network_l2 = NetworkBuilder::new(2, 1)
        .hidden_layer(hidden_size, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .l2(0.01)
        .build();

    train_and_evaluate(&mut network_l2, "L2", &inputs, &targets, epochs);

    // Test 4: Avec L1 (Sparsity)
    println!("\n--- 4. Avec L1 Regularization (lambda=0.01) ---");
    let mut network_l1 = NetworkBuilder::new(2, 1)
        .hidden_layer(hidden_size, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .l1(0.01)
        .build();

    train_and_evaluate(&mut network_l1, "L1", &inputs, &targets, epochs);

    // Test 5: Combiné (Dropout + L2)
    println!("\n--- 5. Dropout + L2 (Combiné) ---");
    let mut network_combined = NetworkBuilder::new(2, 1)
        .hidden_layer(hidden_size, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .dropout(0.2)
        .l2(0.005)
        .build();

    train_and_evaluate(&mut network_combined, "Combined", &inputs, &targets, epochs);

    // Résumé
    println!("\n=== Résumé ===");
    println!("🎯 Régularisation : Techniques pour réduire l'overfitting\n");
    println!("📊 Observations :");
    println!("  • Sans régularisation : Peut sur-apprendre (overfitting)");
    println!("  • Dropout : Force le réseau à être robuste");
    println!("  • L2 : Pénalise les grands poids, modèle plus lisse");
    println!("  • L1 : Encourage la sparsité (poids à zéro)");
    println!("  • Combiné : Souvent la meilleure approche\n");
    println!("💡 Recommandations :");
    println!("  • Dataset petit → Dropout (0.2-0.5) + L2 (0.001-0.01)");
    println!("  • Dataset grand → L2 seul ou Dropout léger");
    println!("  • Besoin de sparsité → L1");
}

fn train_and_evaluate(
    network: &mut Network,
    name: &str,
    inputs: &Vec<ndarray::Array1<f64>>,
    targets: &Vec<ndarray::Array1<f64>>,
    epochs: usize,
) {
    // Training
    network.train_mode();
    for epoch in 0..epochs {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            network.train(input, target);
        }

        if epoch % 1000 == 0 || epoch == epochs - 1 {
            let loss = network.evaluate(inputs, targets);
            println!("  Epoch {:4}: loss = {:.6}", epoch, loss);
        }
    }

    // Evaluation (switch to eval mode to disable dropout)
    network.eval_mode();
    let final_loss = network.evaluate(inputs, targets);
    
    println!("\n  Prédictions finales ({}):", name);
    let mut all_correct = true;
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let prediction = network.predict(input);
        let pred_value = prediction[0];
        let target_value = target[0];
        let correct = (pred_value.round() - target_value).abs() < 0.01;
        all_correct = all_correct && correct;
        println!("    [{:.1}, {:.1}] → {:.4} (target: {:.1}) {}",
            input[0], input[1], pred_value, target_value,
            if correct { "✓" } else { "✗" }
        );
    }
    
    println!("  Loss finale (eval): {:.6}", final_loss);
    println!("  Résultat: {}", if all_correct { "✓ PASSED" } else { "✗ FAILED" });
}
