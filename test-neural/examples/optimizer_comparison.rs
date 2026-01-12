/// Démo comparant différents optimiseurs sur le problème XOR
/// 
/// Compare SGD, Momentum, RMSprop, Adam et AdamW pour montrer
/// les différences de vitesse de convergence.

use test_neural::network::{Network, Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use ndarray::array;

fn main() {
    println!("=== Comparaison d'Optimiseurs sur XOR ===\n");

    // Dataset XOR
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

    // Configuration commune
    let epochs = 2000;
    let hidden_size = 6;

    // Test chaque optimiseur
    let optimizers = vec![
        ("SGD (lr=0.5)", OptimizerType::sgd(0.5)),
        ("Momentum (lr=0.1)", OptimizerType::momentum(0.1)),
        ("RMSprop (lr=0.01)", OptimizerType::rmsprop(0.01)),
        ("Adam (lr=0.01)", OptimizerType::adam(0.01)),
        ("AdamW (lr=0.01, wd=0.01)", OptimizerType::adamw(0.01, 0.01)),
    ];

    for (name, optimizer) in optimizers {
        println!("--- {} ---", name);
        
        // Créer le réseau avec cet optimiseur
        let mut network = Network::new(
            2,
            hidden_size,
            1,
            Activation::Tanh,
            Activation::Sigmoid,
            LossFunction::BinaryCrossEntropy,
            optimizer,
        );

        // Entraînement
        let mut final_loss = 0.0;
        for epoch in 0..epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                network.train(input, target);
            }

            // Calculer la loss tous les 500 epochs
            if epoch % 500 == 0 || epoch == epochs - 1 {
                let loss = network.evaluate(&inputs, &targets);
                final_loss = loss;
                println!("  Epoch {:4}: loss = {:.6}", epoch, loss);
            }
        }

        // Tester les prédictions finales
        println!("\n  Prédictions finales:");
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = network.predict(input);
            let pred_value = prediction[0];
            let target_value = target[0];
            let correct = (pred_value.round() - target_value).abs() < 0.01;
            println!("    [{:.1}, {:.1}] → {:.4} (target: {:.1}) {}",
                input[0], input[1], pred_value, target_value,
                if correct { "✓" } else { "✗" }
            );
        }

        println!("  Loss finale: {:.6}\n", final_loss);
    }

    println!("\n=== Résumé ===");
    println!("• SGD: Simple mais lent, nécessite un learning rate élevé");
    println!("• Momentum: Plus rapide que SGD grâce à l'accumulation de momentum");
    println!("• RMSprop: Adapte le learning rate par paramètre, bonne convergence");
    println!("• Adam: Combine momentum et RMSprop, convergence la plus rapide");
    println!("• AdamW: Adam avec weight decay découplé, meilleure généralisation");
    println!("\n💡 Recommandation: Adam est le standard moderne pour la plupart des cas");
}
