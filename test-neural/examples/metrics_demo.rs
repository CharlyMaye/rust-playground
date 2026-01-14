use test_neural::builder::NetworkBuilder;
use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::metrics::{accuracy, binary_metrics, confusion_matrix_binary, format_confusion_matrix, auc_roc};
use ndarray::array;

fn main() {
    println!("=== Neural Network Metrics Demo ===\n");
    
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
    
    println!("1. Creating and training network on XOR problem...");
    let mut network = NetworkBuilder::new(2, 1)
        .hidden_layer(5, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    
    // Entraînement
    let epochs = 10_000;
    for epoch in 0..epochs {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            network.train(input, target);
        }
        
        if (epoch + 1) % 2000 == 0 {
            let loss = network.evaluate(&inputs, &targets);
            println!("  Epoch {}: loss = {:.4}", epoch + 1, loss);
        }
    }
    
    let final_loss = network.evaluate(&inputs, &targets);
    println!("  Training complete! Final loss: {:.4}\n", final_loss);
    
    // Obtenir les prédictions
    let predictions: Vec<_> = inputs.iter()
        .map(|input| network.predict(input))
        .collect();
    
    println!("2. Predictions:");
    for (i, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
        println!("  {:?} -> {:.3} (expected {:.0})", 
            input, predictions[i][0], target[0]);
    }
    println!();
    
    // Calculer l'accuracy
    println!("3. Accuracy Metric:");
    let acc = accuracy(&predictions, &targets, 0.5);
    println!("  Accuracy: {:.2}% ({}/{})", acc * 100.0, (acc * 4.0) as usize, 4);
    println!();
    
    // Métriques binaires complètes
    println!("4. Detailed Binary Metrics:");
    let metrics = binary_metrics(&predictions, &targets, 0.5);
    println!("  {}", metrics.summary());
    println!();
    
    // Matrice de confusion
    println!("5. Confusion Matrix:");
    let conf_matrix = confusion_matrix_binary(&predictions, &targets, 0.5);
    println!("{}", format_confusion_matrix(&conf_matrix, Some(&["Neg", "Pos"])));
    
    // AUC-ROC
    println!("6. ROC-AUC Score:");
    let auc = auc_roc(&predictions, &targets);
    println!("  AUC: {:.4} (1.0 = perfect, 0.5 = random)", auc);
    println!();
    
    // Test avec différents seuils
    println!("7. Accuracy at Different Thresholds:");
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7] {
        let acc = accuracy(&predictions, &targets, threshold);
        let metrics = binary_metrics(&predictions, &targets, threshold);
        println!("  Threshold {:.1}: Accuracy={:.2}% | Precision={:.3} | Recall={:.3} | F1={:.3}",
            threshold, acc * 100.0, metrics.precision, metrics.recall, metrics.f1_score);
    }
    println!();
    
    // Démonstration sur un cas imparfait
    println!("8. Example with Imperfect Predictions:");
    let imperfect_preds = vec![
        array![0.1],  // Correct (negative)
        array![0.9],  // Correct (positive)
        array![0.4],  // WRONG (should be positive, but < 0.5)
        array![0.2],  // Correct (negative)
    ];
    
    let acc_imperfect = accuracy(&imperfect_preds, &targets, 0.5);
    let metrics_imperfect = binary_metrics(&imperfect_preds, &targets, 0.5);
    
    println!("  Predictions: [0.1, 0.9, 0.4, 0.2]");
    println!("  Targets:     [0.0, 1.0, 1.0, 0.0]");
    println!("  Accuracy: {:.2}% (3/4 correct)", acc_imperfect * 100.0);
    println!("  TP={} FP={} TN={} FN={}",
        metrics_imperfect.true_positives,
        metrics_imperfect.false_positives,
        metrics_imperfect.true_negatives,
        metrics_imperfect.false_negatives);
    println!("  Precision: {:.3} (of predicted positives, how many are correct?)", 
        metrics_imperfect.precision);
    println!("  Recall: {:.3} (of actual positives, how many did we catch?)", 
        metrics_imperfect.recall);
    println!("  F1-Score: {:.3} (harmonic mean of precision and recall)", 
        metrics_imperfect.f1_score);
    
    println!("\n=== Demo Complete ===");
}
