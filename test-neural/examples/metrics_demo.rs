//! Metrics demonstration
//!
//! Shows every metric function available in cma-neural-network:
//! binary accuracy, precision/recall/F1, confusion matrix, ROC curve, AUC-ROC.
//!
//! Run: cargo run --example metrics_demo

use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::dataset::Dataset;
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use cma_neural_network::metrics::{
    accuracy, auc_roc, binary_metrics, confusion_matrix_binary, format_confusion_matrix, roc_curve,
};
use ndarray::array;

fn main() {
    println!("=== Metrics Demo ===\n");

    // --- 1. Build a dataset and train a classifier ---

    // XOR dataset — repeated 100× to give the trainer reasonable volume
    let mut inputs  = Vec::new();
    let mut targets = Vec::new();
    for _ in 0..100 {
        inputs.extend_from_slice(&[array![0.0f32, 0.0], array![0.0, 1.0],
                                    array![1.0, 0.0],    array![1.0, 1.0]]);
        targets.extend_from_slice(&[array![0.0f32], array![1.0], array![1.0], array![0.0]]);
    }
    let mut dataset = Dataset::new(inputs.clone(), targets.clone());
    dataset.shuffle();
    let (mut train, _val) = dataset.split(0.8);

    println!("1. Training network on XOR (TrainingBuilder)...");
    let mut network = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    network.set_seed(42);

    network
        .trainer()
        .train_data(&mut train)
        .epochs(300)
        .batch_size(16)
        .verbose(0)
        .fit();

    // Evaluate on the canonical 4-point XOR set
    let inputs  = vec![array![0.0f32, 0.0], array![0.0, 1.0],
                       array![1.0, 0.0],    array![1.0, 1.0]];
    let targets = vec![array![0.0f32], array![1.0], array![1.0], array![0.0]];

    let final_loss = network.evaluate(&inputs, &targets);
    println!("  Training complete! Final loss: {:.4}\n", final_loss);

    // Get predictions
    let predictions: Vec<_> = inputs.iter()
        .map(|input| network.predict(input))
        .collect();
    
    println!("2. Predictions:");
    for (i, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
        println!("  {:?} -> {:.3} (expected {:.0})", 
            input, predictions[i][0], target[0]);
    }
    println!();

    // Calculate accuracy
    println!("3. Accuracy Metric:");
    let acc = accuracy(&predictions, &targets, 0.5);
    println!("  Accuracy: {:.2}% ({}/{})", acc * 100.0, (acc * 4.0) as usize, 4);
    println!();

    // Complete binary metrics
    println!("4. Detailed Binary Metrics:");
    let metrics = binary_metrics(&predictions, &targets, 0.5);
    println!("  {}", metrics.summary());
    println!();
    
    // Confusion matrix
    println!("5. Confusion Matrix:");
    let conf_matrix = confusion_matrix_binary(&predictions, &targets, 0.5);
    println!("{}", format_confusion_matrix(&conf_matrix, Some(&["Neg", "Pos"])));
    
    // ROC curve (fpr, tpr, thresholds)
    println!("6. ROC Curve (10 threshold points):");
    let (fpr, tpr, _thresholds) = roc_curve(&predictions, &targets, 10);
    for (f, t) in fpr.iter().zip(tpr.iter()) {
        println!("  FPR: {:.2}  TPR: {:.2}", f, t);
    }
    println!();

    // AUC-ROC
    println!("7. ROC-AUC Score:");
    let auc = auc_roc(&predictions, &targets);
    println!("  AUC: {:.4} (1.0 = perfect, 0.5 = random)", auc);
    println!();

    // Test with different thresholds
    println!("8. Accuracy at Different Thresholds:");
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7] {
        let acc = accuracy(&predictions, &targets, threshold);
        let metrics = binary_metrics(&predictions, &targets, threshold);
        println!("  Threshold {:.1}: Accuracy={:.2}% | Precision={:.3} | Recall={:.3} | F1={:.3}",
            threshold, acc * 100.0, metrics.precision, metrics.recall, metrics.f1_score);
    }
    println!();

    // Demonstration on an imperfect case
    println!("9. Example with Imperfect Predictions:");
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
    println!("See docs/guides/02-dense-network.md §5 for full metrics API reference.");
}
