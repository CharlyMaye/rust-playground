//! Getting Started — complete library tour
//!
//! Covers:
//! - NetworkBuilder: layers, activations, optimizers, regularization
//! - TrainingBuilder: dataset, batch training, callbacks, LR scheduler
//! - Evaluation: accuracy, precision/recall/F1, AUC
//! - Serialization: save/load JSON
//!
//! Run: cargo run --example getting_started

use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::callbacks::{
    DeltaMode, EarlyStopping, LRSchedule, LearningRateScheduler, ProgressBar,
};
use cma_neural_network::dataset::Dataset;
use cma_neural_network::io;
use cma_neural_network::metrics::{accuracy, binary_metrics};
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use ndarray::array;
use std::fs;
use std::path::Path;

fn main() {
    // Create data directory for output files
    let data_dir = "examples/data";
    fs::create_dir_all(data_dir).expect("Failed to create data directory");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Test Neural - Getting Started Guide                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════════════
    // 1. DATA PREPARATION
    // ═══════════════════════════════════════════════════════════════════════
    println!("📦 1. Data preparation (XOR problem)\n");

    // Create an extended XOR dataset for training
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..100 {
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
    let (mut train, val) = dataset.split(0.8);

    println!(
        "   Train: {} samples | Validation: {} samples\n",
        train.len(),
        val.len()
    );

    // ═══════════════════════════════════════════════════════════════════════
    // 2. BUILDING A SIMPLE NETWORK
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 2. Building a network with the Builder Pattern\n");

    let _network = NetworkBuilder::new(2, 1) // 2 inputs, 1 output
        .hidden_layer(8, Activation::Tanh) // Hidden layer
        .output_activation(Activation::Sigmoid) // Binary output
        .loss(LossFunction::BinaryCrossEntropy) // Binary classification
        .optimizer(OptimizerType::adam(0.01)) // Adam optimizer
        .build();

    println!("   ✓ Network created: 2 → [8] → 1");
    println!("   ✓ Activation: Tanh → Sigmoid");
    println!("   ✓ Optimizer: Adam (lr=0.01)\n");

    // ═══════════════════════════════════════════════════════════════════════
    // 3. NETWORK WITH REGULARIZATION
    // ═══════════════════════════════════════════════════════════════════════
    println!("🛡️  3. Network with regularization (Dropout + L2)\n");

    let _network_reg = NetworkBuilder::new(2, 1)
        .hidden_layer(16, Activation::ReLU)
        .hidden_layer(8, Activation::ReLU)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.001))
        .dropout(0.2) // 20% of neurons disabled during training
        .l2(0.001) // L2 regularization (weight decay)
        .build();

    println!("   ✓ Architecture: 2 → [16, 8] → 1");
    println!("   ✓ Dropout: 0.2 (prevents overfitting)");
    println!("   ✓ L2: 0.001 (penalizes large weights)\n");

    // ═══════════════════════════════════════════════════════════════════════
    // 4. OPTIMIZER COMPARISON
    // ═══════════════════════════════════════════════════════════════════════
    println!("⚡ 4. Quick optimizer comparison\n");

    let optimizers = vec![
        ("SGD",      OptimizerType::sgd(0.5)),
        ("Momentum", OptimizerType::momentum(0.1)),
        ("Adam",     OptimizerType::adam(0.01)),
    ];

    let test_inputs  = vec![
        array![0.0f32, 0.0], array![0.0, 1.0],
        array![1.0, 0.0],    array![1.0, 1.0],
    ];
    let test_targets = vec![array![0.0f32], array![1.0], array![1.0], array![0.0]];

    for (name, optimizer) in optimizers {
        let mut net = NetworkBuilder::new(2, 1)
            .hidden_layer(8, Activation::Tanh)
            .output_activation(Activation::Sigmoid)
            .loss(LossFunction::BinaryCrossEntropy)
            .optimizer(optimizer)
            .build();
        net.set_seed(42);

        // Quick training via TrainingBuilder (200 epochs, batch_size=4, silent)
        let mut ds = Dataset::new(test_inputs.clone(), test_targets.clone());
        net.trainer()
            .train_data(&mut ds)
            .epochs(200)
            .batch_size(4)
            .verbose(0)
            .fit();

        let loss = net.evaluate(&test_inputs, &test_targets);
        println!("   {:<10} → Final loss: {:.6}", name, loss);
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // 5. TRAINING WITH CALLBACKS
    // ═══════════════════════════════════════════════════════════════════════
    println!("📊 5. Training with callbacks\n");

    let mut network = NetworkBuilder::new(2, 1)
        .hidden_layer(10, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.05))
        .build();

    network.set_seed(42); // Reproducible training

    println!("   Configuration:");
    println!("   • EarlyStopping (patience=15, 0.1% relative improvement)");
    println!("   • LR Scheduler (ReduceOnPlateau)\n");

    let epoch = 1_000;
    let history = network
        .trainer()
        .train_data(&mut train)
        .validation_data(&val)
        .epochs(epoch)
        .batch_size(32)
        .callback(Box::new(
            EarlyStopping::new(15, 0.001).mode(DeltaMode::Relative),
        )) // 0.1% improvement
        .callback(Box::new(ProgressBar::new(epoch)))
        .scheduler(LearningRateScheduler::new(LRSchedule::ReduceOnPlateau {
            patience: 10,
            factor: 0.5,
            min_delta: 0.0001,
        }))
        .fit();

    println!("\n   ✓ Training completed in {} epochs", history.len());
    if let Some((train_loss, val_loss)) = history.last() {
        println!(
            "   ✓ Final loss - Train: {:.6} | Val: {:.6}",
            train_loss,
            val_loss.unwrap_or(0.0)
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 6. EVALUATION AND METRICS
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n📈 6. Evaluation and metrics\n");

    // Note: predict() automatically uses eval mode (no dropout)

    let predictions: Vec<_> = test_inputs
        .iter()
        .map(|input| network.predict(input))
        .collect();

    println!("   Predictions:");
    for (input, (pred, target)) in test_inputs
        .iter()
        .zip(predictions.iter().zip(test_targets.iter()))
    {
        let correct = (pred[0].round() - target[0]).abs() < 0.1;
        println!(
            "   [{:.0}, {:.0}] → {:.3} (expected {:.0}) {}",
            input[0],
            input[1],
            pred[0],
            target[0],
            if correct { "✓" } else { "✗" }
        );
    }

    let acc = accuracy(&predictions, &test_targets, 0.5);
    let metrics = binary_metrics(&predictions, &test_targets, 0.5);

    println!("\n   Metrics:");
    println!("   • Accuracy:  {:.1}%", acc * 100.0);
    println!("   • Precision: {:.3}", metrics.precision);
    println!("   • Recall:    {:.3}", metrics.recall);
    println!("   • F1-Score:  {:.3}", metrics.f1_score);

    // ═══════════════════════════════════════════════════════════════════════
    // 7. MODEL COMPARISON AND SAVE
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n🔄 7. Model comparison\n");

    let model_path = format!("{}/best_model.json", data_dir);
    let current_loss = network.evaluate(&test_inputs, &test_targets);
    let current_acc = acc;

    // Check if a previous model exists
    if Path::new(&model_path).exists() {
        match io::load_json(&model_path) {
            Ok(previous_model) => {
                let previous_loss = previous_model.evaluate(&test_inputs, &test_targets);
                let previous_preds: Vec<_> = test_inputs
                    .iter()
                    .map(|input| previous_model.predict(input))
                    .collect();
                let previous_acc = accuracy(&previous_preds, &test_targets, 0.5);

                println!("   ┌─────────────────────┬───────────────┬───────────────┐");
                println!("   │      Model          │     Loss      │   Accuracy    │");
                println!("   ├─────────────────────┼───────────────┼───────────────┤");
                println!(
                    "   │ Previous (saved)    │   {:.6}    │    {:.1}%      │",
                    previous_loss,
                    previous_acc * 100.0
                );
                println!(
                    "   │ Current  (new)      │   {:.6}    │    {:.1}%      │",
                    current_loss,
                    current_acc * 100.0
                );
                println!("   └─────────────────────┴───────────────┴───────────────┘");

                // Compare and decide
                let loss_improved = current_loss < previous_loss;
                let acc_improved = current_acc > previous_acc;

                println!();
                if loss_improved && acc_improved {
                    println!("   ✅ Analysis: New model is BETTER on all metrics!");
                    println!(
                        "      • Loss improved by {:.2}%",
                        (1.0 - current_loss / previous_loss) * 100.0
                    );
                    println!(
                        "      • Accuracy improved by {:.1} points",
                        (current_acc - previous_acc) * 100.0
                    );
                    println!("   💾 Saving new model...");
                    io::save_json(&network, &model_path).expect("Failed to save model");
                    println!("   ✓ Model saved to {}", model_path);
                } else if loss_improved {
                    println!("   🟡 Analysis: New model has LOWER loss but same/lower accuracy.");
                    println!(
                        "      • Loss: {:.6} → {:.6} (↓ {:.2}%)",
                        previous_loss,
                        current_loss,
                        (1.0 - current_loss / previous_loss) * 100.0
                    );
                    println!(
                        "      • Accuracy: {:.1}% → {:.1}%",
                        previous_acc * 100.0,
                        current_acc * 100.0
                    );
                    println!("   💾 Saving new model (lower loss is preferred)...");
                    io::save_json(&network, &model_path).expect("Failed to save model");
                    println!("   ✓ Model saved to {}", model_path);
                } else if acc_improved {
                    println!("   🟡 Analysis: New model has BETTER accuracy but higher loss.");
                    println!("      • Loss: {:.6} → {:.6}", previous_loss, current_loss);
                    println!(
                        "      • Accuracy: {:.1}% → {:.1}% (↑ {:.1} points)",
                        previous_acc * 100.0,
                        current_acc * 100.0,
                        (current_acc - previous_acc) * 100.0
                    );
                    println!("   💾 Saving new model (better accuracy)...");
                    io::save_json(&network, &model_path).expect("Failed to save model");
                    println!("   ✓ Model saved to {}", model_path);
                } else {
                    println!("   ❌ Analysis: Previous model is still better.");
                    println!(
                        "      • Loss: {:.6} (previous) vs {:.6} (current)",
                        previous_loss, current_loss
                    );
                    println!(
                        "      • Accuracy: {:.1}% (previous) vs {:.1}% (current)",
                        previous_acc * 100.0,
                        current_acc * 100.0
                    );
                    println!("   💾 Keeping previous model.");
                }
            }
            Err(e) => {
                println!("   ⚠️  Could not load previous model: {}", e);
                println!("   💾 Saving current model as new baseline...");
                io::save_json(&network, &model_path).expect("Failed to save model");
                println!("   ✓ Model saved to {}", model_path);
            }
        }
    } else {
        println!("   ℹ️  No previous model found.");
        println!(
            "   Current model: Loss = {:.6} | Accuracy = {:.1}%",
            current_loss,
            current_acc * 100.0
        );
        println!("   💾 Saving as first baseline...");
        io::save_json(&network, &model_path).expect("Failed to save model");
        println!("   ✓ Model saved to {}", model_path);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                        SUMMARY                               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ • NetworkBuilder::new(input, output)                         ║");
    println!("║     .hidden_layer(size, activation)                          ║");
    println!("║     .optimizer(OptimizerType::adam(lr))                      ║");
    println!("║     .dropout(rate).l2(lambda)                                ║");
    println!("║     .build()                                                 ║");
    println!("║                                                              ║");
    println!("║ • network.trainer()                                          ║");
    println!("║     .train_data(&mut dataset)                                ║");
    println!("║     .epochs(100).batch_size(32)                              ║");
    println!("║     .callback(Box::new(...))                                 ║");
    println!("║     .fit()                                                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("📚 Other examples:");
    println!("   cargo run --example serialization   - Save/Load models");
    println!("   cargo run --example minibatch_demo  - Mini-batch training");
    println!("   cargo run --example metrics_demo    - Detailed metrics\n");
}
