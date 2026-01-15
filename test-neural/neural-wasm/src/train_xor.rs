//! XOR Model Training Script
//!
//! This binary trains a neural network on the XOR problem
//! and saves it to neural-wasm/src/xor_model.json

use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use cma_neural_network::dataset::Dataset;
use cma_neural_network::callbacks::{EarlyStopping, DeltaMode, ProgressBar};
use cma_neural_network::metrics::accuracy;
use cma_neural_network::io;
use ndarray::array;
use std::path::Path;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         XOR Neural Network Training                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let model_path = "src/xor_model.json";

    // Check if model already exists
    if Path::new(model_path).exists() {
        println!("⚠️  Model already exists at {}", model_path);
        println!("   Delete it manually if you want to retrain.\n");
        return;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 1. DATA PREPARATION
    // ═══════════════════════════════════════════════════════════════════════
    println!("📦 Preparing XOR dataset...\n");
    
    // Create an extended XOR dataset for training
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for _ in 0..200 {
        inputs.push(array![0.0, 0.0]); targets.push(array![0.0]);
        inputs.push(array![0.0, 1.0]); targets.push(array![1.0]);
        inputs.push(array![1.0, 0.0]); targets.push(array![1.0]);
        inputs.push(array![1.0, 1.0]); targets.push(array![0.0]);
    }
    
    let dataset = Dataset::new(inputs, targets);
    let (train, val) = dataset.split(0.8);
    
    println!("   Training samples: {}", train.len());
    println!("   Validation samples: {}\n", val.len());

    // ═══════════════════════════════════════════════════════════════════════
    // 2. BUILD NETWORK
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 Building network...\n");

    let mut network = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.05))
        .build();

    println!("   Architecture: 2 → [8] → 1");
    println!("   Activation: Tanh → Sigmoid");
    println!("   Optimizer: Adam (lr=0.05)\n");

    // ═══════════════════════════════════════════════════════════════════════
    // 3. TRAIN
    // ═══════════════════════════════════════════════════════════════════════
    println!("🏋️  Training...\n");

    let epochs = 500;
    let history = network.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(epochs)
        .batch_size(32)
        .callback(Box::new(EarlyStopping::new(30, 0.0001).mode(DeltaMode::Relative)))
        .callback(Box::new(ProgressBar::new(epochs)))
        .fit();

    println!("\n   ✅ Training completed in {} epochs", history.len());
    
    if let Some((train_loss, val_loss)) = history.last() {
        println!("   Final loss - Train: {:.6} | Val: {:.6}",
            train_loss, val_loss.unwrap_or(0.0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. EVALUATE
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n📊 Evaluating...\n");

    network.eval_mode();

    let test_inputs = vec![
        array![0.0, 0.0], array![0.0, 1.0],
        array![1.0, 0.0], array![1.0, 1.0],
    ];
    let test_targets = vec![
        array![0.0], array![1.0], array![1.0], array![0.0],
    ];

    let predictions: Vec<_> = test_inputs.iter()
        .map(|input| network.predict(input))
        .collect();

    println!("   XOR Truth Table:");
    println!("   ┌─────┬─────┬──────────┬────────────┐");
    println!("   │  A  │  B  │ Expected │ Prediction │");
    println!("   ├─────┼─────┼──────────┼────────────┤");
    
    for (input, (pred, target)) in test_inputs.iter()
        .zip(predictions.iter().zip(test_targets.iter()))
    {
        let pred_val = pred[0];
        let pred_binary = if pred_val > 0.5 { 1 } else { 0 };
        let expected = target[0] as u8;
        let status = if pred_binary == expected { "✓" } else { "✗" };
        
        println!("   │  {}  │  {}  │    {}     │ {} ({:.2}) {} │",
            input[0] as u8, input[1] as u8, expected, pred_binary, pred_val, status);
    }
    println!("   └─────┴─────┴──────────┴────────────┘");

    let acc = accuracy(&predictions, &test_targets, 0.5);
    println!("\n   Accuracy: {:.1}%", acc * 100.0);

    // ═══════════════════════════════════════════════════════════════════════
    // 5. SAVE MODEL
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n💾 Saving model...\n");

    match io::save_json(&network, model_path) {
        Ok(_) => {
            println!("   ✅ Model saved to {}", model_path);
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║              Training Complete! 🎉                           ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
        }
        Err(e) => {
            eprintln!("   ❌ Failed to save model: {}", e);
            std::process::exit(1);
        }
    }
}
