//! Iris Model Training Script
//!
//! This binary trains a neural network on the Iris classification problem
//! and saves it to neural-wasm/iris/src/iris_model.json

use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use cma_neural_network::dataset::Dataset;
use cma_neural_network::callbacks::{EarlyStopping, DeltaMode, ProgressBar};
use neural_wasm_shared::{calculate_multiclass_accuracy, save_model_with_metadata};
use ndarray::{array, Array1};
use std::path::Path;
use std::error::Error;
use csv::ReaderBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Iris Classification Neural Network Training          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let model_path = "src/iris_model.json";

    // Check if model already exists
    if Path::new(model_path).exists() {
        println!("⚠️  Model already exists at {}", model_path);
        println!("   Delete it manually if you want to retrain.\n");
        return Ok(());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 1. DATA PREPARATION
    // ═══════════════════════════════════════════════════════════════════════
    println!("📦 Preparing Iris dataset...\n");
    
    let iris_data = load_iris_from_csv("data/iris.csv")?;
    println!("   ✅ Loaded {} samples from CSV", iris_data.len());
    
    let inputs: Vec<Array1<f64>> = iris_data.iter().map(|(i, _)| i.clone()).collect();
    let targets: Vec<Array1<f64>> = iris_data.iter().map(|(_, t)| t.clone()).collect();
    
    let dataset = Dataset::new(inputs, targets);
    let (train, val) = dataset.split(0.7);
    
    println!("   Training samples: {} (70%)", train.len());
    println!("   Test samples: {} (30%)\n", val.len());

    // ═══════════════════════════════════════════════════════════════════════
    // 2. BUILD NETWORK
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 Building network...\n");

    let mut network = NetworkBuilder::new(4, 3)
        .hidden_layer(12, Activation::Tanh)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Softmax)
        .loss(LossFunction::CategoricalCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();

    println!("   Architecture: 4 → [12, 8] → 3");
    println!("   Activation: Tanh → Tanh → Softmax");
    println!("   Optimizer: Adam (lr=0.01)\n");

    // ═══════════════════════════════════════════════════════════════════════
    // 3. TRAIN
    // ═══════════════════════════════════════════════════════════════════════
    println!("🏋️  Training...\n");

    let epochs = 2_000;
    let history = network.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(epochs)
        .batch_size(32)
        .callback(Box::new(EarlyStopping::new(100, 0.00001).mode(DeltaMode::Absolute)))
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

    let test_inputs = val.inputs();
    let test_targets = val.targets();
    
    let (correct, total) = calculate_multiclass_accuracy(&network, test_inputs, test_targets);
    let acc = correct as f64 / total as f64;
    
    println!("   Iris Classification Results:");
    println!("   ┌─────────────────────────────────┐");
    println!("   │  Correct: {}/{} ({:.2}%)        │", correct, total, acc * 100.0);
    println!("   └─────────────────────────────────┘");
    
    println!("\n   Test Accuracy: {:.2}%", acc * 100.0);
    
    // ═══════════════════════════════════════════════════════════════════════
    // 5. SAVE MODEL WITH METADATA
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n💾 Saving model with metadata...\n");

    match save_model_with_metadata(network, acc, total, model_path) {
        Ok(_) => {
            println!("   ✅ Model saved to {}", model_path);
            println!("   📊 Accuracy: {:.2}%", acc * 100.0);
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║              Training Complete! 🎉                           ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
        }
        Err(e) => {
            eprintln!("   ❌ Failed to save model: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Load the real Iris dataset from CSV file
/// Dataset source: UCI Machine Learning Repository
/// https://archive.ics.uci.edu/ml/datasets/iris
fn load_iris_from_csv(path: &str) -> Result<Vec<(Array1<f64>, Array1<f64>)>, Box<dyn Error>> {
    let mut data = Vec::new();
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    
    for result in rdr.records() {
        let record = result?;
        
        // Parse the 4 features
        let sepal_length: f64 = record[0].parse()?;
        let sepal_width: f64 = record[1].parse()?;
        let petal_length: f64 = record[2].parse()?;
        let petal_width: f64 = record[3].parse()?;
        
        // Parse species and convert to one-hot encoding
        let species = &record[4];
        let one_hot = match species {
            "setosa" => array![1.0, 0.0, 0.0],
            "versicolor" => array![0.0, 1.0, 0.0],
            "virginica" => array![0.0, 0.0, 1.0],
            _ => return Err(format!("Unknown species: {}", species).into()),
        };
        
        data.push((
            array![sepal_length, sepal_width, petal_length, petal_width],
            one_hot,
        ));
    }
    
    Ok(data)
}
