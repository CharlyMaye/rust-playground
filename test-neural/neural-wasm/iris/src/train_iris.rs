//! Iris Model Training Script
//!
//! This binary trains a neural network on the Iris classification problem
//! and saves it to neural-wasm/iris/src/iris_model.json

use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::callbacks::{DeltaMode, EarlyStopping, ProgressBar};
use cma_neural_network::dataset::Dataset;
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use csv::ReaderBuilder;
use ndarray::{array, Array1};
use neural_wasm_shared::{calculate_multiclass_accuracy, save_model_with_normalization, NormalizationStats};
use std::error::Error;
use std::path::Path;

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

    // Normalize inputs (z-score normalization per feature)
    let (inputs, norm_stats) = normalize_features_with_stats(&inputs);
    println!("   ✅ Features normalized (z-score)");
    println!("   📊 Stats: means={:?}", norm_stats.means);
    println!("   📊 Stats: stds={:?}", norm_stats.stds);

    let mut dataset = Dataset::new(inputs, targets);

    // CRITICAL: Shuffle before split! The CSV is sorted by class (setosa, versicolor, virginica)
    // Without shuffling, the test set would contain only virginica samples!
    dataset.shuffle();
    println!("   ✅ Dataset shuffled");

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
    let history = network
        .trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(epochs)
        .batch_size(32)
        .callback(Box::new(
            EarlyStopping::new(100, 0.00001).mode(DeltaMode::Relative),
        ))
        .callback(Box::new(ProgressBar::new(epochs)))
        .fit();

    println!("\n   ✅ Training completed in {} epochs", history.len());

    if let Some((train_loss, val_loss)) = history.last() {
        println!(
            "   Final loss - Train: {:.6} | Val: {:.6}",
            train_loss,
            val_loss.unwrap_or(0.0)
        );
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
    println!(
        "   │  Correct: {}/{} ({:.2}%)        │",
        correct,
        total,
        acc * 100.0
    );
    println!("   └─────────────────────────────────┘");

    println!("\n   Test Accuracy: {:.2}%", acc * 100.0);

    // ═══════════════════════════════════════════════════════════════════════
    // 5. SAVE MODEL WITH METADATA
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n💾 Saving model with metadata...\n");

    match save_model_with_normalization(network, acc, total, Some(norm_stats), model_path) {
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
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;

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

/// Normalize features using z-score normalization (mean=0, std=1)
/// Returns normalized data AND the normalization statistics for inference
fn normalize_features_with_stats(inputs: &[Array1<f64>]) -> (Vec<Array1<f64>>, NormalizationStats) {
    if inputs.is_empty() {
        return (vec![], NormalizationStats::new(vec![], vec![]));
    }

    let n_features = inputs[0].len();
    let n_samples = inputs.len() as f64;

    // Calculate mean for each feature
    let mut means = vec![0.0; n_features];
    for input in inputs {
        for (i, &val) in input.iter().enumerate() {
            means[i] += val;
        }
    }
    for mean in &mut means {
        *mean /= n_samples;
    }

    // Calculate standard deviation for each feature
    let mut stds = vec![0.0; n_features];
    for input in inputs {
        for (i, &val) in input.iter().enumerate() {
            stds[i] += (val - means[i]).powi(2);
        }
    }
    for std in &mut stds {
        *std = (*std / n_samples).sqrt();
        // Prevent division by zero
        if *std < 1e-8 {
            *std = 1.0;
        }
    }

    // Normalize each input
    let normalized = inputs
        .iter()
        .map(|input| {
            Array1::from_vec(
                input
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| (val - means[i]) / stds[i])
                    .collect(),
            )
        })
        .collect();
    
    (normalized, NormalizationStats::new(means, stds))
}
