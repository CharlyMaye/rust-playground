//! ResNet-MNIST CNN Training
//!
//! Proper ResNet (He et al., 2015) with residual blocks for MNIST.
//! Uses ResNetBuilder from cma_models for flexible architecture configuration.
//!
//! Architecture (via ResNetBuilder::mnist()):
//! - Input: 28x28x1 (grayscale)
//! - Stem: Conv 1→16, 3x3 → 28x28x16
//! - Stage 1: 2× BasicBlock(16→16) → 28x28x16
//! - Stage 2: 2× BasicBlock(16→32, stride=2) → 14x14x32
//! - Stage 3: 2× BasicBlock(32→64, stride=2) → 7x7x64
//! - Global Average Pooling → 64
//! - FC: 64 → 10

use cma_cnn::Float;
use cma_cnn::Tensor4D;
use cma_models::resnet::{ResNet, ResNetBuilder};
use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::callbacks::{DeltaMode, EarlyStopping, LRSchedule, LearningRateScheduler, ProgressBar};
use cma_neural_network::dataset::Dataset;
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use ndarray::Array1;
use neural_wasm_shared::{load_mnist_from_csv, normalize_features_with_stats, save_model_binary};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       ResNet-MNIST CNN (He et al., 2015) with Skip Conn      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let model_path = "src/resnet_model.bin";

    if std::path::Path::new(model_path).exists() {
        println!("⚠️  Model already exists at {}", model_path);
        println!("   Delete it manually if you want to retrain.\n");
        return Ok(());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 1. DATA PREPARATION
    // ═══════════════════════════════════════════════════════════════════════
    println!("📦 Preparing MNIST dataset...\n");

    let mnist_data = load_mnist_from_csv("../mnist/data/mnist.csv")?;
    println!("   ✅ Loaded {} samples from CSV", mnist_data.len());

    let inputs: Vec<Array1<Float>> = mnist_data.iter().map(|(i, _)| i.clone()).collect();
    let targets: Vec<Array1<Float>> = mnist_data.iter().map(|(_, t)| t.clone()).collect();

    let (inputs, norm_stats) = normalize_features_with_stats(&inputs);
    println!("   ✅ Features normalized (z-score)");

    let mut dataset = Dataset::new(inputs, targets);
    dataset.shuffle();
    println!("   ✅ Dataset shuffled");

    let (train, val) = dataset.split(0.8);
    println!("   Training: {} | Validation: {}\n", train.len(), val.len());

    // ═══════════════════════════════════════════════════════════════════════
    // 2. BUILD ResNet using cma_models::ResNetBuilder
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 Building ResNet-MNIST using ResNetBuilder...\n");

    // Use the library's builder with MNIST preset
    let resnet = ResNetBuilder::mnist().build();
    resnet.summary();

    let flat_size = resnet.output_features();

    // FC classifier - Higher LR because features are from random CNN
    let mut classifier = NetworkBuilder::new(flat_size, 10)
        .hidden_layer(128, Activation::ReLU)  // Larger hidden layer
        .hidden_layer(64, Activation::ReLU)   // Additional layer
        .output_activation(Activation::Softmax)
        .loss(LossFunction::CategoricalCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))  // Higher LR for feature learning
        .dropout(0.3)  // Regularization
        .build();

    println!("\n   FC Classifier: {} → 64 → 10", flat_size);

    // ═══════════════════════════════════════════════════════════════════════
    // 3. TRAIN (CNN features are fixed, only FC is trained)
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n🏋️  Training (ResNet forward + FC backprop)...\n");

    println!("   📊 Extracting ResNet features...");
    let train_features = extract_resnet_features(&resnet, train.inputs());
    let val_features = extract_resnet_features(&resnet, val.inputs());
    println!(
        "   ✅ Features extracted: {} training, {} validation",
        train_features.len(),
        val_features.len()
    );

    let mut train_fc = Dataset::new(train_features, train.targets().to_vec());
    let val_fc = Dataset::new(val_features, val.targets().to_vec());

    let epochs = 500;
    let history = classifier
        .trainer()
        .parallel() // Enable multi-threaded training
        .train_data(&mut train_fc)
        .validation_data(&val_fc)
        .epochs(epochs)
        .batch_size(64)  // Smaller batch for better gradient estimates
        .max_grad_norm(1.0) // Tighter gradient clipping
        .scheduler(LearningRateScheduler::new(
            LRSchedule::ReduceOnPlateau {
                patience: 25,   // Wait longer before reducing LR
                factor: 0.5,
                min_delta: 0.001,  // Larger threshold for improvement
            },
        ))
        .callback(Box::new(
            EarlyStopping::new(50, 0.005).mode(DeltaMode::Relative),  // More patience
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

    classifier.eval_mode();

    let mut correct = 0;
    let total = val_fc.len();

    for i in 0..total {
        let output = classifier.predict(&val_fc.inputs()[i]);
        let predicted = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let expected = val_fc.targets()[i]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if predicted == expected {
            correct += 1;
        }
    }

    let acc = correct as Float / total as Float;

    println!("   ResNet-MNIST Classification Results:");
    println!("   ┌─────────────────────────────────────┐");
    println!(
        "   │  Correct: {}/{} ({:.2}%)      │",
        correct,
        total,
        acc * 100.0
    );
    println!("   └─────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════
    // 5. SAVE MODEL
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n💾 Saving model...\n");

    match save_model_binary(classifier, acc, total, Some(norm_stats), model_path) {
        Ok(_) => {
            println!("   ✅ Model saved to {}", model_path);
            println!("   📊 Accuracy: {:.2}%", acc * 100.0);
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║              ResNet-MNIST Training Complete! 🎉              ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
        }
        Err(e) => {
            eprintln!("   ❌ Failed to save model: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Extract ResNet features for all samples
fn extract_resnet_features(resnet: &ResNet, inputs: &[Array1<Float>]) -> Vec<Array1<Float>> {
    inputs
        .iter()
        .map(|flat_input| {
            // Reshape flat 784 → [1, 1, 28, 28]
            let pixels: Vec<Float> = flat_input.to_vec();
            let tensor = Tensor4D::from_array(
                ndarray::Array4::from_shape_vec((1, 1, 28, 28), pixels)
                    .expect("Failed to reshape input"),
            );

            // Forward through ResNet (returns flattened 64-dim vector)
            let features = resnet.forward(&tensor);

            // Flatten to Array1
            let flat = features.flatten();
            Array1::from_vec(flat.row(0).to_vec())
        })
        .collect()
}
