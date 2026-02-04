//! LeNet-5 CNN Training for MNIST
//!
//! This binary trains a LeNet-5 convolutional neural network on MNIST
//! and saves the model for WASM deployment.
//!
//! Architecture (adapted from LeCun et al., 1998):
//! - Input: 28x28x1 (grayscale)
//! - C1: Conv2D 1→6, 5x5, padding=2 → 28x28x6
//! - S2: MaxPool 2x2 → 14x14x6
//! - C3: Conv2D 6→16, 5x5 → 10x10x16
//! - S4: MaxPool 2x2 → 5x5x16
//! - C5: Conv2D 16→120, 5x5 → 1x1x120
//! - Flatten → 120
//! - FC: 120 → 84 → 10

use cma_cnn::Float;
use cma_cnn::{
    ActivationLayer, AvgPool2D, Conv2D, Flatten, Layer, MaxPool2D, Sequential, Tensor4D,
    TensorShape,
};
use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::callbacks::{
    DeltaMode, EarlyStopping, LRSchedule, LearningRateScheduler, ProgressBar,
};
use cma_neural_network::dataset::Dataset;
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use ndarray::Array1;
use neural_wasm_shared::{load_mnist_from_csv, normalize_features_with_stats, save_model_binary};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     LeNet-5 CNN for MNIST - Training (LeCun et al., 1998)    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let model_path = "src/lenet_model.bin";

    // Check if model already exists
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

    // Normalize inputs (z-score normalization per feature)
    let (inputs, norm_stats) = normalize_features_with_stats(&inputs);
    println!("   ✅ Features normalized (z-score)");

    let mut dataset = Dataset::new(inputs, targets);
    dataset.shuffle();
    println!("   ✅ Dataset shuffled");

    let (train, val) = dataset.split(0.8);
    println!("   Training samples: {} (80%)", train.len());
    println!("   Validation samples: {} (20%)\n", val.len());

    // ═══════════════════════════════════════════════════════════════════════
    // 2. BUILD LeNet-5 CNN
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 Building LeNet-5 CNN...\n");

    // Build CNN feature extractor
    let cnn = Sequential::new()
        // C1: Conv 1→6, 5x5, padding=2 (same) → 28x28x6
        .add_conv2d(Conv2D::new(1, 6, 5, 1, 2))
        .add_activation(ActivationLayer::relu())
        // S2: AvgPool 2x2 → 14x14x6 (LeNet original used average pooling)
        .add_avgpool(AvgPool2D::new(2, 2))
        // C3: Conv 6→16, 5x5 → 10x10x16
        .add_conv2d(Conv2D::new(6, 16, 5, 1, 0))
        .add_activation(ActivationLayer::relu())
        // S4: AvgPool 2x2 → 5x5x16
        .add_avgpool(AvgPool2D::new(2, 2))
        // C5: Conv 16→120, 5x5 → 1x1x120
        .add_conv2d(Conv2D::new(16, 120, 5, 1, 0))
        .add_activation(ActivationLayer::relu())
        // Flatten
        .add_flatten();

    // Calculate flattened size
    let input_shape = TensorShape::new(1, 1, 28, 28);
    let output_shape = cnn.output_shape(input_shape);
    let flat_size = output_shape.width; // After flatten: [1, 1, 1, 120]

    println!("   CNN Architecture:");
    cnn.summary(input_shape);
    println!("\n   Flattened output size: {}", flat_size);

    // Build FC classifier - Higher LR for random CNN features
    let mut classifier = NetworkBuilder::new(flat_size, 10)
        .hidden_layer(128, Activation::ReLU)
        .hidden_layer(84, Activation::ReLU)
        .output_activation(Activation::Softmax)
        .loss(LossFunction::CategoricalCrossEntropy)
        .optimizer(OptimizerType::adam(0.01)) // Higher LR
        .dropout(0.3)
        .build();

    println!("   FC Classifier: {} → 84 → 10", flat_size);

    // ═══════════════════════════════════════════════════════════════════════
    // 3. TRAIN (CNN features are fixed, only FC is trained)
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n🏋️  Training (CNN forward + FC backprop)...\n");

    // Pre-compute CNN features for all samples
    println!("   📊 Extracting CNN features...");
    let train_features = extract_cnn_features(&cnn, train.inputs());
    let val_features = extract_cnn_features(&cnn, val.inputs());
    println!(
        "   ✅ Features extracted: {} training, {} validation",
        train_features.len(),
        val_features.len()
    );

    // Create datasets with CNN features
    let mut train_fc = Dataset::new(train_features, train.targets().to_vec());
    let val_fc = Dataset::new(val_features, val.targets().to_vec());

    let epochs = 500;
    let history = classifier
        .trainer()
        .parallel() // Enable multi-threaded training
        .train_data(&mut train_fc)
        .validation_data(&val_fc)
        .epochs(epochs)
        .batch_size(64) // Smaller batch
        .max_grad_norm(1.0)
        .scheduler(LearningRateScheduler::new(LRSchedule::ReduceOnPlateau {
            patience: 25,
            factor: 0.5,
            min_delta: 0.001,
        }))
        .callback(Box::new(
            EarlyStopping::new(50, 0.005).mode(DeltaMode::Relative),
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

    println!("   LeNet-5 MNIST Classification Results:");
    println!("   ┌─────────────────────────────────┐");
    println!(
        "   │  Correct: {}/{} ({:.2}%)      │",
        correct,
        total,
        acc * 100.0
    );
    println!("   └─────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════
    // 5. SAVE MODEL
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n💾 Saving model...\n");

    // Note: We save only the FC classifier for now
    // The CNN weights would need a separate serialization format
    match save_model_binary(classifier, acc, total, Some(norm_stats), model_path) {
        Ok(_) => {
            println!("   ✅ Model saved to {}", model_path);
            println!("   📊 Accuracy: {:.2}%", acc * 100.0);
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║              LeNet-5 Training Complete! 🎉                   ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
        }
        Err(e) => {
            eprintln!("   ❌ Failed to save model: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Extract CNN features for all samples
fn extract_cnn_features(cnn: &Sequential, inputs: &[Array1<Float>]) -> Vec<Array1<Float>> {
    inputs
        .iter()
        .map(|flat_input| {
            // Reshape flat 784 → [1, 1, 28, 28]
            let pixels: Vec<Float> = flat_input.to_vec();
            let tensor = Tensor4D::from_array(
                ndarray::Array4::from_shape_vec((1, 1, 28, 28), pixels)
                    .expect("Failed to reshape input"),
            );

            // Forward through CNN
            let features = cnn.forward(&tensor);

            // Flatten to Array1
            let flat = features.flatten();
            Array1::from_vec(flat.row(0).to_vec())
        })
        .collect()
}
