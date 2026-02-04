//! ResNet-Micro CNN Training for MNIST
//!
//! Minimal ResNet (He et al., 2015) for MNIST - very fast version.
//! Uses only 1 conv layer + 1 residual block to keep training fast.
//!
//! Architecture:
//! - Input: 28x28x1
//! - Conv 1→16, 3x3 + ReLU + MaxPool → 14x14x16
//! - Flatten → 3136
//! - FC: 3136 → 64 → 10

use cma_cnn::{ActivationLayer, Conv2D, Layer, MaxPool2D, Sequential, Tensor4D, TensorShape};
use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::callbacks::{DeltaMode, EarlyStopping, ProgressBar};
use cma_neural_network::dataset::Dataset;
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use ndarray::Array1;
use neural_wasm_shared::{load_mnist_from_csv, normalize_features_with_stats, save_model_binary};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     ResNet-Micro CNN for MNIST (He et al., 2015 style)       ║");
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

    let inputs: Vec<Array1<f64>> = mnist_data.iter().map(|(i, _)| i.clone()).collect();
    let targets: Vec<Array1<f64>> = mnist_data.iter().map(|(_, t)| t.clone()).collect();

    let (inputs, norm_stats) = normalize_features_with_stats(&inputs);
    println!("   ✅ Features normalized");

    let mut dataset = Dataset::new(inputs, targets);
    dataset.shuffle();

    let (train, val) = dataset.split(0.8);
    println!("   Training: {} | Validation: {}\n", train.len(), val.len());

    // ═══════════════════════════════════════════════════════════════════════
    // 2. BUILD ResNet-Micro CNN (minimal for speed)
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 Building ResNet-Micro CNN...\n");

    // Very simple: 1 conv + pool → flatten
    let cnn = Sequential::named("ResNet-Micro")
        .add_conv2d(Conv2D::new(1, 16, 3, 1, 1)) // 28x28x16
        .add_activation(ActivationLayer::relu())
        .add_maxpool(MaxPool2D::new(2, 2)) // 14x14x16
        .add_flatten();

    let input_shape = TensorShape::new(1, 1, 28, 28);
    let output_shape = cnn.output_shape(input_shape);
    let flat_size = output_shape.width;

    println!("   CNN Architecture:");
    cnn.summary(input_shape);
    println!("\n   Flattened output size: {}", flat_size);

    let mut classifier = NetworkBuilder::new(flat_size, 10)
        .hidden_layer(64, Activation::ReLU)
        .output_activation(Activation::Softmax)
        .loss(LossFunction::CategoricalCrossEntropy)
        .optimizer(OptimizerType::adam(0.001))
        .build();

    println!("   FC Classifier: {} → 64 → 10", flat_size);

    // ═══════════════════════════════════════════════════════════════════════
    // 3. TRAIN
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n🏋️  Training...\n");

    println!("   📊 Extracting CNN features...");
    let train_features = extract_cnn_features(&cnn, train.inputs());
    let val_features = extract_cnn_features(&cnn, val.inputs());
    println!("   ✅ Features extracted");

    let mut train_fc = Dataset::new(train_features, train.targets().to_vec());
    let val_fc = Dataset::new(val_features, val.targets().to_vec());

    let epochs = 200;
    let history = classifier
        .trainer()
        .train_data(&mut train_fc)
        .validation_data(&val_fc)
        .epochs(epochs)
        .batch_size(128)
        .callback(Box::new(EarlyStopping::new(20, 0.001).mode(DeltaMode::Relative)))
        .callback(Box::new(ProgressBar::new(epochs)))
        .fit();

    println!("\n   ✅ Training completed in {} epochs", history.len());

    // ═══════════════════════════════════════════════════════════════════════
    // 4. EVALUATE
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n📊 Evaluating...\n");

    classifier.eval_mode();

    let mut correct = 0;
    let total = val_fc.len();

    for i in 0..total {
        let output = classifier.predict(&val_fc.inputs()[i]);
        let predicted = output.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx).unwrap();
        let expected = val_fc.targets()[i].iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx).unwrap();
        if predicted == expected { correct += 1; }
    }

    let acc = correct as f64 / total as f64;

    println!("   ResNet-Micro MNIST: {}/{} ({:.2}%)", correct, total, acc * 100.0);

    // ═══════════════════════════════════════════════════════════════════════
    // 5. SAVE MODEL
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n💾 Saving model...\n");

    match save_model_binary(classifier, acc, total, Some(norm_stats), model_path) {
        Ok(_) => {
            println!("   ✅ Model saved to {}", model_path);
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║           ResNet-Micro Training Complete! 🎉                 ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
        }
        Err(e) => {
            eprintln!("   ❌ Failed to save model: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn extract_cnn_features(cnn: &Sequential, inputs: &[Array1<f64>]) -> Vec<Array1<f64>> {
    inputs.iter().map(|flat_input| {
        let pixels: Vec<f64> = flat_input.to_vec();
        let tensor = Tensor4D::from_array(
            ndarray::Array4::from_shape_vec((1, 1, 28, 28), pixels).expect("reshape failed"),
        );
        let features = cnn.forward(&tensor);
        let flat = features.flatten();
        Array1::from_vec(flat.row(0).to_vec())
    }).collect()
}
