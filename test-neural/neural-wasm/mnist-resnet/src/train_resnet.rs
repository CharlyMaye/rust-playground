//! ResNet-MNIST: End-to-End CNN Training with Autograd
//!
//! Trains a deep CNN end-to-end using cma_autograd's automatic differentiation,
//! then exports trained weights to cma-cnn format for WASM inference.
//!
//! Architecture (ResNet-style plain network):
//! - Stem:     Conv(1→16, 3×3, p=1) → BN → ReLU           28×28×16
//! - Stage 1:  Conv(16→16, 3×3, p=1) → BN → ReLU          28×28×16
//! - Stage 2a: Conv(16→32, 3×3, s=2, p=1) → BN → ReLU     14×14×32
//! - Stage 2b: Conv(32→32, 3×3, p=1) → BN → ReLU          14×14×32
//! - Stage 3a: Conv(32→64, 3×3, s=2, p=1) → BN → ReLU     7×7×64
//! - Stage 3b: Conv(64→64, 3×3, p=1) → BN → ReLU          7×7×64
//! - Head:     GlobalAvgPool → Flatten → Linear(64→10)
//!
//! Pipeline:
//! 1. Train autograd Sequential (end-to-end backprop through all layers)
//! 2. Export CNN → cma-cnn Sequential (via export_cnn_to_inference)
//! 3. Extract features with exported CNN
//! 4. Quick-train FC head (cma-neural-network Network)
//! 5. Save full model: save_cnn_model_binary(cnn, classifier, metadata)

use cma_autograd::prelude::*;
use cma_cnn::Float;
use cma_cnn::Tensor4D;
use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::callbacks::{
    DeltaMode, EarlyStopping, LRSchedule, LearningRateScheduler, ProgressBar,
};
use cma_neural_network::dataset::Dataset;
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use ndarray::{Array1, ArrayD, IxDyn};
use neural_wasm_shared::{load_mnist_from_csv, normalize_features_with_stats, save_cnn_model_binary};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     ResNet-MNIST: End-to-End CNN Training with Autograd     ║");
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

    // Deterministic shuffle
    let n = inputs.len();
    let split = (n as f64 * 0.8) as usize;
    let mut indices: Vec<usize> = (0..n).collect();
    {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        for i in (1..n).rev() {
            let mut h = DefaultHasher::new();
            i.hash(&mut h);
            let j = h.finish() as usize % (i + 1);
            indices.swap(i, j);
        }
    }

    // Convert to autograd format: [1, 1, 28, 28] images, [1, 10] targets
    let train_inputs: Vec<ArrayD<Float>> = indices[..split]
        .iter()
        .map(|&i| ArrayD::from_shape_vec(IxDyn(&[1, 1, 28, 28]), inputs[i].to_vec()).unwrap())
        .collect();
    let train_targets: Vec<ArrayD<Float>> = indices[..split]
        .iter()
        .map(|&i| ArrayD::from_shape_vec(IxDyn(&[1, 10]), targets[i].to_vec()).unwrap())
        .collect();
    let val_inputs: Vec<ArrayD<Float>> = indices[split..]
        .iter()
        .map(|&i| ArrayD::from_shape_vec(IxDyn(&[1, 1, 28, 28]), inputs[i].to_vec()).unwrap())
        .collect();
    let val_targets: Vec<ArrayD<Float>> = indices[split..]
        .iter()
        .map(|&i| ArrayD::from_shape_vec(IxDyn(&[1, 10]), targets[i].to_vec()).unwrap())
        .collect();

    println!(
        "   Training: {} | Validation: {}\n",
        train_inputs.len(),
        val_inputs.len()
    );

    // ═══════════════════════════════════════════════════════════════════════
    // 2. BUILD CNN MODEL (autograd end-to-end trainable)
    // ═══════════════════════════════════════════════════════════════════════
    println!("🔧 Building ResNet-style CNN (autograd)...\n");

    let mut model = CnnBuilder::resnet_mnist(10);

    model.train();
    model.summary();
    println!("   Total parameters: {}\n", model.num_parameters());

    // ═══════════════════════════════════════════════════════════════════════
    // 3. TRAIN END-TO-END (autograd backprop through all layers)
    // ═══════════════════════════════════════════════════════════════════════
    println!("🏋️  Training end-to-end with autograd backprop...\n");

    let params: Vec<Parameter> = model.parameters().into_iter().cloned().collect();
    let mut optimizer = Adam::new(params, 0.001);

    let history = model.trainer(&mut optimizer)
        .train_data(&train_inputs, &train_targets)
        .validation_data(&val_inputs, &val_targets)
        .loss_fn(cross_entropy_loss)
        .epochs(10)
        .batch_size(64)
        .early_stopping(3)
        .fit();

    let last = history.last().unwrap();
    println!(
        "\n   ✅ Autograd training done — {} epochs",
        history.len()
    );
    println!(
        "   Train loss: {:.4} | Val loss: {:.4}",
        last.train_loss,
        last.val_loss.unwrap_or(0.0)
    );
    if let Some(acc) = last.val_accuracy {
        println!("   Val accuracy: {:.2}%", acc * 100.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. EXPORT CNN → cma-cnn Sequential
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n🔄 Exporting trained CNN to cma-cnn format...\n");

    model.eval();

    let exported = cma_autograd::export::export_cnn_to_inference(&model)
        .expect("Failed to export CNN to cma-cnn");

    println!(
        "   ✅ CNN exported ({} layers in cma-cnn Sequential)",
        exported.cnn.layers().len()
    );

    // ═══════════════════════════════════════════════════════════════════════
    // 5. EXTRACT FEATURES with exported CNN + TRAIN FC HEAD
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n📊 Extracting features with trained CNN...\n");

    // Collect flat inputs in shuffled order for feature extraction
    let train_flat: Vec<Array1<Float>> = indices[..split]
        .iter()
        .map(|&i| inputs[i].clone())
        .collect();
    let val_flat: Vec<Array1<Float>> = indices[split..]
        .iter()
        .map(|&i| inputs[i].clone())
        .collect();
    let train_tgt: Vec<Array1<Float>> = indices[..split]
        .iter()
        .map(|&i| targets[i].clone())
        .collect();
    let val_tgt: Vec<Array1<Float>> = indices[split..]
        .iter()
        .map(|&i| targets[i].clone())
        .collect();

    let train_features = extract_cnn_features(&exported.cnn, &train_flat);
    let val_features = extract_cnn_features(&exported.cnn, &val_flat);

    let feature_dim = train_features[0].len();
    println!(
        "   ✅ Features: {}D, {} train, {} val",
        feature_dim,
        train_features.len(),
        val_features.len()
    );

    // Train FC classifier head
    println!("\n🎯 Training FC classifier on trained CNN features...\n");

    let mut classifier = NetworkBuilder::new(feature_dim, 10)
        .hidden_layer(128, Activation::ReLU)
        .hidden_layer(64, Activation::ReLU)
        .dropout(0.3)
        .output_activation(Activation::Softmax)
        .loss(LossFunction::CategoricalCrossEntropy)
        .optimizer(OptimizerType::adam(0.001))
        .build();

    let mut train_fc = Dataset::new(train_features, train_tgt);
    let val_fc = Dataset::new(val_features, val_tgt);

    let fc_epochs = 500;
    let fc_history = classifier
        .trainer()
        .parallel()
        .train_data(&mut train_fc)
        .validation_data(&val_fc)
        .epochs(fc_epochs)
        .batch_size(64)
        .max_grad_norm(1.0)
        .scheduler(LearningRateScheduler::new(LRSchedule::ReduceOnPlateau {
            patience: 25,
            factor: 0.5,
            min_delta: 0.001,
        }))
        .callback(Box::new(
            EarlyStopping::new(50, 0.005).mode(DeltaMode::Relative),
        ))
        .callback(Box::new(ProgressBar::new(fc_epochs)))
        .fit();

    println!(
        "\n   ✅ FC training completed in {} epochs",
        fc_history.len()
    );

    // ═══════════════════════════════════════════════════════════════════════
    // 6. EVALUATE
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n📊 Evaluating final model...\n");

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
    // 7. SAVE MODEL (CNN + FC + metadata)
    // ═══════════════════════════════════════════════════════════════════════
    println!("\n💾 Saving model...\n");

    match save_cnn_model_binary(
        exported.cnn,
        classifier,
        acc,
        total,
        Some(norm_stats),
        model_path,
    ) {
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

/// Extract CNN features for a batch of flat MNIST inputs
fn extract_cnn_features(
    cnn: &cma_cnn::sequential::Sequential,
    inputs: &[Array1<Float>],
) -> Vec<Array1<Float>> {
    inputs
        .iter()
        .map(|flat_input| {
            let tensor = Tensor4D::from_array(
                ndarray::Array4::from_shape_vec((1, 1, 28, 28), flat_input.to_vec())
                    .expect("Failed to reshape MNIST input to 28×28"),
            );
            let features = cnn.forward(&tensor);
            let flat = features.flatten();
            Array1::from_vec(flat.row(0).to_vec())
        })
        .collect()
}
