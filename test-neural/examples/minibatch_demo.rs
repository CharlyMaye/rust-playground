//! Mini-batch training demonstration
//!
//! Compares three training approaches:
//!   1. Sample-by-sample  — network.train() per example
//!   2. Manual batches    — network.train_batch() with Dataset::batches()
//!   3. TrainingBuilder   — .batch_size(N).fit() (the recommended high-level API)
//!
//! Run: cargo run --example minibatch_demo

use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::dataset::Dataset;
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use cma_neural_network::Float;
use ndarray::array;
use std::time::Instant;

fn main() {
    println!("=== Mini-Batch Training Demonstration ===\n");

    // Build a 1000-sample XOR-like dataset with small noise
    println!("📊 Building dataset...");
    let mut inputs  = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..250 {
        inputs.push(array![0.0_f32 + rand::random::<Float>() * 0.1,
                            0.0_f32 + rand::random::<Float>() * 0.1]);
        targets.push(array![0.0_f32]);

        inputs.push(array![0.0_f32 + rand::random::<Float>() * 0.1,
                            1.0_f32 - rand::random::<Float>() * 0.1]);
        targets.push(array![1.0_f32]);

        inputs.push(array![1.0_f32 - rand::random::<Float>() * 0.1,
                            0.0_f32 + rand::random::<Float>() * 0.1]);
        targets.push(array![1.0_f32]);

        inputs.push(array![1.0_f32 - rand::random::<Float>() * 0.1,
                            1.0_f32 - rand::random::<Float>() * 0.1]);
        targets.push(array![0.0_f32]);
    }

    println!("Dataset: {} samples\n", inputs.len());

    // Create dataset and split
    let dataset = Dataset::new(inputs, targets);
    let (train_dataset, test_dataset) = dataset.split(0.8);

    println!("Train: {} samples", train_dataset.len());
    println!("Test:  {} samples\n", test_dataset.len());

    // ===== 1. Single-sample training (baseline) =====
    println!("--- 1. Sample-by-sample training ---");
    let mut network_single = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.001))
        .build();

    network_single.set_seed(42); // Reproducible training

    let start = Instant::now();
    let epochs = 50;

    for epoch in 0..epochs {
        // Train on each example individually
        for (input, target) in train_dataset
            .inputs()
            .iter()
            .zip(train_dataset.targets().iter())
        {
            network_single.train(input, target);
        }

        if (epoch + 1) % 10 == 0 {
            let loss = network_single.evaluate(train_dataset.inputs(), train_dataset.targets());
            println!("  Epoch {}: loss = {:.6}", epoch + 1, loss);
        }
    }

    let duration_single = start.elapsed();
    let test_loss_single = network_single.evaluate(test_dataset.inputs(), test_dataset.targets());

    println!("✓ Time: {:.2}s", duration_single.as_secs_f64());
    println!("✓ Final loss (test): {:.6}\n", test_loss_single);

    // ===== 2. Mini-batch training (batch_size = 32) =====
    println!("--- 2. Mini-batch training (batch_size=32) ---");
    let mut network_batch32 = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01)) // Increased learning rate for batch training
        .build();

    network_batch32.set_seed(42); // Reproducible training

    let start = Instant::now();
    let batch_size = 32;

    let mut train_data_shuffleable = train_dataset.clone();

    for epoch in 0..epochs {
        // Shuffle before each epoch
        train_data_shuffleable.shuffle();

        // Train on batches
        for (batch_inputs, batch_targets) in train_data_shuffleable.batches(batch_size) {
            network_batch32.train_batch(&batch_inputs, &batch_targets);
        }

        if (epoch + 1) % 10 == 0 {
            let loss = network_batch32.evaluate(train_dataset.inputs(), train_dataset.targets());
            println!("  Epoch {}: loss = {:.6}", epoch + 1, loss);
        }
    }

    let duration_batch32 = start.elapsed();
    let test_loss_batch32 = network_batch32.evaluate(test_dataset.inputs(), test_dataset.targets());

    println!("✓ Time: {:.2}s", duration_batch32.as_secs_f64());
    println!("✓ Final loss (test): {:.6}", test_loss_batch32);
    println!(
        "✓ Speedup: {:.2}x faster\n",
        duration_single.as_secs_f64() / duration_batch32.as_secs_f64()
    );

    // ===== 3. Mini-batch training (batch_size = 64) =====
    println!("--- 3. Mini-batch training (batch_size=64) ---");
    let mut network_batch64 = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();

    network_batch64.set_seed(42); // Reproducible training

    let start = Instant::now();
    let batch_size = 64;

    for epoch in 0..epochs {
        train_data_shuffleable.shuffle();

        for (batch_inputs, batch_targets) in train_data_shuffleable.batches(batch_size) {
            network_batch64.train_batch(&batch_inputs, &batch_targets);
        }

        if (epoch + 1) % 10 == 0 {
            let loss = network_batch64.evaluate(train_dataset.inputs(), train_dataset.targets());
            println!("  Epoch {}: loss = {:.6}", epoch + 1, loss);
        }
    }

    let duration_batch64 = start.elapsed();
    let test_loss_batch64 = network_batch64.evaluate(test_dataset.inputs(), test_dataset.targets());

    println!("✓ Time: {:.2}s", duration_batch64.as_secs_f64());
    println!("✓ Final loss (test): {:.6}", test_loss_batch64);
    println!(
        "✓ Speedup: {:.2}x faster\n",
        duration_single.as_secs_f64() / duration_batch64.as_secs_f64()
    );

    // ===== 4. Mini-batch training (batch_size = 128) =====
    println!("--- 4. Mini-batch training (batch_size=128) ---");
    let mut network_batch128 = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();

    network_batch128.set_seed(42); // Reproducible training

    let start = Instant::now();
    let batch_size = 128;

    for epoch in 0..epochs {
        train_data_shuffleable.shuffle();

        for (batch_inputs, batch_targets) in train_data_shuffleable.batches(batch_size) {
            network_batch128.train_batch(&batch_inputs, &batch_targets);
        }

        if (epoch + 1) % 10 == 0 {
            let loss = network_batch128.evaluate(train_dataset.inputs(), train_dataset.targets());
            println!("  Epoch {}: loss = {:.6}", epoch + 1, loss);
        }
    }

    let duration_batch128 = start.elapsed();
    let test_loss_batch128 =
        network_batch128.evaluate(test_dataset.inputs(), test_dataset.targets());

    println!("✓ Time: {:.2}s", duration_batch128.as_secs_f64());
    println!("✓ Final loss (test): {:.6}", test_loss_batch128);
    println!(
        "✓ Speedup: {:.2}x faster\n",
        duration_single.as_secs_f64() / duration_batch128.as_secs_f64()
    );

    // ===== 5. TrainingBuilder (recommended high-level API) =====
    println!("--- 5. TrainingBuilder (high-level, batch_size=32) ---");
    let mut network_builder = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    network_builder.set_seed(42);

    let mut train_for_builder = train_dataset.clone();
    let start = Instant::now();
    network_builder
        .trainer()
        .train_data(&mut train_for_builder)
        .epochs(epochs)
        .batch_size(32)
        .verbose(0)
        .fit();
    let duration_builder = start.elapsed();
    let test_loss_builder = network_builder.evaluate(test_dataset.inputs(), test_dataset.targets());

    println!("✓ Time: {:.2}s", duration_builder.as_secs_f64());
    println!("✓ Final loss (test): {:.6}", test_loss_builder);
    println!(
        "✓ Speedup: {:.2}x faster (vs single-sample)\n",
        duration_single.as_secs_f64() / duration_builder.as_secs_f64()
    );

    // ===== Summary =====
    println!("\n=== Performance Summary ===");
    println!(
        "  • 1. Sample-by-sample:       {:.2}s (baseline)",
        duration_single.as_secs_f64()
    );
    println!(
        "  • 2. train_batch (size=32):  {:.2}s ({:.1}x speedup)",
        duration_batch32.as_secs_f64(),
        duration_single.as_secs_f64() / duration_batch32.as_secs_f64()
    );
    println!(
        "  • 3. train_batch (size=64):  {:.2}s ({:.1}x speedup)",
        duration_batch64.as_secs_f64(),
        duration_single.as_secs_f64() / duration_batch64.as_secs_f64()
    );
    println!(
        "  • 4. train_batch (size=128): {:.2}s ({:.1}x speedup)",
        duration_batch128.as_secs_f64(),
        duration_single.as_secs_f64() / duration_batch128.as_secs_f64()
    );
    println!(
        "  • 5. TrainingBuilder (32):   {:.2}s ({:.1}x speedup) ← recommended",
        duration_builder.as_secs_f64(),
        duration_single.as_secs_f64() / duration_builder.as_secs_f64()
    );
    println!("\nSee docs/guides/02-dense-network.md §3 for TrainingBuilder API reference.");
}
