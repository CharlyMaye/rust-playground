//! Model serialization demo
//!
//! Demonstrates:
//! - Saving and loading models in JSON and binary formats
//! - File size comparison between formats
//! - Verifying loaded models produce identical predictions
//!
//! Run: cargo run --example serialization

use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
use cma_neural_network::dataset::Dataset;
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use cma_neural_network::io;
use ndarray::array;
use std::fs;

fn main() {
    println!("=== Serialization Demo ===\n");

    let data_dir = "examples/data";
    fs::create_dir_all(data_dir).expect("Failed to create data directory");

    // --- 1. Build and train ---
    println!("1. Building and training a network (TrainingBuilder)...");

    let test_inputs  = vec![
        array![0.0f32, 0.0], array![0.0, 1.0],
        array![1.0, 0.0],    array![1.0, 1.0],
    ];
    let test_targets = vec![array![0.0f32], array![1.0], array![1.0], array![0.0]];

    // Repeat XOR 50× so TrainingBuilder has enough samples for mini-batching
    let mut raw_inputs  = Vec::new();
    let mut raw_targets = Vec::new();
    for _ in 0..50 {
        raw_inputs.extend_from_slice(&test_inputs);
        raw_targets.extend_from_slice(&test_targets);
    }
    let mut dataset = Dataset::new(raw_inputs, raw_targets);
    dataset.shuffle();

    let mut network = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    network.set_seed(42);

    network
        .trainer()
        .train_data(&mut dataset)
        .epochs(300)
        .batch_size(16)
        .verbose(0)
        .fit();

    let train_loss = network.evaluate(&test_inputs, &test_targets);
    println!("  Training complete. Loss: {:.4}\n", train_loss);

    // Verify predictions before saving
    println!("2. Predictions before saving:");
    for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
        let pred = network.predict(input);
        println!("  {:?} -> {:.3}  (expected {:.0})",
            input.as_slice().unwrap(), pred[0], target[0]);
    }
    // --- 3. Save to JSON ---
    println!("\n3. Saving to JSON...");
    let json_path = format!("{}/xor_model.json", data_dir);
    match io::save_json(&network, &json_path) {
        Ok(_)  => println!("  Saved to {}", json_path),
        Err(e) => eprintln!("  Error: {}", e),
    }

    // --- 4. Save to binary ---
    println!("\n4. Saving to binary...");
    let bin_path = format!("{}/xor_model.bin", data_dir);
    match io::save_binary(&network, &bin_path) {
        Ok(_)  => println!("  Saved to {}", bin_path),
        Err(e) => eprintln!("  Error: {}", e),
    }

    // --- 5. File size comparison ---
    println!("\n5. File size comparison:");
    let (json_bytes, bin_bytes) = io::get_serialized_size(&network);
    println!("  JSON:   {} bytes", json_bytes);
    println!("  Binary: {} bytes", bin_bytes);
    println!("  Ratio:  {:.2}× (binary is more compact)",
        json_bytes as f64 / bin_bytes as f64);

    // --- 6. Load from JSON and verify ---
    println!("\n6. Loading from JSON and verifying...");
    match io::load_json(&json_path) {
        Ok(loaded) => {
            println!("  Loaded from {}", json_path);
            for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
                let pred = loaded.predict(input);
                let ok   = (pred[0].round() - target[0]).abs() < 0.1;
                println!("  {:?} -> {:.3}  (expected {:.0}) {}",
                    input.as_slice().unwrap(), pred[0], target[0],
                    if ok { "✓" } else { "✗" });
            }
            let loaded_loss = loaded.evaluate(&test_inputs, &test_targets);
            println!("  Loaded loss: {:.4}", loaded_loss);
        }
        Err(e) => eprintln!("  Error loading JSON: {}", e),
    }

    // --- 7. Load from binary and verify ---
    println!("\n7. Loading from binary and verifying...");
    match io::load_binary(&bin_path) {
        Ok(loaded) => {
            println!("  Loaded from {}", bin_path);
            for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
                let pred = loaded.predict(input);
                let ok   = (pred[0].round() - target[0]).abs() < 0.1;
                println!("  {:?} -> {:.3}  (expected {:.0}) {}",
                    input.as_slice().unwrap(), pred[0], target[0],
                    if ok { "✓" } else { "✗" });
            }
            let loaded_loss = loaded.evaluate(&test_inputs, &test_targets);
            println!("  Loaded loss: {:.4}", loaded_loss);
        }
        Err(e) => eprintln!("  Error loading binary: {}", e),
    }

    println!("\n=== Demo Complete ===");
    println!("Generated files:");
    println!("  - {} (human-readable JSON)", json_path);
    println!("  - {} (compact binary)", bin_path);
    println!("See docs/guides/02-dense-network.md §6 for full IO API reference.");
}
