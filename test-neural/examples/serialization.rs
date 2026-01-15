use cma_neural_network::builder::NetworkBuilder;
use cma_neural_network::network::{Activation, LossFunction};
use cma_neural_network::optimizer::OptimizerType;
use cma_neural_network::io;
use ndarray::array;
use std::fs;

fn main() {
    println!("=== Neural Network Serialization Demo ===\n");

    // Create data directory for output files
    let data_dir = "examples/data";
    fs::create_dir_all(data_dir).expect("Failed to create data directory");
    
    // 1. Create and train a network
    println!("1. Creating and training network on XOR...");
    let mut network = NetworkBuilder::new(2, 1)
        .hidden_layer(5, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    
    let inputs = vec![
        array![0.0, 0.0],
        array![0.0, 1.0],
        array![1.0, 0.0],
        array![1.0, 1.0],
    ];
    
    let targets = vec![
        array![0.0],
        array![1.0],
        array![1.0],
        array![0.0],
    ];
    
    // Train the network
    for epoch in 0..10_000 {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            network.train(input, target);
        }
        
        if (epoch + 1) % 2_000 == 0 {
            let loss = network.evaluate(&inputs, &targets);
            println!("  Epoch {}: loss = {:.4}", epoch + 1, loss);
        }
    }
    
    let final_loss = network.evaluate(&inputs, &targets);
    println!("  Training complete! Final loss: {:.4}\n", final_loss);
    
    // Test predictions before saving
    println!("2. Testing predictions before saving:");
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let prediction = network.predict(input);
        println!("  {:?} -> {:.3} (expected {:.0})", 
            input.as_slice().unwrap(), prediction[0], target[0]);
    }
    
    // 2. Save to JSON
    println!("\n3. Saving to JSON...");
    let json_path = format!("{}/xor_model.json", data_dir);
    match io::save_json(&network, &json_path) {
        Ok(_) => println!("  ✓ Saved to {}", json_path),
        Err(e) => println!("  ✗ Error: {}", e),
    }
    
    // 3. Save to binary
    println!("\n4. Saving to binary...");
    let bin_path = format!("{}/xor_model.bin", data_dir);
    match io::save_binary(&network, &bin_path) {
        Ok(_) => println!("  ✓ Saved to {}", bin_path),
        Err(e) => println!("  ✗ Error: {}", e),
    }
    
    // 4. Compare sizes
    println!("\n5. Comparing file sizes:");
    let (json_size, bin_size) = io::get_serialized_size(&network);
    println!("  JSON: {} bytes", json_size);
    println!("  Binary: {} bytes", bin_size);
    println!("  Compression ratio: {:.2}x (binary is more compact)", 
        json_size as f64 / bin_size as f64);
    
    // 5. Load from JSON
    println!("\n6. Loading from JSON...");
    match io::load_json(&json_path) {
        Ok(loaded_network) => {
            println!("  ✓ Loaded from {}", json_path);
            
            println!("\n7. Testing loaded network (JSON):");
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let prediction = loaded_network.predict(input);
                let matches = (prediction[0] - target[0]).abs() < 0.1;
                println!("  {:?} -> {:.3} (expected {:.0}) {}", 
                    input.as_slice().unwrap(), 
                    prediction[0], 
                    target[0],
                    if matches { "✓" } else { "✗" });
            }
            
            let loaded_loss = loaded_network.evaluate(&inputs, &targets);
            println!("  Loaded network loss: {:.4}", loaded_loss);
        },
        Err(e) => println!("  ✗ Error loading: {}", e),
    }
    
    // 6. Load from binary
    println!("\n8. Loading from binary...");
    match io::load_binary(&bin_path) {
        Ok(loaded_network) => {
            println!("  ✓ Loaded from {}", bin_path);
            
            println!("\n9. Testing loaded network (Binary):");
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let prediction = loaded_network.predict(input);
                let matches = (prediction[0] - target[0]).abs() < 0.1;
                println!("  {:?} -> {:.3} (expected {:.0}) {}", 
                    input.as_slice().unwrap(), 
                    prediction[0], 
                    target[0],
                    if matches { "✓" } else { "✗" });
            }
            
            let loaded_loss = loaded_network.evaluate(&inputs, &targets);
            println!("  Loaded network loss: {:.4}", loaded_loss);
        },
        Err(e) => println!("  ✗ Error loading: {}", e),
    }
    
    println!("\n=== Demo Complete ===");
    println!("\nGenerated files:");
    println!("  - {} (human-readable)", json_path);
    println!("  - {} (compact binary)", bin_path);
}
