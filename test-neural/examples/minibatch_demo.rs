/// Demonstration of mini-batch training
/// 
/// This example shows:
/// - How to use the Dataset structure
/// - Comparison between single-sample and mini-batch training
/// - Impact of different batch sizes on training speed and convergence
/// - Benefits of shuffling data between epochs

use test_neural::network::{Network, Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::dataset::Dataset;
use ndarray::array;
use std::time::Instant;

fn main() {
    println!("=== Démonstration du Mini-Batch Training ===\n");
    
    // Create a synthetic dataset (larger than XOR for demonstrating batch benefits)
    println!("📊 Création du dataset...");
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    // Generate 1000 XOR-like examples with variations
    for _ in 0..250 {
        // XOR: 00 -> 0
        inputs.push(array![0.0 + rand::random::<f64>() * 0.1, 0.0 + rand::random::<f64>() * 0.1]);
        targets.push(array![0.0]);
        
        // XOR: 01 -> 1
        inputs.push(array![0.0 + rand::random::<f64>() * 0.1, 1.0 - rand::random::<f64>() * 0.1]);
        targets.push(array![1.0]);
        
        // XOR: 10 -> 1
        inputs.push(array![1.0 - rand::random::<f64>() * 0.1, 0.0 + rand::random::<f64>() * 0.1]);
        targets.push(array![1.0]);
        
        // XOR: 11 -> 0
        inputs.push(array![1.0 - rand::random::<f64>() * 0.1, 1.0 - rand::random::<f64>() * 0.1]);
        targets.push(array![0.0]);
    }
    
    println!("Dataset créé: {} exemples\n", inputs.len());
    
    // Create dataset and split
    let dataset = Dataset::new(inputs, targets);
    let (train_dataset, test_dataset) = dataset.split(0.8);
    
    println!("Train: {} exemples", train_dataset.len());
    println!("Test:  {} exemples\n", test_dataset.len());
    
    // ===== 1. Single-sample training (baseline) =====
    println!("--- 1. Entraînement échantillon par échantillon ---");
    let mut network_single = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.001)
    );
    
    let start = Instant::now();
    let epochs = 50;
    
    for epoch in 0..epochs {
        // Train on each example individually
        for (input, target) in train_dataset.inputs().iter().zip(train_dataset.targets().iter()) {
            network_single.train(input, target);
        }
        
        if (epoch + 1) % 10 == 0 {
            let loss = network_single.evaluate(train_dataset.inputs(), train_dataset.targets());
            println!("  Epoch {}: loss = {:.6}", epoch + 1, loss);
        }
    }
    
    let duration_single = start.elapsed();
    let test_loss_single = network_single.evaluate(test_dataset.inputs(), test_dataset.targets());
    
    println!("✓ Temps: {:.2}s", duration_single.as_secs_f64());
    println!("✓ Loss finale (test): {:.6}\n", test_loss_single);
    
    // ===== 2. Mini-batch training (batch_size = 32) =====
    println!("--- 2. Entraînement par mini-batch (batch_size=32) ---");
    let mut network_batch32 = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.01)  // Increased learning rate for batch training
    );
    
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
    
    println!("✓ Temps: {:.2}s", duration_batch32.as_secs_f64());
    println!("✓ Loss finale (test): {:.6}", test_loss_batch32);
    println!("✓ Speedup: {:.2}x plus rapide\n", duration_single.as_secs_f64() / duration_batch32.as_secs_f64());
    
    // ===== 3. Mini-batch training (batch_size = 64) =====
    println!("--- 3. Entraînement par mini-batch (batch_size=64) ---");
    let mut network_batch64 = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.01)  // Increased learning rate for batch training
    );
    
    let start = Instant::now();
    let batch_size = 64;
    
    let mut train_data_shuffleable = train_dataset.clone();
    
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
    
    println!("✓ Temps: {:.2}s", duration_batch64.as_secs_f64());
    println!("✓ Loss finale (test): {:.6}", test_loss_batch64);
    println!("✓ Speedup: {:.2}x plus rapide\n", duration_single.as_secs_f64() / duration_batch64.as_secs_f64());
    
    // ===== 4. Mini-batch training (batch_size = 128) =====
    println!("--- 4. Entraînement par mini-batch (batch_size=128) ---");
    let mut network_batch128 = Network::new(
        2, 8, 1,
        Activation::Tanh,
        Activation::Sigmoid,
        LossFunction::BinaryCrossEntropy,
        OptimizerType::adam(0.01)  // Increased learning rate for batch training
    );
    
    let start = Instant::now();
    let batch_size = 128;
    
    let mut train_data_shuffleable = train_dataset.clone();
    
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
    let test_loss_batch128 = network_batch128.evaluate(test_dataset.inputs(), test_dataset.targets());
    
    println!("✓ Temps: {:.2}s", duration_batch128.as_secs_f64());
    println!("✓ Loss finale (test): {:.6}", test_loss_batch128);
    println!("✓ Speedup: {:.2}x plus rapide\n", duration_single.as_secs_f64() / duration_batch128.as_secs_f64());
    
    // ===== Summary =====
    println!("=== Résumé ===");
    println!("\n📈 Temps d'entraînement:");
    println!("  • Échantillon par échantillon: {:.2}s", duration_single.as_secs_f64());
    println!("  • Mini-batch (32):             {:.2}s ({:.1}x speedup)", 
             duration_batch32.as_secs_f64(), 
             duration_single.as_secs_f64() / duration_batch32.as_secs_f64());
    println!("  • Mini-batch (64):             {:.2}s ({:.1}x speedup)", 
             duration_batch64.as_secs_f64(), 
             duration_single.as_secs_f64() / duration_batch64.as_secs_f64());
    println!("  • Mini-batch (128):            {:.2}s ({:.1}x speedup)", 
             duration_batch128.as_secs_f64(), 
             duration_single.as_secs_f64() / duration_batch128.as_secs_f64());
    
    println!("\n🎯 Loss finale (test):");
    println!("  • Échantillon par échantillon: {:.6}", test_loss_single);
    println!("  • Mini-batch (32):             {:.6}", test_loss_batch32);
    println!("  • Mini-batch (64):             {:.6}", test_loss_batch64);
    println!("  • Mini-batch (128):            {:.6}", test_loss_batch128);
    
    println!("\n💡 Recommandations:");
    println!("  • Pour datasets < 1000:    batch_size = 16-32");
    println!("  • Pour datasets 1000-10k:  batch_size = 32-64");
    println!("  • Pour datasets > 10k:     batch_size = 64-128");
    println!("  • Toujours shuffle entre epochs!");
    println!("  • Mini-batch = meilleur compromis vitesse/stabilité");
}
