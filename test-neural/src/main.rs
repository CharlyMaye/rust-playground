mod network;
use network::{Network, Activation, LossFunction};
use ndarray::array;

fn main() {
    println!("=== Testing All Loss Functions ===\n");
    
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
    
    let learning_rate = 0.5;
    let epochs = 50_000;
    
    // Test 1: MSE
    test_loss("MSE", 
        LossFunction::MSE,
        Activation::Tanh,
        Activation::Sigmoid,
        &inputs, &targets, learning_rate, epochs);
    
    // Test 2: MAE (needs lower lr and more epochs due to constant gradients)
    test_loss("MAE",
        LossFunction::MAE,
        Activation::Tanh,
        Activation::Sigmoid,
        &inputs, &targets, 0.2, 150_000);
    
    // Test 3: Binary Cross-Entropy
    test_loss("Binary Cross-Entropy",
        LossFunction::BinaryCrossEntropy,
        Activation::Tanh,
        Activation::Sigmoid,
        &inputs, &targets, learning_rate, epochs);
    
    // Test 4: Huber
    test_loss("Huber",
        LossFunction::Huber,
        Activation::Tanh,
        Activation::Sigmoid,
        &inputs, &targets, learning_rate, epochs);
    
    println!("\n=== Testing Different Activation Combinations ===\n");
    
    // Test 5: LeakyReLU + Sigmoid + MSE (using LeakyReLU to avoid dying neurons)
    test_loss("LeakyReLU + Sigmoid + MSE",
        LossFunction::MSE,
        Activation::LeakyReLU,
        Activation::Sigmoid,
        &inputs, &targets, learning_rate, epochs);
    
    // Test 6: GELU + Sigmoid + BCE
    test_loss("GELU + Sigmoid + BCE",
        LossFunction::BinaryCrossEntropy,
        Activation::GELU,
        Activation::Sigmoid,
        &inputs, &targets, learning_rate, epochs);
    
    println!("\n=== All Tests Completed ===");
}

fn test_loss(
    name: &str,
    loss: LossFunction,
    hidden_act: Activation,
    output_act: Activation,
    inputs: &Vec<ndarray::Array1<f64>>,
    targets: &Vec<ndarray::Array1<f64>>,
    learning_rate: f64,
    epochs: usize,
) {
    let mut network = Network::new(2, 5, 1, hidden_act, output_act, loss);
    
    println!("--- {} ---", name);
    let initial_loss = network.evaluate(inputs, targets);
    println!("Initial loss: {:.4}", initial_loss);
    
    for _ in 0..epochs {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            network.train(input, target, learning_rate);
        }
    }
    
    let final_loss = network.evaluate(inputs, targets);
    println!("Final loss: {:.4}", final_loss);
    
    println!("Results:");
    let mut all_correct = true;
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let (_, _, output) = network.forward(input);
        let prediction = if output[0] > 0.5 { 1.0 } else { 0.0 };
        let correct = (prediction - target[0]).abs() < 0.1;
        all_correct = all_correct && correct;
        println!("  {:?} -> {:.3} (expected {:.0}) {}", 
            input, output[0], target[0], 
            if correct { "✓" } else { "✗" });
    }
    
    if all_correct {
        println!("Status: ✓ PASSED\n");
    } else {
        println!("Status: ✗ FAILED\n");
    }
}