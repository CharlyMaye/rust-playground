use cma_neural_network::{
    builder::{NetworkBuilder, NetworkTrainer},
    network::{Activation, LossFunction},
    optimizer::OptimizerType,
    dataset::Dataset,
    callbacks::{EarlyStopping, DeltaMode, ProgressBar},
};
use ndarray::{array, Array1};
use std::fs;

fn main() {
    println!("🌸 Training Iris Classification Neural Network");
    println!("{}", "=".repeat(60));

    // Famous Iris dataset (simplified version with 30 samples per class)
    let iris_data = get_iris_dataset();
    
    // Split into training and validation sets
    let split_idx = (iris_data.len() as f64 * 0.8) as usize;
    let train_data = &iris_data[..split_idx];
    let test_data = &iris_data[split_idx..];
    
    println!("\n📊 Dataset:");
    println!("   Training samples: {}", train_data.len());
    println!("   Testing samples:  {}", test_data.len());
    println!("   Input features:   4 (sepal length, sepal width, petal length, petal width)");
    println!("   Output classes:   3 (Setosa, Versicolor, Virginica)");

    // Build the network: 4 inputs -> 8 hidden -> 3 outputs
    println!("\n🏗️  Building network architecture:");
    println!("   Input layer:  4 neurons");
    println!("   Hidden layer: 8 neurons (ReLU)");
    println!("   Output layer: 3 neurons (Sigmoid)");
    
    let mut network = NetworkBuilder::new(4, 3)
        .hidden_layer(8, Activation::ReLU)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();

    // Create dataset
    let train_inputs: Vec<Array1<f64>> = train_data.iter()
        .map(|(i, _)| i.clone())
        .collect();
    let train_targets: Vec<Array1<f64>> = train_data.iter()
        .map(|(_, t)| t.clone())
        .collect();
    
    let train_dataset = Dataset::new(train_inputs, train_targets);
    
    // Configure callbacks
    let early_stopping = EarlyStopping::new(50, 0.0001)
        .mode(DeltaMode::Absolute);
    
    let epochs = 1000;
    let progress = ProgressBar::new(epochs);

    println!("\n🎯 Training configuration:");
    println!("   Learning rate: 0.01");
    println!("   Batch size:    16");
    println!("   Max epochs:    {}", epochs);
    println!("   Early stopping: 50 epochs patience");

    println!("\n🚀 Starting training...\n");
    
    // Train the network
    let _history = network.trainer()
        .train_data(&train_dataset)
        .epochs(epochs)
        .batch_size(16)
        .callback(Box::new(early_stopping))
        .callback(Box::new(progress))
        .fit();
    
    println!("\n✅ Training completed!");
    
    // Evaluate on test set
    let mut correct = 0;
    let mut total = 0;
    
    for (input, expected) in test_data {
        let output = network.predict(input);
        let predicted_class = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        let expected_class = expected.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        if predicted_class == expected_class {
            correct += 1;
        }
        total += 1;
    }
    
    let accuracy = (correct as f64 / total as f64) * 100.0;
    println!("\n📊 Test Set Accuracy: {:.2}% ({}/{})", accuracy, correct, total);
    
    // Save the model
    let model_json = serde_json::to_string_pretty(&network)
        .expect("Failed to serialize network");
    
    fs::write("src/iris_model.json", model_json)
        .expect("Failed to write model file");
    
    println!("\n💾 Model saved to: src/iris_model.json");
    println!("\n{}", "=".repeat(60));
    println!("🎉 Training successful!");
}

/// Returns the Iris dataset as (input, one-hot-encoded output)
fn get_iris_dataset() -> Vec<(Array1<f64>, Array1<f64>)> {
    // Simplified Iris dataset (30 samples per class = 90 total)
    let mut data = Vec::new();
    
    // Setosa (class 0): Small petals, distinctive
    let setosa = vec![
        [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
        [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.8, 4.0, 1.2, 0.2],
        [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3],
        [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3], [5.4, 3.4, 1.7, 0.2],
        [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1.0, 0.2], [5.1, 3.3, 1.7, 0.5],
        [4.8, 3.4, 1.9, 0.2], [5.0, 3.0, 1.6, 0.2], [5.0, 3.4, 1.6, 0.4],
        [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2],
    ];
    
    // Versicolor (class 1): Medium size
    let versicolor = vec![
        [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
        [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1.0], [6.6, 2.9, 4.6, 1.3],
        [5.2, 2.7, 3.9, 1.4], [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5],
        [6.0, 2.2, 4.0, 1.0], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3],
        [6.7, 3.1, 4.4, 1.4], [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 4.1, 1.0],
        [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1], [5.9, 3.2, 4.8, 1.8],
        [6.1, 2.8, 4.0, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
        [6.4, 2.9, 4.3, 1.3], [6.6, 3.0, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4],
        [6.7, 3.0, 5.0, 1.7], [6.0, 2.9, 4.5, 1.5], [5.7, 2.6, 3.5, 1.0],
    ];
    
    // Virginica (class 2): Large petals
    let virginica = vec![
        [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1],
        [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2], [7.6, 3.0, 6.6, 2.1],
        [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8], [6.7, 2.5, 5.8, 1.8],
        [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2.0], [6.4, 2.7, 5.3, 1.9],
        [6.8, 3.0, 5.5, 2.1], [5.7, 2.5, 5.0, 2.0], [5.8, 2.8, 5.1, 2.4],
        [6.4, 3.2, 5.3, 2.3], [6.5, 3.0, 5.5, 1.8], [7.7, 3.8, 6.7, 2.2],
        [7.7, 2.6, 6.9, 2.3], [6.0, 2.2, 5.0, 1.5], [6.9, 3.2, 5.7, 2.3],
        [5.6, 2.8, 4.9, 2.0], [7.7, 2.8, 6.7, 2.0], [6.3, 2.7, 4.9, 1.8],
        [6.7, 3.3, 5.7, 2.1], [7.2, 3.2, 6.0, 1.8], [6.2, 2.8, 4.8, 1.8],
        [6.1, 3.0, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1], [7.2, 3.0, 5.8, 1.6],
    ];
    
    // Convert to network format
    for measurements in setosa {
        data.push((
            array![measurements[0], measurements[1], measurements[2], measurements[3]],
            array![1.0, 0.0, 0.0]  // One-hot: Setosa
        ));
    }
    
    for measurements in versicolor {
        data.push((
            array![measurements[0], measurements[1], measurements[2], measurements[3]],
            array![0.0, 1.0, 0.0]  // One-hot: Versicolor
        ));
    }
    
    for measurements in virginica {
        data.push((
            array![measurements[0], measurements[1], measurements[2], measurements[3]],
            array![0.0, 0.0, 1.0]  // One-hot: Virginica
        ));
    }
    
    data
}
