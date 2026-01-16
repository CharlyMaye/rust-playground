use cma_neural_network::{
    builder::{NetworkBuilder, NetworkTrainer},
    network::{Activation, LossFunction},
    optimizer::OptimizerType,
    dataset::Dataset,
    callbacks::{EarlyStopping, DeltaMode, ProgressBar},
};
use ndarray::{array, Array1};
use std::fs;
use std::error::Error;
use csv::ReaderBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    println!("🌸 Training Iris Classification Neural Network");
    println!("{}", "=".repeat(60));

    // Load the real Iris dataset from CSV
    println!("\n📂 Loading Iris dataset from CSV...");
    let iris_data = load_iris_from_csv("data/iris.csv")?;
    println!("   ✅ Loaded {} samples", iris_data.len());
    
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
    
    let epochs = 100_000;
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
    
    Ok(())
}

/// Load the real Iris dataset from CSV file
/// Dataset source: UCI Machine Learning Repository
/// https://archive.ics.uci.edu/ml/datasets/iris
fn load_iris_from_csv(path: &str) -> Result<Vec<(Array1<f64>, Array1<f64>)>, Box<dyn Error>> {
    let mut data = Vec::new();
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    
    for result in rdr.records() {
        let record = result?;
        
        // Parse the 4 features
        let sepal_length: f64 = record[0].parse()?;
        let sepal_width: f64 = record[1].parse()?;
        let petal_length: f64 = record[2].parse()?;
        let petal_width: f64 = record[3].parse()?;
        
        // Parse species and convert to one-hot encoding
        let species = &record[4];
        let one_hot = match species {
            "setosa" => array![1.0, 0.0, 0.0],
            "versicolor" => array![0.0, 1.0, 0.0],
            "virginica" => array![0.0, 0.0, 1.0],
            _ => return Err(format!("Unknown species: {}", species).into()),
        };
        
        data.push((
            array![sepal_length, sepal_width, petal_length, petal_width],
            one_hot,
        ));
    }
    
    Ok(data)
}
