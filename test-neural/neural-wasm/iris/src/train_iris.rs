use cma_neural_network::{
    builder::{NetworkBuilder, NetworkTrainer},
    network::{Activation, LossFunction, Network},
    optimizer::OptimizerType,
    dataset::Dataset,
    callbacks::{EarlyStopping, DeltaMode, ProgressBar},
};
use neural_wasm_shared::{ModelWithMetadata, ModelMetadata};
use ndarray::{array, Array1};
use std::fs;
use std::error::Error;
use csv::ReaderBuilder;
use chrono::Local;

fn main() -> Result<(), Box<dyn Error>> {
    println!("🌸 Training Iris Classification Neural Network");
    println!("{}", "=".repeat(60));

    // Load the real Iris dataset from CSV
    println!("\n📂 Loading Iris dataset from CSV...");
    let iris_data = load_iris_from_csv("data/iris.csv")?;
    println!("   ✅ Loaded {} samples", iris_data.len());
    
    // Split into training (70%) and test (30%) sets
    let inputs: Vec<Array1<f64>> = iris_data.iter().map(|(i, _)| i.clone()).collect();
    let targets: Vec<Array1<f64>> = iris_data.iter().map(|(_, t)| t.clone()).collect();
    
    let dataset = Dataset::new(inputs, targets);
    let (train_dataset, test_dataset) = dataset.split(0.7);
    
    println!("\n📊 Dataset:");
    println!("   Training samples: {} (70%)", train_dataset.len());
    println!("   Test samples:     {} (30%)", test_dataset.len());
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

    // Configure callbacks  
    // Use larger patience to allow the model to learn properly
    let early_stopping = EarlyStopping::new(200, 0.00001)
        .mode(DeltaMode::Absolute);
    
    let epochs = 5_000;
    let progress = ProgressBar::new(epochs);

    println!("\n🎯 Training configuration:");
    println!("   Learning rate: 0.01");
    println!("   Batch size:    16");
    println!("   Max epochs:    {}", epochs);
    println!("   Early stopping: 200 epochs patience (prevents premature stopping)");

    println!("\n🚀 Starting training...\n");
    
    // Train the network with validation
    let _history = network.trainer()
        .train_data(&train_dataset)
        .validation_data(&test_dataset)
        .epochs(epochs)
        .batch_size(16)
        .callback(Box::new(early_stopping))
        .callback(Box::new(progress))
        .fit();
    
    println!("\n✅ Training completed!");
    
    // Evaluate on test set
    network.eval_mode();
    
    let test_inputs = test_dataset.inputs();
    let test_targets = test_dataset.targets();
    
    let mut correct = 0;
    let total = test_dataset.len();
    
    for i in 0..total {
        let output = network.predict(&test_inputs[i]);
        let predicted_class = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        let expected_class = test_targets[i].iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        if predicted_class == expected_class {
            correct += 1;
        }
    }
    
    let accuracy = (correct as f64 / total as f64) * 100.0;
    println!("\n📊 Test Set Accuracy: {:.2}% ({}/{})", accuracy, correct, total);
    
    // Save the model with metadata
    let model_with_metadata = ModelWithMetadata {
        network,
        metadata: ModelMetadata {
            accuracy,
            test_samples: total,
            trained_at: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        },
    };
    
    let model_json = serde_json::to_string_pretty(&model_with_metadata)
        .expect("Failed to serialize model with metadata");
    
    fs::write("src/iris_model.json", model_json)
        .expect("Failed to write model file");
    
    println!("\n💾 Model saved to: src/iris_model.json");
    println!("   ✅ Accuracy {:.2}% automatically saved in metadata", accuracy);
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
