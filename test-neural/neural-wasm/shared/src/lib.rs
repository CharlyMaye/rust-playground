use serde::{Serialize, Deserialize};
use cma_neural_network::network::Network;
use ndarray;
use chrono;

/// Model metadata saved during training
#[derive(Serialize, Deserialize)]
pub struct ModelMetadata {
    pub accuracy: f64,
    pub test_samples: usize,
    pub trained_at: String,
}

/// Model wrapper with metadata
#[derive(Serialize, Deserialize)]
pub struct ModelWithMetadata {
    pub network: Network,
    pub metadata: ModelMetadata,
}

/// Information about a trained model
#[derive(Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub architecture: String,
    pub accuracy: f64,
    pub description: String,
    pub test_samples: usize,
    pub trained_at: String,
}

/// Information about a network layer
#[derive(Serialize)]
pub struct LayerInfo {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: String,
    pub shape: [usize; 2],
}

/// Information about network weights
#[derive(Serialize)]
pub struct WeightsInfo {
    pub layers: Vec<LayerInfo>,
}

/// Activation information for visualization
#[derive(Serialize)]
pub struct ActivationInfo {
    pub inputs: Vec<f64>,
    pub layers: Vec<LayerActivation>,
    pub output: f64,
}

#[derive(Serialize)]
pub struct LayerActivation {
    pub activation: Vec<f64>,
}

/// Convert confidence to percentage
pub fn confidence_to_percentage(value: f64) -> f64 {
    (value * 100.0).max(0.0).min(100.0)
}

/// Softmax function for multi-class probability
pub fn softmax(values: &[f64]) -> Vec<f64> {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = values.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exp_values.iter().sum();
    exp_values.iter().map(|&v| v / sum).collect()
}

/// Calculate accuracy for multi-class classification
pub fn calculate_multiclass_accuracy(
    network: &Network,
    inputs: &[ndarray::Array1<f64>],
    targets: &[ndarray::Array1<f64>],
) -> (usize, usize) {
    let mut correct = 0;
    let total = inputs.len();
    
    for i in 0..total {
        let output = network.predict(&inputs[i]);
        let predicted_class = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        let expected_class = targets[i].iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        if predicted_class == expected_class {
            correct += 1;
        }
    }
    
    (correct, total)
}

/// Save model with metadata to JSON file
pub fn save_model_with_metadata(
    network: Network,
    accuracy: f64,
    test_samples: usize,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_with_metadata = ModelWithMetadata {
        network,
        metadata: ModelMetadata {
            accuracy,
            test_samples,
            trained_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        },
    };

    let model_json = serde_json::to_string_pretty(&model_with_metadata)?;
    std::fs::write(path, model_json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let values = vec![1.0, 2.0, 3.0];
        let result = softmax(&values);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_to_percentage() {
        assert_eq!(confidence_to_percentage(0.5), 50.0);
        assert_eq!(confidence_to_percentage(1.0), 100.0);
        assert_eq!(confidence_to_percentage(0.0), 0.0);
    }
}
