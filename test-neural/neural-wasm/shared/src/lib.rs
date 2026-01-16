use serde::{Serialize, Deserialize};
use cma_neural_network::network::Network;

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
