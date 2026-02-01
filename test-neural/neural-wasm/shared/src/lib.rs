use chrono;
use cma_neural_network::network::Network;
use ndarray;
use serde::{Deserialize, Serialize};

/// Normalization statistics for input features
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NormalizationStats {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

impl NormalizationStats {
    pub fn new(means: Vec<f64>, stds: Vec<f64>) -> Self {
        Self { means, stds }
    }

    /// Normalize a single input using these statistics
    pub fn normalize(&self, input: &[f64]) -> Vec<f64> {
        input
            .iter()
            .enumerate()
            .map(|(i, &val)| (val - self.means[i]) / self.stds[i])
            .collect()
    }
}

/// Model metadata saved during training
#[derive(Serialize, Deserialize)]
pub struct ModelMetadata {
    pub accuracy: f64,
    pub test_samples: usize,
    pub trained_at: String,
    #[serde(default)]
    pub normalization: Option<NormalizationStats>,
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

// ===== Common Response Structures =====

/// Prediction result for any classifier
#[derive(Serialize)]
pub struct PredictionResult {
    pub class_name: String,
    pub class_index: usize,
    pub probabilities: Vec<f64>,
    pub confidence: f64,
}

/// Layer activation data for visualization
#[derive(Serialize)]
pub struct LayerActivation {
    pub pre_activation: Vec<f64>,
    pub activation: Vec<f64>,
    pub function: String,
}

/// Full activation response for network visualization
#[derive(Serialize)]
pub struct ActivationsResponse {
    pub inputs: Vec<f64>,
    pub layers: Vec<LayerActivation>,
    pub output: Vec<f64>,
}

/// Generic test result for any classifier
#[derive(Serialize)]
pub struct TestResult {
    pub inputs: Vec<f64>,
    pub expected_class: String,
    pub expected_index: usize,
    pub predicted_class: String,
    pub predicted_index: usize,
    pub probabilities: Vec<f64>,
    pub confidence: f64,
    pub is_correct: bool,
}

// ===== Utility Functions =====

/// Find the class with highest probability
pub fn find_max_class(probs: &[f64]) -> (usize, f64) {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &val)| (idx, val))
        .unwrap_or((0, 0.0))
}

/// Validate input size
pub fn validate_input_size(input: &[f64], expected: usize) -> Result<(), String> {
    if input.len() != expected {
        Err(format!("Expected {} inputs, got {}", expected, input.len()))
    } else {
        Ok(())
    }
}

/// Build a prediction result from probabilities
pub fn build_prediction_result(probs: &[f64], class_names: &[String]) -> PredictionResult {
    let (class_index, confidence) = find_max_class(probs);
    let class_name = class_names
        .get(class_index)
        .cloned()
        .unwrap_or_else(|| class_index.to_string());

    PredictionResult {
        class_name,
        class_index,
        probabilities: probs.to_vec(),
        confidence, // ratio 0-1, frontend multiplies by 100 for display
    }
}

/// Build a test result from prediction
pub fn build_test_result(
    inputs: Vec<f64>,
    expected_index: usize,
    probs: &[f64],
    class_names: &[String],
) -> TestResult {
    let (predicted_index, confidence) = find_max_class(probs);

    let expected_class = class_names
        .get(expected_index)
        .cloned()
        .unwrap_or_else(|| expected_index.to_string());
    let predicted_class = class_names
        .get(predicted_index)
        .cloned()
        .unwrap_or_else(|| predicted_index.to_string());

    TestResult {
        inputs,
        expected_class,
        expected_index,
        predicted_class,
        predicted_index,
        probabilities: probs.to_vec(),
        confidence, // ratio 0-1, frontend multiplies by 100 for display
        is_correct: expected_index == predicted_index,
    }
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
        let predicted_class = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let expected_class = targets[i]
            .iter()
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
    save_model_with_normalization(network, accuracy, test_samples, None, path)
}

/// Save model with metadata and normalization statistics to JSON file
pub fn save_model_with_normalization(
    network: Network,
    accuracy: f64,
    test_samples: usize,
    normalization: Option<NormalizationStats>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_with_metadata = ModelWithMetadata {
        network,
        metadata: ModelMetadata {
            accuracy,
            test_samples,
            trained_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            normalization,
        },
    };

    let model_json = serde_json::to_string_pretty(&model_with_metadata)?;
    std::fs::write(path, model_json)?;
    Ok(())
}

/// Save model with metadata to binary file (compact format for WASM)
pub fn save_model_binary(
    network: Network,
    accuracy: f64,
    test_samples: usize,
    normalization: Option<NormalizationStats>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_with_metadata = ModelWithMetadata {
        network,
        metadata: ModelMetadata {
            accuracy,
            test_samples,
            trained_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            normalization,
        },
    };

    let model_bin = bincode::serialize(&model_with_metadata)?;
    std::fs::write(path, &model_bin)?;

    Ok(())
}

/// Load model from binary bytes (for WASM with include_bytes!)
pub fn load_model_from_bytes(
    bytes: &[u8],
) -> Result<ModelWithMetadata, Box<dyn std::error::Error>> {
    let model: ModelWithMetadata = bincode::deserialize(bytes)?;
    Ok(model)
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
