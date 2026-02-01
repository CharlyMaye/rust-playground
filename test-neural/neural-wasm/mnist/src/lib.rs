//! WebAssembly Neural Network for MNIST
//!
//! This module exposes a pre-trained MNIST neural network via WebAssembly.
//! Uses cma_neural_network for all neural network operations.

use cma_neural_network::network::Network;
use ndarray::Array1;
use neural_wasm_shared::{
    build_prediction_result, build_test_result, load_model_from_bytes, ActivationsResponse,
    LayerActivation, LayerInfo, ModelInfo, NormalizationStats, TestResult, WeightsInfo,
};
use wasm_bindgen::prelude::*;

// Embed the pre-trained model at compile time (binary format for smaller size)
const MODEL_BIN: &[u8] = include_bytes!("mnist_model.bin");

// Class names for MNIST digits
const CLASS_NAMES: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

// ===== Main Network Struct =====

/// MNIST Neural Network exposed to JavaScript
#[wasm_bindgen]
pub struct MnistNetwork {
    network: Network,
    accuracy: f64,
    test_samples: usize,
    trained_at: String,
    normalization: Option<NormalizationStats>,
}

#[wasm_bindgen]
impl MnistNetwork {
    /// Create a new MNIST network by loading the embedded model
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<MnistNetwork, JsValue> {
        let model = load_model_from_bytes(MODEL_BIN)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;

        Ok(MnistNetwork {
            network: model.network,
            accuracy: model.metadata.accuracy,
            test_samples: model.metadata.test_samples,
            trained_at: model.metadata.trained_at,
            normalization: model.metadata.normalization,
        })
    }

    /// Predict MNIST digit from pixel array
    /// Accepts 784 pixels (28x28 image) or normalized values
    /// Returns JSON with digit prediction (0-9), probabilities, and confidence
    #[wasm_bindgen]
    pub fn predict(&self, pixels: &[f64]) -> String {
        if pixels.len() != 784 {
            return serde_json::json!({
                "error": format!("Expected 784 pixels, got {}", pixels.len())
            })
            .to_string();
        }

        let normalized = self.normalize_input(pixels);
        let input = Array1::from_vec(normalized);
        let output = self.network.predict(&input);
        let probs = output.to_vec();
        let class_names: Vec<String> = CLASS_NAMES.iter().map(|s| s.to_string()).collect();

        let result = build_prediction_result(&probs, &class_names);
        serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
    }

    /// Normalize input pixels using stored normalization statistics
    fn normalize_input(&self, pixels: &[f64]) -> Vec<f64> {
        if let Some(ref norm) = self.normalization {
            norm.normalize(pixels)
        } else {
            pixels.to_vec()
        }
    }

    /// Get class probabilities for 784 pixels
    #[wasm_bindgen]
    pub fn get_probabilities(&self, pixels: &[f64]) -> String {
        if pixels.len() != 784 {
            return serde_json::json!({"error": "Expected 784 pixels"}).to_string();
        }

        let normalized = self.normalize_input(pixels);
        let input = Array1::from_vec(normalized);
        let output = self.network.predict(&input);
        serde_json::to_string(&output.to_vec()).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get class names (digits 0-9)
    #[wasm_bindgen]
    pub fn get_class_names(&self) -> String {
        serde_json::to_string(&vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
            .unwrap_or_else(|_| "[]".to_string())
    }

    // Private helper methods
    fn predict_probs(&self, pixels: &[f64]) -> Vec<f64> {
        if pixels.len() != 784 {
            return vec![0.0; 10];
        }
        let normalized = self.normalize_input(pixels);
        let input = Array1::from_vec(normalized);
        let output = self.network.predict(&input);
        output.to_vec()
    }

    /// Test with sample MNIST digits
    /// Returns results with digit predictions (0-9)
    #[wasm_bindgen]
    pub fn test_all(&self) -> String {
        let test_samples = get_mnist_test_samples();
        let class_names: Vec<String> = CLASS_NAMES.iter().map(|s| s.to_string()).collect();

        let results: Vec<TestResult> = test_samples
            .iter()
            .map(|(pixels, expected)| {
                let probs = self.predict_probs(pixels);
                build_test_result(pixels.clone(), *expected as usize, &probs, &class_names)
            })
            .collect();

        serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get model info with accuracy and metadata
    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        let info = ModelInfo {
            name: "MNIST Classifier".to_string(),
            architecture: self.network.architecture_string(),
            accuracy: self.accuracy * 100.0,
            description: "Binary classification using MNIST dataset".to_string(),
            test_samples: self.test_samples,
            trained_at: self.trained_at.clone(),
        };
        serde_json::to_string(&info).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get all weights and biases as JSON
    #[wasm_bindgen]
    pub fn get_weights(&self) -> String {
        let layers = self.network.get_layers_info();
        let response = WeightsInfo {
            layers: layers
                .iter()
                .map(|(weights, biases, activation_name)| {
                    let weights_2d: Vec<Vec<f64>> =
                        weights.rows().into_iter().map(|row| row.to_vec()).collect();

                    LayerInfo {
                        weights: weights_2d,
                        biases: biases.to_vec(),
                        activation: activation_name.to_string(),
                        shape: [weights.nrows(), weights.ncols()],
                    }
                })
                .collect(),
        };

        serde_json::to_string(&response).unwrap_or_else(|_| r#"{"layers":[]}"#.to_string())
    }

    /// Run inference and return all neuron activations for visualization
    #[wasm_bindgen]
    pub fn get_activations(&self, pixels: &[f64]) -> String {
        if pixels.len() != 784 {
            return r#"{"inputs":[],"layers":[],"output":[]}"#.to_string();
        }

        let normalized = self.normalize_input(pixels);
        let input = Array1::from_vec(normalized);
        let activations = self.network.get_all_activations(&input);

        let output_probs = activations
            .last()
            .map(|(_, post, _)| post.to_vec())
            .unwrap_or_else(|| vec![0.0; 10]);

        let response = ActivationsResponse {
            inputs: pixels.to_vec(),
            layers: activations
                .iter()
                .map(|(pre, post, activation_name)| LayerActivation {
                    pre_activation: pre.to_vec(),
                    activation: post.to_vec(),
                    function: activation_name.to_string(),
                })
                .collect(),
            output: output_probs,
        };

        serde_json::to_string(&response)
            .unwrap_or_else(|_| r#"{"inputs":[],"layers":[],"output":[]}"#.to_string())
    }
}

/// Initialize the module
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Sample MNIST test data (10 digits with normalized pixel values)
/// These are simplified representations - real MNIST pixels would be 0-255
fn get_mnist_test_samples() -> Vec<(Vec<f64>, u8)> {
    vec![
        (vec_with_first_n(vec![0.5; 784], 5), 0),
        (vec_with_first_n(vec![0.3; 784], 5), 1),
        (vec_with_first_n(vec![0.7; 784], 5), 2),
        (vec_with_first_n(vec![0.4; 784], 5), 3),
        (vec_with_first_n(vec![0.6; 784], 5), 4),
        (vec_with_first_n(vec![0.2; 784], 5), 5),
        (vec_with_first_n(vec![0.8; 784], 5), 6),
        (vec_with_first_n(vec![0.45; 784], 5), 7),
        (vec_with_first_n(vec![0.55; 784], 5), 8),
        (vec_with_first_n(vec![0.65; 784], 5), 9),
    ]
}

fn vec_with_first_n(mut v: Vec<f64>, n: usize) -> Vec<f64> {
    for i in 0..n.min(v.len()) {
        v[i] = v[i] * 2.0;
    }
    v
}
