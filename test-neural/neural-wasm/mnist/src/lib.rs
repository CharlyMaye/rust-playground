//! WebAssembly Neural Network for MNIST
//!
//! This module exposes a pre-trained MNIST neural network via WebAssembly.
//! Uses cma_neural_network for all neural network operations.

use cma_neural_network::network::Network;
use ndarray::array;
use neural_wasm_shared::{load_model_from_bytes, ModelInfo};
use serde::Serialize;
use wasm_bindgen::prelude::*;

// Embed the pre-trained model at compile time (binary format for smaller size)
const MODEL_BIN: &[u8] = include_bytes!("mnist_model.bin");

// ===== JSON Response Structures =====

#[derive(Serialize)]
struct TestResult {
    a: u8,
    b: u8,
    expected: u8,
    prediction: u8,
    raw: f64,
    confidence: f64,
}

#[derive(Serialize)]
struct LayerWeights {
    weights: Vec<f64>,
    biases: Vec<f64>,
    activation: String,
    shape: [usize; 2],
}

#[derive(Serialize)]
struct WeightsResponse {
    layers: Vec<LayerWeights>,
}

#[derive(Serialize)]
struct LayerActivation {
    pre_activation: Vec<f64>,
    activation: Vec<f64>,
    function: String,
}

#[derive(Serialize)]
struct ActivationsResponse {
    inputs: [f64; 2],
    layers: Vec<LayerActivation>,
    output: f64,
}

// ===== Main Network Struct =====

/// MNIST Neural Network exposed to JavaScript
#[wasm_bindgen]
pub struct MnistNetwork {
    network: Network,
    accuracy: f64,
    test_samples: usize,
    trained_at: String,
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
        })
    }

    /// Predict MNIST result for two binary inputs
    /// Returns JSON with prediction details
    #[wasm_bindgen]
    pub fn predict(&self, x1: f64, x2: f64) -> String {
        let input = array![x1, x2];
        let output = self.network.predict(&input);
        let raw = output[0];
        let prediction = if raw > 0.5 { 1 } else { 0 };
        let confidence = (raw - 0.5).abs() * 2.0;

        let result = serde_json::json!({
            "prediction": prediction,
            "raw": raw,
            "confidence": confidence,
            "probabilities": [1.0 - raw, raw]
        });

        serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get class probabilities
    #[wasm_bindgen]
    pub fn get_probabilities(&self, x1: f64, x2: f64) -> String {
        let input = array![x1, x2];
        let output = self.network.predict(&input);
        let raw = output[0];
        let probs = vec![1.0 - raw, raw];
        serde_json::to_string(&probs).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get class names
    #[wasm_bindgen]
    pub fn get_class_names(&self) -> String {
        serde_json::to_string(&vec!["0", "1"]).unwrap_or_else(|_| "[]".to_string())
    }

    // Private helper methods
    fn predict_binary(&self, x1: f64, x2: f64) -> u8 {
        let input = array![x1, x2];
        let output = self.network.predict(&input);
        if output[0] > 0.5 {
            1
        } else {
            0
        }
    }

    fn predict_raw(&self, x1: f64, x2: f64) -> f64 {
        let input = array![x1, x2];
        let output = self.network.predict(&input);
        output[0]
    }

    fn confidence(&self, x1: f64, x2: f64) -> f64 {
        let raw = self.predict_raw(x1, x2);
        (raw - 0.5).abs() * 2.0 * 100.0
    }

    /// Test all MNIST combinations and return results as JSON string
    #[wasm_bindgen]
    pub fn test_all(&self) -> String {
        let results: Vec<TestResult> = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            .iter()
            .map(|&(a, b)| {
                let expected = if (a as u8) ^ (b as u8) == 1 { 1 } else { 0 };
                TestResult {
                    a: a as u8,
                    b: b as u8,
                    expected,
                    prediction: self.predict_binary(a, b),
                    raw: self.predict_raw(a, b),
                    confidence: self.confidence(a, b),
                }
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
        let response = WeightsResponse {
            layers: layers
                .iter()
                .map(|(weights, biases, activation_name)| {
                    let shape = weights.shape();
                    LayerWeights {
                        weights: weights.iter().cloned().collect(),
                        biases: biases.iter().cloned().collect(),
                        activation: activation_name.to_string(),
                        shape: [shape[0], shape[1]],
                    }
                })
                .collect(),
        };

        serde_json::to_string(&response).unwrap_or_else(|_| r#"{"layers":[]}"#.to_string())
    }

    /// Run inference and return all neuron activations for visualization
    #[wasm_bindgen]
    pub fn get_activations(&self, x1: f64, x2: f64) -> String {
        let input = array![x1, x2];
        let activations = self.network.get_all_activations(&input);

        let output = activations
            .last()
            .map(|(_, post, _)| post[0])
            .unwrap_or(0.0);

        let response = ActivationsResponse {
            inputs: [x1, x2],
            layers: activations
                .iter()
                .map(|(pre, post, activation_name)| LayerActivation {
                    pre_activation: pre.iter().cloned().collect(),
                    activation: post.iter().cloned().collect(),
                    function: activation_name.to_string(),
                })
                .collect(),
            output,
        };

        serde_json::to_string(&response)
            .unwrap_or_else(|_| r#"{"inputs":[0,0],"layers":[],"output":0}"#.to_string())
    }
}

/// Initialize the module
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}
