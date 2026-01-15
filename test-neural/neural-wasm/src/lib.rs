//! WebAssembly Neural Network for XOR
//!
//! This module exposes a pre-trained XOR neural network via WebAssembly.
//! Uses cma_neural_network for all neural network operations.

use wasm_bindgen::prelude::*;
use cma_neural_network::network::Network;
use ndarray::array;
use serde::Serialize;
use std::sync::OnceLock;

// Embed the pre-trained model at compile time
const MODEL_JSON: &str = include_str!("xor_model.json");

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

/// XOR Neural Network exposed to JavaScript
#[wasm_bindgen]
pub struct XorNetwork {
    network: Network,
}

#[wasm_bindgen]
impl XorNetwork {
    /// Create a new XOR network by loading the embedded model
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<XorNetwork, JsValue> {
        let network: Network = serde_json::from_str(MODEL_JSON)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;
        
        Ok(XorNetwork { network })
    }

    /// Predict XOR result for two binary inputs
    /// Returns 0 or 1 (rounded prediction)
    #[wasm_bindgen]
    pub fn predict(&self, x1: f64, x2: f64) -> u8 {
        let input = array![x1, x2];
        let output = self.network.predict(&input);
        if output[0] > 0.5 { 1 } else { 0 }
    }

    /// Get raw prediction value (0.0 to 1.0)
    #[wasm_bindgen]
    pub fn predict_raw(&self, x1: f64, x2: f64) -> f64 {
        let input = array![x1, x2];
        let output = self.network.predict(&input);
        output[0]
    }

    /// Get confidence percentage (0-100)
    #[wasm_bindgen]
    pub fn confidence(&self, x1: f64, x2: f64) -> f64 {
        let raw = self.predict_raw(x1, x2);
        let distance_from_uncertain = (raw - 0.5).abs();
        distance_from_uncertain * 2.0 * 100.0
    }

    /// Test all XOR combinations and return results as JSON string
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
                    prediction: self.predict(a, b),
                    raw: self.predict_raw(a, b),
                    confidence: self.confidence(a, b),
                }
            })
            .collect();
        
        serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get model info
    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        format!("XOR Network: {}", self.network.architecture_string())
    }

    /// Get all weights and biases as JSON
    #[wasm_bindgen]
    pub fn get_weights(&self) -> String {
        let layers = self.network.get_layers_info();
        let response = WeightsResponse {
            layers: layers.iter().map(|(weights, biases, activation_name)| {
                let shape = weights.shape();
                LayerWeights {
                    weights: weights.iter().cloned().collect(),
                    biases: biases.iter().cloned().collect(),
                    activation: activation_name.to_string(),
                    shape: [shape[0], shape[1]],
                }
            }).collect(),
        };
        
        serde_json::to_string(&response).unwrap_or_else(|_| r#"{"layers":[]}"#.to_string())
    }

    /// Run inference and return all neuron activations for visualization
    #[wasm_bindgen]
    pub fn get_activations(&self, x1: f64, x2: f64) -> String {
        let input = array![x1, x2];
        let activations = self.network.get_all_activations(&input);
        
        let output = activations.last().map(|(_, post, _)| post[0]).unwrap_or(0.0);
        
        let response = ActivationsResponse {
            inputs: [x1, x2],
            layers: activations.iter().map(|(pre, post, activation_name)| {
                LayerActivation {
                    pre_activation: pre.iter().cloned().collect(),
                    activation: post.iter().cloned().collect(),
                    function: activation_name.to_string(),
                }
            }).collect(),
            output,
        };
        
        serde_json::to_string(&response).unwrap_or_else(|_| r#"{"inputs":[0,0],"layers":[],"output":0}"#.to_string())
    }
}

/// Initialize the module
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// ===== Singleton for convenience functions =====

/// Global singleton instance for quick functions (parsed once, reused)
static SINGLETON_NETWORK: OnceLock<Network> = OnceLock::new();

/// Get or initialize the singleton network instance
fn get_singleton() -> &'static Network {
    SINGLETON_NETWORK.get_or_init(|| {
        serde_json::from_str(MODEL_JSON).expect("Failed to parse embedded model")
    })
}

/// Quick predict function (uses singleton - no parsing overhead)
#[wasm_bindgen]
pub fn xor_predict(x1: f64, x2: f64) -> u8 {
    let input = array![x1, x2];
    let output = get_singleton().predict(&input);
    if output[0] > 0.5 { 1 } else { 0 }
}

/// Quick raw predict function (uses singleton - no parsing overhead)
#[wasm_bindgen]
pub fn xor_predict_raw(x1: f64, x2: f64) -> f64 {
    let input = array![x1, x2];
    let output = get_singleton().predict(&input);
    output[0]
}