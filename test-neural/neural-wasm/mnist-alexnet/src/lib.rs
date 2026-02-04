//! WebAssembly AlexNet-Mini CNN for MNIST
//!
//! This module exposes an AlexNet-Mini CNN trained on MNIST via WebAssembly.
//! Architecture adapted from Krizhevsky et al., 2012 for 28x28 grayscale images.

use cma_cnn::{ActivationLayer, BatchNorm2D, Conv2D, MaxPool2D, Sequential, Tensor4D, TensorShape};
use cma_neural_network::network::Network;
use ndarray::Array1;
use neural_wasm_shared::{
    build_prediction_result, build_test_result, load_model_from_bytes, LayerInfo, ModelInfo,
    NormalizationStats, TestResult, WeightsInfo,
};
use wasm_bindgen::prelude::*;

// Embed the pre-trained model at compile time (only when not training)
#[cfg(not(feature = "training"))]
const MODEL_BIN: &[u8] = include_bytes!("alexnet_model.bin");

#[cfg(feature = "training")]
const MODEL_BIN: &[u8] = &[];

// Class names for MNIST digits
const CLASS_NAMES: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

// ===== AlexNet-Mini CNN Structure =====

/// Build the AlexNet-Mini CNN feature extractor
fn build_alexnet_cnn() -> Sequential {
    Sequential::named("AlexNet-Mini")
        // Block 1: Conv2D 1→64, 3x3, pad=1 → 28x28x64
        .add_conv2d(Conv2D::new(1, 64, 3, 1, 1))
        .add_batchnorm(BatchNorm2D::new(64))
        .add_activation(ActivationLayer::relu())
        .add_maxpool(MaxPool2D::new(2, 2)) // → 14x14x64
        // Block 2: Conv2D 64→128, 3x3, pad=1 → 14x14x128
        .add_conv2d(Conv2D::new(64, 128, 3, 1, 1))
        .add_batchnorm(BatchNorm2D::new(128))
        .add_activation(ActivationLayer::relu())
        .add_maxpool(MaxPool2D::new(2, 2)) // → 7x7x128
        // Block 3: Conv2D 128→256, 3x3, pad=1 → 7x7x256
        .add_conv2d(Conv2D::new(128, 256, 3, 1, 1))
        .add_batchnorm(BatchNorm2D::new(256))
        .add_activation(ActivationLayer::relu())
        // Block 4: Conv2D 256→256, 3x3, pad=1 → 7x7x256
        .add_conv2d(Conv2D::new(256, 256, 3, 1, 1))
        .add_batchnorm(BatchNorm2D::new(256))
        .add_activation(ActivationLayer::relu())
        .add_maxpool(MaxPool2D::new(2, 2)) // → 3x3x256
        // Flatten
        .add_flatten()
}

// ===== Main Network Struct =====

/// AlexNet-Mini MNIST Neural Network exposed to JavaScript
#[wasm_bindgen]
pub struct MnistAlexNetNetwork {
    cnn: Sequential,
    classifier: Network,
    accuracy: f64,
    test_samples: usize,
    trained_at: String,
    normalization: Option<NormalizationStats>,
}

#[wasm_bindgen]
impl MnistAlexNetNetwork {
    /// Create a new AlexNet-Mini MNIST network by loading the embedded model
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<MnistAlexNetNetwork, JsValue> {
        let model = load_model_from_bytes(MODEL_BIN)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;

        let cnn = build_alexnet_cnn();

        Ok(MnistAlexNetNetwork {
            cnn,
            classifier: model.network,
            accuracy: model.metadata.accuracy,
            test_samples: model.metadata.test_samples,
            trained_at: model.metadata.trained_at,
            normalization: model.metadata.normalization,
        })
    }

    /// Predict MNIST digit from pixel array using AlexNet-Mini CNN
    /// Accepts 784 pixels (28x28 image)
    /// Returns JSON with digit prediction (0-9), probabilities, and confidence
    #[wasm_bindgen]
    pub fn predict(&self, pixels: &[f64]) -> String {
        if pixels.len() != 784 {
            return serde_json::json!({
                "error": format!("Expected 784 pixels, got {}", pixels.len())
            })
            .to_string();
        }

        let probs = self.forward(pixels);
        let class_names: Vec<String> = CLASS_NAMES.iter().map(|s| s.to_string()).collect();

        let result = build_prediction_result(&probs, &class_names);
        serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
    }

    /// Full forward pass: normalize → CNN → FC classifier
    fn forward(&self, pixels: &[f64]) -> Vec<f64> {
        // Normalize input
        let normalized = self.normalize_input(pixels);

        // Reshape to [1, 1, 28, 28]
        let tensor = Tensor4D::from_array(
            ndarray::Array4::from_shape_vec((1, 1, 28, 28), normalized)
                .expect("Failed to reshape input"),
        );

        // CNN forward pass
        let features = self.cnn.forward(&tensor);

        // Flatten to Array1
        let flat = features.flatten();
        let fc_input = Array1::from_vec(flat.row(0).to_vec());

        // FC classifier
        let output = self.classifier.predict(&fc_input);
        output.to_vec()
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

        let probs = self.forward(pixels);
        serde_json::to_string(&probs).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get class names (digits 0-9)
    #[wasm_bindgen]
    pub fn get_class_names(&self) -> String {
        serde_json::to_string(&CLASS_NAMES.to_vec()).unwrap_or_else(|_| "[]".to_string())
    }

    /// Test with sample MNIST digits
    #[wasm_bindgen]
    pub fn test_all(&self) -> String {
        let test_samples = get_mnist_test_samples();
        let class_names: Vec<String> = CLASS_NAMES.iter().map(|s| s.to_string()).collect();

        let results: Vec<TestResult> = test_samples
            .iter()
            .map(|(pixels, expected)| {
                let probs = self.forward(pixels);
                build_test_result(pixels.clone(), *expected as usize, &probs, &class_names)
            })
            .collect();

        serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get model info with accuracy and metadata
    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        let cnn_arch = format!(
            "AlexNet-Mini: Conv(1→64,3x3)→BN→Pool→Conv(64→128)→BN→Pool→Conv(128→256)→BN→Conv(256→256)→BN→Pool→FC(512)→10"
        );

        let info = ModelInfo {
            name: "AlexNet-Mini MNIST Classifier".to_string(),
            architecture: cnn_arch,
            accuracy: self.accuracy * 100.0,
            description: "AlexNet-Mini CNN for MNIST (Krizhevsky et al., 2012 style)".to_string(),
            test_samples: self.test_samples,
            trained_at: self.trained_at.clone(),
        };
        serde_json::to_string(&info).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get FC classifier weights and biases as JSON
    #[wasm_bindgen]
    pub fn get_weights(&self) -> String {
        let layers = self.classifier.get_layers_info();
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

    /// Get CNN architecture summary
    #[wasm_bindgen]
    pub fn get_cnn_summary(&self) -> String {
        let input_shape = TensorShape::new(1, 1, 28, 28);
        let output_shape = self.cnn.output_shape(input_shape);

        serde_json::json!({
            "name": "AlexNet-Mini",
            "input_shape": [1, 1, 28, 28],
            "output_shape": [output_shape.batch, output_shape.channels, output_shape.height, output_shape.width],
            "num_cnn_parameters": self.cnn.num_parameters(),
            "layers": [
                {"name": "Conv2D", "config": "1→64, 3x3, pad=1"},
                {"name": "BatchNorm2D", "config": "64"},
                {"name": "ReLU", "config": ""},
                {"name": "MaxPool2D", "config": "2x2"},
                {"name": "Conv2D", "config": "64→128, 3x3, pad=1"},
                {"name": "BatchNorm2D", "config": "128"},
                {"name": "ReLU", "config": ""},
                {"name": "MaxPool2D", "config": "2x2"},
                {"name": "Conv2D", "config": "128→256, 3x3, pad=1"},
                {"name": "BatchNorm2D", "config": "256"},
                {"name": "ReLU", "config": ""},
                {"name": "Conv2D", "config": "256→256, 3x3, pad=1"},
                {"name": "BatchNorm2D", "config": "256"},
                {"name": "ReLU", "config": ""},
                {"name": "MaxPool2D", "config": "2x2"},
                {"name": "Flatten", "config": "→2304"}
            ]
        }).to_string()
    }
}

/// Initialize the module
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Sample MNIST test data
fn get_mnist_test_samples() -> Vec<(Vec<f64>, u8)> {
    vec![
        (vec![0.0; 784], 0),
        (vec![0.0; 784], 1),
        (vec![0.0; 784], 2),
        (vec![0.0; 784], 3),
        (vec![0.0; 784], 4),
    ]
}
