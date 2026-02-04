//! WebAssembly VGG-Tiny CNN for MNIST
//!
//! VGG-style architecture (Simonyan & Zisserman, 2014) adapted for MNIST.
//! Uses stacked 3x3 convolutions as per the original VGG design.

use cma_cnn::{ActivationLayer, Conv2D, MaxPool2D, Sequential, Tensor4D, TensorShape};
use cma_neural_network::network::Network;
use ndarray::Array1;
use neural_wasm_shared::{
    build_prediction_result, build_test_result, load_model_from_bytes, LayerInfo, ModelInfo,
    NormalizationStats, TestResult, WeightsInfo,
};
use wasm_bindgen::prelude::*;

#[cfg(not(feature = "training"))]
const MODEL_BIN: &[u8] = include_bytes!("vgg_model.bin");

#[cfg(feature = "training")]
const MODEL_BIN: &[u8] = &[];

const CLASS_NAMES: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

fn build_vgg_cnn() -> Sequential {
    Sequential::named("VGG-Tiny")
        // Block 1: 2x Conv 3x3, 32 filters
        .add_conv2d(Conv2D::new(1, 32, 3, 1, 1))
        .add_activation(ActivationLayer::relu())
        .add_conv2d(Conv2D::new(32, 32, 3, 1, 1))
        .add_activation(ActivationLayer::relu())
        .add_maxpool(MaxPool2D::new(2, 2))
        // Block 2: 2x Conv 3x3, 64 filters
        .add_conv2d(Conv2D::new(32, 64, 3, 1, 1))
        .add_activation(ActivationLayer::relu())
        .add_conv2d(Conv2D::new(64, 64, 3, 1, 1))
        .add_activation(ActivationLayer::relu())
        .add_maxpool(MaxPool2D::new(2, 2))
        .add_flatten()
}

#[wasm_bindgen]
pub struct MnistVGGNetwork {
    cnn: Sequential,
    classifier: Network,
    accuracy: f64,
    test_samples: usize,
    trained_at: String,
    normalization: Option<NormalizationStats>,
}

#[wasm_bindgen]
impl MnistVGGNetwork {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<MnistVGGNetwork, JsValue> {
        let model = load_model_from_bytes(MODEL_BIN)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;

        let cnn = build_vgg_cnn();

        Ok(MnistVGGNetwork {
            cnn,
            classifier: model.network,
            accuracy: model.metadata.accuracy,
            test_samples: model.metadata.test_samples,
            trained_at: model.metadata.trained_at,
            normalization: model.metadata.normalization,
        })
    }

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

    fn forward(&self, pixels: &[f64]) -> Vec<f64> {
        let normalized = self.normalize_input(pixels);
        let tensor = Tensor4D::from_array(
            ndarray::Array4::from_shape_vec((1, 1, 28, 28), normalized)
                .expect("Failed to reshape input"),
        );
        let features = self.cnn.forward(&tensor);
        let flat = features.flatten();
        let fc_input = Array1::from_vec(flat.row(0).to_vec());
        let output = self.classifier.predict(&fc_input);
        output.to_vec()
    }

    fn normalize_input(&self, pixels: &[f64]) -> Vec<f64> {
        if let Some(ref norm) = self.normalization {
            norm.normalize(pixels)
        } else {
            pixels.to_vec()
        }
    }

    #[wasm_bindgen]
    pub fn get_probabilities(&self, pixels: &[f64]) -> String {
        if pixels.len() != 784 {
            return serde_json::json!({"error": "Expected 784 pixels"}).to_string();
        }
        let probs = self.forward(pixels);
        serde_json::to_string(&probs).unwrap_or_else(|_| "[]".to_string())
    }

    #[wasm_bindgen]
    pub fn get_class_names(&self) -> String {
        serde_json::to_string(&CLASS_NAMES.to_vec()).unwrap_or_else(|_| "[]".to_string())
    }

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

    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        let info = ModelInfo {
            name: "VGG-Tiny MNIST Classifier".to_string(),
            architecture: "VGG-Tiny: Conv(1→32,3x3)×2→Pool→Conv(32→64,3x3)×2→Pool→FC(128)→10"
                .to_string(),
            accuracy: self.accuracy * 100.0,
            description: "VGG-style CNN for MNIST (Simonyan & Zisserman, 2014 style)".to_string(),
            test_samples: self.test_samples,
            trained_at: self.trained_at.clone(),
        };
        serde_json::to_string(&info).unwrap_or_else(|_| "{}".to_string())
    }

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

    #[wasm_bindgen]
    pub fn get_cnn_summary(&self) -> String {
        let input_shape = TensorShape::new(1, 1, 28, 28);
        let output_shape = self.cnn.output_shape(input_shape);

        serde_json::json!({
            "name": "VGG-Tiny",
            "input_shape": [1, 1, 28, 28],
            "output_shape": [output_shape.batch, output_shape.channels, output_shape.height, output_shape.width],
            "num_cnn_parameters": self.cnn.num_parameters(),
            "layers": [
                {"name": "Conv2D", "config": "1→32, 3x3, pad=1"},
                {"name": "ReLU", "config": ""},
                {"name": "Conv2D", "config": "32→32, 3x3, pad=1"},
                {"name": "ReLU", "config": ""},
                {"name": "MaxPool2D", "config": "2x2"},
                {"name": "Conv2D", "config": "32→64, 3x3, pad=1"},
                {"name": "ReLU", "config": ""},
                {"name": "Conv2D", "config": "64→64, 3x3, pad=1"},
                {"name": "ReLU", "config": ""},
                {"name": "MaxPool2D", "config": "2x2"},
                {"name": "Flatten", "config": "→3136"}
            ]
        }).to_string()
    }
}

#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

fn get_mnist_test_samples() -> Vec<(Vec<f64>, u8)> {
    vec![
        (vec![0.0; 784], 0),
        (vec![0.0; 784], 1),
        (vec![0.0; 784], 2),
        (vec![0.0; 784], 3),
        (vec![0.0; 784], 4),
    ]
}
