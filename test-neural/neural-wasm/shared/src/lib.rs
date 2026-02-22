use chrono;
use cma_cnn::sequential::Sequential as CnnSequential;
use cma_neural_network::network::Network;
use cma_neural_network::{Dim, Float};
use ndarray;
use serde::{Deserialize, Serialize};

/// Normalization statistics for input features
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NormalizationStats {
    pub means: Vec<Float>,
    pub stds: Vec<Float>,
}

impl NormalizationStats {
    pub fn new(means: Vec<Float>, stds: Vec<Float>) -> Self {
        Self { means, stds }
    }

    /// Normalize a single input using these statistics
    pub fn normalize(&self, input: &[Float]) -> Vec<Float> {
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
    pub accuracy: Float,
    pub test_samples: Dim,
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

/// CNN model wrapper: CNN feature extractor + FC classifier + metadata
///
/// This is the new format for models that use trained CNN weights (ResNet, LeNet, etc.).
/// Unlike `ModelWithMetadata` which only saves the FC head, this saves the full pipeline.
#[derive(Serialize, Deserialize)]
pub struct CnnModelWithMetadata {
    /// Trained CNN feature extractor (Conv2D, BatchNorm2D, Pool, etc.)
    pub cnn: CnnSequential,
    /// FC classifier head
    pub classifier: Network,
    /// Training metadata (accuracy, normalization stats, etc.)
    pub metadata: ModelMetadata,
}

/// Save a CNN model (feature extractor + classifier) to binary file
pub fn save_cnn_model_binary(
    cnn: CnnSequential,
    classifier: Network,
    accuracy: Float,
    test_samples: usize,
    normalization: Option<NormalizationStats>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = CnnModelWithMetadata {
        cnn,
        classifier,
        metadata: ModelMetadata {
            accuracy,
            test_samples: test_samples as Dim,
            trained_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            normalization,
        },
    };

    let bin = bincode::serialize(&model)?;
    std::fs::write(path, &bin)?;

    let size_kb = bin.len() as f64 / 1024.0;
    eprintln!(
        "Saved CNN model to {} ({:.1} KB, accuracy={:.1}%)",
        path,
        size_kb,
        accuracy * 100.0
    );

    Ok(())
}

/// Load a CNN model from binary bytes (for WASM with include_bytes!)
pub fn load_cnn_model_from_bytes(
    bytes: &[u8],
) -> Result<CnnModelWithMetadata, Box<dyn std::error::Error>> {
    let model: CnnModelWithMetadata = bincode::deserialize(bytes)?;
    Ok(model)
}

/// Information about a trained model
#[derive(Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub architecture: String,
    pub accuracy: Float,
    pub description: String,
    pub test_samples: usize,
    pub trained_at: String,
}

/// Information about a network layer
#[derive(Serialize)]
pub struct LayerInfo {
    pub weights: Vec<Vec<Float>>,
    pub biases: Vec<Float>,
    pub activation: String,
    pub shape: [usize; 2],
}

/// Information about network weights
#[derive(Serialize)]
pub struct WeightsInfo {
    pub layers: Vec<LayerInfo>,
}

/// Architecture summary for any model (FC, CNN, ResNet, etc.)
#[derive(Serialize)]
pub struct ArchitectureSummary {
    pub name: String,
    pub model_type: String, // "fc", "cnn", "resnet"
    pub input_shape: Vec<usize>,
    pub output_features: usize,
    pub num_parameters: usize,
    pub layers: Vec<LayerSummary>,
}

/// Layer summary for architecture display
#[derive(Serialize)]
pub struct LayerSummary {
    pub name: String,
    pub config: String,
}

impl ArchitectureSummary {
    /// Create a summary for a fully-connected network
    pub fn fc(name: &str, input_size: usize, architecture: &str, num_params: usize) -> Self {
        Self {
            name: name.to_string(),
            model_type: "fc".to_string(),
            input_shape: vec![input_size],
            output_features: 0,
            num_parameters: num_params,
            layers: vec![LayerSummary {
                name: "FC".to_string(),
                config: architecture.to_string(),
            }],
        }
    }

    /// Create a summary for a CNN
    pub fn cnn(
        name: &str,
        input_shape: Vec<usize>,
        output_features: usize,
        num_params: usize,
        layers: Vec<(&str, &str)>,
    ) -> Self {
        Self {
            name: name.to_string(),
            model_type: "cnn".to_string(),
            input_shape,
            output_features,
            num_parameters: num_params,
            layers: layers
                .into_iter()
                .map(|(n, c)| LayerSummary {
                    name: n.to_string(),
                    config: c.to_string(),
                })
                .collect(),
        }
    }
}

// ===== Common Response Structures =====

/// Prediction result for any classifier
#[derive(Serialize)]
pub struct PredictionResult {
    pub class_name: String,
    pub class_index: usize,
    pub probabilities: Vec<Float>,
    pub confidence: Float,
}

/// Layer activation data for visualization
#[derive(Serialize)]
pub struct LayerActivation {
    pub pre_activation: Vec<Float>,
    pub activation: Vec<Float>,
    pub function: String,
}

/// Full activation response for network visualization
#[derive(Serialize)]
pub struct ActivationsResponse {
    pub inputs: Vec<Float>,
    pub layers: Vec<LayerActivation>,
    pub output: Vec<Float>,
}

// ===== CNN Activation Types =====

/// One CNN layer's intermediate output for visualization
#[derive(Serialize)]
pub struct CnnLayerActivation {
    /// Layer type name: "Conv2D", "MaxPool2D", "ReLU", "BatchNorm2D", "Flatten", etc.
    pub layer_type: String,
    /// Human-readable config: "1→32, 3×3, s=1, p=1"
    pub config: String,
    /// Output shape [channels, height, width]
    pub shape: Vec<usize>,
    /// Flattened activation data (C×H×W values for a single sample)
    pub activations: Vec<Float>,
}

/// Full CNN forward pass result with all intermediate activations
#[derive(Serialize)]
pub struct CnnActivationsResponse {
    /// Input shape [channels, height, width]
    pub input_shape: Vec<usize>,
    /// Per-layer intermediate activations
    pub layers: Vec<CnnLayerActivation>,
    /// Output shape of the last CNN layer
    pub output_shape: Vec<usize>,
}

/// Build CNN activations response from a CnnSequential forward pass
pub fn build_cnn_activations(
    cnn: &CnnSequential,
    input: &cma_cnn::Tensor4D,
) -> CnnActivationsResponse {
    let input_shape = input.shape();
    let intermediates = cnn.forward_with_intermediates(input);

    let layers: Vec<CnnLayerActivation> = intermediates
        .iter()
        .map(|(layer_type, config, tensor)| {
            let shape = tensor.shape();
            // Extract data for sample 0 only (batch=0)
            let sample_data: Vec<Float> = tensor
                .data()
                .slice(ndarray::s![0, .., .., ..])
                .iter()
                .copied()
                .collect();

            CnnLayerActivation {
                layer_type: layer_type.clone(),
                config: config.clone(),
                shape: vec![shape.channels, shape.height, shape.width],
                activations: sample_data,
            }
        })
        .collect();

    let output_shape = if let Some(last) = intermediates.last() {
        let s = last.2.shape();
        vec![s.channels, s.height, s.width]
    } else {
        vec![]
    };

    CnnActivationsResponse {
        input_shape: vec![input_shape.channels, input_shape.height, input_shape.width],
        layers,
        output_shape,
    }
}

/// Generic test result for any classifier
#[derive(Serialize)]
pub struct TestResult {
    pub inputs: Vec<Float>,
    pub expected_class: String,
    pub expected_index: usize,
    pub predicted_class: String,
    pub predicted_index: usize,
    pub probabilities: Vec<Float>,
    pub confidence: Float,
    pub is_correct: bool,
}

// ===== Utility Functions =====

/// Find the class with highest probability
pub fn find_max_class(probs: &[Float]) -> (usize, Float) {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &val)| (idx, val))
        .unwrap_or((0, 0.0))
}

/// Validate input size
pub fn validate_input_size(input: &[Float], expected: usize) -> Result<(), String> {
    if input.len() != expected {
        Err(format!("Expected {} inputs, got {}", expected, input.len()))
    } else {
        Ok(())
    }
}

/// Build a prediction result from probabilities
pub fn build_prediction_result(probs: &[Float], class_names: &[String]) -> PredictionResult {
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
    inputs: Vec<Float>,
    expected_index: usize,
    probs: &[Float],
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
pub fn confidence_to_percentage(value: Float) -> Float {
    (value * 100.0).max(0.0).min(100.0)
}

/// Softmax function for multi-class probability
pub fn softmax(values: &[Float]) -> Vec<Float> {
    let max = values.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
    let exp_values: Vec<Float> = values.iter().map(|&v| (v - max).exp()).collect();
    let sum: Float = exp_values.iter().sum();
    exp_values.iter().map(|&v| v / sum).collect()
}

/// Calculate accuracy for multi-class classification
pub fn calculate_multiclass_accuracy(
    network: &Network,
    inputs: &[ndarray::Array1<Float>],
    targets: &[ndarray::Array1<Float>],
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
    accuracy: Float,
    test_samples: usize,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    save_model_with_normalization(network, accuracy, test_samples, None, path)
}

/// Save model with metadata and normalization statistics to JSON file
pub fn save_model_with_normalization(
    network: Network,
    accuracy: Float,
    test_samples: usize,
    normalization: Option<NormalizationStats>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_with_metadata = ModelWithMetadata {
        network,
        metadata: ModelMetadata {
            accuracy,
            test_samples: test_samples as Dim,
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
    accuracy: Float,
    test_samples: usize,
    normalization: Option<NormalizationStats>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_with_metadata = ModelWithMetadata {
        network,
        metadata: ModelMetadata {
            accuracy,
            test_samples: test_samples as Dim,
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
        let sum: Float = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_to_percentage() {
        assert_eq!(confidence_to_percentage(0.5), 50.0);
        assert_eq!(confidence_to_percentage(1.0), 100.0);
        assert_eq!(confidence_to_percentage(0.0), 0.0);
    }

    #[test]
    fn test_cnn_model_roundtrip() {
        use cma_cnn::layers::{Conv2D, MaxPool2D, ActivationLayer};
        use cma_cnn::sequential::Sequential as CnnSeq;

        // Build a small CNN
        let cnn = CnnSeq::new()
            .add_conv2d(Conv2D::new(1, 4, 3, 1, 0))
            .add_activation(ActivationLayer::relu())
            .add_maxpool(MaxPool2D::new(2, 2))
            .add_flatten();

        // Build a small FC classifier
        use cma_neural_network::builder::NetworkBuilder;
        use cma_neural_network::network::Activation;
        let classifier = NetworkBuilder::new(36, 10)
            .hidden_layer(32, Activation::ReLU)
            .output_activation(Activation::Softmax)
            .build();

        // Build the combined model
        let model = CnnModelWithMetadata {
            cnn: cnn.clone(),
            classifier,
            metadata: ModelMetadata {
                accuracy: 0.95,
                test_samples: 1000,
                trained_at: "2025-01-01 00:00:00".to_string(),
                normalization: Some(NormalizationStats::new(vec![0.5; 36], vec![0.25; 36])),
            },
        };

        // Serialize to bincode
        let bytes = bincode::serialize(&model).expect("serialize failed");
        assert!(bytes.len() > 0, "Serialized bytes should not be empty");

        // Deserialize
        let loaded: CnnModelWithMetadata =
            bincode::deserialize(&bytes).expect("deserialize failed");

        // Verify metadata
        assert!((loaded.metadata.accuracy - 0.95).abs() < 1e-6);
        assert_eq!(loaded.metadata.test_samples, 1000);
        assert!(loaded.metadata.normalization.is_some());

        // Verify CNN has same structure (layer count)
        assert_eq!(loaded.cnn.layers().len(), model.cnn.layers().len());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MNIST Data Loading (shared across all MNIST variants)
// ═══════════════════════════════════════════════════════════════════════════

/// Load the MNIST dataset from CSV file (OpenML format)
/// Format: 784 pixel values (0-255), then label (0-9) as last column
/// Dataset source: https://www.openml.org/d/554
pub fn load_mnist_from_csv(
    path: &str,
) -> Result<Vec<(ndarray::Array1<Float>, ndarray::Array1<Float>)>, Box<dyn std::error::Error>> {
    use csv::ReaderBuilder;

    let mut data = Vec::new();
    let mut rdr = ReaderBuilder::new().has_headers(false).from_path(path)?;

    for result in rdr.records() {
        let record = result?;

        if record.len() != 785 {
            return Err(format!(
                "Expected 785 columns (784 pixels + 1 label), got {}",
                record.len()
            )
            .into());
        }

        // Parse the 784 pixel values (columns 0-783)
        let mut pixels = Vec::with_capacity(784);
        for i in 0..784 {
            let pixel: Float = record[i].parse()?;
            pixels.push(pixel);
        }

        // Parse label (last column, index 784) and convert to one-hot encoding (10 classes)
        let label: usize = record[784].parse()?;
        if label > 9 {
            return Err(format!("Invalid label: {} (expected 0-9)", label).into());
        }

        let mut one_hot = vec![0.0; 10];
        one_hot[label] = 1.0;

        data.push((
            ndarray::Array1::from_vec(pixels),
            ndarray::Array1::from_vec(one_hot),
        ));
    }

    Ok(data)
}

/// Normalize features using z-score normalization (mean=0, std=1)
/// Returns normalized data AND the normalization statistics for inference
pub fn normalize_features_with_stats(
    inputs: &[ndarray::Array1<Float>],
) -> (Vec<ndarray::Array1<Float>>, NormalizationStats) {
    if inputs.is_empty() {
        return (vec![], NormalizationStats::new(vec![], vec![]));
    }

    let n_features = inputs[0].len();
    let n_samples = inputs.len() as Float;

    // Calculate mean for each feature
    let mut means = vec![0.0; n_features];
    for input in inputs {
        for (i, &val) in input.iter().enumerate() {
            means[i] += val;
        }
    }
    for mean in &mut means {
        *mean /= n_samples;
    }

    // Calculate standard deviation for each feature
    let mut stds = vec![0.0; n_features];
    for input in inputs {
        for (i, &val) in input.iter().enumerate() {
            stds[i] += (val - means[i]).powi(2);
        }
    }
    for std in &mut stds {
        *std = (*std / n_samples).sqrt();
        // Prevent division by zero
        if *std < 1e-8 {
            *std = 1.0;
        }
    }

    // Normalize each input
    let normalized = inputs
        .iter()
        .map(|input| {
            ndarray::Array1::from_vec(
                input
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| (val - means[i]) / stds[i])
                    .collect(),
            )
        })
        .collect();

    (normalized, NormalizationStats::new(means, stds))
}

/// Reshape flat MNIST pixels to 2D image format for CNN
/// Input: 784 values (flattened 28x28)
/// Output: [1, 28, 28] for single channel grayscale
pub fn reshape_mnist_to_image(pixels: &[Float]) -> ndarray::Array3<Float> {
    assert_eq!(pixels.len(), 784, "MNIST images must be 784 pixels");
    ndarray::Array3::from_shape_vec((1, 28, 28), pixels.to_vec())
        .expect("Failed to reshape MNIST pixels")
}

/// Returns a set of placeholder MNIST test samples (one per digit class).
///
/// Used by WASM network structs to populate the `test_all()` demo endpoint.
pub fn get_mnist_test_samples() -> Vec<(Vec<Float>, u8)> {
    vec![
        (vec![0.5; 784], 0),
        (vec![0.3; 784], 1),
        (vec![0.7; 784], 2),
        (vec![0.4; 784], 3),
        (vec![0.6; 784], 4),
        (vec![0.2; 784], 5),
        (vec![0.8; 784], 6),
        (vec![0.45; 784], 7),
        (vec![0.55; 784], 8),
        (vec![0.65; 784], 9),
    ]
}

/// Generates a complete `#[wasm_bindgen]` CNN MNIST network struct with all methods.
///
/// # Parameters
/// - `struct_name`: identifier for the generated struct
/// - `model_file`: literal path to the model binary (for `include_bytes!`)
/// - `model_info_name`: long display name returned by `model_info()`
/// - `description`: description string in `model_info()`
/// - `arch_name`: short architecture name in `get_architecture()`
/// - `output_features`: CNN output feature count before the FC classifier
///
/// # Example
/// ```rust,ignore
/// neural_wasm_shared::define_cnn_mnist_network! {
///     struct_name = MnistLeNetNetwork,
///     model_file = "lenet_model.bin",
///     model_info_name = "LeNet-5 MNIST Classifier",
///     description = "LeNet-5 CNN (LeCun et al., 1998)",
///     arch_name = "LeNet-5",
///     output_features = 120,
/// }
/// ```
#[macro_export]
macro_rules! define_cnn_mnist_network {
    (
        struct_name = $struct_name:ident,
        model_file = $model_file:literal,
        model_info_name = $model_info_name:literal,
        description = $description:literal,
        arch_name = $arch_name:literal,
        output_features = $output_features:expr $(,)?
    ) => {
        use cma_cnn::sequential::Sequential as CnnSequential;
        use cma_cnn::{Float, Tensor4D};
        use cma_neural_network::network::Network;
        use ndarray::Array1;
        use neural_wasm_shared::{
            build_cnn_activations, build_prediction_result, build_test_result,
            get_mnist_test_samples, load_cnn_model_from_bytes, ActivationsResponse,
            ArchitectureSummary, LayerActivation, LayerInfo, LayerSummary, ModelInfo,
            NormalizationStats, TestResult, WeightsInfo,
        };
        use wasm_bindgen::prelude::*;

        #[cfg(not(feature = "training"))]
        const MODEL_BIN: &[u8] = include_bytes!($model_file);

        #[cfg(feature = "training")]
        const MODEL_BIN: &[u8] = &[];

        const CLASS_NAMES: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

        #[wasm_bindgen]
        pub struct $struct_name {
            cnn: CnnSequential,
            classifier: Network,
            accuracy: Float,
            test_samples: usize,
            trained_at: String,
            normalization: Option<NormalizationStats>,
        }

        #[wasm_bindgen]
        impl $struct_name {
            #[wasm_bindgen(constructor)]
            pub fn new() -> Result<$struct_name, JsValue> {
                let model = load_cnn_model_from_bytes(MODEL_BIN)
                    .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;
                Ok($struct_name {
                    cnn: model.cnn,
                    classifier: model.classifier,
                    accuracy: model.metadata.accuracy,
                    test_samples: model.metadata.test_samples as usize,
                    trained_at: model.metadata.trained_at,
                    normalization: model.metadata.normalization,
                })
            }

            #[wasm_bindgen]
            pub fn predict(&self, pixels: &[Float]) -> String {
                if pixels.len() != 784 {
                    return serde_json::json!({"error": format!("Expected 784 pixels, got {}", pixels.len())}).to_string();
                }
                let probs = self.forward(pixels);
                let class_names: Vec<String> = CLASS_NAMES.iter().map(|s| s.to_string()).collect();
                let result = build_prediction_result(&probs, &class_names);
                serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
            }

            fn forward(&self, pixels: &[Float]) -> Vec<Float> {
                let normalized = self.normalize_input(pixels);
                let tensor = Tensor4D::from_array(
                    ndarray::Array4::from_shape_vec((1, 1, 28, 28), normalized)
                        .expect("reshape failed"),
                );
                let features = self.cnn.forward(&tensor);
                let flat = features.flatten();
                let fc_input = Array1::from_vec(flat.row(0).to_vec());
                self.classifier.predict(&fc_input).to_vec()
            }

            fn normalize_input(&self, pixels: &[Float]) -> Vec<Float> {
                if let Some(ref norm) = self.normalization {
                    norm.normalize(pixels)
                } else {
                    pixels.to_vec()
                }
            }

            #[wasm_bindgen]
            pub fn get_probabilities(&self, pixels: &[Float]) -> String {
                if pixels.len() != 784 {
                    return serde_json::json!({"error": "Expected 784 pixels"}).to_string();
                }
                serde_json::to_string(&self.forward(pixels)).unwrap_or_else(|_| "[]".to_string())
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
                let num_cnn_layers = self.cnn.layers().len();
                let info = ModelInfo {
                    name: $model_info_name.to_string(),
                    architecture: format!("CNN: {} layers \u{2192} FC(\u{2192}10)", num_cnn_layers),
                    accuracy: self.accuracy * 100.0,
                    description: $description.to_string(),
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
                            let weights_2d: Vec<Vec<Float>> = weights
                                .rows()
                                .into_iter()
                                .map(|row| row.to_vec())
                                .collect();
                            LayerInfo {
                                weights: weights_2d,
                                biases: biases.to_vec(),
                                activation: activation_name.to_string(),
                                shape: [weights.nrows(), weights.ncols()],
                            }
                        })
                        .collect(),
                };
                serde_json::to_string(&response)
                    .unwrap_or_else(|_| r#"{"layers":[]}"#.to_string())
            }

            #[wasm_bindgen]
            pub fn get_architecture(&self) -> String {
                let layers: Vec<LayerSummary> = self
                    .cnn
                    .layers()
                    .iter()
                    .map(|layer| LayerSummary {
                        name: layer.type_name().to_string(),
                        config: layer.config_string(),
                    })
                    .collect();
                let num_params = self.cnn.num_parameters()
                    + self
                        .classifier
                        .get_layers_info()
                        .iter()
                        .map(|(w, b, _)| w.len() + b.len())
                        .sum::<usize>();
                let summary = ArchitectureSummary {
                    name: $arch_name.to_string(),
                    model_type: "cnn".to_string(),
                    input_shape: vec![1, 1, 28, 28],
                    output_features: $output_features,
                    num_parameters: num_params,
                    layers,
                };
                serde_json::to_string(&summary).unwrap_or_else(|_| "{}".to_string())
            }

            /// Get CNN intermediate activations for visualization.
            #[wasm_bindgen]
            pub fn get_cnn_activations(&self, pixels: &[Float]) -> String {
                if pixels.len() != 784 {
                    return serde_json::json!({"error": format!("Expected 784 pixels, got {}", pixels.len())}).to_string();
                }
                let normalized = self.normalize_input(pixels);
                let tensor = Tensor4D::from_array(
                    ndarray::Array4::from_shape_vec((1, 1, 28, 28), normalized)
                        .expect("reshape failed"),
                );
                let response = build_cnn_activations(&self.cnn, &tensor);
                serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string())
            }

            /// Get FC classifier activations for visualization.
            #[wasm_bindgen]
            pub fn get_activations(&self, pixels: &[Float]) -> String {
                if pixels.len() != 784 {
                    return r#"{"inputs":[],"layers":[],"output":[]}"#.to_string();
                }
                let normalized = self.normalize_input(pixels);
                let tensor = Tensor4D::from_array(
                    ndarray::Array4::from_shape_vec((1, 1, 28, 28), normalized)
                        .expect("reshape failed"),
                );
                let features = self.cnn.forward(&tensor);
                let flat = features.flatten();
                let fc_input = Array1::from_vec(flat.row(0).to_vec());
                let activations = self.classifier.get_all_activations(&fc_input);
                let output_probs = activations
                    .last()
                    .map(|(_, post, _)| post.to_vec())
                    .unwrap_or_else(|| vec![0.0; 10]);
                let response = ActivationsResponse {
                    inputs: flat.row(0).to_vec(),
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

        #[wasm_bindgen(start)]
        pub fn main() {
            #[cfg(feature = "console_error_panic_hook")]
            console_error_panic_hook::set_once();
        }
    };
}
