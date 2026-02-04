use cma_neural_network::network::Network;
use cma_neural_network::Float;
use ndarray::array;
use neural_wasm_shared::{
    build_prediction_result, build_test_result, load_model_from_bytes, ActivationsResponse,
    ArchitectureSummary, LayerActivation, LayerInfo, LayerSummary, ModelInfo, NormalizationStats,
    TestResult, WeightsInfo,
};
use wasm_bindgen::prelude::*;

// Embed the pre-trained model at compile time (binary format for smaller size)
const MODEL_BIN: &[u8] = include_bytes!("iris_model.bin");

#[wasm_bindgen]
pub struct IrisClassifier {
    network: Network,
    classes: Vec<String>,
    accuracy: Float,
    test_samples: usize,
    trained_at: String,
    normalization: Option<NormalizationStats>,
}

#[wasm_bindgen]
impl IrisClassifier {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<IrisClassifier, JsValue> {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        let model = load_model_from_bytes(MODEL_BIN)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;

        let classes = vec![
            "Setosa".to_string(),
            "Versicolor".to_string(),
            "Virginica".to_string(),
        ];

        Ok(IrisClassifier {
            network: model.network,
            classes,
            accuracy: model.metadata.accuracy,
            test_samples: model.metadata.test_samples,
            trained_at: model.metadata.trained_at,
            normalization: model.metadata.normalization,
        })
    }

    /// Normalize input features using stored statistics
    fn normalize_input(
        &self,
        sepal_length: Float,
        sepal_width: Float,
        petal_length: Float,
        petal_width: Float,
    ) -> [Float; 4] {
        if let Some(ref norm) = self.normalization {
            let raw = [sepal_length, sepal_width, petal_length, petal_width];
            let normalized = norm.normalize(&raw);
            [normalized[0], normalized[1], normalized[2], normalized[3]]
        } else {
            // No normalization stats - use raw values (backward compatibility)
            [sepal_length, sepal_width, petal_length, petal_width]
        }
    }

    /// Predict iris species from measurements
    /// Parameters: sepal_length, sepal_width, petal_length, petal_width (in cm)
    #[wasm_bindgen]
    pub fn predict(
        &self,
        sepal_length: Float,
        sepal_width: Float,
        petal_length: Float,
        petal_width: Float,
    ) -> String {
        let normalized = self.normalize_input(sepal_length, sepal_width, petal_length, petal_width);
        let input = array![normalized[0], normalized[1], normalized[2], normalized[3]];
        let output = self.network.predict(&input);

        // Network already uses Softmax output activation - output IS probabilities
        let probs = output.to_vec();
        let result = build_prediction_result(&probs, &self.classes);

        serde_json::to_string(&result).unwrap()
    }

    /// Get class probabilities for a prediction
    #[wasm_bindgen]
    pub fn get_probabilities(
        &self,
        sepal_length: Float,
        sepal_width: Float,
        petal_length: Float,
        petal_width: Float,
    ) -> String {
        let normalized = self.normalize_input(sepal_length, sepal_width, petal_length, petal_width);
        let input = array![normalized[0], normalized[1], normalized[2], normalized[3]];
        let output = self.network.predict(&input);
        // Network already uses Softmax output activation - output IS probabilities
        serde_json::to_string(&output.to_vec()).unwrap()
    }

    /// Test all samples from the dataset
    #[wasm_bindgen]
    pub fn test_all(&self) -> String {
        let test_data = get_iris_test_samples();
        let results: Vec<TestResult> = test_data
            .iter()
            .map(|(inputs, expected_idx)| {
                let input = array![inputs[0], inputs[1], inputs[2], inputs[3]];
                let output = self.network.predict(&input);
                let probs = output.to_vec();

                build_test_result(inputs.to_vec(), *expected_idx, &probs, &self.classes)
            })
            .collect();

        serde_json::to_string(&results).unwrap()
    }

    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        // Accuracy is loaded from the model metadata (saved during training)
        let info = ModelInfo {
            name: "Iris Species Classifier".to_string(),
            architecture: self.network.architecture_string(),
            accuracy: self.accuracy * 100.0,
            description: "Classifies iris flowers into three species: Setosa, Versicolor, and Virginica based on sepal and petal measurements".to_string(),
            test_samples: self.test_samples,
            trained_at: self.trained_at.clone(),
        };
        serde_json::to_string(&info).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_weights(&self) -> String {
        let layers_info = self.network.get_layers_info();
        let layers: Vec<LayerInfo> = layers_info
            .iter()
            .map(|(weights, biases, activation_name)| {
                let weights_2d: Vec<Vec<Float>> =
                    weights.rows().into_iter().map(|row| row.to_vec()).collect();

                LayerInfo {
                    weights: weights_2d,
                    biases: biases.to_vec(),
                    activation: activation_name.to_string(),
                    shape: [weights.nrows(), weights.ncols()],
                }
            })
            .collect();

        let weights_info = WeightsInfo { layers };

        serde_json::to_string(&weights_info).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_class_names(&self) -> String {
        serde_json::to_string(&self.classes).unwrap()
    }

    /// Get layer-by-layer activations for visualization
    #[wasm_bindgen]
    pub fn get_activations(
        &self,
        sepal_length: Float,
        sepal_width: Float,
        petal_length: Float,
        petal_width: Float,
    ) -> String {
        let input = array![sepal_length, sepal_width, petal_length, petal_width];
        let activations = self.network.get_all_activations(&input);

        let output = self.network.predict(&input);
        let probs = output.to_vec(); // Network already uses Softmax

        let response = ActivationsResponse {
            inputs: vec![sepal_length, sepal_width, petal_length, petal_width],
            layers: activations
                .iter()
                .map(|(pre, post, activation_name)| LayerActivation {
                    pre_activation: pre.to_vec(),
                    activation: post.to_vec(),
                    function: activation_name.to_string(),
                })
                .collect(),
            output: probs,
        };

        serde_json::to_string(&response)
            .unwrap_or_else(|_| r#"{"inputs":[],"layers":[],"output":[]}"#.to_string())
    }

    /// Get architecture summary
    #[wasm_bindgen]
    pub fn get_architecture(&self) -> String {
        let layers_info = self.network.get_layers_info();
        let layers: Vec<LayerSummary> = layers_info
            .iter()
            .enumerate()
            .map(|(i, (weights, _, activation))| LayerSummary {
                name: format!("FC{}", i + 1),
                config: format!("{}→{} ({})", weights.ncols(), weights.nrows(), activation),
            })
            .collect();

        let num_params: usize = layers_info.iter().map(|(w, b, _)| w.len() + b.len()).sum();

        let summary = ArchitectureSummary {
            name: "Iris Classifier".to_string(),
            model_type: "fc".to_string(),
            input_shape: vec![4],
            output_features: 3,
            num_parameters: num_params,
            layers,
        };

        serde_json::to_string(&summary).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Sample iris data for testing
fn get_iris_test_samples() -> Vec<([Float; 4], usize)> {
    vec![
        // Setosa samples (class 0)
        ([5.1, 3.5, 1.4, 0.2], 0),
        ([4.9, 3.0, 1.4, 0.2], 0),
        ([5.0, 3.6, 1.4, 0.2], 0),
        ([4.6, 3.1, 1.5, 0.2], 0),
        // Versicolor samples (class 1)
        ([7.0, 3.2, 4.7, 1.4], 1),
        ([6.4, 3.2, 4.5, 1.5], 1),
        ([6.9, 3.1, 4.9, 1.5], 1),
        ([5.5, 2.3, 4.0, 1.3], 1),
        // Virginica samples (class 2)
        ([6.3, 3.3, 6.0, 2.5], 2),
        ([5.8, 2.7, 5.1, 1.9], 2),
        ([7.1, 3.0, 5.9, 2.1], 2),
        ([6.5, 3.0, 5.8, 2.2], 2),
    ]
}
