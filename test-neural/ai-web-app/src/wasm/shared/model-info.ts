/**
 * Metadata about a trained neural network model.
 */
export interface ModelInfo {
  /** Display name of the model */
  name: string;
  /** Architecture description (e.g., "2 → 4 → 1") */
  architecture: string;
  /** Model accuracy as a percentage (0-100) */
  accuracy: number;
  /** Human-readable description of the model's purpose */
  description: string;
  /** Number of test samples used for validation */
  test_samples: number;
  /** Timestamp when the model was trained */
  trained_at: Date;
}

/**
 * Result of a single test case (unified for all classifiers).
 */
export interface TestResult {
  /** Input values fed to the network */
  inputs: number[];
  /** Expected class name */
  expected_class: string;
  /** Predicted class name */
  predicted_class: string;
  /** Probability distribution for each class */
  probabilities: number[];
  /** Confidence score (0-1) */
  confidence: number;
  /** Whether prediction matches expected */
  is_correct: boolean;
}

/**
 * Represents a single layer in a neural network.
 */
export type NeuralNetworkLayer = {
  /** Weight matrix for this layer */
  weights: number[];
  /** Bias vector for this layer */
  biases: number[];
  /** Activation function name (e.g., "relu", "sigmoid") */
  activation: string;
  /** Layer dimensions [input_size, output_size] */
  shape: number[];
};

/**
 * Collection of all layers in a neural network.
 */
export type NeuralNetworkLayers = {
  /** Array of network layers from input to output */
  layers: NeuralNetworkLayer[];
};

/**
 * Activation data for all layers during a forward pass.
 * @typeParam TIn - Type of input values
 * @typeParam TOut - Type of output values
 */
export type Activation<TIn = number, TOut = number> = {
  /** Input values fed to the network */
  inputs: TIn[];
  /** Activation data for each layer */
  layers: {
    /** Values before activation function */
    pre_activation: number[];
    /** Values after activation function */
    activation: number[];
    /** Name of the activation function */
    function: string;
  }[];
  /** Final network output */
  output: TOut[];
};

/**
 * Prediction result from any classifier (unified).
 */
export interface PredictionResult {
  /** Predicted class name */
  class_name: string;
  /** Predicted class index */
  class_index: number;
  /** Probability distribution for each class */
  probabilities: number[];
  /** Confidence score (0-1) */
  confidence: number;
}

/**
 * @deprecated Use PredictionResult instead
 */
export type IrisPrediction = PredictionResult;

/**
 * @deprecated Use TestResult instead
 */
export type IrisTestResult = TestResult;

/**
 * Layer summary for architecture display.
 */
export interface LayerSummary {
  /** Layer name (e.g., "Conv2D", "FC1", "ReLU") */
  name: string;
  /** Layer configuration (e.g., "1→32, 3x3, pad=1") */
  config: string;
}

/**
 * Architecture summary for any model (FC, CNN, ResNet, etc.)
 * Unified format returned by get_architecture() from all WASM modules.
 */
export interface ArchitectureSummary {
  /** Model name (e.g., "LeNet-5", "VGG-Tiny", "XOR Network") */
  name: string;
  /** Model type: "fc", "cnn", or "resnet" */
  model_type: 'fc' | 'cnn' | 'resnet';
  /** Input shape dimensions (e.g., [1, 1, 28, 28] for CNN, [784] for FC) */
  input_shape: number[];
  /** Number of output features */
  output_features: number;
  /** Total number of trainable parameters */
  num_parameters: number;
  /** Layer descriptions */
  layers: LayerSummary[];
}

/**
 * One CNN layer's intermediate activation output.
 * Returned by get_cnn_activations() from CNN WASM modules.
 */
export interface CnnLayerActivation {
  /** Layer type: "Conv2D", "ReLU", "MaxPool2D", "BatchNorm2D", "Flatten", etc. */
  layer_type: string;
  /** Human-readable config: "1→32, 3×3, s=1, p=1" */
  config: string;
  /** Output shape [channels, height, width] */
  shape: number[];
  /** Flattened activation data (C×H×W values for a single sample) */
  activations: number[];
}

/**
 * Full CNN forward-pass result with all intermediate activations.
 * Returned by get_cnn_activations() from CNN WASM modules.
 * Non-CNN modules return `{ error: string }` instead.
 */
export interface CnnActivationsResponse {
  /** Input shape [channels, height, width] */
  input_shape: number[];
  /** Per-layer intermediate activations */
  layers: CnnLayerActivation[];
  /** Output shape of the last CNN layer */
  output_shape: number[];
}
