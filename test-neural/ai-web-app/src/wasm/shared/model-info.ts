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
