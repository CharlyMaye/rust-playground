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
 * Result of a single XOR test case.
 */
export interface XORTestResult {
  /** First input value (0 or 1) */
  a: number;
  /** Second input value (0 or 1) */
  b: number;
  /** Expected XOR output */
  expected: number;
  /** Model's binary prediction */
  prediction: number;
  /** Raw network output before thresholding */
  raw: number;
  /** Confidence score (0-1) */
  confidence: number;
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
 * Prediction result from the XOR network.
 */
export type XorPrediction = {
  /** Confidence score (0-1) */
  confidence: number;
  /** Binary prediction (0 or 1) */
  prediction: number;
  /** Probability distribution [P(0), P(1)] */
  probabilities: [number, number];
  /** Raw network output */
  raw: number;
};

/**
 * Prediction result from the Iris classifier.
 */
export type IrisPrediction = {
  /** Predicted class name (setosa, versicolor, virginica) */
  class: string;
  /** Predicted class index (0, 1, or 2) */
  class_idx: number;
  /** Probability distribution for each class */
  probabilities: [number, number, number];
  /** Confidence score (0-100) */
  confidence: number;
};

/**
 * Result of a single Iris test case.
 */
export interface IrisTestResult {
  /** Sepal length in cm */
  sepal_length: number;
  /** Sepal width in cm */
  sepal_width: number;
  /** Petal length in cm */
  petal_length: number;
  /** Petal width in cm */
  petal_width: number;
  /** Expected class label */
  expected: string;
  /** Predicted class label */
  predicted: string;
  /** Whether prediction matches expected (snake_case) */
  is_correct: boolean;
  /** Whether prediction matches expected (camelCase alias) */
  correct: boolean;
  /** Confidence score */
  confidence: number;
  /** Probability distribution for each class */
  probabilities: [number, number, number];
}
