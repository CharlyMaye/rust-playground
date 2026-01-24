export interface ModelInfo {
  name: string;
  architecture: string;
  accuracy: number;
  description: string;
  test_samples: number;
  trained_at: Date;
}

export interface XORTestResult {
  a: number;
  b: number;
  expected: number;
  prediction: number;
  raw: number;
  confidence: number;
}

export type NeuralNetworkLayer = {
  weights: number[];
  biases: number[];
  activation: string;
  shape: number[];
};
export type NeuralNetworkLayers = {
  layers: NeuralNetworkLayer[];
};

export type Activation<TIn = number, TOut = number> = {
  inputs: TIn[];
  layers: {
    pre_activation: number[];
    activation: number[];
    function: string;
  }[];
  output: TOut[];
};

export type XorPrediction = {
  confidence: number;
  prediction: number;
  probabilities: [number, number];
  raw: number;
};

export type IrisPrediction = {
  class: string;
  class_idx: number;
  probabilities: [number, number, number];
  confidence: number;
};

export interface IrisTestResult {
  sepal_length: number;
  sepal_width: number;
  petal_length: number;
  petal_width: number;
  expected: string;
  predicted: string;
  is_correct: boolean;
  correct: boolean; // Alias pour compatibilité
  confidence: number;
  probabilities: [number, number, number];
}
