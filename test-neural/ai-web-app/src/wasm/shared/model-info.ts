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
