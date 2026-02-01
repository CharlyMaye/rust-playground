/**
 * Example usage of NetworkVisualization component
 * This file demonstrates how to use the new visualization system.
 *
 * You can copy this code to your actual components (mnist-digit, xor, iris, etc.)
 */

import { Component, computed, signal } from '@angular/core';
import { ConfigurableNetworkVisualization } from './configurable-network-visualization';
import { LayerWeights, NetworkArchitecture } from './renderers';

/**
 * Example: Simple XOR Network
 */
@Component({
  selector: 'app-example-xor',
  imports: [ConfigurableNetworkVisualization],
  template: `
    <app-configurable-network
      [architecture]="architecture()"
      [weights]="weights()"
      preset="small-network"
    />
  `,
})
export class ExampleXOR {
  // XOR network: 2 inputs → 2 hidden (ReLU) → 1 output (Sigmoid)
  public readonly architecture = signal<NetworkArchitecture>({
    inputs: [1.0, 0.0], // Input: [A=1, B=0]
    layers: [
      {
        size: 2,
        activations: [0.89, 0.12],
        activationFunction: 'ReLU',
        isOutput: false,
      },
      {
        size: 1,
        activations: [0.91], // Output: should be ~1 (true XOR)
        activationFunction: 'Sigmoid',
        isOutput: true,
      },
    ],
  });

  public readonly weights = signal<LayerWeights[]>([
    // Weights from input to hidden layer
    {
      weights: [
        [0.8, -0.5], // Hidden neuron 1
        [-0.6, 0.9], // Hidden neuron 2
      ],
    },
    // Weights from hidden to output layer
    {
      weights: [
        [1.2, 0.8], // Output neuron
      ],
    },
  ]);
}

/**
 * Example: MNIST Network (Large)
 */
@Component({
  selector: 'app-example-mnist',
  imports: [ConfigurableNetworkVisualization],
  template: `
    <app-configurable-network
      [architecture]="architecture()"
      [weights]="weights()"
      preset="mnist"
    />
  `,
})
export class ExampleMNIST {
  // MNIST: 784 inputs → 128 hidden → 64 hidden → 10 output
  public readonly architecture = computed<NetworkArchitecture>(() => {
    // Generate dummy data for 784 inputs (28x28 pixels)
    const inputs = Array(784)
      .fill(0)
      .map(() => Math.random() * 0.5);

    return {
      inputs,
      layers: [
        {
          size: 128,
          activations: Array(128)
            .fill(0)
            .map(() => Math.random()),
          activationFunction: 'ReLU',
          isOutput: false,
        },
        {
          size: 64,
          activations: Array(64)
            .fill(0)
            .map(() => Math.random()),
          activationFunction: 'ReLU',
          isOutput: false,
        },
        {
          size: 10,
          activations: [0.01, 0.02, 0.03, 0.92, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0],
          activationFunction: 'Softmax',
          isOutput: true,
        },
      ],
    };
  });

  public readonly weights = computed<LayerWeights[]>(() => {
    // Generate dummy weights (in real app, these come from WASM)
    return [
      // Input → Hidden1: 784 × 128 = 100,352 weights
      {
        weights: Array(128)
          .fill(0)
          .map(() =>
            Array(784)
              .fill(0)
              .map(() => (Math.random() - 0.5) * 2),
          ),
      },
      // Hidden1 → Hidden2: 128 × 64 = 8,192 weights
      {
        weights: Array(64)
          .fill(0)
          .map(() =>
            Array(128)
              .fill(0)
              .map(() => (Math.random() - 0.5) * 2),
          ),
      },
      // Hidden2 → Output: 64 × 10 = 640 weights
      {
        weights: Array(10)
          .fill(0)
          .map(() =>
            Array(64)
              .fill(0)
              .map(() => (Math.random() - 0.5) * 2),
          ),
      },
    ];
  });
}

/**
 * Example: Integration with WASM (Real Use Case)
 */
@Component({
  selector: 'app-example-wasm-integration',
  imports: [ConfigurableNetworkVisualization],
  template: `
    <app-configurable-network
      [architecture]="networkArchitecture()"
      [weights]="networkWeights()"
      [autoConfig]="true"
    />
  `,
})
export class ExampleWasmIntegration {
  // Import the adapter functions
  // import { activationToArchitecture, neuralNetworkLayersToWeights } from './adapter';

  // Assuming you have these from your WASM service
  // public readonly activations = computed(() => { ... });
  // public readonly weights = this.wasmService.mnistWeights;

  // Convert to new format using adapters
  public readonly networkArchitecture = computed(() => {
    // const acts = this.activations();
    // if (!acts) return null;
    // return activationToArchitecture(acts);

    // Dummy implementation for example
    return {
      inputs: [0.5, 0.8],
      layers: [
        { size: 2, activations: [0.6, 0.3], activationFunction: 'ReLU', isOutput: false },
        { size: 1, activations: [0.7], activationFunction: 'Sigmoid', isOutput: true },
      ],
    };
  });

  public readonly networkWeights = computed(() => {
    // const wts = this.weights();
    // if (!wts) return null;
    // return neuralNetworkLayersToWeights(wts);

    // Dummy implementation for example
    return [
      {
        weights: [
          [0.5, -0.3],
          [0.2, 0.9],
        ],
      },
      { weights: [[0.6, 0.1]] },
    ];
  });
}

/**
 * Example: Dynamic Updates (Interactive)
 */
@Component({
  selector: 'app-example-interactive',
  imports: [ConfigurableNetworkVisualization],
  template: `
    <div>
      <button (click)="updateInputs()">Update Inputs</button>
      <app-configurable-network
        [architecture]="architecture()"
        [weights]="weights()"
        preset="interactive"
      />
    </div>
  `,
})
export class ExampleInteractive {
  private inputValues = signal<number[]>([0.5, 0.5]);

  public readonly architecture = computed<NetworkArchitecture>(() => ({
    inputs: this.inputValues(),
    layers: [
      {
        size: 2,
        activations: this.inputValues().map((v) => Math.tanh(v)),
        activationFunction: 'Tanh',
        isOutput: false,
      },
      {
        size: 1,
        activations: [this.inputValues().reduce((a, b) => a + b, 0) / 2],
        activationFunction: 'Sigmoid',
        isOutput: true,
      },
    ],
  }));

  public readonly weights = signal<LayerWeights[]>([
    {
      weights: [
        [0.8, -0.5],
        [-0.6, 0.9],
      ],
    },
    { weights: [[1.0, 0.5]] },
  ]);

  public updateInputs(): void {
    // Generate random inputs
    this.inputValues.set([Math.random(), Math.random()]);
    // The visualization will automatically update thanks to signals!
  }
}
