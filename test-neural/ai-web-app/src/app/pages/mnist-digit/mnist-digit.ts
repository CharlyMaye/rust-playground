import { DecimalPipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { WasmFacade } from '@cma/wasm/shared';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';
import { NeuralNetworkModelVizualizer } from '../../ui/neural-network-model-vizualizer/neural-network-model-vizualizer';

/**
 * MNIST digit classifier demo page.
 * Placeholder for future handwritten digit recognition feature.
 */
@Component({
  selector: 'app-mnist-digit',
  imports: [DecimalPipe, Loader, ModelInfoComponent, NeuralNetworkModelVizualizer],
  templateUrl: './mnist-digit.html',
  styleUrl: './mnist-digit.scss',
  host: { class: 'page container' },
})
export class MnistDigit {
  private readonly wasmService = inject(WasmFacade);

  /** Whether the WASM module is currently loading */
  public readonly xorIsLoading = this.wasmService.mnistWasmResource.isLoading;
  /** XOR network instance */
  public readonly xorNetwork = this.wasmService.mnistNetwork;
  /** Model metadata */
  public readonly xorModelInfo = this.wasmService.mnistModelInfo;
  /** Network architecture */
  public readonly xorArchitecture = this.wasmService.mnistArchitecture;
  /** Network weights */
  public readonly xorWeights = this.wasmService.mnistWeights;
  /** Test results for all XOR combinations */
  public readonly xorTestAll = this.wasmService.mnistTestAll;

  /** First input value (0 or 1) */
  public readonly inputA = signal(0);
  /** Second input value (0 or 1) */
  public readonly inputB = signal(0);

  /** Current prediction output from the network */
  public readonly output = computed(() => {
    const network = this.xorNetwork();
    if (!network) {
      return null;
    }
    // const inputA = this.inputA();
    // const inputB = this.inputB();
    // // TODO - modifier le code cote neural network
    // const prediction = network.predict(inputA, inputB);
    // const output = JSON.parse(prediction) as XorPrediction;
    // return output;
    return null;
  });

  /** Layer activations for the current input */
  public readonly activations = computed(() => {
    const network = this.xorNetwork();
    if (!network) {
      return null;
    }
    // const inputA = this.inputA();
    // const inputB = this.inputB();
    // const acts = JSON.parse(network.get_activations(inputA, inputB)) as Activation<number, number>;
    // // TODO - modifier le code cote neural network
    // acts.output = [acts.output as unknown as number];
    // return acts;
    return null;
  });

  /** Formatted prediction value for display */
  public readonly predictionDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    return 'N/A';
    // return output.prediction;
  });

  /** Formatted confidence value for display */
  public readonly confidenceDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    return 'N/A';
    // return (output.confidence * 100).toFixed(1) + '% confidence';
  });
}
