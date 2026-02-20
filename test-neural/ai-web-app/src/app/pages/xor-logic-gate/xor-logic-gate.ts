import { DecimalPipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { Activation, PredictionResult, WasmFacade } from '@cma/wasm/shared';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';
import {
  activationToArchitecture,
  neuralNetworkLayersToWeights,
} from '../../ui/network-visualization/adapter';
import { ConfigurableNetworkVisualization } from '../../ui/network-visualization/configurable-network-visualization';

/**
 * Interactive XOR logic gate demo page.
 * Demonstrates a neural network trained to compute the XOR function.
 */
@Component({
  selector: 'app-xor-logic-gate',
  imports: [DecimalPipe, Loader, ModelInfoComponent, ConfigurableNetworkVisualization],
  templateUrl: './xor-logic-gate.html',
  styleUrl: './xor-logic-gate.scss',
  host: { class: 'page container' },
})
export class XorLogicGate {
  private readonly wasmService = inject(WasmFacade);

  /** Whether the WASM module is currently loading */
  public readonly xorIsLoading = this.wasmService.xorWasmResource.isLoading;
  /** XOR network instance */
  public readonly xorNetwork = this.wasmService.xorNetwork;
  /** Model metadata */
  public readonly xorModelInfo = this.wasmService.xorModelInfo;
  /** Network architecture */
  public readonly xorArchitecture = this.wasmService.xorArchitecture;
  /** Network weights */
  public readonly xorWeights = this.wasmService.xorWeights;
  /** Test results for all XOR combinations */
  public readonly xorTestAll = this.wasmService.xorTestAll;

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
    const inputA = this.inputA();
    const inputB = this.inputB();
    const prediction = network.predict(inputA, inputB);
    const output = JSON.parse(prediction) as PredictionResult;
    return output;
  });

  /** Layer activations for the current input */
  public readonly activations = computed(() => {
    const network = this.xorNetwork();
    if (!network) {
      return null;
    }
    const inputA = this.inputA();
    const inputB = this.inputB();
    const acts = JSON.parse(network.get_activations(inputA, inputB)) as Activation<number, number>;
    return acts;
  });

  /** Network architecture for visualization (converted from activations) */
  public readonly networkArchitecture = computed(() => {
    const acts = this.activations();
    if (!acts) return null;
    return activationToArchitecture(acts);
  });

  /** Network weights for visualization (converted from WASM weights) */
  public readonly networkWeights = computed(() => {
    const wts = this.xorWeights();
    if (!wts) return null;
    return neuralNetworkLayersToWeights(wts);
  });

  /** Activation functions used in the network for display */
  public readonly activationFunctions = computed(() => {
    const wts = this.xorWeights();
    if (!wts?.layers) return undefined;
    const activations = wts.layers.map((l) => this.capitalizeFirst(l.activation));
    const unique = [...new Set(activations)];
    return unique.join(' → ');
  });

  private capitalizeFirst(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  /** Formatted prediction value for display */
  public readonly predictionDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    return output.class_index;
  });

  /** Formatted confidence value for display */
  public readonly confidenceDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    return (output.confidence * 100).toFixed(1) + '% confidence';
  });

  /**
   * Toggles the specified input between 0 and 1.
   * @param type - Which input to toggle ('A' or 'B')
   */
  public toggleInput(type: 'A' | 'B'): void {
    if (type === 'A') {
      this.inputA.set(this.inputA() === 0 ? 1 : 0);
    } else {
      this.inputB.set(this.inputB() === 0 ? 1 : 0);
    }
  }
}
