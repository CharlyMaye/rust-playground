import { DecimalPipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { WasmFacade } from '@cma/wasm/shared/wasm';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';

import { NeuralNetworkModelVizualizer } from '../../ui/neural-network-model-vizualizer/neural-network-model-vizualizer';

type NetworkPrediction = {
  confidence: number;
  prediction: number;
  probabilities: [number, number];
  raw: number;
};

@Component({
  selector: 'app-xor-logic-gate',
  imports: [DecimalPipe, Loader, ModelInfoComponent, NeuralNetworkModelVizualizer],
  templateUrl: './xor-logic-gate.html',
  styleUrl: './xor-logic-gate.scss',
  host: { class: 'container' },
})
export class XorLogicGate {
  private readonly wasmService = inject(WasmFacade);
  public readonly xorIsLoading = this.wasmService.xorWasmResource.isLoading;
  public readonly xorNetwork = this.wasmService.xorNetwork;
  public readonly xorModelInfo = this.wasmService.xorModelInfo;
  public readonly xorArchitecture = this.wasmService.xorArchitecture;
  public readonly xorWeights = this.wasmService.xorWeights;
  public readonly xorTestAll = this.wasmService.xorTestAll;

  public readonly inputA = signal(0);
  public readonly inputB = signal(0);
  public readonly output = computed(() => {
    const network = this.xorNetwork();
    if (!network) {
      return null;
    }
    const inputA = this.inputA();
    const inputB = this.inputB();
    const prediction = network.predict(inputA, inputB);
    const output = JSON.parse(prediction) as NetworkPrediction;
    console.log('XOR Prediction:', output);
    return output;
  });
  public readonly activations = computed(() => {
    const network = this.xorNetwork();
    if (!network) {
      return null;
    }
    const inputA = this.inputA();
    const inputB = this.inputB();
    const acts = JSON.parse(network.get_activations(inputA, inputB));
    return acts;
  });
  public readonly predictionDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    return output.prediction;
  });
  public readonly confidenceDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    return (output.confidence * 100).toFixed(1) + '% confidence';
  });

  public toggleInput(type: 'A' | 'B'): void {
    if (type === 'A') {
      this.inputA.set(this.inputA() === 0 ? 1 : 0);
    } else {
      this.inputB.set(this.inputB() === 0 ? 1 : 0);
    }
    console.log(this.xorWeights());
  }
}
