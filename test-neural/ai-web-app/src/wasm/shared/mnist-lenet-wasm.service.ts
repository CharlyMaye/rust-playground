import { DOCUMENT, isPlatformBrowser } from '@angular/common';
import {
  computed,
  inject,
  Injectable,
  PLATFORM_ID,
  resource,
  ResourceLoaderParams,
  ResourceRef,
  Signal,
  signal,
  WritableSignal,
} from '@angular/core';
import init, {
  InitOutput as InitMNISTLeNetOutput,
  MnistLeNetNetwork,
} from '@cma/wasm/mnist_lenet_wasm/neural_wasm_mnist_lenet.js';
import { ModelInfo, NeuralNetworkLayers, TestResult } from './model-info';

/**
 * CNN Architecture info for visualization
 */
export interface CNNArchitectureInfo {
  cnn_summary: string;
  fc_architecture: number[];
}

/**
 * Service for loading and interacting with the LeNet-5 CNN WASM neural network.
 * Handles WASM module initialization and provides reactive access to network data.
 */
@Injectable({
  providedIn: 'root',
})
export class MNISTLeNetWasmService {
  private readonly _document = inject(DOCUMENT);
  private readonly _platformId = inject(PLATFORM_ID);

  protected readonly _wasmPath: WritableSignal<string> = signal('');

  constructor() {
    const base = this.computeWasmBase();
    this._wasmPath.set(`${base}wasm/mnist_lenet_wasm/neural_wasm_mnist_lenet_bg.wasm`);
  }

  private computeWasmBase(): string {
    if (!isPlatformBrowser(this._platformId)) {
      return '/';
    }
    const b = this._document.querySelector('base')?.getAttribute('href') ?? '/';
    if (b === './') return './';
    return b.endsWith('/') ? b : b + '/';
  }

  /** Resource managing WASM module loading state */
  public readonly wasmResource: ResourceRef<InitMNISTLeNetOutput | undefined> = resource({
    params: this._wasmPath,
    loader: (param: ResourceLoaderParams<string>) => init(param.params),
    defaultValue: undefined,
  });

  /** LeNet-5 network instance, available after WASM initialization */
  public readonly network = computed(() => {
    const initOutput = this.wasmResource.value();
    if (!initOutput) {
      return undefined;
    }
    return new MnistLeNetNetwork();
  });

  /** Model metadata including name, accuracy, and description */
  public readonly modelInfo = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const modelInfoJson: string = network.model_info();
    const modelInfo: ModelInfo = JSON.parse(modelInfoJson);
    return modelInfo;
  });

  /** CNN architecture summary */
  public readonly cnnSummary = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    return network.get_cnn_summary();
  });

  /** FC classifier architecture as an array of layer sizes */
  public readonly architecture = computed(() => {
    const modelInfo = this.modelInfo();
    if (!modelInfo) {
      return undefined;
    }
    // For CNN, architecture format is "CNN(LeNet-5) → 120 → 84 → 10"
    return modelInfo.architecture.split('→').map((layer) => {
      const trimmedLayer = layer.trim();
      if (trimmedLayer.startsWith('[') && trimmedLayer.endsWith(']')) {
        return trimmedLayer
          .slice(1, -1)
          .split(',')
          .map((numStr) => Number(numStr.trim()));
      }
      const num = Number(trimmedLayer);
      return isNaN(num) ? trimmedLayer : num;
    });
  });

  /** Network weights and biases for FC classifier layers */
  public readonly weights: Signal<NeuralNetworkLayers | undefined> = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const weightsJson: string = network.get_weights();
    const weights = JSON.parse(weightsJson) as NeuralNetworkLayers;
    return weights;
  });

  /** Test results for MNIST samples */
  public readonly testAll: Signal<TestResult[] | undefined> = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const testResultsJson: string = network.test_all();
    const testResults = JSON.parse(testResultsJson) as TestResult[];
    return testResults;
  });
}
