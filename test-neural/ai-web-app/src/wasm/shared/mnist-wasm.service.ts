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
  InitOutput as InitMNISTOutput,
  MnistNetwork,
} from '@cma/wasm/mnist_wasm/neural_wasm_mnist.js';
import { ArchitectureSummary, ModelInfo, NeuralNetworkLayers, TestResult } from './model-info';

/**
 * Service for loading and interacting with the MNIST WASM neural network.
 * Handles WASM module initialization and provides reactive access to network data.
 */
@Injectable({
  providedIn: 'root',
})
export class MNISTWasmService {
  private readonly _document = inject(DOCUMENT);
  private readonly _platformId = inject(PLATFORM_ID);

  protected readonly _wasPath: WritableSignal<string> = signal('');

  constructor() {
    const base = this.computeWasmBase();
    this._wasPath.set(`${base}wasm/mnist_wasm/neural_wasm_mnist_bg.wasm`);
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
  public readonly wasmResource: ResourceRef<InitMNISTOutput | undefined> = resource({
    params: this._wasPath,
    loader: (param: ResourceLoaderParams<string>) => init(param.params),
    defaultValue: undefined,
  });

  /** MNIST network instance, available after WASM initialization */
  public readonly network = computed(() => {
    const initOutput = this.wasmResource.value();
    if (!initOutput) {
      return undefined;
    }
    return new MnistNetwork();
  });

  /** Model metadata including name, accuracy, and description */
  public readonly modelInfo = computed(() => {
    const mnistNetwork = this.network();
    if (!mnistNetwork) {
      return undefined;
    }
    const modelInfoJson: string = mnistNetwork.model_info();
    const modelInfo: ModelInfo = JSON.parse(modelInfoJson);
    return modelInfo;
  });

  /** Architecture summary (unified format for all models) */
  public readonly architectureSummary = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const json: string = network.get_architecture();
    return JSON.parse(json) as ArchitectureSummary;
  });

  /** Network architecture as an array of layer sizes */
  public readonly architecture = computed(() => {
    const modelInfo = this.modelInfo();
    if (!modelInfo) {
      return undefined;
    }
    return modelInfo.architecture.split('→').map((layer) => {
      const trimmedLayer = layer.trim();
      if (trimmedLayer.startsWith('[') && trimmedLayer.endsWith(']')) {
        return trimmedLayer
          .slice(1, -1)
          .split(',')
          .map((numStr) => Number(numStr.trim()));
      }
      return Number(trimmedLayer);
    });
  });

  /** Network weights and biases for all layers */
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
    const mnistNetwork = this.network();
    if (!mnistNetwork) {
      return undefined;
    }
    const testResultsJson: string = mnistNetwork.test_all();
    const testResults = JSON.parse(testResultsJson) as TestResult[];
    return testResults;
  });
}
