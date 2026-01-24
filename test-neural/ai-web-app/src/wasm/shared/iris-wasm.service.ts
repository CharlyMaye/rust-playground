import {
  computed,
  Injectable,
  resource,
  ResourceLoaderParams,
  ResourceRef,
  Signal,
  signal,
} from '@angular/core';
import init, {
  InitOutput as InitIraisOutput,
  IrisClassifier,
} from '@cma/wasm/iris_wasm/neural_wasm_iris.js';
import { IrisTestResult, ModelInfo, NeuralNetworkLayers } from './model-info';

/**
 * Service for loading and interacting with the Iris WASM neural network classifier.
 * Handles WASM module initialization and provides reactive access to network data.
 */
@Injectable({
  providedIn: 'root',
})
export class IrisWasmService {
  protected readonly _wasPath = signal('/wasm/iris_wasm/neural_wasm_iris_bg.wasm');

  /** Resource managing WASM module loading state */
  public readonly wasmResource: ResourceRef<InitIraisOutput | undefined> = resource({
    params: this._wasPath,
    loader: (param: ResourceLoaderParams<string>) => init(param.params),
    defaultValue: undefined,
  });

  /** Iris classifier instance, available after WASM initialization */
  public readonly network = computed(() => {
    const initOutput = this.wasmResource.value();
    if (!initOutput) {
      return undefined;
    }
    return new IrisClassifier();
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

  /** Test results for validation samples */
  public readonly testAll = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const testResultsJson: string = network.test_all();
    const testResults = JSON.parse(testResultsJson) as IrisTestResult[];
    return testResults;
  });
}
