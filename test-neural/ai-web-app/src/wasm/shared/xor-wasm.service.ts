import { computed, effect, Injectable, resource, ResourceLoaderParams, ResourceRef, Signal, signal } from '@angular/core';
import init, { InitOutput as InitXorOutput, XorNetwork} from '@cma/wasm/xor_wasm/neural_wasm_xor.js';
import { ModelInfo, NeuralNetworkLayers, XORTestResult } from './model-info';

@Injectable({
  providedIn: 'root',
})
export class XorWasmService {
  protected readonly _wasPath = signal('/wasm/xor_wasm/neural_wasm_xor_bg.wasm');
  public readonly wasmResource: ResourceRef<InitXorOutput | undefined> = resource({
    params: this._wasPath,
    loader: (param: ResourceLoaderParams<string>) =>  init(param.params),
    defaultValue: undefined,
  });

  public readonly network = computed(() => {
    const initOutput = this.wasmResource.value();
    if (!initOutput) {
      return undefined;
    }
    return new XorNetwork();
  });

  public readonly modelInfo = computed(() => {
    const xorNetwork = this.network();
    if (!xorNetwork) {
      return undefined;
    }
    const modelInfoJson: string = xorNetwork.model_info();
    const modelInfo: ModelInfo = JSON.parse(modelInfoJson);
    return modelInfo;
  });

  public readonly architecture = computed(() => {
    const modelInfo = this.modelInfo();
    if (!modelInfo) {
      return undefined;
    }
    return modelInfo.architecture.split('→')
      .map(layer => {
        const trimmedLayer = layer.trim();
        if (trimmedLayer.startsWith('[') && trimmedLayer.endsWith(']')) {
          return trimmedLayer.slice(1, -1).split(',').map(numStr => Number(numStr.trim()));
        }
        return Number(trimmedLayer);
      });
  });
  public readonly weights: Signal<NeuralNetworkLayers | undefined> = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const weightsJson: string = network.get_weights();
    const weights = JSON.parse(weightsJson) as NeuralNetworkLayers;
    return weights;
  });

  public readonly testAll: Signal<XORTestResult[] | undefined> = computed(() => {
    const xorNetwork = this.network();
    if (!xorNetwork) {
      return undefined;
    }
    const testResultsJson: string = xorNetwork.test_all();
    const testResults = JSON.parse(testResultsJson);
    return testResults;
  });
}
