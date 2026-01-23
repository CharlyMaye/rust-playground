import { computed, effect, Injectable, resource, ResourceLoaderParams, ResourceRef, signal } from '@angular/core';
import init, { InitOutput as InitIraisOutput, IrisClassifier} from '@cma/wasm/iris_wasm/neural_wasm_iris.js';
import { ModelInfo } from './model-info';

@Injectable({
  providedIn: 'root',
})
export class IrisWasmService {
  protected readonly _wasPath = signal('/wasm/iris_wasm/neural_wasm_iris_bg.wasm');
  public readonly wasmResource: ResourceRef<InitIraisOutput | undefined> = resource({
    params: this._wasPath,
    loader: (param: ResourceLoaderParams<string>) =>  init(param.params),
    defaultValue: undefined,
  });

  public readonly network = computed(() => {
    const initOutput = this.wasmResource.value();
    if (!initOutput) {
      return undefined;
    }
    console.log('Iris Classifier model wasm output:', initOutput);
    return new IrisClassifier();
  });
  public readonly modelInfo = computed(() => {
    const irisClassifier = this.network();
    if (!irisClassifier) {
      return undefined;
    }
    const modelInfoJson: string = irisClassifier.model_info();
    const modelInfo: ModelInfo = JSON.parse(modelInfoJson);
    console.log('Iris Classifier model info:', modelInfo);
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
 
}
