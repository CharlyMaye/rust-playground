import { computed, effect, Injectable, resource, ResourceLoaderParams, ResourceRef, signal } from '@angular/core';
import init, { InitOutput as InitXorOutput, XorNetwork} from '@cma/wasm/xor_wasm/neural_wasm_xor.js';
import { ModelInfo } from './model-info';

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

  public readonly modelInfo = computed(() => {
    const initOutput = this.wasmResource.value();
    if (!initOutput) {
      return undefined;
    }
    console.log('XOR Network model wasm output:', initOutput);
    const  xorNetwork = new XorNetwork();
    const modelInfoJson: string = xorNetwork.model_info();
    const modelInfo: ModelInfo = JSON.parse(modelInfoJson);
    console.log('XOR Network model info:', modelInfo);
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

  constructor() {
    effect(() => {
      console.log('XOR Network architecture changed:', this.architecture());
    });
  }
 
}
