import { computed, Injectable, resource, ResourceRef } from '@angular/core';
import init, { InitOutput as InitXorOutput, XorNetwork} from '@cma/wasm/xor_wasm/neural_wasm_xor.js';
import initIris, { InitOutput as InitIraisOutput, IrisClassifier} from '@cma/wasm/iris_wasm/neural_wasm_iris.js';

@Injectable({
  providedIn: 'root',
})
export class IrisWasmService {
  public readonly irisWasmResource: ResourceRef<InitIraisOutput | undefined> = resource({
    loader: async () => {
      const wasmlPath = '/wasm/iris_wasm/neural_wasm_iris_bg.wasm';
      // const fetchResponse = await fetch(wasmlPath);
      const initResponse = await initIris(wasmlPath);
      return initResponse;            
    },
    defaultValue: undefined,
  });
  public readonly irisModelInfo = computed(() => {
    const initOutput = this.irisWasmResource.value();
    if (!initOutput) {
      return;
    }
    const  irisClassifier = new IrisClassifier();

    // Get model info
    const modelInfoJson = irisClassifier.model_info();
    const modelInfo = JSON.parse(modelInfoJson);
    console.log('Iris Classifier model info:', modelInfo, initOutput);
    return modelInfo;
  });
  
}
