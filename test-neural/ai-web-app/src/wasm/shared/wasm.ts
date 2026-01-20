import { Injectable, resource, ResourceRef } from '@angular/core';
import init, { InitOutput as InitXorOutput} from '@cma/wasm/xor_wasm/neural_wasm_xor.js';
import initIris, { InitOutput as InitIraisOutput} from '@cma/wasm/iris_wasm/neural_wasm_iris.js';

@Injectable({
  providedIn: 'root',
})
export class WasmService {

  public readonly xorWasmResource: ResourceRef<InitXorOutput | undefined> = resource({
    loader: async () => {
      const wasmlPath = '/wasm/xor_wasm/neural_wasm_xor_bg.wasm';
      const fetchResponse = await fetch(wasmlPath);
      console.log('status:', fetchResponse.status);
      const initResponse = await init(wasmlPath);
      return initResponse;            
    }
  });
  public readonly irisWasmResource: ResourceRef<InitIraisOutput | undefined> = resource({
    loader: async () => {
      const wasmlPath = '/wasm/iris_wasm/neural_wasm_iris_bg.wasm';
      const fetchResponse = await fetch(wasmlPath);
      console.log('status:', fetchResponse.status);
      const initResponse = await initIris(wasmlPath);
      return initResponse;            
    }
  });
  
}
