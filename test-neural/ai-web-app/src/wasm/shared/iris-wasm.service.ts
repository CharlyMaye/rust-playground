import { Injectable } from '@angular/core';
import init, {
  InitOutput as InitIrisOutput,
  IrisClassifier,
} from '@cma/wasm/iris_wasm/neural_wasm_iris.js';
import { BaseWasmService } from './base-wasm.service';

/**
 * Service for loading and interacting with the Iris WASM neural network classifier.
 */
@Injectable({ providedIn: 'root' })
export class IrisWasmService extends BaseWasmService<IrisClassifier, InitIrisOutput> {
  protected override wasmFilePath(base: string): string {
    return `${base}wasm/iris_wasm/neural_wasm_iris_bg.wasm`;
  }

  protected override loadModule(path: string): Promise<InitIrisOutput> {
    return init({ module_or_path: path });
  }

  protected override createNetwork(): IrisClassifier {
    return new IrisClassifier();
  }
}
