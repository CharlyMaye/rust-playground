import { inject, Injectable, ResourceRef } from '@angular/core';
import { InitOutput as InitXorOutput } from '@cma/wasm/xor_wasm/neural_wasm_xor.js';
import { InitOutput as InitIraisOutput } from '@cma/wasm/iris_wasm/neural_wasm_iris.js';
import { XorWasmService } from './wor-wasm.service';
import { IrisWasmService } from './iris-wasm.service';

@Injectable({
  providedIn: 'root',
})
export class WasmFacade {
  private readonly _xor = inject(XorWasmService);
  private _iris = inject(IrisWasmService);

  public readonly xorWasmResource: ResourceRef<InitXorOutput | undefined> = this._xor.wasmResource;
  public readonly xorNetwork = this._xor.network;
  public readonly xorModelInfo = this._xor.modelInfo;
  public readonly xorArchitecture = this._xor.architecture;
  public readonly xorWeights = this._xor.weights;
  public readonly xorTestAll = this._xor.testAll;

  public readonly irisWasmResource: ResourceRef<InitIraisOutput | undefined> = this._iris.wasmResource;
  public readonly irisNetwork = this._iris.network;
  public readonly irisModelInfo = this._iris.modelInfo;
  public readonly irisArchitecture = this._iris.architecture;
  public readonly irisWeights = this._iris.weights;
  public readonly irisTestAll = this._iris.testAll;
}
