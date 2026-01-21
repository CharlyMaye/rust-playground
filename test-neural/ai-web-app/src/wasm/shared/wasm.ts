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

  public readonly xorWasmResource: ResourceRef<InitXorOutput | undefined> = this._xor.xorWasmResource;
  public readonly xorModelInfo = this._xor.xorModelInfo;

  public readonly irisWasmResource: ResourceRef<InitIraisOutput | undefined> = this._iris.irisWasmResource;
  public readonly irisModelInfo = this._iris.irisModelInfo;
  
}
