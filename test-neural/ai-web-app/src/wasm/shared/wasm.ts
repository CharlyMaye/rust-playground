import { inject, Injectable, ResourceRef } from '@angular/core';
import { InitOutput as InitIraisOutput } from '@cma/wasm/iris_wasm/neural_wasm_iris.js';
import { InitOutput as InitXorOutput } from '@cma/wasm/xor_wasm/neural_wasm_xor.js';
import { IrisWasmService } from './iris-wasm.service';
import { XorWasmService } from './xor-wasm.service';

/**
 * Facade service providing unified access to all WASM neural network modules.
 * Acts as a single entry point for XOR and Iris classifiers.
 */
@Injectable({
  providedIn: 'root',
})
export class WasmFacade {
  private readonly _xor = inject(XorWasmService);
  private readonly _iris = inject(IrisWasmService);

  /** Resource for XOR WASM module initialization */
  public readonly xorWasmResource: ResourceRef<InitXorOutput | undefined> = this._xor.wasmResource;
  /** XOR neural network instance */
  public readonly xorNetwork = this._xor.network;
  /** XOR model metadata */
  public readonly xorModelInfo = this._xor.modelInfo;
  /** XOR network architecture as layer sizes */
  public readonly xorArchitecture = this._xor.architecture;
  /** XOR network weights and biases */
  public readonly xorWeights = this._xor.weights;
  /** XOR test results for all input combinations */
  public readonly xorTestAll = this._xor.testAll;

  /** Resource for Iris WASM module initialization */
  public readonly irisWasmResource: ResourceRef<InitIraisOutput | undefined> =
    this._iris.wasmResource;
  /** Iris classifier network instance */
  public readonly irisNetwork = this._iris.network;
  /** Iris model metadata */
  public readonly irisModelInfo = this._iris.modelInfo;
  /** Iris network architecture as layer sizes */
  public readonly irisArchitecture = this._iris.architecture;
  /** Iris network weights and biases */
  public readonly irisWeights = this._iris.weights;
  /** Iris test results for validation samples */
  public readonly irisTestAll = this._iris.testAll;
}
