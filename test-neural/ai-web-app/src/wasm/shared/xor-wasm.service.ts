import { Injectable } from '@angular/core';
import init, {
  InitOutput as InitXorOutput,
  XorNetwork,
} from '@cma/wasm/xor_wasm/neural_wasm_xor.js';
import { BaseWasmService } from './base-wasm.service';

/**
 * Service for loading and interacting with the XOR WASM neural network.
 */
@Injectable({ providedIn: 'root' })
export class XorWasmService extends BaseWasmService<XorNetwork, InitXorOutput> {
  protected override wasmFilePath(base: string): string {
    return `${base}wasm/xor_wasm/neural_wasm_xor_bg.wasm`;
  }

  protected override loadModule(path: string): Promise<InitXorOutput> {
    return init({ module_or_path: path });
  }

  protected override createNetwork(): XorNetwork {
    return new XorNetwork();
  }
}
