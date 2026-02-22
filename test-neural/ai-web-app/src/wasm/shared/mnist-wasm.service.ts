import { Injectable } from '@angular/core';
import init, {
  InitOutput as InitMNISTOutput,
  MnistNetwork,
} from '@cma/wasm/mnist_wasm/neural_wasm_mnist.js';
import { BaseWasmService } from './base-wasm.service';

/**
 * Service for loading and interacting with the MNIST WASM neural network.
 */
@Injectable({ providedIn: 'root' })
export class MNISTWasmService extends BaseWasmService<MnistNetwork, InitMNISTOutput> {
  protected override wasmFilePath(base: string): string {
    return `${base}wasm/mnist_wasm/neural_wasm_mnist_bg.wasm`;
  }

  protected override loadModule(path: string): Promise<InitMNISTOutput> {
    return init({ module_or_path: path });
  }

  protected override createNetwork(): MnistNetwork {
    return new MnistNetwork();
  }
}
