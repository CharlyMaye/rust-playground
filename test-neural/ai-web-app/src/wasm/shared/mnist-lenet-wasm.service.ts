import { Injectable } from '@angular/core';
import init, {
  InitOutput as InitMNISTLeNetOutput,
  MnistLeNetNetwork,
} from '@cma/wasm/mnist_lenet_wasm/neural_wasm_mnist_lenet.js';
import { BaseWasmService } from './base-wasm.service';

/**
 * Service for loading and interacting with the LeNet-5 CNN WASM neural network.
 */
@Injectable({ providedIn: 'root' })
export class MNISTLeNetWasmService extends BaseWasmService<
  MnistLeNetNetwork,
  InitMNISTLeNetOutput
> {
  protected override wasmFilePath(base: string): string {
    return `${base}wasm/mnist_lenet_wasm/neural_wasm_mnist_lenet_bg.wasm`;
  }

  protected override loadModule(path: string): Promise<InitMNISTLeNetOutput> {
    return init({ module_or_path: path });
  }

  protected override createNetwork(): MnistLeNetNetwork {
    return new MnistLeNetNetwork();
  }
}
