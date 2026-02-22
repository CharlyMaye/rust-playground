import { Injectable } from '@angular/core';
import init, {
  InitOutput as InitMNISTAlexNetOutput,
  MnistAlexNetNetwork,
} from '@cma/wasm/mnist_alexnet_wasm/neural_wasm_mnist_alexnet.js';
import { BaseWasmService } from './base-wasm.service';

/**
 * Service for loading and interacting with the AlexNet-Mini CNN WASM neural network.
 */
@Injectable({ providedIn: 'root' })
export class MNISTAlexNetWasmService extends BaseWasmService<
  MnistAlexNetNetwork,
  InitMNISTAlexNetOutput
> {
  protected override wasmFilePath(base: string): string {
    return `${base}wasm/mnist_alexnet_wasm/neural_wasm_mnist_alexnet_bg.wasm`;
  }

  protected override loadModule(path: string): Promise<InitMNISTAlexNetOutput> {
    return init({ module_or_path: path });
  }

  protected override createNetwork(): MnistAlexNetNetwork {
    return new MnistAlexNetNetwork();
  }
}
