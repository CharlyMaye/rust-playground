import { Injectable } from '@angular/core';
import init, {
  InitOutput as InitMNISTResNetOutput,
  MnistResNetNetwork,
} from '@cma/wasm/mnist_resnet_wasm/neural_wasm_mnist_resnet.js';
import { BaseWasmService } from './base-wasm.service';

/**
 * Service for loading and interacting with the ResNet-Micro CNN WASM neural network.
 */
@Injectable({ providedIn: 'root' })
export class MNISTResNetWasmService extends BaseWasmService<
  MnistResNetNetwork,
  InitMNISTResNetOutput
> {
  protected override wasmFilePath(base: string): string {
    return `${base}wasm/mnist_resnet_wasm/neural_wasm_mnist_resnet_bg.wasm`;
  }

  protected override loadModule(path: string): Promise<InitMNISTResNetOutput> {
    return init({ module_or_path: path });
  }

  protected override createNetwork(): MnistResNetNetwork {
    return new MnistResNetNetwork();
  }
}
