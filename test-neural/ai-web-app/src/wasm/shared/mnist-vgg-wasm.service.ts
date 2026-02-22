import { Injectable } from '@angular/core';
import init, {
  InitOutput as InitMNISTVggOutput,
  MnistVGGNetwork,
} from '@cma/wasm/mnist_vgg_wasm/neural_wasm_mnist_vgg.js';
import { BaseWasmService } from './base-wasm.service';

/**
 * Service for loading and interacting with the VGG-Tiny CNN WASM neural network.
 */
@Injectable({ providedIn: 'root' })
export class MNISTVggWasmService extends BaseWasmService<MnistVGGNetwork, InitMNISTVggOutput> {
  protected override wasmFilePath(base: string): string {
    return `${base}wasm/mnist_vgg_wasm/neural_wasm_mnist_vgg_bg.wasm`;
  }

  protected override loadModule(path: string): Promise<InitMNISTVggOutput> {
    return init({ module_or_path: path });
  }

  protected override createNetwork(): MnistVGGNetwork {
    return new MnistVGGNetwork();
  }
}
