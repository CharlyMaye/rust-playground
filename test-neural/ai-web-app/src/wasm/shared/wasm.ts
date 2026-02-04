import { inject, Injectable, ResourceRef } from '@angular/core';
import { InitOutput as InitIraisOutput } from '@cma/wasm/iris_wasm/neural_wasm_iris.js';
import { InitOutput as InitMNISTLeNetOutput } from '@cma/wasm/mnist_lenet_wasm/neural_wasm_mnist_lenet.js';
import { InitOutput as InitMNISTResNetOutput } from '@cma/wasm/mnist_resnet_wasm/neural_wasm_mnist_resnet.js';
import { InitOutput as InitMNISTOutput } from '@cma/wasm/mnist_wasm/neural_wasm_mnist.js';
import { InitOutput as InitXorOutput } from '@cma/wasm/xor_wasm/neural_wasm_xor.js';
import { IrisWasmService } from './iris-wasm.service';
import { MNISTLeNetWasmService } from './mnist-lenet-wasm.service';
import { MNISTResNetWasmService } from './mnist-resnet-wasm.service';
import { MNISTWasmService } from './mnist-wasm.service';
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
  private readonly _mnist = inject(MNISTWasmService);
  private readonly _mnistLeNet = inject(MNISTLeNetWasmService);
  private readonly _mnistResNet = inject(MNISTResNetWasmService);

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

  /** Resource for MNIST WASM module initialization */
  public readonly mnistWasmResource: ResourceRef<InitMNISTOutput | undefined> =
    this._mnist.wasmResource;
  /** MNIST classifier network instance */
  public readonly mnistNetwork = this._mnist.network;
  /** MNIST model metadata */
  public readonly mnistModelInfo = this._mnist.modelInfo;
  /** MNIST network architecture as layer sizes */
  public readonly mnistArchitecture = this._mnist.architecture;
  /** MNIST network weights and biases */
  public readonly mnistWeights = this._mnist.weights;
  /** MNIST test results for validation samples */
  public readonly mnistTestAll = this._mnist.testAll;

  // ====== LeNet-5 CNN for MNIST ======
  /** Resource for MNIST LeNet-5 WASM module initialization */
  public readonly mnistLeNetWasmResource: ResourceRef<InitMNISTLeNetOutput | undefined> =
    this._mnistLeNet.wasmResource;
  /** LeNet-5 CNN network instance */
  public readonly mnistLeNetNetwork = this._mnistLeNet.network;
  /** LeNet-5 model metadata */
  public readonly mnistLeNetModelInfo = this._mnistLeNet.modelInfo;
  /** LeNet-5 CNN architecture summary */
  public readonly mnistLeNetCnnSummary = this._mnistLeNet.cnnSummary;
  /** LeNet-5 FC architecture as layer sizes */
  public readonly mnistLeNetArchitecture = this._mnistLeNet.architecture;
  /** LeNet-5 FC network weights and biases */
  public readonly mnistLeNetWeights = this._mnistLeNet.weights;
  /** LeNet-5 test results for validation samples */
  public readonly mnistLeNetTestAll = this._mnistLeNet.testAll;

  // ====== ResNet-Micro CNN for MNIST ======
  /** Resource for MNIST ResNet-Micro WASM module initialization */
  public readonly mnistResNetWasmResource: ResourceRef<InitMNISTResNetOutput | undefined> =
    this._mnistResNet.wasmResource;
  /** ResNet-Micro CNN network instance */
  public readonly mnistResNetNetwork = this._mnistResNet.network;
  /** ResNet-Micro model metadata */
  public readonly mnistResNetModelInfo = this._mnistResNet.modelInfo;
  /** ResNet-Micro CNN architecture summary */
  public readonly mnistResNetCnnSummary = this._mnistResNet.cnnSummary;
  /** ResNet-Micro FC architecture as layer sizes */
  public readonly mnistResNetArchitecture = this._mnistResNet.architecture;
  /** ResNet-Micro FC network weights and biases */
  public readonly mnistResNetWeights = this._mnistResNet.weights;
  /** ResNet-Micro test results for validation samples */
  public readonly mnistResNetTestAll = this._mnistResNet.testAll;
}
