import { DecimalPipe } from '@angular/common';
import { Component, inject } from '@angular/core';
import { RouterLinkWithHref } from '@angular/router';
import { WasmFacade } from '@cma/wasm/shared';
import { About } from '../../ui/about/about';
import { Loader } from '../../ui/loader/loader';

/**
 * Home page displaying available neural network demos.
 * Shows cards for XOR, Iris classifier, and MNIST demos.
 */
@Component({
  selector: 'app-home',
  imports: [DecimalPipe, About, Loader, RouterLinkWithHref],
  templateUrl: './home.html',
  styleUrl: './home.scss',
  host: { class: 'page container' },
})
export class Home {
  private readonly wasmService = inject(WasmFacade);

  /** Whether XOR WASM module is loading */
  public readonly xorIsLoading = this.wasmService.xorWasmResource.isLoading;
  /** Whether Iris WASM module is loading */
  public readonly irisIsLoading = this.wasmService.irisWasmResource.isLoading;
  /** Whether MNIST WASM module is loading */
  public readonly mnistIsLoading = this.wasmService.mnistWasmResource.isLoading;
  /** Whether MNIST LeNet-5 WASM module is loading */
  public readonly mnistLeNetIsLoading = this.wasmService.mnistLeNetWasmResource.isLoading;
  /** Whether MNIST ResNet WASM module is loading */
  public readonly mnistResNetIsLoading = this.wasmService.mnistResNetWasmResource.isLoading;
  /** Whether MNIST AlexNet-Mini WASM module is loading */
  public readonly mnistAlexNetIsLoading = this.wasmService.mnistAlexNetWasmResource.isLoading;
  /** Whether MNIST VGG-Tiny WASM module is loading */
  public readonly mnistVggIsLoading = this.wasmService.mnistVggWasmResource.isLoading;

  /** XOR model metadata */
  public readonly xorModelInfo = this.wasmService.xorModelInfo;
  /** XOR network architecture */
  public readonly xorArchitecture = this.wasmService.xorArchitecture;

  /** Iris model metadata */
  public readonly irisModelInfo = this.wasmService.irisModelInfo;
  /** Iris network architecture */
  public readonly irisArchitecture = this.wasmService.irisArchitecture;

  /** MNIST model metadata */
  public readonly mnistModelInfo = this.wasmService.mnistModelInfo;
  /** MNIST network architecture */
  public readonly mnistArchitecture = this.wasmService.mnistArchitecture;

  /** MNIST LeNet-5 model metadata */
  public readonly mnistLeNetModelInfo = this.wasmService.mnistLeNetModelInfo;
  /** MNIST LeNet-5 network architecture */
  public readonly mnistLeNetArchitecture = this.wasmService.mnistLeNetArchitecture;

  /** MNIST ResNet model metadata */
  public readonly mnistResNetModelInfo = this.wasmService.mnistResNetModelInfo;
  /** MNIST ResNet network architecture */
  public readonly mnistResNetArchitecture = this.wasmService.mnistResNetArchitecture;

  /** MNIST AlexNet-Mini model metadata */
  public readonly mnistAlexNetModelInfo = this.wasmService.mnistAlexNetModelInfo;
  /** MNIST AlexNet-Mini network architecture */
  public readonly mnistAlexNetArchitecture = this.wasmService.mnistAlexNetArchitecture;

  /** MNIST VGG-Tiny model metadata */
  public readonly mnistVggModelInfo = this.wasmService.mnistVggModelInfo;
  /** MNIST VGG-Tiny network architecture */
  public readonly mnistVggArchitecture = this.wasmService.mnistVggArchitecture;
}
