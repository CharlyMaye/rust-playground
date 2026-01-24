import { DecimalPipe } from '@angular/common';
import { Component, inject } from '@angular/core';
import { RouterLinkWithHref } from '@angular/router';
import { WasmFacade } from '@cma/wasm/shared';
import { About } from '../../ui/about/about';
import { Loader } from '../../ui/loader/loader';

/**
 * Home page displaying available neural network demos.
 * Shows cards for XOR, Iris classifier, and upcoming MNIST demos.
 */
@Component({
  selector: 'app-home',
  imports: [DecimalPipe, About, Loader, RouterLinkWithHref],
  templateUrl: './home.html',
  styleUrl: './home.scss',
  host: { class: 'container' },
})
export class Home {
  private readonly wasmService = inject(WasmFacade);

  /** Whether XOR WASM module is loading */
  public readonly xorIsLoading = this.wasmService.xorWasmResource.isLoading;
  /** Whether Iris WASM module is loading */
  public readonly irisIsLoading = this.wasmService.irisWasmResource.isLoading;

  /** XOR model metadata */
  public readonly xorModelInfo = this.wasmService.xorModelInfo;
  /** XOR network architecture */
  public readonly xorArchitecture = this.wasmService.xorArchitecture;

  /** Iris model metadata */
  public readonly irisModelInfo = this.wasmService.irisModelInfo;
  /** Iris network architecture */
  public readonly irisArchitecture = this.wasmService.irisArchitecture;
}
