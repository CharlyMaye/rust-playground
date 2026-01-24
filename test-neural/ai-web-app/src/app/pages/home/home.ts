import { DecimalPipe } from '@angular/common';
import { Component, inject } from '@angular/core';
import { RouterLinkWithHref } from '@angular/router';
import { WasmFacade } from '@cma/wasm/shared/wasm';
import { About } from '../../ui/about/about';
import { Loader } from '../../ui/loader/loader';

@Component({
  selector: 'app-home',
  imports: [DecimalPipe, About, Loader, RouterLinkWithHref],
  templateUrl: './home.html',
  styleUrl: './home.scss',
  host: { class: 'container' },
})
export class Home {
  private readonly wasmService = inject(WasmFacade);
  public readonly xorIsLoading = this.wasmService.xorWasmResource.isLoading;
  public readonly irisIsLoading = this.wasmService.irisWasmResource.isLoading;

  public readonly xorModelInfo = this.wasmService.xorModelInfo;
  public readonly xorArchitecture = this.wasmService.xorArchitecture;

  public readonly irisModelInfo = this.wasmService.irisModelInfo;
  public readonly irisArchitecture = this.wasmService.irisArchitecture;
}
