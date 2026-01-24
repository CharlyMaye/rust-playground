import { DecimalPipe } from '@angular/common';
import { Component, inject, signal, Signal } from '@angular/core';
import { RouterLinkWithHref } from '@angular/router';
import { WasmFacade } from '@cma/wasm/shared/wasm';
import { About } from '../../ui/about/about';
import { Loader } from '../../ui/loader/loader';
import { PageFooter } from '../../ui/page-footer/page-footer';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';

@Component({
  selector: 'app-home',
  imports: [PageTitle, DecimalPipe, About, Loader, PageFooter, RouterLinkWithHref],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home {
  private readonly wasmService = inject(WasmFacade);
  public readonly xorIsLoading = this.wasmService.xorWasmResource.isLoading;
  public readonly irisIsLoading = this.wasmService.irisWasmResource.isLoading;

  public readonly pageTitleOptions: Signal<PageTitleOptions> = signal({
    title: 'Neural Networks',
    subtitle: 'Interactive WebAssembly Demos',
    icon: '🧠',
  });

  public readonly xorModelInfo = this.wasmService.xorModelInfo;
  public readonly xorArchitecture = this.wasmService.xorArchitecture;

  public readonly irisModelInfo = this.wasmService.irisModelInfo;
  public readonly irisArchitecture = this.wasmService.irisArchitecture;
}
