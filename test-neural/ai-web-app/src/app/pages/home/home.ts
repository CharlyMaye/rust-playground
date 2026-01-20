import { Component, effect, inject, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';
import { PageFooter } from '../../ui/page-footer/page-footer';
import { About } from '../../ui/about/about';
import { RouterLinkWithHref } from '@angular/router';
import { WasmService } from '@cma/wasm/shared/wasm';


@Component({
  selector: 'app-home',
  imports: [PageTitle, About, PageFooter, RouterLinkWithHref],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home {
  private readonly wasmService = inject(WasmService);
  public readonly xorIsLoading = this.wasmService.xorWasmResource.isLoading;
  public readonly irisIsLoading = this.wasmService.irisWasmResource.isLoading;

  public readonly pageTitleOptions: Signal<PageTitleOptions> = signal({
    title: 'Neural Networks',
    subtitle: 'Interactive WebAssembly Demos',
    icon: '🧠',
  });

  public readonly xorInitOutput = this.wasmService.xorWasmResource.value.asReadonly();
  public readonly irisInitOutput = this.wasmService.irisWasmResource.value.asReadonly();

  constructor() {
    effect  (() => {
      console.log('WASM Resource loaded:', this.xorInitOutput());
    });
    effect  (() => {
      console.log('WASM Resource loaded:', this.irisInitOutput());
    });
  }
  public ngOnInit() {
    this.loadModelAccuracies();
  }

  async loadModelAccuracies() {
  }
}
