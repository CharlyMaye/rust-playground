import { Component, effect, inject, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';
import { PageFooter } from '../../ui/page-footer/page-footer';
import { About } from '../../ui/about/about';
import { RouterLinkWithHref } from '@angular/router';
import { WasmService } from '@cma/wasm/shared/wasm';
import { Loader } from '../../ui/loader/loader';
import { IrisClassifier } from '@cma/wasm/iris_wasm';
import { XorNetwork } from '@cma/wasm/xor_wasm';


@Component({
  selector: 'app-home',
  imports: [PageTitle, About, Loader, PageFooter, RouterLinkWithHref],
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

  public readonly xorModelInfo = this.wasmService.xorModelInfo;
  public readonly irisModelInfo = this.wasmService.irisModelInfo;

  constructor() {
    effect  (() => {
      const initOutput = this.xorModelInfo();
      if (!initOutput) {
        return;
      }
    });
    effect  (() => {
      const initOutput = this.irisModelInfo();
      if (!initOutput) {
        return;
      }
    });
  }

}
