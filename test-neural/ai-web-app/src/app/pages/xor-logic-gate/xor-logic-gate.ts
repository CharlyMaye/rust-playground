import { Component, inject, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';
import { PageFooter } from '../../ui/page-footer/page-footer';
import { NavigationBack } from '../../ui/navigation-back/navigation-back';
import { Loader } from '../../ui/loader/loader';
import { WasmFacade } from '@cma/wasm/shared/wasm';

@Component({
  selector: 'app-xor-logic-gate',
  imports: [PageTitle, Loader, NavigationBack, PageFooter],
  templateUrl: './xor-logic-gate.html',
  styleUrl: './xor-logic-gate.scss',
})
export class XorLogicGate {
  private readonly wasmService = inject(WasmFacade);
  public readonly xorIsLoading = this.wasmService.xorWasmResource.isLoading;
  public readonly xorModelInfo = this.wasmService.xorModelInfo;
  public readonly xorArchitecture = this.wasmService.xorArchitecture;

  public readonly pageTitleOptions: Signal<PageTitleOptions> = signal({
    title: 'Neural XOR',
    subtitle: 'WebAssembly Neural Network Demo',
    icon: '🧠',
  });
}
