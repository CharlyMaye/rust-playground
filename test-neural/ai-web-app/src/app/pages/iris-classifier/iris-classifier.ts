import { Component, inject, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';
import { PageFooter } from '../../ui/page-footer/page-footer';
import { NavigationBack } from '../../ui/navigation-back/navigation-back';
import { Loader } from '../../ui/loader/loader';
import { WasmFacade } from '@cma/wasm/shared/wasm';
import { ModelInfoComponent } from '../../ui/model-info/model-info';

@Component({
  selector: 'app-iris-classifier',
  imports: [PageTitle, Loader, NavigationBack, ModelInfoComponent, PageFooter],
  templateUrl: './iris-classifier.html',
  styleUrl: './iris-classifier.scss',
})
export class IrisClassifier {
  private readonly wasmService = inject(WasmFacade);
  public readonly irisIsLoading = this.wasmService.irisWasmResource.isLoading;
  public readonly irisModelInfo = this.wasmService.irisModelInfo;
  public readonly irisArchitecture = this.wasmService.irisArchitecture;

  public readonly pageTitleOptions: Signal<PageTitleOptions> = signal({
    title: 'Iris Classifier',
    subtitle: 'Multi-class Neural Network Classification',
    icon: '🌸',
  });
}
