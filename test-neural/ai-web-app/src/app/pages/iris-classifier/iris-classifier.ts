import { Component, inject } from '@angular/core';
import { WasmFacade } from '@cma/wasm/shared/wasm';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';

@Component({
  selector: 'app-iris-classifier',
  imports: [Loader, ModelInfoComponent],
  templateUrl: './iris-classifier.html',
  styleUrl: './iris-classifier.scss',
  host: { class: 'container' },
})
export class IrisClassifier {
  private readonly wasmService = inject(WasmFacade);
  public readonly irisIsLoading = this.wasmService.irisWasmResource.isLoading;
  public readonly irisNetwork = this.wasmService.irisNetwork;
  public readonly irisModelInfo = this.wasmService.irisModelInfo;
  public readonly irisArchitecture = this.wasmService.irisArchitecture;
  public readonly irisWeights = this.wasmService.irisWeights;
  public readonly irisTestAll = this.wasmService.irisTestAll;
}
