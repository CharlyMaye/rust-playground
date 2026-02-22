import { ChangeDetectionStrategy, Component, inject, Signal } from '@angular/core';
import { WasmFacade } from '@cma/wasm/shared';
import { CanvasDraw } from 'src/app/ui/canvas-draw/canvas-draw';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';
import { ConfigurableNetworkVisualization } from '../../ui/network-visualization';
import { CnnMnistNetwork, CnnMnistPage, CnnPageConfig } from '../cnn-mnist-page';

/**
 * LeNet-5 CNN MNIST digit classifier demo page.
 */
@Component({
  selector: 'app-mnist-lenet',
  imports: [CanvasDraw, ConfigurableNetworkVisualization, Loader, ModelInfoComponent],
  templateUrl: '../cnn-mnist-page.html',
  styleUrl: './mnist-lenet.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'page container' },
})
export class MnistLeNet extends CnnMnistPage {
  private readonly wasmService = inject(WasmFacade);

  protected override readonly modelTag = '[LeNet]';

  public override readonly pageConfig: CnnPageConfig = {
    cardTitle: '🧠 LeNet-5 CNN - Draw a Digit (0-9)',
    cardSubtitle:
      'Convolutional neural network (LeNet-5 architecture from 1998) for handwritten digit recognition',
    architectureLabel: 'LeNet-5',
    architectureSubtitle:
      'Convolutional layers extract visual features, followed by fully connected classifier',
    loadingMessage: 'Loading LeNet-5 CNN...',
    showErrorBlock: false,
  };

  public override readonly isLoading = this.wasmService.mnistLeNetWasmResource.isLoading;
  public override readonly hasError = this.wasmService.mnistLeNetWasmResource
    .error as Signal<unknown>;
  public override readonly network = this.wasmService.mnistLeNetNetwork as Signal<
    CnnMnistNetwork | undefined
  >;
  public override readonly modelInfo = this.wasmService.mnistLeNetModelInfo;
  public override readonly architectureSummary = this.wasmService.mnistLeNetArchitectureSummary;
  public override readonly weights = this.wasmService.mnistLeNetWeights;
}
