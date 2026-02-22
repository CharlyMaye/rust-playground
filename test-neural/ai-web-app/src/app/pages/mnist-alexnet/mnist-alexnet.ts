import { ChangeDetectionStrategy, Component, inject, Signal } from '@angular/core';
import { WasmFacade } from '@cma/wasm/shared';
import { CanvasDraw } from 'src/app/ui/canvas-draw/canvas-draw';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';
import { ConfigurableNetworkVisualization } from '../../ui/network-visualization';
import { CnnMnistNetwork, CnnMnistPage, CnnPageConfig } from '../cnn-mnist-page';

/**
 * AlexNet-Mini CNN MNIST digit classifier demo page.
 */
@Component({
  selector: 'app-mnist-alexnet',
  imports: [CanvasDraw, ConfigurableNetworkVisualization, Loader, ModelInfoComponent],
  templateUrl: '../cnn-mnist-page.html',
  styleUrl: './mnist-alexnet.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'page container' },
})
export class MnistAlexNet extends CnnMnistPage {
  private readonly wasmService = inject(WasmFacade);

  protected override readonly modelTag = '[AlexNet]';

  public override readonly pageConfig: CnnPageConfig = {
    cardTitle: '🔥 AlexNet-Mini CNN - Draw a Digit (0-9)',
    cardSubtitle:
      'AlexNet-style CNN (Krizhevsky et al., 2012) adapted for MNIST handwritten digit recognition',
    architectureLabel: 'AlexNet-Mini',
    architectureSubtitle:
      'Multiple convolutional layers with batch normalization extract visual features',
    loadingMessage: 'Loading AlexNet-Mini CNN...',
    showErrorBlock: true,
    errorTitle: 'AlexNet-Mini Model Not Ready',
    errorMessage: 'The AlexNet-Mini model has not been trained yet.',
    trainingCommand:
      'cd neural-wasm/mnist-alexnet && cargo run --bin train_alexnet --features training --release',
  };

  public override readonly isLoading = this.wasmService.mnistAlexNetWasmResource.isLoading;
  public override readonly hasError = this.wasmService.mnistAlexNetWasmResource
    .error as Signal<unknown>;
  public override readonly network = this.wasmService.mnistAlexNetNetwork as Signal<
    CnnMnistNetwork | undefined
  >;
  public override readonly modelInfo = this.wasmService.mnistAlexNetModelInfo;
  public override readonly architectureSummary = this.wasmService.mnistAlexNetArchitectureSummary;
  public override readonly weights = this.wasmService.mnistAlexNetWeights;
}
