import { ChangeDetectionStrategy, Component, inject, Signal } from '@angular/core';
import { WasmFacade } from '@cma/wasm/shared';
import { CanvasDraw } from 'src/app/ui/canvas-draw/canvas-draw';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';
import { ConfigurableNetworkVisualization } from '../../ui/network-visualization';
import { CnnMnistNetwork, CnnMnistPage, CnnPageConfig } from '../cnn-mnist-page';

/**
 * VGG-Tiny CNN MNIST digit classifier demo page.
 */
@Component({
  selector: 'app-mnist-vgg',
  imports: [CanvasDraw, ConfigurableNetworkVisualization, Loader, ModelInfoComponent],
  templateUrl: '../cnn-mnist-page.html',
  styleUrl: './mnist-vgg.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'page container' },
})
export class MnistVgg extends CnnMnistPage {
  private readonly wasmService = inject(WasmFacade);

  protected override readonly modelTag = '[VGG]';

  public override readonly pageConfig: CnnPageConfig = {
    cardTitle: '📚 VGG-Tiny CNN - Draw a Digit (0-9)',
    cardSubtitle:
      'VGG-style CNN (Simonyan & Zisserman, 2014) with stacked 3×3 convolutions for digit recognition',
    architectureLabel: 'VGG-Tiny',
    architectureSubtitle: 'Stacked 3×3 convolutions for deep feature extraction',
    loadingMessage: 'Loading VGG-Tiny CNN...',
    showErrorBlock: true,
    errorTitle: 'VGG-Tiny Model Not Ready',
    errorMessage: 'The VGG-Tiny model has not been trained yet.',
    trainingCommand:
      'cd neural-wasm/mnist-vgg && cargo run --bin train_vgg --features training --release',
  };

  public override readonly isLoading = this.wasmService.mnistVggWasmResource.isLoading;
  public override readonly hasError = this.wasmService.mnistVggWasmResource
    .error as Signal<unknown>;
  public override readonly network = this.wasmService.mnistVggNetwork as Signal<
    CnnMnistNetwork | undefined
  >;
  public override readonly modelInfo = this.wasmService.mnistVggModelInfo;
  public override readonly architectureSummary = this.wasmService.mnistVggArchitectureSummary;
  public override readonly weights = this.wasmService.mnistVggWeights;
}
