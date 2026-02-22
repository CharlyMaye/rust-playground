import { ChangeDetectionStrategy, Component, inject, Signal } from '@angular/core';
import { WasmFacade } from '@cma/wasm/shared';
import { CanvasDraw } from 'src/app/ui/canvas-draw/canvas-draw';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';
import { ConfigurableNetworkVisualization } from '../../ui/network-visualization';
import { CnnMnistNetwork, CnnMnistPage, CnnPageConfig } from '../cnn-mnist-page';

/**
 * ResNet-Micro CNN MNIST digit classifier demo page.
 */
@Component({
  selector: 'app-mnist-resnet',
  imports: [CanvasDraw, ConfigurableNetworkVisualization, Loader, ModelInfoComponent],
  templateUrl: '../cnn-mnist-page.html',
  styleUrl: './mnist-resnet.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'page container' },
})
export class MnistResNet extends CnnMnistPage {
  private readonly wasmService = inject(WasmFacade);

  protected override readonly modelTag = '[ResNet]';

  public override readonly pageConfig: CnnPageConfig = {
    cardTitle: '🔗 ResNet-Micro CNN - Draw a Digit (0-9)',
    cardSubtitle: 'ResNet-style architecture (He et al., 2015) - uses residual connections concept',
    architectureLabel: 'ResNet-Micro',
    architectureSubtitle: 'Minimal ResNet-style for fast inference',
    loadingMessage: 'Loading ResNet-Micro CNN...',
    showErrorBlock: false,
  };

  public override readonly isLoading = this.wasmService.mnistResNetWasmResource.isLoading;
  public override readonly hasError = this.wasmService.mnistResNetWasmResource
    .error as Signal<unknown>;
  public override readonly network = this.wasmService.mnistResNetNetwork as Signal<
    CnnMnistNetwork | undefined
  >;
  public override readonly modelInfo = this.wasmService.mnistResNetModelInfo;
  public override readonly architectureSummary = this.wasmService.mnistResNetArchitectureSummary;
  public override readonly weights = this.wasmService.mnistResNetWeights;
}
