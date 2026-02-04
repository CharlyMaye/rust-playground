import { Component, computed, inject, signal } from '@angular/core';
import { PredictionResult, WasmFacade } from '@cma/wasm/shared';
import { CanvasDraw } from 'src/app/ui/canvas-draw/canvas-draw';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';

/**
 * ResNet-Micro CNN MNIST digit classifier demo page.
 */
@Component({
  selector: 'app-mnist-resnet',
  imports: [CanvasDraw, Loader, ModelInfoComponent],
  templateUrl: './mnist-resnet.html',
  styleUrl: './mnist-resnet.scss',
  host: { class: 'page container' },
})
export class MnistResNet {
  private readonly wasmService = inject(WasmFacade);

  public readonly isLoading = this.wasmService.mnistResNetWasmResource.isLoading;
  public readonly network = this.wasmService.mnistResNetNetwork;
  public readonly modelInfo = this.wasmService.mnistResNetModelInfo;
  public readonly cnnSummary = this.wasmService.mnistResNetCnnSummary;
  public readonly architecture = this.wasmService.mnistResNetArchitecture;
  public readonly weights = this.wasmService.mnistResNetWeights;
  public readonly testAll = this.wasmService.mnistResNetTestAll;

  public readonly drawnDigit = signal<number[][]>([]);

  public readonly output = computed(() => {
    const network = this.network();
    const digitData = this.drawnDigit();

    if (!network || digitData.length === 0) {
      return null;
    }

    const flattenedInput = new Float64Array(digitData.flat());
    const prediction = network.predict(flattenedInput);
    const output = JSON.parse(prediction) as PredictionResult;
    return output;
  });

  public readonly predictionDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'Draw a digit';
    }
    return `Digit: ${output.class_name}`;
  });

  public readonly confidenceDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return '';
    }
    return (output.confidence * 100).toFixed(1) + '% confidence';
  });

  public readonly cnnSummaryDisplay = computed(() => {
    const summary = this.cnnSummary();
    if (!summary) {
      return 'Loading CNN architecture...';
    }
    return summary;
  });

  public onDrawingChanged(gridData: number[][]): void {
    this.drawnDigit.set(gridData);
  }

  public clearCanvas(): void {
    this.drawnDigit.set([]);
  }
}
