import { Component, computed, effect, inject, signal, untracked } from '@angular/core';
import {
  ArchitectureSummary,
  CnnActivationsResponse,
  PredictionResult,
  WasmFacade,
} from '@cma/wasm/shared';
import { CanvasDraw } from 'src/app/ui/canvas-draw/canvas-draw';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';
import {
  cnnActivationsToLayerVizArray,
  CnnLayerViz,
  ConfigurableNetworkVisualization,
} from '../../ui/network-visualization';

/**
 * ResNet-Micro CNN MNIST digit classifier demo page.
 */
@Component({
  selector: 'app-mnist-resnet',
  imports: [CanvasDraw, ConfigurableNetworkVisualization, Loader, ModelInfoComponent],
  templateUrl: './mnist-resnet.html',
  styleUrl: './mnist-resnet.scss',
  host: { class: 'page container' },
})
export class MnistResNet {
  private readonly wasmService = inject(WasmFacade);


  public readonly isLoading = this.wasmService.mnistResNetWasmResource.isLoading;
  public readonly network = this.wasmService.mnistResNetNetwork;
  public readonly modelInfo = this.wasmService.mnistResNetModelInfo;
  public readonly architectureSummary = this.wasmService.mnistResNetArchitectureSummary;
  public readonly architecture = this.wasmService.mnistResNetArchitecture;
  public readonly weights = this.wasmService.mnistResNetWeights;
  public readonly testAll = this.wasmService.mnistResNetTestAll;

  public readonly drawnDigit = signal<number[][]>([]);

  /** Committed digit data (mouseup only) - for expensive CNN computations */
  public readonly committedDigit = signal<number[][]>([]);

  public readonly output = computed(() => {
    const network = this.network();
    const digitData = this.drawnDigit();

    if (!network || digitData.length === 0) {
      return null;
    }

    const flattenedInput = new Float32Array(digitData.flat());
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

  /** CNN intermediate feature maps activations (populated by Web Worker) */
  public readonly cnnActivations = signal<CnnActivationsResponse | undefined>(undefined);

  /** Per-layer CNN visualization data (one canvas per layer) */
  public readonly cnnLayers = computed<CnnLayerViz[]>(() => {
    const activations = this.cnnActivations();
    const digitData = this.committedDigit();
    if (!activations || digitData.length === 0) return [];
    return cnnActivationsToLayerVizArray(activations, digitData.flat());
  });

  private readonly cnnEffect = effect(() => {
    const network = this.network();
    const digitData = this.committedDigit();

    if (!network || digitData.length === 0) {
      untracked(() => this.cnnActivations.set(undefined));
      return;
    }

    untracked(() => {
      try {
        const input = new Float32Array(digitData.flat());
        const json: string = network.get_cnn_activations(input);
        const parsed = JSON.parse(json);
        this.cnnActivations.set('error' in parsed ? undefined : parsed);
      } catch (e) {
        console.warn('[ResNet] CNN activations failed:', e);
        this.cnnActivations.set(undefined);
      }
    });
  });

  public readonly architectureSummaryDisplay = computed(() => {
    const summary = this.architectureSummary();
    if (!summary) {
      return 'Loading architecture...';
    }
    return this.formatArchitectureSummary(summary);
  });

  private formatArchitectureSummary(summary: ArchitectureSummary): string {
    const lines = [
      `${summary.name} (${summary.model_type.toUpperCase()})`,
      `Input: ${summary.input_shape.join('×')}`,
      `Parameters: ${summary.num_parameters.toLocaleString()}`,
      '',
      'Layers:',
      ...summary.layers.map((l) => `  ${l.name}: ${l.config}`),
    ];
    return lines.join('\n');
  }

  public onDrawingChanged(gridData: number[][]): void {
    this.drawnDigit.set(gridData);
  }

  /** Handle end-of-stroke: trigger expensive CNN activation computation */
  public onDrawingComplete(gridData: number[][]): void {
    this.committedDigit.set(gridData);
  }

  public clearCanvas(): void {
    this.drawnDigit.set([]);
    this.committedDigit.set([]);
  }
}
