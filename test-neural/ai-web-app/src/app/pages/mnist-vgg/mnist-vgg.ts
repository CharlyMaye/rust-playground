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
 * VGG-Tiny CNN MNIST digit classifier demo page.
 * Uses VGG-style architecture for handwritten digit recognition.
 */
@Component({
  selector: 'app-mnist-vgg',
  imports: [CanvasDraw, ConfigurableNetworkVisualization, Loader, ModelInfoComponent],
  templateUrl: './mnist-vgg.html',
  styleUrl: './mnist-vgg.scss',
  host: { class: 'page container' },
})
export class MnistVgg {
  private readonly wasmService = inject(WasmFacade);

  /** Whether the WASM module is currently loading */
  public readonly isLoading = this.wasmService.mnistVggWasmResource.isLoading;
  /** Whether the WASM module failed to load (model not trained yet) */
  public readonly hasError = this.wasmService.mnistVggWasmResource.error;
  /** VGG-Tiny network instance */
  public readonly network = this.wasmService.mnistVggNetwork;
  /** Model metadata */
  public readonly modelInfo = this.wasmService.mnistVggModelInfo;
  /** Architecture summary (unified format) */
  public readonly architectureSummary = this.wasmService.mnistVggArchitectureSummary;
  /** FC architecture */
  public readonly architecture = this.wasmService.mnistVggArchitecture;
  /** FC weights */
  public readonly weights = this.wasmService.mnistVggWeights;
  /** Test results for MNIST samples */
  public readonly testAll = this.wasmService.mnistVggTestAll;

  /** Current drawn digit data (28x28 grid) - updates on every stroke for fast prediction */
  public readonly drawnDigit = signal<number[][]>([]);

  /** Committed digit data (mouseup only) - for expensive CNN computations */
  public readonly committedDigit = signal<number[][]>([]);

  /** Current prediction output from the network */
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

  /** Formatted prediction value for display */
  public readonly predictionDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'Draw a digit';
    }
    return `Digit: ${output.class_name}`;
  });

  /** Formatted confidence value for display */
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
        console.warn('[VGG] CNN activations failed:', e);
        this.cnnActivations.set(undefined);
      }
    });
  });

  /** Formatted architecture summary for display */
  public readonly architectureSummaryDisplay = computed(() => {
    const summary = this.architectureSummary();
    if (!summary) {
      return 'Loading architecture...';
    }
    return this.formatArchitectureSummary(summary);
  });

  /** Format architecture summary for display */
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

  /**
   * Handle drawing changes from the canvas
   * @param gridData - The 28x28 grid with pixel intensities (0-1)
   */
  public onDrawingChanged(gridData: number[][]): void {
    this.drawnDigit.set(gridData);
  }

  /**
   * Handle end-of-stroke: trigger expensive CNN activation computation
   */
  public onDrawingComplete(gridData: number[][]): void {
    this.committedDigit.set(gridData);
  }

  /**
   * Clear the canvas and reset prediction
   */
  public clearCanvas(): void {
    this.drawnDigit.set([]);
    this.committedDigit.set([]);
  }
}
