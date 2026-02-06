import { Component, computed, inject, signal } from '@angular/core';
import { ArchitectureSummary, PredictionResult, WasmFacade } from '@cma/wasm/shared';
import { CanvasDraw } from 'src/app/ui/canvas-draw/canvas-draw';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';

/**
 * LeNet-5 CNN MNIST digit classifier demo page.
 * Uses convolutional neural network for handwritten digit recognition.
 */
@Component({
  selector: 'app-mnist-lenet',
  imports: [CanvasDraw, Loader, ModelInfoComponent],
  templateUrl: './mnist-lenet.html',
  styleUrl: './mnist-lenet.scss',
  host: { class: 'page container' },
})
export class MnistLeNet {
  private readonly wasmService = inject(WasmFacade);

  /** Whether the WASM module is currently loading */
  public readonly isLoading = this.wasmService.mnistLeNetWasmResource.isLoading;
  /** LeNet-5 network instance */
  public readonly network = this.wasmService.mnistLeNetNetwork;
  /** Model metadata */
  public readonly modelInfo = this.wasmService.mnistLeNetModelInfo;
  /** Architecture summary (unified format) */
  public readonly architectureSummary = this.wasmService.mnistLeNetArchitectureSummary;
  /** FC architecture */
  public readonly architecture = this.wasmService.mnistLeNetArchitecture;
  /** FC weights */
  public readonly weights = this.wasmService.mnistLeNetWeights;
  /** Test results for MNIST samples */
  public readonly testAll = this.wasmService.mnistLeNetTestAll;

  /** Current drawn digit data (28x28 grid) */
  public readonly drawnDigit = signal<number[][]>([]);

  /** Current prediction output from the network */
  public readonly output = computed(() => {
    const network = this.network();
    const digitData = this.drawnDigit();

    if (!network || digitData.length === 0) {
      return null;
    }

    // Flatten the 28x28 grid into a 784-element Float32Array for MNIST
    const flattenedInput = new Float32Array(digitData.flat());

    // Call the WASM predict function with flattened input
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
   * Clear the canvas and reset prediction
   */
  public clearCanvas(): void {
    this.drawnDigit.set([]);
  }
}
