import { DecimalPipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { PredictionResult, WasmFacade } from '@cma/wasm/shared';
import { CanvasDraw } from 'src/app/ui/canvas-draw/canvas-draw';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';
import { NeuralNetworkModelVizualizer } from '../../ui/neural-network-model-vizualizer/neural-network-model-vizualizer';

/**
 * MNIST digit classifier demo page.
 * Placeholder for future handwritten digit recognition feature.
 */
@Component({
  selector: 'app-mnist-digit',
  imports: [DecimalPipe, CanvasDraw, Loader, ModelInfoComponent, NeuralNetworkModelVizualizer],
  templateUrl: './mnist-digit.html',
  styleUrl: './mnist-digit.scss',
  host: { class: 'page container' },
})
export class MnistDigit {
  private readonly wasmService = inject(WasmFacade);

  /** Whether the WASM module is currently loading */
  public readonly isLoading = this.wasmService.mnistWasmResource.isLoading;
  /** MNIST network instance */
  public readonly network = this.wasmService.mnistNetwork;
  /** Model metadata */
  public readonly modelInfo = this.wasmService.mnistModelInfo;
  /** Network architecture */
  public readonly architecture = this.wasmService.mnistArchitecture;
  /** Network weights */
  public readonly weights = this.wasmService.mnistWeights;
  /** Test results for all MNIST combinations */
  public readonly testAll = this.wasmService.mnistTestAll;

  /** Current drawn digit data (28x28 grid) */
  public readonly drawnDigit = signal<number[][]>([]);

  /** Current prediction output from the network */
  public readonly output = computed(() => {
    const network = this.network();
    const digitData = this.drawnDigit();

    if (!network || digitData.length === 0) {
      return null;
    }

    // Flatten the 28x28 grid into a 784-element Float64Array for MNIST
    const flattenedInput = new Float64Array(digitData.flat());

    // Call the WASM predict function with flattened input
    const prediction = network.predict(flattenedInput);
    const output = JSON.parse(prediction) as PredictionResult;
    return output;
  });

  /** Layer activations for the current input */
  public readonly activations = computed(() => {
    const network = this.network();
    const digitData = this.drawnDigit();

    if (!network || digitData.length === 0) {
      return null;
    }

    const flattenedInput = new Float64Array(digitData.flat());
    const acts = JSON.parse(network.get_activations(flattenedInput));
    return acts;
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
