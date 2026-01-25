import { Component, computed, inject, signal } from '@angular/core';
import { form, FormField, max, min } from '@angular/forms/signals';
import { Activation, IrisPrediction, WasmFacade } from '@cma/wasm/shared';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';
import { NeuralNetworkModelVizualizer } from '../../ui/neural-network-model-vizualizer/neural-network-model-vizualizer';

/**
 * Form state for Iris flower measurements.
 */
interface IrisFormState {
  sepalLength: number;
  sepalWidth: number;
  petalLength: number;
  petalWidth: number;
}

/**
 * Interactive Iris flower classifier demo page.
 * Demonstrates a neural network trained to classify iris flowers
 * into setosa, versicolor, or virginica species.
 */
@Component({
  selector: 'app-iris-classifier',
  imports: [FormField, Loader, ModelInfoComponent, NeuralNetworkModelVizualizer],
  templateUrl: './iris-classifier.html',
  styleUrl: './iris-classifier.scss',
  host: { class: 'page container' },
})
export class IrisClassifier {
  private readonly wasmService = inject(WasmFacade);

  /** Whether the WASM module is currently loading */
  public readonly irisIsLoading = this.wasmService.irisWasmResource.isLoading;
  /** Iris classifier network instance */
  public readonly irisNetwork = this.wasmService.irisNetwork;
  /** Model metadata */
  public readonly irisModelInfo = this.wasmService.irisModelInfo;
  /** Network architecture */
  public readonly irisArchitecture = this.wasmService.irisArchitecture;
  /** Network weights */
  public readonly irisWeights = this.wasmService.irisWeights;
  /** Test results for validation samples */
  public readonly irisTestAll = this.wasmService.irisTestAll;

  private readonly _showTestSamplesResult = signal(false);

  /** Whether to display test sample results */
  public readonly showTestSamplesResult = computed(() => this._showTestSamplesResult());

  /** Count of correct predictions in test samples */
  public readonly numberOfCorrectPredictions = computed(() => {
    const testResults = this.irisTestAll();
    if (!testResults) {
      return 0;
    }
    return testResults.filter((result) => result.correct).length;
  });

  private readonly _preset = signal({
    setosa: { sepalLength: 5.1, sepalWidth: 3.5, petalLength: 1.4, petalWidth: 0.2 },
    versicolor: { sepalLength: 7.0, sepalWidth: 3.2, petalLength: 4.7, petalWidth: 1.4 },
    virginica: { sepalLength: 6.3, sepalWidth: 3.3, petalLength: 6.0, petalWidth: 2.5 },
  });

  // TODO - calculate this from _preset
  private readonly _selectedPreset = signal<'setosa' | 'versicolor' | 'virginica'>('setosa');

  /** Current form state with flower measurements */
  public readonly irisFormState = signal<IrisFormState>({
    sepalLength: 5.1,
    sepalWidth: 3.5,
    petalLength: 1.4,
    petalWidth: 0.2,
  });

  /** Reactive form with validation */
  public readonly irisForm = form<IrisFormState>(
    this.irisFormState,
    (schemaPath) => {
      min(schemaPath.sepalLength, 0);
      max(schemaPath.sepalLength, 10);
      min(schemaPath.sepalWidth, 0);
      max(schemaPath.sepalWidth, 10);

      min(schemaPath.petalLength, 0);
      max(schemaPath.petalLength, 10);
      min(schemaPath.petalWidth, 0);
      max(schemaPath.petalWidth, 10);
    },
    {},
  );

  private readonly formInputs = computed(() => ({
    sepalLength: this.irisForm.sepalLength().value(),
    sepalWidth: this.irisForm.sepalWidth().value(),
    petalLength: this.irisForm.petalLength().value(),
    petalWidth: this.irisForm.petalWidth().value(),
  }));

  /** Current prediction output from the network */
  public readonly output = computed(() => {
    const network = this.irisNetwork();
    if (!network) {
      return null;
    }
    const { sepalLength, sepalWidth, petalLength, petalWidth } = this.formInputs();
    const resultJSON = network.predict(sepalLength, sepalWidth, petalLength, petalWidth);
    const result = JSON.parse(resultJSON) as IrisPrediction;
    return result;
  });

  /** Layer activations for the current input */
  public readonly activations = computed(() => {
    const network = this.irisNetwork();
    if (!network) {
      return null;
    }
    const { sepalLength, sepalWidth, petalLength, petalWidth } = this.formInputs();
    const activationData = JSON.parse(
      network.get_activations(sepalLength, sepalWidth, petalLength, petalWidth),
    ) as Activation<number, number>;
    return activationData;
  });

  /** Formatted prediction class for display */
  public readonly predictionDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    return output.class;
  });

  /** Formatted confidence value for display */
  public readonly confidenceDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    // TODO - homogénéiser l'échelle de confiance : on a soit 0-1, soit un pourcentage
    return output.confidence.toFixed(1) + '% confidence';
  });

  /** Probability percentages for each class */
  public readonly probability = computed(() => {
    const output = this.output();
    if (!output) {
      return [(0 * 100).toFixed(1), (0 * 100).toFixed(1), (0 * 100).toFixed(1)];
    }
    return [
      (output.probabilities[0] * 100).toFixed(1),
      (output.probabilities[1] * 100).toFixed(1),
      (output.probabilities[2] * 100).toFixed(1),
    ] as [string, string, string];
  });

  /**
   * Loads preset flower measurements for a given species.
   * @param preset - Species to load ('setosa', 'versicolor', or 'virginica')
   */
  public loadPreset(preset: 'setosa' | 'versicolor' | 'virginica'): void {
    this._selectedPreset.set(preset);
    const values = this._preset()[preset];
    this.irisFormState.update(() => ({
      sepalLength: values.sepalLength,
      sepalWidth: values.sepalWidth,
      petalLength: values.petalLength,
      petalWidth: values.petalWidth,
    }));
  }

  /** Runs prediction on all test samples and displays results */
  public testAllSamples(): void {
    this.activations();
    this._showTestSamplesResult.set(true);
  }
}
