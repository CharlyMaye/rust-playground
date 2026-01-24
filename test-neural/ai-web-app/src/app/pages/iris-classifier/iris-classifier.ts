import { Component, computed, effect, inject, signal } from '@angular/core';
import { form, FormField, max, min } from '@angular/forms/signals';
import { WasmFacade } from '@cma/wasm/shared/wasm';
import { Loader } from '../../ui/loader/loader';
import { ModelInfoComponent } from '../../ui/model-info/model-info';

interface IrisFormState {
  sepalLength: number;
  sepalWidth: number;
  petalLength: number;
  petalWidth: number;
}
type NetworkPrediction = {
  class: string;
  class_idx: number;
  probabilities: [number, number, number];
  confidence: number;
};

@Component({
  selector: 'app-iris-classifier',
  imports: [FormField, Loader, ModelInfoComponent],
  templateUrl: './iris-classifier.html',
  styleUrl: './iris-classifier.scss',
  host: { class: 'container' },
})
export class IrisClassifier {
  private readonly wasmService = inject(WasmFacade);
  public readonly irisIsLoading = this.wasmService.irisWasmResource.isLoading;
  public readonly irisNetwork = this.wasmService.irisNetwork;
  public readonly irisModelInfo = this.wasmService.irisModelInfo;
  public readonly irisArchitecture = this.wasmService.irisArchitecture;
  public readonly irisWeights = this.wasmService.irisWeights;
  public readonly irisTestAll = this.wasmService.irisTestAll;

  private readonly _showTestSamplesResult = signal(false);
  public readonly showTestSamplesResult = computed(() => this._showTestSamplesResult());
  public readonly numberOfCorrectPredictions = computed(() => {
    const testResults = this.irisTestAll();
    if (!testResults) {
      return 0;
    }
    let correctCount = 0;
    for (const result of testResults) {
      if (result.correct) {
        correctCount++;
      }
    }
    return correctCount;
  });
  private readonly _preset = signal({
    setosa: { sepalLength: 5.1, sepalWidth: 3.5, petalLength: 1.4, petalWidth: 0.2 },
    versicolor: { sepalLength: 7.0, sepalWidth: 3.2, petalLength: 4.7, petalWidth: 1.4 },
    virginica: { sepalLength: 6.3, sepalWidth: 3.3, petalLength: 6.0, petalWidth: 2.5 },
  });
  // TODO - calculate this from _preset
  private readonly _selectedPreset = signal<'setosa' | 'versicolor' | 'virginica'>('setosa');

  public readonly irisFormState = signal<IrisFormState>({
    sepalLength: 5.1,
    sepalWidth: 3.5,
    petalLength: 1.4,
    petalWidth: 0.2,
  });

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
  public readonly output = computed(() => {
    const network = this.irisNetwork();
    if (!network) {
      return null;
    }
    const sepalLength = this.irisForm.sepalLength().value();
    const sepalWidth = this.irisForm.sepalWidth().value();
    const petalLength = this.irisForm.petalLength().value();
    const petalWidth = this.irisForm.petalWidth().value();
    const resultJSON = network.predict(sepalLength, sepalWidth, petalLength, petalWidth);
    const result = JSON.parse(resultJSON) as NetworkPrediction;
    console.log('Iris Prediction:', result);
    return result;
  });
  public readonly predictionDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    return output.class;
  });
  public readonly confidenceDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    // TODO - homogénéiser l'échelle de confiance : on a soit 0-1, soit un pourcentage
    return (
      output.confidence /** 100*/
        .toFixed(1) + '% confidence'
    );
  });

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

  constructor() {
    effect(() => {
      console.log('hi', this.output());
    });
  }

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
  public testAllSamples(): void {
    this._showTestSamplesResult.set(true);
  }
}
