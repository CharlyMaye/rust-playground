import { computed, effect, signal, Signal, untracked } from '@angular/core';
import {
  Activation,
  ArchitectureSummary,
  CnnActivationsResponse,
  ModelInfo,
  NeuralNetworkLayers,
  PredictionResult,
} from '@cma/wasm/shared';
import {
  activationToArchitecture,
  cnnActivationsToLayerVizArray,
  CnnLayerViz,
  neuralNetworkLayersToWeights,
} from '../ui/network-visualization';

/**
 * Minimal interface for a WASM CNN network that supports digit prediction.
 * All concrete CNN network classes (LeNet, ResNet, AlexNet, VGG) satisfy this.
 */
export interface CnnMnistNetwork {
  /** Returns JSON-encoded PredictionResult for the given input. */
  predict(input: Float32Array): string;
  /** Returns JSON-encoded CNN layer activations. */
  get_cnn_activations(input: Float32Array): string;
  /** Returns JSON-encoded FC layer activations. */
  get_activations(input: Float32Array): string;
}

/**
 * Per-model display strings for the CNN MNIST demo page template.
 */
export interface CnnPageConfig {
  /** Predictor card title (icon + model name). */
  cardTitle: string;
  /** Predictor card subtitle (model description). */
  cardSubtitle: string;
  /** Short model name used in the architecture card title. */
  architectureLabel: string;
  /** Architecture card subtitle. */
  architectureSubtitle: string;
  /** Message shown inside the loader. */
  loadingMessage: string;
  /** When true the "model not ready" error card is rendered. */
  showErrorBlock: boolean;
  /** Title for the error card (only relevant when showErrorBlock=true). */
  errorTitle?: string;
  /** Body text for the error card. */
  errorMessage?: string;
  /** Shell command shown in the error card to trigger training. */
  trainingCommand?: string;
}

/**
 * Abstract base class shared by all four CNN MNIST demo page components.
 *
 * Concrete pages (LeNet, ResNet, AlexNet, VGG) extend this class and provide:
 * - The pageConfig display strings
 * - The model-specific signal references (network, modelInfo, etc.)
 *
 * All reactive logic — signals, computeds, effects, event handlers — lives here.
 */
export abstract class CnnMnistPage {
  // ---------------------------------------------------------------------------
  // Abstract: provided by concrete subclasses
  // ---------------------------------------------------------------------------

  /** Human-readable tag for console warnings, e.g. "[LeNet]". */
  protected abstract readonly modelTag: string;

  /** Display strings specific to this model's page. */
  public abstract readonly pageConfig: CnnPageConfig;

  /** Whether the WASM module is loading. */
  public abstract readonly isLoading: Signal<boolean>;

  /** Resource error signal — truthy when the model file could not be loaded. */
  public abstract readonly hasError: Signal<unknown>;

  /** The CNN network instance once loaded. */
  public abstract readonly network: Signal<CnnMnistNetwork | undefined>;

  /** Model metadata (name, accuracy, description). */
  public abstract readonly modelInfo: Signal<ModelInfo | undefined>;

  /** Unified architecture summary. */
  public abstract readonly architectureSummary: Signal<ArchitectureSummary | undefined>;

  /** FC layer weights for the network visualization. */
  public abstract readonly weights: Signal<NeuralNetworkLayers | undefined>;

  // ---------------------------------------------------------------------------
  // Shared mutable state
  // ---------------------------------------------------------------------------

  /** Raw 28×28 grid from the canvas — updated on every pointer move for fast prediction. */
  public readonly drawnDigit = signal<number[][]>([]);

  /** 28×28 grid committed at pointer-up — used for expensive CNN computations. */
  public readonly committedDigit = signal<number[][]>([]);

  /** CNN layer feature-map activations, updated per committed stroke. */
  public readonly cnnActivations = signal<CnnActivationsResponse | undefined>(undefined);

  /** FC classifier activation data, updated per committed stroke. */
  public readonly fcActivations = signal<Activation | undefined>(undefined);

  // ---------------------------------------------------------------------------
  // Derived state
  // ---------------------------------------------------------------------------

  /** Prediction result from the network for the current drawn digit. */
  public readonly output = computed(() => {
    const network = this.network();
    const digitData = this.drawnDigit();
    if (!network || digitData.length === 0) return null;
    const input = new Float32Array(digitData.flat());
    return JSON.parse(network.predict(input)) as PredictionResult;
  });

  /** Display text for the predicted digit class. */
  public readonly predictionDisplay = computed(() => {
    const out = this.output();
    return out ? `Digit: ${out.class_name}` : 'Draw a digit';
  });

  /** Display text for the prediction confidence. */
  public readonly confidenceDisplay = computed(() => {
    const out = this.output();
    return out ? `${(out.confidence * 100).toFixed(1)}% confidence` : '';
  });

  /** Per-layer CNN feature-map visualization data. */
  public readonly cnnLayers = computed<CnnLayerViz[]>(() => {
    const activations = this.cnnActivations();
    const digitData = this.committedDigit();
    if (!activations || digitData.length === 0) return [];
    return cnnActivationsToLayerVizArray(activations, digitData.flat());
  });

  /** FC network architecture formatted for the visualization component. */
  public readonly fcNetworkArchitecture = computed(() => {
    const acts = this.fcActivations();
    return acts ? activationToArchitecture(acts) : null;
  });

  /** FC network weights formatted for the visualization component. */
  public readonly fcNetworkWeights = computed(() => {
    const wts = this.weights();
    return wts ? neuralNetworkLayersToWeights(wts) : null;
  });

  /** Architecture summary formatted as a multiline string for the <pre> block. */
  public readonly architectureSummaryDisplay = computed(() => {
    const summary = this.architectureSummary();
    return summary ? this.formatArchitectureSummary(summary) : 'Loading architecture...';
  });

  /** Comma-separated list of activation functions found in the architecture. */
  public readonly activationFunctions = computed(() => {
    const summary = this.architectureSummary();
    if (!summary?.layers) return undefined;
    const known = ['ReLU', 'Tanh', 'Sigmoid', 'Softmax', 'LeakyReLU', 'ELU', 'GELU'];
    const found = [...new Set(summary.layers.map((l) => l.name).filter((n) => known.includes(n)))];
    return found.length > 0 ? found.join(' → ') : undefined;
  });

  // ---------------------------------------------------------------------------
  // Side effects (lazy — run after construction, after all fields are set)
  // ---------------------------------------------------------------------------

  /** Recomputes CNN feature-map activations whenever the committed digit changes. */
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
        const parsed = JSON.parse(network.get_cnn_activations(input));
        this.cnnActivations.set('error' in parsed ? undefined : parsed);
      } catch (e) {
        console.warn(`${this.modelTag} CNN activations failed:`, e);
        this.cnnActivations.set(undefined);
      }
    });
  });

  /** Recomputes FC classifier activations whenever the committed digit changes. */
  private readonly fcEffect = effect(() => {
    const network = this.network();
    const digitData = this.committedDigit();

    if (!network || digitData.length === 0) {
      untracked(() => this.fcActivations.set(undefined));
      return;
    }

    untracked(() => {
      try {
        const input = new Float32Array(digitData.flat());
        const parsed = JSON.parse(network.get_activations(input));
        this.fcActivations.set('error' in parsed ? undefined : parsed);
      } catch (e) {
        console.warn(`${this.modelTag} FC activations failed:`, e);
        this.fcActivations.set(undefined);
      }
    });
  });

  // ---------------------------------------------------------------------------
  // Event handlers (bound in template)
  // ---------------------------------------------------------------------------

  /** Called on each pointer move — updates the prediction in real time. */
  public onDrawingChanged(gridData: number[][]): void {
    this.drawnDigit.set(gridData);
  }

  /** Called on pointer-up — triggers the expensive CNN activation computation. */
  public onDrawingComplete(gridData: number[][]): void {
    this.committedDigit.set(gridData);
  }

  /** Clears the canvas and resets all prediction state. */
  public clearCanvas(): void {
    this.drawnDigit.set([]);
    this.committedDigit.set([]);
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private formatArchitectureSummary(summary: ArchitectureSummary): string {
    return [
      `${summary.name} (${summary.model_type.toUpperCase()})`,
      `Input: ${summary.input_shape.join('×')}`,
      `Parameters: ${summary.num_parameters.toLocaleString()}`,
      '',
      'Layers:',
      ...summary.layers.map((l) => `  ${l.name}: ${l.config}`),
    ].join('\n');
  }
}
