import { DOCUMENT, isPlatformBrowser } from '@angular/common';
import {
  computed,
  DestroyRef,
  inject,
  Injectable,
  PLATFORM_ID,
  resource,
  ResourceLoaderParams,
  ResourceRef,
  signal,
  WritableSignal,
} from '@angular/core';
import init, {
  InitOutput as InitMNISTVggOutput,
  MnistVGGNetwork,
} from '@cma/wasm/mnist_vgg_wasm/neural_wasm_mnist_vgg.js';
import { ArchitectureSummary, ModelInfo, NeuralNetworkLayers, TestResult } from './model-info';

/**
 * Service for loading and interacting with the VGG-Tiny CNN WASM neural network.
 * Handles WASM module initialization and provides reactive access to network data.
 */
@Injectable({
  providedIn: 'root',
})
export class MNISTVggWasmService {
  private readonly _document = inject(DOCUMENT);
  private readonly _platformId = inject(PLATFORM_ID);
  private readonly _destroyRef = inject(DestroyRef);

  protected readonly _wasmPath: WritableSignal<string> = signal('');

  /** Cached network instance to avoid recreating on each computed access */
  private _networkInstance: MnistVGGNetwork | null = null;

  constructor() {
    const base = this.computeWasmBase();
    this._wasmPath.set(`${base}wasm/mnist_vgg_wasm/neural_wasm_mnist_vgg_bg.wasm`);

    // Cleanup WASM memory when service is destroyed
    this._destroyRef.onDestroy(() => {
      this.dispose();
    });
  }

  /**
   * Dispose of WASM resources and free memory.
   * Called automatically on service destroy.
   */
  public dispose(): void {
    if (this._networkInstance) {
      try {
        this._networkInstance.free();
      } catch {
        // Ignore errors if already freed
      }
      this._networkInstance = null;
    }
  }

  private computeWasmBase(): string {
    if (!isPlatformBrowser(this._platformId)) {
      return '/';
    }
    const b = this._document.querySelector('base')?.getAttribute('href') ?? '/';
    if (b === './') return './';
    return b.endsWith('/') ? b : b + '/';
  }

  /** Resource managing WASM module loading state */
  public readonly wasmResource: ResourceRef<InitMNISTVggOutput | undefined> = resource({
    params: this._wasmPath,
    loader: (param: ResourceLoaderParams<string>) => init({ module_or_path: param.params }),
    defaultValue: undefined,
  });

  /** VGG-Tiny network instance, available after WASM initialization */
  public readonly network = computed(() => {
    const initOutput = this.wasmResource.value();
    if (!initOutput) {
      return undefined;
    }
    // Reuse existing instance or create new one
    if (!this._networkInstance) {
      this._networkInstance = new MnistVGGNetwork();
    }
    return this._networkInstance;
  });

  /** Model metadata including name, accuracy, and description */
  public readonly modelInfo = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const modelInfoJson: string = network.model_info();
    const modelInfo: ModelInfo = JSON.parse(modelInfoJson);
    return modelInfo;
  });

  /** Architecture summary (unified format for all models) */
  public readonly architectureSummary = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const json: string = network.get_architecture();
    return JSON.parse(json) as ArchitectureSummary;
  });

  /** FC classifier architecture as an array of layer sizes */
  public readonly architecture = computed(() => {
    const modelInfo = this.modelInfo();
    if (!modelInfo) {
      return undefined;
    }
    return modelInfo.architecture.split('→').map((layer) => {
      const trimmedLayer = layer.trim();
      if (trimmedLayer.startsWith('[') && trimmedLayer.endsWith(']')) {
        return trimmedLayer;
      }
      const num = parseInt(trimmedLayer, 10);
      return isNaN(num) ? trimmedLayer : num;
    });
  });

  /** FC classifier weights and biases */
  public readonly weights = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const weightsJson: string = network.get_weights();
    const weights: NeuralNetworkLayers = JSON.parse(weightsJson);
    return weights;
  });

  /** Test results for MNIST samples */
  public readonly testAll = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const testResultsJson: string = network.test_all();
    const testResults: TestResult[] = JSON.parse(testResultsJson);
    return testResults;
  });
}
