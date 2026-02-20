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
  Signal,
  signal,
  WritableSignal,
} from '@angular/core';
import init, {
  InitOutput as InitMNISTResNetOutput,
  MnistResNetNetwork,
} from '@cma/wasm/mnist_resnet_wasm/neural_wasm_mnist_resnet.js';
import { ArchitectureSummary, ModelInfo, NeuralNetworkLayers, TestResult } from './model-info';

/**
 * Service for loading and interacting with the ResNet-Micro CNN WASM neural network.
 */
@Injectable({
  providedIn: 'root',
})
export class MNISTResNetWasmService {
  private readonly _document = inject(DOCUMENT);
  private readonly _platformId = inject(PLATFORM_ID);
  private readonly _destroyRef = inject(DestroyRef);

  protected readonly _wasmPath: WritableSignal<string> = signal('');

  /** Cached network instance to avoid recreating on each computed access */
  private _networkInstance: MnistResNetNetwork | null = null;

  constructor() {
    const base = this.computeWasmBase();
    this._wasmPath.set(`${base}wasm/mnist_resnet_wasm/neural_wasm_mnist_resnet_bg.wasm`);

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

  public readonly wasmResource: ResourceRef<InitMNISTResNetOutput | undefined> = resource({
    params: this._wasmPath,
    loader: (param: ResourceLoaderParams<string>) => init({ module_or_path: param.params }),
    defaultValue: undefined,
  });

  /** ResNet-Micro network instance, available after WASM initialization */
  public readonly network = computed(() => {
    const initOutput = this.wasmResource.value();
    if (!initOutput) {
      return undefined;
    }
    // Reuse existing instance or create new one
    if (!this._networkInstance) {
      this._networkInstance = new MnistResNetNetwork();
    }
    return this._networkInstance;
  });

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

  public readonly architecture = computed(() => {
    const modelInfo = this.modelInfo();
    if (!modelInfo) {
      return undefined;
    }
    return modelInfo.architecture.split('→').map((layer) => {
      const trimmedLayer = layer.trim();
      if (trimmedLayer.startsWith('[') && trimmedLayer.endsWith(']')) {
        return trimmedLayer
          .slice(1, -1)
          .split(',')
          .map((numStr) => Number(numStr.trim()));
      }
      const num = Number(trimmedLayer);
      return isNaN(num) ? trimmedLayer : num;
    });
  });

  public readonly weights: Signal<NeuralNetworkLayers | undefined> = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const weightsJson: string = network.get_weights();
    const weights = JSON.parse(weightsJson) as NeuralNetworkLayers;
    return weights;
  });

  public readonly testAll: Signal<TestResult[] | undefined> = computed(() => {
    const network = this.network();
    if (!network) {
      return undefined;
    }
    const testResultsJson: string = network.test_all();
    const testResults = JSON.parse(testResultsJson) as TestResult[];
    return testResults;
  });
}
