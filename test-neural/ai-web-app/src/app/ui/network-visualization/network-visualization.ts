import {
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  computed,
  effect,
  input,
  signal,
  viewChild,
} from '@angular/core';
import {
  INetworkRenderer,
  LayerWeights,
  NetworkArchitecture,
  NetworkLayoutCalculator,
  NetworkRenderData,
  RendererFactory,
} from './renderers';

/**
 * High-performance neural network visualization component.
 *
 * Uses a modular rendering architecture that supports multiple backends:
 * - Canvas 2D (default, high performance)
 * - WebGL (future, GPU acceleration)
 * - WebGPU (future, next-gen graphics)
 *
 * Features:
 * - Handles 100K+ connections smoothly
 * - Reactive updates via Angular signals
 * - Respects CSS theme variables
 * - Extensible renderer architecture
 */
@Component({
  selector: 'app-network-visualization',
  imports: [],
  templateUrl: './network-visualization.html',
  styleUrl: './network-visualization.scss',
  host: {
    class: 'card',
  },
})
export class NetworkVisualization implements OnInit, OnDestroy {
  /** Canvas element reference (signal-based) */
  public readonly canvasRef = viewChild<ElementRef<HTMLCanvasElement>>('networkCanvas');

  /** Network architecture (layer sizes, activations, etc.) */
  public readonly architecture = input<NetworkArchitecture | null>(null);

  /** Network weights for connections */
  public readonly weights = input<LayerWeights[] | null>(null);

  /** Whether to show debug information */
  public readonly debug = input<boolean>(false);

  /** Current renderer instance */
  private renderer: INetworkRenderer | null = null;

  /** Layout calculator instance */
  private layoutCalculator: NetworkLayoutCalculator;

  /** Loading state */
  public readonly isLoading = signal(false);

  /** Computed render data */
  public readonly renderData = computed<NetworkRenderData | null>(() => {
    const arch = this.architecture();
    const wts = this.weights();

    if (!arch || !wts) {
      return null;
    }

    return this.layoutCalculator.calculateLayout(arch, wts);
  });

  constructor() {
    this.layoutCalculator = new NetworkLayoutCalculator();

    // Initialize renderer when canvas becomes available
    effect(() => {
      const canvasEl = this.canvasRef();
      if (canvasEl && !this.renderer) {
        this.initializeRenderer();
      }
    });

    // Auto-render when data changes
    effect(() => {
      const data = this.renderData();
      if (this.renderer && data) {
        this.renderer.render(data);
      } else if (this.renderer) {
        this.renderer.clear();
      }
    });
  }

  ngOnInit(): void {
    // Subscribe to data changes and re-render
    // Angular signals will automatically track dependencies
  }

  ngOnDestroy(): void {
    this.destroyRenderer();
  }

  /**
   * Initialize the renderer
   */
  private initializeRenderer(): void {
    const canvasEl = this.canvasRef();
    if (!canvasEl) {
      console.warn('Canvas element not found');
      return;
    }

    try {
      const canvas = canvasEl.nativeElement;

      // Create renderer with factory (auto-detects best available)
      this.renderer = RendererFactory.createAuto(canvas, {
        antialias: true,
        debug: this.debug(),
      });

      console.log(`Initialized ${this.renderer.getType()} renderer`);

      // Render initial data if available
      const data = this.renderData();
      if (data) {
        this.renderer.render(data);
      }
    } catch (error) {
      console.error('Failed to initialize renderer:', error);
    }
  }

  /**
   * Destroy the renderer and clean up resources
   */
  private destroyRenderer(): void {
    if (this.renderer) {
      this.renderer.destroy();
      this.renderer = null;
    }
  }

  /**
   * Handle canvas click (for future interactivity)
   */
  public onCanvasClick(event: MouseEvent): void {
    // Future: Implement click handling for neuron selection, etc.
    console.log('Canvas clicked at:', event.offsetX, event.offsetY);
  }
}
