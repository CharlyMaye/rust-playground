import {
  Component,
  computed,
  effect,
  ElementRef,
  input,
  OnDestroy,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import {
  calculateViewport,
  INetworkRenderer,
  LayerWeights,
  NetworkArchitecture,
  NetworkLayoutCalculator,
  NetworkRenderData,
  RendererFactory,
  RendererPreference,
  Viewport,
} from './renderers';

/**
 * High-performance neural network visualization component.
 *
 * Architecture: Content-First Rendering
 * - Layout is calculated in natural coordinates (fixed dimensions for readability)
 * - Viewport calculates scale to fit display canvas
 * - Renderer applies uniform transformation
 *
 * Angular Patterns:
 * - Signals for reactive state management
 * - computed() for derived state
 * - effect() with untracked() for side effects
 */
@Component({
  selector: 'app-network-visualization',
  imports: [],
  templateUrl: './network-visualization.html',
  styleUrl: './network-visualization.scss',
  host: { class: 'card' },
})
export class NetworkVisualization implements OnDestroy {
  // ============================================================================
  // Inputs (Signals)
  // ============================================================================

  /** Network architecture (layer sizes, activations) */
  readonly architecture = input<NetworkArchitecture | null>(null);

  /** Network weights for connections */
  readonly weights = input<LayerWeights[] | null>(null);

  /** Preferred renderer type */
  readonly rendererType = input<RendererPreference>('canvas2d');

  /** Enable debug mode */
  readonly debug = input<boolean>(false);

  // ============================================================================
  // Internal State (Signals)
  // ============================================================================

  /** Canvas element reference */
  readonly canvasRef = viewChild<ElementRef<HTMLCanvasElement>>('networkCanvas');

  /** Current renderer instance */
  private renderer: INetworkRenderer | null = null;

  /** Layout calculator (updated with aspect ratio) */
  private readonly layoutCalculator = new NetworkLayoutCalculator();

  /** Current renderer type for display */
  readonly currentRendererType = signal<string>('none');

  /** Display dimensions (CSS pixels) */
  private readonly displayWidth = signal(500);
  private readonly displayHeight = signal(280);

  /** Aspect ratio for layout calculation */
  private readonly aspectRatio = computed(() => this.displayWidth() / this.displayHeight());

  // ============================================================================
  // Computed State (Derived)
  // ============================================================================

  /**
   * Computed render data in natural coordinates.
   * Recalculates when architecture, weights, or aspect ratio changes.
   */
  readonly renderData = computed<NetworkRenderData | null>(() => {
    const arch = this.architecture();
    const wts = this.weights();
    const ratio = this.aspectRatio();

    if (!arch || !wts) return null;

    // Update layout calculator with current aspect ratio
    this.layoutCalculator.updateDimensions({ targetAspectRatio: ratio });

    return this.layoutCalculator.calculateLayout(arch, wts);
  });

  /**
   * Computed viewport with scale and offset.
   * Recalculates when render data or display size changes.
   */
  readonly viewport = computed<Viewport | null>(() => {
    const data = this.renderData();
    if (!data) return null;

    return calculateViewport(data.naturalBounds, this.displayWidth(), this.displayHeight());
  });

  // ============================================================================
  // Constructor with Effects
  // ============================================================================

  constructor() {
    // Effect: Initialize renderer when canvas becomes available
    effect(() => {
      const canvasEl = this.canvasRef();
      const type = this.rendererType();

      untracked(() => {
        if (canvasEl && !this.renderer) {
          this.initializeRenderer(canvasEl.nativeElement, type);
        }
      });
    });

    // Effect: Render when data or viewport changes
    effect(() => {
      const data = this.renderData();
      const viewport = this.viewport();

      untracked(() => {
        if (this.renderer && data && viewport) {
          this.renderer.render(data, viewport);
        } else if (this.renderer) {
          this.renderer.clear();
        }
      });
    });

    // Effect: Update debug config when debug input changes
    effect(() => {
      const isDebug = this.debug();

      untracked(() => {
        if (this.renderer) {
          this.renderer.updateConfig({ debug: isDebug });
          // Re-render with new config
          const data = this.renderData();
          const viewport = this.viewport();
          if (data && viewport) {
            this.renderer.render(data, viewport);
          }
        }
      });
    });
  }

  ngOnDestroy(): void {
    this.destroyRenderer();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  /**
   * Initialize the renderer with the specified type
   */
  private initializeRenderer(canvas: HTMLCanvasElement, type: RendererPreference): void {
    try {
      // Get display dimensions from canvas
      const rect = canvas.getBoundingClientRect();
      this.displayWidth.set(rect.width);
      this.displayHeight.set(rect.height);

      // Create renderer
      this.renderer = RendererFactory.create(canvas, [type], {
        antialias: true,
        debug: this.debug(),
      });

      this.currentRendererType.set(this.renderer.getType());

      if (this.debug()) {
        console.log(`[NetworkVisualization] Initialized ${this.renderer.getType()} renderer`);
      }

      // Initial render if data available
      const data = this.renderData();
      const viewport = this.viewport();
      if (data && viewport) {
        this.renderer.render(data, viewport);
      }
    } catch (error) {
      console.error('[NetworkVisualization] Failed to initialize renderer:', error);
    }
  }

  /**
   * Destroy renderer and clean up resources
   */
  private destroyRenderer(): void {
    if (this.renderer) {
      this.renderer.destroy();
      this.renderer = null;
    }
  }

  // ============================================================================
  // Public Methods
  // ============================================================================

  /**
   * Handle canvas click (for future interactivity)
   */
  onCanvasClick(event: MouseEvent): void {
    if (this.debug()) {
      console.log('[NetworkVisualization] Canvas clicked:', {
        x: event.offsetX,
        y: event.offsetY,
      });
    }
  }
}
