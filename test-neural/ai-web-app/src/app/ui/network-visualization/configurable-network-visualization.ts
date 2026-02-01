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
  ConfigurableLayoutCalculator,
  ConfigurableRenderData,
  INetworkRenderer,
  LayerWeights,
  NetworkArchitecture,
  RendererPreference,
  Viewport,
} from './renderers';

import {
  DEFAULT_VISUALIZATION_CONFIG,
  PresetName,
  VisualizationConfig,
} from './config/visualization-config';

import { NetworkVisualizationBuilder } from './config/visualization-builder';
import { ConfigurableCanvas2DRenderer } from './renderers/configurable-canvas2d-renderer';
import { ConfigurableWebGLRenderer } from './renderers/configurable-webgl-renderer';

/**
 * Configurable neural network visualization component.
 *
 * Supports the Builder pattern for flexible configuration:
 * - Presets for common network types (XOR, Iris, MNIST, etc.)
 * - Custom layer representations (neurons, heatmap, bar, sampled)
 * - Connection filtering strategies
 * - Level of Detail (LOD) for zoom
 *
 * Architecture: Content-First Rendering
 * - Layout is calculated in natural coordinates
 * - Viewport calculates scale to fit display canvas
 * - Renderer applies uniform transformation
 *
 * Usage:
 * ```typescript
 * // With preset
 * <app-configurable-network [architecture]="arch" [weights]="weights" preset="mnist" />
 *
 * // With custom config
 * const config = NetworkVisualizationBuilder.create()
 *   .forLayer(0, { representation: 'heatmap', shape: [28, 28] })
 *   .forHiddenLayers({ representation: 'bar' })
 *   .forOutputLayer({ representation: 'neurons' })
 *   .withConnections('none')
 *   .build();
 *
 * <app-configurable-network [architecture]="arch" [weights]="weights" [config]="config" />
 * ```
 */
@Component({
  selector: 'app-configurable-network',
  imports: [],
  templateUrl: './configurable-network-visualization.html',
  styleUrl: './configurable-network-visualization.scss',
  host: { class: 'card' },
})
export class ConfigurableNetworkVisualization implements OnDestroy {
  // ============================================================================
  // Inputs (Signals)
  // ============================================================================

  /** Network architecture (layer sizes, activations) */
  readonly architecture = input<NetworkArchitecture | null>(null);

  /** Network weights for connections */
  readonly weights = input<LayerWeights[] | null>(null);

  /** Preset name - auto-configures based on network type */
  readonly preset = input<PresetName | null>(null);

  /** Custom configuration (overrides preset) */
  readonly config = input<VisualizationConfig | null>(null);

  /** Enable auto-configuration based on network analysis */
  readonly autoConfig = input<boolean>(true);

  /** Enable debug mode */
  readonly debug = input<boolean>(false);

  // ============================================================================
  // Internal State (Signals)
  // ============================================================================

  /** Canvas element reference */
  readonly canvasRef = viewChild<ElementRef<HTMLCanvasElement>>('networkCanvas');

  /** Current renderer instance */
  private renderer: INetworkRenderer | null = null;

  /** Current renderer type for display */
  readonly currentRendererType = signal<string>('none');

  /** Display dimensions (CSS pixels) */
  private readonly displayWidth = signal(500);
  private readonly displayHeight = signal(280);

  /** Aspect ratio for layout calculation */
  private readonly aspectRatio = computed(() => this.displayWidth() / this.displayHeight());

  // ============================================================================
  // Computed Configuration
  // ============================================================================

  /**
   * Resolved configuration from preset, custom config, or auto-config.
   * Priority: config > preset > auto-config > default
   */
  readonly resolvedConfig = computed<VisualizationConfig>(() => {
    // Highest priority: explicit config
    const customConfig = this.config();
    if (customConfig) {
      return customConfig;
    }

    // Second priority: preset
    const presetName = this.preset();
    if (presetName) {
      return NetworkVisualizationBuilder.fromPreset(presetName).build();
    }

    // Third priority: auto-configure based on network
    if (this.autoConfig()) {
      const arch = this.architecture();
      if (arch) {
        return this.buildAutoConfig(arch);
      }
    }

    // Default configuration
    return DEFAULT_VISUALIZATION_CONFIG;
  });

  /**
   * Build auto-configuration based on network architecture
   */
  private buildAutoConfig(arch: NetworkArchitecture): VisualizationConfig {
    // Analyze network to determine best configuration
    const builder = NetworkVisualizationBuilder.forNetwork(arch);

    // Add debug if enabled
    if (this.debug()) {
      builder.withDebug(true);
    }

    return builder.build();
  }

  // ============================================================================
  // Computed Layout Calculator
  // ============================================================================

  /**
   * Layout calculator configured for current settings
   */
  readonly layoutCalculator = computed(() => {
    const config = this.resolvedConfig();
    return new ConfigurableLayoutCalculator(config);
  });

  // ============================================================================
  // Computed Render Data
  // ============================================================================

  /**
   * Computed render data in natural coordinates.
   * Recalculates when architecture, weights, or config changes.
   */
  readonly renderData = computed<ConfigurableRenderData | null>(() => {
    const arch = this.architecture();
    const wts = this.weights();
    const calculator = this.layoutCalculator();

    if (!arch || !wts) return null;

    return calculator.calculateLayout(arch, wts);
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
      const config = this.resolvedConfig();

      untracked(() => {
        if (canvasEl) {
          this.initializeRenderer(canvasEl.nativeElement, config);
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

    // Effect: Update renderer when config changes
    effect(() => {
      const config = this.resolvedConfig();
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
   * Initialize the renderer based on configuration
   */
  private initializeRenderer(canvas: HTMLCanvasElement, config: VisualizationConfig): void {
    // Destroy existing renderer if present
    this.destroyRenderer();

    try {
      // Get display dimensions from canvas
      const rect = canvas.getBoundingClientRect();
      this.displayWidth.set(rect.width || 500);
      this.displayHeight.set(rect.height || 280);

      // Choose renderer based on config
      const rendererType = this.determineRenderer(config);

      const rendererConfig = {
        antialias: config.rendering.antialias,
        debug: config.rendering.debug || this.debug(),
      };

      if (rendererType === 'webgl') {
        this.renderer = new ConfigurableWebGLRenderer(canvas, rendererConfig);
      } else {
        this.renderer = new ConfigurableCanvas2DRenderer(canvas, rendererConfig);
      }

      this.currentRendererType.set(this.renderer.getType());

      if (this.debug()) {
        console.log(
          `[ConfigurableNetworkVisualization] Initialized ${this.renderer.getType()} renderer`,
        );
        console.log(`[ConfigurableNetworkVisualization] Config:`, config);
      }

      // Initial render if data available
      const data = this.renderData();
      const viewport = this.viewport();
      if (data && viewport) {
        this.renderer.render(data, viewport);
      }
    } catch (error) {
      console.error('[ConfigurableNetworkVisualization] Failed to initialize renderer:', error);
    }
  }

  /**
   * Determine best renderer based on config and data
   */
  private determineRenderer(config: VisualizationConfig): RendererPreference {
    if (config.rendering.renderer !== 'auto') {
      return config.rendering.renderer as RendererPreference;
    }

    // Auto-select based on connection count
    const data = this.renderData();
    if (data) {
      const connectionCount = data.connections.length;
      const threshold = config.rendering.webglThreshold ?? 5000;

      if (connectionCount > threshold) {
        return 'webgl';
      }
    }

    return 'canvas2d';
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
      console.log('[ConfigurableNetworkVisualization] Canvas clicked:', {
        x: event.offsetX,
        y: event.offsetY,
      });
    }
  }

  /**
   * Get layer types as a display string
   */
  getLayerTypes(): string {
    const data = this.renderData() as ConfigurableRenderData;
    if (!data?.layerElements) return '';
    return data.layerElements.map((e) => e.type).join(', ');
  }

  /**
   * Get current configuration analysis
   */
  getConfigAnalysis(): {
    preset: PresetName | null;
    layerRepresentations: string[];
    connectionStrategy: string;
    totalNeurons: number;
    totalConnections: number;
  } {
    const config = this.resolvedConfig();
    const data = this.renderData();

    return {
      preset: this.preset(),
      layerRepresentations:
        (data as ConfigurableRenderData)?.layerElements?.map((e) => e.type) ?? [],
      connectionStrategy: config.connections.strategy,
      totalNeurons: data?.neurons.length ?? 0,
      totalConnections: data?.connections.length ?? 0,
    };
  }
}
