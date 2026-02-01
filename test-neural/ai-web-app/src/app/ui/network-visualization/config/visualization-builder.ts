/**
 * Fluent builder for network visualization configuration.
 *
 * Allows declarative, composable configuration of how neural networks
 * are visualized, from small XOR networks to large MNIST models.
 *
 * @example
 * ```typescript
 * const config = new NetworkVisualizationBuilder()
 *   .usePreset('mnist')
 *   .withConnections('on-hover')
 *   .forLayer(0, { representation: 'heatmap', shape: [28, 28] })
 *   .build();
 * ```
 */

import { NetworkArchitecture } from '../renderers';
import { PRESETS } from './presets';
import {
  CanvasConfig,
  ClickBehavior,
  ConnectionConfig,
  ConnectionStrategy,
  DEFAULT_CANVAS_CONFIG,
  DEFAULT_CONNECTION_CONFIG,
  DEFAULT_INTERACTION_CONFIG,
  DEFAULT_LAYER_CONFIG,
  DEFAULT_LAYOUT_CONFIG,
  DEFAULT_LOD_CONFIG,
  DEFAULT_NEURON_SIZE_CONFIG,
  DEFAULT_RENDERING_CONFIG,
  HoverBehavior,
  InteractionConfig,
  LayerConfig,
  LayerRepresentation,
  LayerRule,
  LayoutConfig,
  LayoutStrategy,
  LODConfig,
  LODLevel,
  NeuronSizeConfig,
  NeuronSizeStrategy,
  PanConfig,
  PresetName,
  RendererType,
  RenderingConfig,
  SpacingStrategy,
  VisualizationConfig,
  ZoomConfig,
} from './visualization-config';

/**
 * Fluent builder for creating VisualizationConfig.
 */
export class NetworkVisualizationBuilder {
  private layout: LayoutConfig = { ...DEFAULT_LAYOUT_CONFIG };
  private defaultLayerConfig: LayerConfig = { ...DEFAULT_LAYER_CONFIG };
  private layerOverrides: Map<number, Partial<LayerConfig>> = new Map();
  private layerRules: LayerRule[] = [];
  private connections: ConnectionConfig = { ...DEFAULT_CONNECTION_CONFIG };
  private neuronSize: NeuronSizeConfig = { ...DEFAULT_NEURON_SIZE_CONFIG };
  private canvas: CanvasConfig = { ...DEFAULT_CANVAS_CONFIG };
  private interaction: InteractionConfig = { ...DEFAULT_INTERACTION_CONFIG };
  private lod: LODConfig = { ...DEFAULT_LOD_CONFIG };
  private rendering: RenderingConfig = { ...DEFAULT_RENDERING_CONFIG };

  // ============================================================================
  // Static Factory Methods
  // ============================================================================

  /**
   * Create a new builder instance
   */
  static create(): NetworkVisualizationBuilder {
    return new NetworkVisualizationBuilder();
  }

  /**
   * Create a builder starting from a preset
   */
  static fromPreset(preset: PresetName): NetworkVisualizationBuilder {
    return new NetworkVisualizationBuilder().usePreset(preset);
  }

  /**
   * Analyze a network and create a builder with suggested configuration
   */
  static forNetwork(architecture: NetworkArchitecture): NetworkVisualizationBuilder {
    const builder = new NetworkVisualizationBuilder();
    return builder.analyzeAndConfigure(architecture);
  }

  // ============================================================================
  // Layout Configuration
  // ============================================================================

  /**
   * Set the layout strategy for positioning neurons
   */
  withLayoutStrategy(strategy: LayoutStrategy): this {
    this.layout = { ...this.layout, strategy };
    return this;
  }

  /**
   * Set the spacing strategy between neurons
   */
  withSpacing(strategy: SpacingStrategy, options?: { min?: number; max?: number }): this {
    this.layout = {
      ...this.layout,
      spacing: strategy,
      minSpacing: options?.min,
      maxSpacing: options?.max,
    };
    return this;
  }

  /**
   * Set complete layout configuration
   */
  withLayout(config: Partial<LayoutConfig>): this {
    this.layout = { ...this.layout, ...config };
    return this;
  }

  // ============================================================================
  // Layer Configuration
  // ============================================================================

  /**
   * Set default configuration for all layers
   */
  withDefaultLayerConfig(config: Partial<LayerConfig>): this {
    this.defaultLayerConfig = { ...this.defaultLayerConfig, ...config };
    return this;
  }

  /**
   * Configure a specific layer by index.
   * Use negative indices for layers from the end (-1 = last layer).
   */
  forLayer(index: number, config: Partial<LayerConfig>): this {
    const existing = this.layerOverrides.get(index) || {};
    this.layerOverrides.set(index, { ...existing, ...config });
    return this;
  }

  /**
   * Configure multiple layers at once
   */
  forLayers(indices: number[], config: Partial<LayerConfig>): this {
    for (const index of indices) {
      this.forLayer(index, config);
    }
    return this;
  }

  /**
   * Configure the input layer (first layer)
   */
  forInputLayer(config: Partial<LayerConfig>): this {
    return this.forLayer(0, config);
  }

  /**
   * Configure the output layer (last layer)
   */
  forOutputLayer(config: Partial<LayerConfig>): this {
    return this.forLayer(-1, config);
  }

  /**
   * Configure hidden layers (all except first and last)
   */
  forHiddenLayers(config: Partial<LayerConfig>): this {
    // This will be resolved at build time based on actual layer count
    this.layerRules.push({
      threshold: -1, // Special marker for hidden layers
      config: { ...DEFAULT_LAYER_CONFIG, ...config },
    });
    return this;
  }

  /**
   * Auto-configure layers above a certain size threshold
   */
  forLargeLayers(threshold: number, config: Partial<LayerConfig>): this {
    this.layerRules.push({
      threshold,
      config: { ...DEFAULT_LAYER_CONFIG, ...config },
    });
    return this;
  }

  /**
   * Set layer representation shorthand
   */
  withLayerRepresentation(representation: LayerRepresentation): this {
    this.defaultLayerConfig = { ...this.defaultLayerConfig, representation };
    return this;
  }

  // ============================================================================
  // Connection Configuration
  // ============================================================================

  /**
   * Set connection display strategy
   */
  withConnections(strategy: ConnectionStrategy): this {
    this.connections = { ...this.connections, strategy };
    return this;
  }

  /**
   * Set connection weight threshold (for 'strong' strategy)
   */
  withConnectionThreshold(threshold: number): this {
    this.connections = { ...this.connections, threshold };
    return this;
  }

  /**
   * Set maximum connections to display (for 'sampled' strategy)
   */
  withConnectionSampling(maxCount: number): this {
    this.connections = { ...this.connections, maxCount };
    return this;
  }

  /**
   * Set connection opacity
   */
  withConnectionOpacity(opacity: number, byWeight = true): this {
    this.connections = { ...this.connections, opacity, opacityByWeight: byWeight };
    return this;
  }

  /**
   * Set connection stroke width
   */
  withConnectionWidth(width: number, byWeight = false): this {
    this.connections = { ...this.connections, strokeWidth: width, widthByWeight: byWeight };
    return this;
  }

  /**
   * Set complete connection configuration
   */
  withConnectionConfig(config: Partial<ConnectionConfig>): this {
    this.connections = { ...this.connections, ...config };
    return this;
  }

  // ============================================================================
  // Neuron Size Configuration
  // ============================================================================

  /**
   * Set neuron sizing strategy
   */
  withNeuronSize(strategy: NeuronSizeStrategy): this {
    this.neuronSize = { ...this.neuronSize, strategy };
    return this;
  }

  /**
   * Set fixed neuron size
   */
  withFixedNeuronSize(size: number): this {
    this.neuronSize = { ...this.neuronSize, strategy: 'fixed', fixedSize: size };
    return this;
  }

  /**
   * Set neuron size bounds (for adaptive sizing)
   */
  withNeuronSizeBounds(min: number, max: number): this {
    this.neuronSize = { ...this.neuronSize, minSize: min, maxSize: max };
    return this;
  }

  /**
   * Set complete neuron size configuration
   */
  withNeuronSizeConfig(config: Partial<NeuronSizeConfig>): this {
    this.neuronSize = { ...this.neuronSize, ...config };
    return this;
  }

  // ============================================================================
  // Canvas Configuration
  // ============================================================================

  /**
   * Set fixed canvas size
   */
  withCanvasSize(width: number, height: number): this {
    this.canvas = { ...this.canvas, sizeStrategy: 'fixed', width, height };
    return this;
  }

  /**
   * Set canvas to fill container
   */
  withFillContainer(maxWidth?: number, maxHeight?: number): this {
    this.canvas = {
      ...this.canvas,
      sizeStrategy: 'fill-container',
      maxWidth,
      maxHeight,
    };
    return this;
  }

  /**
   * Set canvas aspect ratio
   */
  withAspectRatio(ratio: number | 'auto'): this {
    this.canvas = { ...this.canvas, aspectRatio: ratio };
    return this;
  }

  /**
   * Set complete canvas configuration
   */
  withCanvasConfig(config: Partial<CanvasConfig>): this {
    this.canvas = { ...this.canvas, ...config };
    return this;
  }

  // ============================================================================
  // Interaction Configuration
  // ============================================================================

  /**
   * Enable/configure zoom
   */
  withZoom(config: ZoomConfig | boolean): this {
    if (typeof config === 'boolean') {
      this.interaction = {
        ...this.interaction,
        zoom: { enabled: config },
      };
    } else {
      this.interaction = { ...this.interaction, zoom: config };
    }
    return this;
  }

  /**
   * Enable/configure pan
   */
  withPan(config: PanConfig | boolean): this {
    if (typeof config === 'boolean') {
      this.interaction = {
        ...this.interaction,
        pan: { enabled: config },
      };
    } else {
      this.interaction = { ...this.interaction, pan: config };
    }
    return this;
  }

  /**
   * Set hover behavior
   */
  withHover(behavior: HoverBehavior): this {
    this.interaction = { ...this.interaction, hover: behavior };
    return this;
  }

  /**
   * Set click behavior
   */
  withClick(behavior: ClickBehavior): this {
    this.interaction = { ...this.interaction, click: behavior };
    return this;
  }

  /**
   * Set complete interaction configuration
   */
  withInteraction(config: Partial<InteractionConfig>): this {
    this.interaction = { ...this.interaction, ...config };
    return this;
  }

  // ============================================================================
  // Level of Detail Configuration
  // ============================================================================

  /**
   * Enable LOD with levels
   */
  withLOD(levels: LODLevel[]): this {
    this.lod = { enabled: true, levels };
    return this;
  }

  /**
   * Add a single LOD level
   */
  addLODLevel(level: LODLevel): this {
    this.lod = {
      enabled: true,
      levels: [...this.lod.levels, level],
    };
    return this;
  }

  /**
   * Disable LOD
   */
  withoutLOD(): this {
    this.lod = { enabled: false, levels: [] };
    return this;
  }

  // ============================================================================
  // Rendering Configuration
  // ============================================================================

  /**
   * Set renderer type
   */
  withRenderer(renderer: RendererType): this {
    this.rendering = { ...this.rendering, renderer };
    return this;
  }

  /**
   * Enable/disable antialiasing
   */
  withAntialias(enabled: boolean): this {
    this.rendering = { ...this.rendering, antialias: enabled };
    return this;
  }

  /**
   * Enable/disable debug mode
   */
  withDebug(enabled: boolean): this {
    this.rendering = { ...this.rendering, debug: enabled };
    return this;
  }

  /**
   * Set threshold for auto-switching to WebGL
   */
  withWebGLThreshold(threshold: number): this {
    this.rendering = { ...this.rendering, webglThreshold: threshold };
    return this;
  }

  /**
   * Set complete rendering configuration
   */
  withRenderingConfig(config: Partial<RenderingConfig>): this {
    this.rendering = { ...this.rendering, ...config };
    return this;
  }

  // ============================================================================
  // Presets
  // ============================================================================

  /**
   * Apply a preset configuration
   */
  usePreset(preset: PresetName): this {
    const presetConfig = PRESETS[preset];
    if (!presetConfig) {
      console.warn(`Unknown preset: ${preset}, using defaults`);
      return this;
    }

    // Apply preset values
    if (presetConfig.layout) this.layout = { ...this.layout, ...presetConfig.layout };
    if (presetConfig.defaultLayerConfig) {
      this.defaultLayerConfig = { ...this.defaultLayerConfig, ...presetConfig.defaultLayerConfig };
    }
    if (presetConfig.layerRules) {
      this.layerRules = [...this.layerRules, ...presetConfig.layerRules];
    }
    if (presetConfig.connections) {
      this.connections = { ...this.connections, ...presetConfig.connections };
    }
    if (presetConfig.neuronSize) {
      this.neuronSize = { ...this.neuronSize, ...presetConfig.neuronSize };
    }
    if (presetConfig.canvas) {
      this.canvas = { ...this.canvas, ...presetConfig.canvas };
    }
    if (presetConfig.interaction) {
      this.interaction = { ...this.interaction, ...presetConfig.interaction };
    }
    if (presetConfig.lod) {
      this.lod = { ...this.lod, ...presetConfig.lod };
    }
    if (presetConfig.rendering) {
      this.rendering = { ...this.rendering, ...presetConfig.rendering };
    }

    // Apply layer overrides from preset
    if (presetConfig.layerOverrides) {
      for (const [index, config] of presetConfig.layerOverrides) {
        this.forLayer(index, config);
      }
    }

    return this;
  }

  // ============================================================================
  // Auto-Configuration
  // ============================================================================

  /**
   * Analyze network and auto-configure based on its characteristics
   */
  analyzeAndConfigure(architecture: NetworkArchitecture): this {
    // Extract layer sizes from architecture
    const layerSizes = [architecture.inputs.length, ...architecture.layers.map((l) => l.size)];

    const totalNeurons = layerSizes.reduce((sum, size) => sum + size, 0);
    const totalConnections = layerSizes
      .slice(0, -1)
      .reduce((sum, size, i) => sum + size * layerSizes[i + 1], 0);

    // Find largest layer
    const maxLayerSize = Math.max(...layerSizes);
    const maxLayerIndex = layerSizes.indexOf(maxLayerSize);

    // Choose preset based on network characteristics
    if (totalNeurons <= 20 && totalConnections <= 50) {
      this.usePreset('small-network');
    } else if (totalNeurons <= 100 && totalConnections <= 1000) {
      this.usePreset('medium-network');
    } else if (maxLayerSize === 784) {
      // MNIST-like
      this.usePreset('mnist');
    } else if (maxLayerSize >= 100) {
      this.usePreset('large-mlp');
    } else {
      this.usePreset('medium-network');
    }

    // Auto-configure large layers
    for (let i = 0; i < layerSizes.length; i++) {
      const size = layerSizes[i];

      if (size === 784) {
        // MNIST input layer - use heatmap
        this.forLayer(i, {
          representation: 'heatmap',
          shape: [28, 28],
          showValues: false,
        });
      } else if (size > 200) {
        // Very large layer - use bar
        this.forLayer(i, {
          representation: 'bar',
          showValues: false,
        });
      } else if (size > 50) {
        // Large layer - use sampled
        this.forLayer(i, {
          representation: 'sampled',
          sampleCount: 20,
          showValues: false,
        });
      }
    }

    // Output layer always shows neurons
    this.forOutputLayer({
      representation: 'neurons',
      showValues: true,
    });

    // Adjust connections based on count
    if (totalConnections > 50000) {
      this.withConnections('none');
    } else if (totalConnections > 10000) {
      this.withConnections('on-hover');
    } else if (totalConnections > 1000) {
      this.withConnections('strong').withConnectionThreshold(0.2);
    }

    return this;
  }

  // ============================================================================
  // Build
  // ============================================================================

  /**
   * Build the final configuration
   */
  build(): VisualizationConfig {
    return {
      layout: { ...this.layout },
      defaultLayerConfig: { ...this.defaultLayerConfig },
      layerOverrides: new Map(this.layerOverrides),
      layerRules: [...this.layerRules],
      connections: { ...this.connections },
      neuronSize: { ...this.neuronSize },
      canvas: { ...this.canvas },
      interaction: { ...this.interaction },
      lod: { ...this.lod },
      rendering: { ...this.rendering },
    };
  }

  /**
   * Clone this builder
   */
  clone(): NetworkVisualizationBuilder {
    const clone = new NetworkVisualizationBuilder();
    clone.layout = { ...this.layout };
    clone.defaultLayerConfig = { ...this.defaultLayerConfig };
    clone.layerOverrides = new Map(this.layerOverrides);
    clone.layerRules = [...this.layerRules];
    clone.connections = { ...this.connections };
    clone.neuronSize = { ...this.neuronSize };
    clone.canvas = { ...this.canvas };
    clone.interaction = { ...this.interaction };
    clone.lod = { ...this.lod };
    clone.rendering = { ...this.rendering };
    return clone;
  }
}
