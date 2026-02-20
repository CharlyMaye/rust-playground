/**
 * Configuration types for NetworkVisualizationBuilder.
 *
 * These types define all possible configuration options for
 * visualizing neural networks at different scales.
 */

// ============================================================================
// Layout Strategies
// ============================================================================

/**
 * Layout strategy for positioning neurons
 */
export type LayoutStrategy = 'column' | 'row' | 'grid' | 'spiral' | 'hierarchical';

/**
 * Spacing strategy for neurons within a layer
 */
export type SpacingStrategy = 'fixed' | 'adaptive' | 'proportional';

/**
 * Layout configuration
 */
export interface LayoutConfig {
  /** Primary layout strategy */
  readonly strategy: LayoutStrategy;
  /** Spacing between neurons */
  readonly spacing: SpacingStrategy;
  /** Minimum spacing in pixels (for adaptive) */
  readonly minSpacing?: number;
  /** Maximum spacing in pixels (for adaptive) */
  readonly maxSpacing?: number;
}

// ============================================================================
// Layer Representation
// ============================================================================

/**
 * How a layer of neurons should be represented
 */
export type LayerRepresentation =
  | 'neurons' // Individual neuron circles
  | 'sampled' // Subset of neurons
  | 'bar' // Single colored bar
  | 'heatmap' // 2D heatmap grid
  | 'feature-maps' // Multi-channel 2D heatmaps (CNN feature maps)
  | 'histogram' // Distribution histogram
  | 'stats' // Statistics only (min/max/avg)
  | 'collapsed'; // Single node representing the layer

/**
 * Configuration for a single layer's representation
 */
export interface LayerConfig {
  /** How to represent this layer */
  readonly representation: LayerRepresentation;
  /** For 'sampled': number of neurons to show */
  readonly sampleCount?: number;
  /** For 'heatmap' or 'grid': shape [rows, cols] or [rows, cols, channels] */
  readonly shape?: readonly number[];
  /** For 'feature-maps': number of channels */
  readonly channels?: number;
  /** For 'feature-maps': max channels to display */
  readonly maxChannels?: number;
  /** For 'histogram': number of bins */
  readonly bins?: number;
  /** Whether to show layer label */
  readonly showLabel?: boolean;
  /** Whether to show neuron values */
  readonly showValues?: boolean;
  /** Custom color scheme */
  readonly colorScheme?: 'default' | 'grayscale' | 'viridis' | 'coolwarm';
}

/**
 * Rule for automatically configuring layers based on size
 */
export interface LayerRule {
  /** Layer size threshold */
  readonly threshold: number;
  /** Configuration to apply */
  readonly config: LayerConfig;
}

// ============================================================================
// Connection Configuration
// ============================================================================

/**
 * Strategy for displaying connections
 */
export type ConnectionStrategy =
  | 'all' // Show all connections
  | 'strong' // Only connections above threshold
  | 'sampled' // Random sample
  | 'on-hover' // Show only when hovering a neuron
  | 'none'; // Hide all connections

/**
 * Connection display configuration
 */
export interface ConnectionConfig {
  /** Display strategy */
  readonly strategy: ConnectionStrategy;
  /** Weight threshold for 'strong' strategy (absolute value) */
  readonly threshold?: number;
  /** Max connections for 'sampled' strategy */
  readonly maxCount?: number;
  /** Base opacity for connections */
  readonly opacity?: number;
  /** Whether opacity scales with weight */
  readonly opacityByWeight?: boolean;
  /** Base stroke width */
  readonly strokeWidth?: number;
  /** Whether stroke width scales with weight */
  readonly widthByWeight?: boolean;
}

// ============================================================================
// Neuron Size Configuration
// ============================================================================

/**
 * Strategy for neuron sizing
 */
export type NeuronSizeStrategy = 'fixed' | 'adaptive' | 'by-activation';

/**
 * Neuron size configuration
 */
export interface NeuronSizeConfig {
  /** Sizing strategy */
  readonly strategy: NeuronSizeStrategy;
  /** Fixed size in natural pixels */
  readonly fixedSize?: number;
  /** Minimum size for adaptive */
  readonly minSize?: number;
  /** Maximum size for adaptive */
  readonly maxSize?: number;
}

// ============================================================================
// Canvas/Viewport Configuration
// ============================================================================

/**
 * Canvas sizing strategy
 */
export type CanvasSizeStrategy = 'fixed' | 'adaptive' | 'fill-container';

/**
 * Canvas configuration
 */
export interface CanvasConfig {
  /** Sizing strategy */
  readonly sizeStrategy: CanvasSizeStrategy;
  /** Fixed width (for 'fixed' strategy) */
  readonly width?: number;
  /** Fixed height (for 'fixed' strategy) */
  readonly height?: number;
  /** Aspect ratio (for 'adaptive') */
  readonly aspectRatio?: number | 'auto';
  /** Maximum width */
  readonly maxWidth?: number;
  /** Maximum height */
  readonly maxHeight?: number;
}

// ============================================================================
// Interaction Configuration
// ============================================================================

/**
 * Zoom configuration
 */
export interface ZoomConfig {
  readonly enabled: boolean;
  readonly min?: number;
  readonly max?: number;
  readonly initial?: number;
  readonly step?: number;
}

/**
 * Pan configuration
 */
export interface PanConfig {
  readonly enabled: boolean;
  /** Constrain to bounds */
  readonly constrained?: boolean;
}

/**
 * Hover behavior
 */
export type HoverBehavior = 'none' | 'highlight' | 'details' | 'connections';

/**
 * Click behavior
 */
export type ClickBehavior = 'none' | 'expand' | 'focus' | 'info';

/**
 * Interaction configuration
 */
export interface InteractionConfig {
  readonly zoom: ZoomConfig;
  readonly pan: PanConfig;
  readonly hover: HoverBehavior;
  readonly click: ClickBehavior;
}

// ============================================================================
// Level of Detail (LOD)
// ============================================================================

/**
 * LOD level configuration
 */
export interface LODLevel {
  /** Zoom range [min, max] for this level */
  readonly zoomRange: readonly [number, number];
  /** Layer configuration at this zoom level */
  readonly layerConfig?: Partial<LayerConfig>;
  /** Connection configuration at this zoom level */
  readonly connectionConfig?: Partial<ConnectionConfig>;
  /** Neuron size config at this zoom level */
  readonly neuronSizeConfig?: Partial<NeuronSizeConfig>;
}

/**
 * LOD configuration
 */
export interface LODConfig {
  readonly enabled: boolean;
  readonly levels: readonly LODLevel[];
}

// ============================================================================
// Render Configuration
// ============================================================================

/**
 * Renderer type
 */
export type RendererType = 'auto' | 'canvas2d' | 'webgl' | 'webgpu';

/**
 * Render configuration
 */
export interface RenderingConfig {
  readonly renderer: RendererType;
  readonly antialias: boolean;
  readonly debug: boolean;
  /** Max connections before auto-switching to WebGL */
  readonly webglThreshold?: number;
}

// ============================================================================
// Complete Visualization Configuration
// ============================================================================

/**
 * Complete configuration for network visualization.
 * This is the output of NetworkVisualizationBuilder.build()
 */
export interface VisualizationConfig {
  /** Layout configuration */
  readonly layout: LayoutConfig;

  /** Default layer configuration */
  readonly defaultLayerConfig: LayerConfig;

  /** Per-layer overrides (index → config) */
  readonly layerOverrides: ReadonlyMap<number, Partial<LayerConfig>>;

  /** Rules for auto-configuring large layers */
  readonly layerRules: readonly LayerRule[];

  /** Connection configuration */
  readonly connections: ConnectionConfig;

  /** Neuron size configuration */
  readonly neuronSize: NeuronSizeConfig;

  /** Canvas configuration */
  readonly canvas: CanvasConfig;

  /** Interaction configuration */
  readonly interaction: InteractionConfig;

  /** LOD configuration */
  readonly lod: LODConfig;

  /** Rendering configuration */
  readonly rendering: RenderingConfig;
}

// ============================================================================
// Preset Names
// ============================================================================

/**
 * Available preset names
 */
export type PresetName =
  | 'small-network'
  | 'medium-network'
  | 'mnist'
  | 'cifar'
  | 'large-mlp'
  | 'cnn'
  | 'architecture-only'
  | 'interactive'
  | 'presentation'
  | 'debug';

// ============================================================================
// Network Analysis
// ============================================================================

/**
 * Result of analyzing a network architecture
 */
export interface NetworkAnalysis {
  /** Total number of neurons */
  readonly totalNeurons: number;
  /** Total number of connections */
  readonly totalConnections: number;
  /** Layer sizes */
  readonly layerSizes: readonly number[];
  /** Largest layer info */
  readonly largestLayer: {
    readonly index: number;
    readonly size: number;
  };
  /** Suggested preset */
  readonly suggestedPreset: PresetName;
  /** Warnings about the network size */
  readonly warnings: readonly string[];
  /** Auto-generated configuration */
  readonly autoConfig: Partial<VisualizationConfig>;
}

// ============================================================================
// Default Configurations
// ============================================================================

export const DEFAULT_LAYOUT_CONFIG: LayoutConfig = {
  strategy: 'column',
  spacing: 'fixed',
};

export const DEFAULT_LAYER_CONFIG: LayerConfig = {
  representation: 'neurons',
  showLabel: true,
  showValues: true,
  colorScheme: 'default',
};

export const DEFAULT_CONNECTION_CONFIG: ConnectionConfig = {
  strategy: 'all',
  threshold: 0.1,
  maxCount: 10000,
  opacity: 0.6,
  opacityByWeight: true,
  strokeWidth: 1,
  widthByWeight: false,
};

export const DEFAULT_NEURON_SIZE_CONFIG: NeuronSizeConfig = {
  strategy: 'fixed',
  fixedSize: 40,
  minSize: 4,
  maxSize: 60,
};

export const DEFAULT_CANVAS_CONFIG: CanvasConfig = {
  sizeStrategy: 'fixed',
  width: 500,
  height: 280,
  aspectRatio: 'auto',
};

export const DEFAULT_INTERACTION_CONFIG: InteractionConfig = {
  zoom: { enabled: false },
  pan: { enabled: false },
  hover: 'none',
  click: 'none',
};

export const DEFAULT_LOD_CONFIG: LODConfig = {
  enabled: false,
  levels: [],
};

export const DEFAULT_RENDERING_CONFIG: RenderingConfig = {
  renderer: 'auto',
  antialias: true,
  debug: false,
  webglThreshold: 5000,
};

export const DEFAULT_VISUALIZATION_CONFIG: VisualizationConfig = {
  layout: DEFAULT_LAYOUT_CONFIG,
  defaultLayerConfig: DEFAULT_LAYER_CONFIG,
  layerOverrides: new Map(),
  layerRules: [],
  connections: DEFAULT_CONNECTION_CONFIG,
  neuronSize: DEFAULT_NEURON_SIZE_CONFIG,
  canvas: DEFAULT_CANVAS_CONFIG,
  interaction: DEFAULT_INTERACTION_CONFIG,
  lod: DEFAULT_LOD_CONFIG,
  rendering: DEFAULT_RENDERING_CONFIG,
};
