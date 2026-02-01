/**
 * Network visualization configuration module.
 *
 * Provides a fluent Builder pattern for configuring how neural networks
 * are visualized, with presets for common network types.
 *
 * @example
 * ```typescript
 * import { NetworkVisualizationBuilder } from './config';
 *
 * // Use a preset
 * const config = NetworkVisualizationBuilder
 *   .fromPreset('mnist')
 *   .withDebug(true)
 *   .build();
 *
 * // Or auto-configure based on architecture
 * const config = NetworkVisualizationBuilder
 *   .forNetwork(architecture)
 *   .build();
 *
 * // Or build from scratch
 * const config = new NetworkVisualizationBuilder()
 *   .withLayoutStrategy('column')
 *   .forLayer(0, { representation: 'heatmap', shape: [28, 28] })
 *   .withConnections('on-hover')
 *   .withZoom({ enabled: true, min: 0.5, max: 5 })
 *   .build();
 * ```
 */

// Types
export type {
  CanvasConfig,
  CanvasSizeStrategy,
  ClickBehavior,
  ConnectionConfig,
  ConnectionStrategy,
  HoverBehavior,
  InteractionConfig,
  LayerConfig,
  LayerRepresentation,
  LayerRule,
  LayoutConfig,
  LayoutStrategy,
  LODConfig,
  LODLevel,
  NetworkAnalysis,
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

// Default configs
export {
  DEFAULT_CANVAS_CONFIG,
  DEFAULT_CONNECTION_CONFIG,
  DEFAULT_INTERACTION_CONFIG,
  DEFAULT_LAYER_CONFIG,
  DEFAULT_LAYOUT_CONFIG,
  DEFAULT_LOD_CONFIG,
  DEFAULT_NEURON_SIZE_CONFIG,
  DEFAULT_RENDERING_CONFIG,
  DEFAULT_VISUALIZATION_CONFIG,
} from './visualization-config';

// Builder
export { NetworkVisualizationBuilder } from './visualization-builder';

// Presets
export { getPreset, getPresetNames, PRESETS } from './presets';
export type { PresetConfig } from './presets';
