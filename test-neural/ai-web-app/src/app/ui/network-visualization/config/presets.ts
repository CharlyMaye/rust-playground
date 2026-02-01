/**
 * Predefined visualization presets for common network types.
 *
 * Each preset provides a complete configuration optimized for
 * a specific type of network or use case.
 */

import {
  CanvasConfig,
  ConnectionConfig,
  InteractionConfig,
  LayerConfig,
  LayerRule,
  LayoutConfig,
  LODConfig,
  NeuronSizeConfig,
  PresetName,
  RenderingConfig,
} from './visualization-config';

/**
 * Partial configuration for a preset.
 * Only includes values that differ from defaults.
 */
export interface PresetConfig {
  readonly layout?: Partial<LayoutConfig>;
  readonly defaultLayerConfig?: Partial<LayerConfig>;
  readonly layerOverrides?: ReadonlyMap<number, Partial<LayerConfig>>;
  readonly layerRules?: readonly LayerRule[];
  readonly connections?: Partial<ConnectionConfig>;
  readonly neuronSize?: Partial<NeuronSizeConfig>;
  readonly canvas?: Partial<CanvasConfig>;
  readonly interaction?: Partial<InteractionConfig>;
  readonly lod?: Partial<LODConfig>;
  readonly rendering?: Partial<RenderingConfig>;
}

/**
 * All available presets
 */
export const PRESETS: Record<PresetName, PresetConfig> = {
  // ===========================================================================
  // Small Network (XOR, AND, OR)
  // ===========================================================================
  'small-network': {
    layout: {
      strategy: 'column',
      spacing: 'fixed',
    },
    defaultLayerConfig: {
      representation: 'neurons',
      showLabel: true,
      showValues: true,
    },
    connections: {
      strategy: 'all',
      opacity: 0.7,
      opacityByWeight: true,
    },
    neuronSize: {
      strategy: 'fixed',
      fixedSize: 40,
    },
    canvas: {
      sizeStrategy: 'fixed',
      width: 500,
      height: 280,
    },
    interaction: {
      zoom: { enabled: false },
      pan: { enabled: false },
      hover: 'none',
      click: 'none',
    },
  },

  // ===========================================================================
  // Medium Network (Iris, small MLPs)
  // ===========================================================================
  'medium-network': {
    layout: {
      strategy: 'column',
      spacing: 'fixed',
    },
    defaultLayerConfig: {
      representation: 'neurons',
      showLabel: true,
      showValues: true,
    },
    connections: {
      strategy: 'strong',
      threshold: 0.15,
      opacity: 0.5,
      opacityByWeight: true,
    },
    neuronSize: {
      strategy: 'fixed',
      fixedSize: 36,
    },
    canvas: {
      sizeStrategy: 'fixed',
      width: 600,
      height: 400,
    },
    interaction: {
      zoom: { enabled: true, min: 0.5, max: 3 },
      pan: { enabled: false },
      hover: 'highlight',
      click: 'none',
    },
  },

  // ===========================================================================
  // MNIST (784-128-64-10)
  // ===========================================================================
  mnist: {
    layout: {
      strategy: 'column',
      spacing: 'adaptive',
      minSpacing: 2,
      maxSpacing: 15,
    },
    defaultLayerConfig: {
      representation: 'bar',
      showLabel: true,
      showValues: false,
    },
    layerOverrides: new Map([
      // Input layer: 28x28 heatmap
      [
        0,
        {
          representation: 'heatmap',
          shape: [28, 28],
          showValues: false,
          colorScheme: 'grayscale',
        },
      ],
      // Output layer: individual neurons
      [
        -1,
        {
          representation: 'neurons',
          showValues: true,
        },
      ],
    ]),
    layerRules: [
      // Layers > 100 neurons: use bar
      {
        threshold: 100,
        config: {
          representation: 'bar',
          showValues: false,
        },
      },
    ],
    connections: {
      strategy: 'on-hover',
      opacity: 0.4,
      maxCount: 100,
    },
    neuronSize: {
      strategy: 'adaptive',
      minSize: 4,
      maxSize: 40,
    },
    canvas: {
      sizeStrategy: 'fixed',
      width: 800,
      height: 500,
    },
    interaction: {
      zoom: { enabled: true, min: 0.3, max: 5, initial: 1 },
      pan: { enabled: true, constrained: true },
      hover: 'connections',
      click: 'focus',
    },
    rendering: {
      renderer: 'webgl',
      antialias: true,
      debug: false,
    },
  },

  // ===========================================================================
  // CIFAR (32x32x3 input images)
  // ===========================================================================
  cifar: {
    layout: {
      strategy: 'column',
      spacing: 'adaptive',
    },
    defaultLayerConfig: {
      representation: 'bar',
      showLabel: true,
      showValues: false,
    },
    layerOverrides: new Map([
      // Input layer: 32x32 RGB heatmap
      [
        0,
        {
          representation: 'heatmap',
          shape: [32, 32, 3],
          showValues: false,
        },
      ],
      // Output layer
      [
        -1,
        {
          representation: 'neurons',
          showValues: true,
        },
      ],
    ]),
    connections: {
      strategy: 'none',
    },
    neuronSize: {
      strategy: 'adaptive',
      minSize: 2,
      maxSize: 30,
    },
    canvas: {
      sizeStrategy: 'fixed',
      width: 900,
      height: 600,
    },
    interaction: {
      zoom: { enabled: true, min: 0.2, max: 8 },
      pan: { enabled: true },
      hover: 'details',
      click: 'focus',
    },
    rendering: {
      renderer: 'webgl',
      antialias: true,
      debug: false,
    },
  },

  // ===========================================================================
  // Large MLP
  // ===========================================================================
  'large-mlp': {
    layout: {
      strategy: 'column',
      spacing: 'proportional',
    },
    defaultLayerConfig: {
      representation: 'bar',
      showLabel: true,
      showValues: false,
    },
    layerOverrides: new Map([
      // Output layer always shows neurons
      [
        -1,
        {
          representation: 'neurons',
          showValues: true,
        },
      ],
    ]),
    layerRules: [
      {
        threshold: 50,
        config: {
          representation: 'bar',
          showValues: false,
        },
      },
    ],
    connections: {
      strategy: 'none',
    },
    neuronSize: {
      strategy: 'adaptive',
      minSize: 4,
      maxSize: 30,
    },
    canvas: {
      sizeStrategy: 'adaptive',
      aspectRatio: 16 / 9,
      maxWidth: 1200,
      maxHeight: 800,
    },
    interaction: {
      zoom: { enabled: true, min: 0.5, max: 4 },
      pan: { enabled: true },
      hover: 'details',
      click: 'expand',
    },
    rendering: {
      renderer: 'webgl',
      antialias: true,
      debug: false,
    },
  },

  // ===========================================================================
  // CNN (Convolutional Neural Network)
  // ===========================================================================
  cnn: {
    layout: {
      strategy: 'hierarchical',
      spacing: 'adaptive',
    },
    defaultLayerConfig: {
      representation: 'heatmap',
      showLabel: true,
      showValues: false,
    },
    layerOverrides: new Map([
      [
        -1,
        {
          representation: 'neurons',
          showValues: true,
        },
      ],
    ]),
    connections: {
      strategy: 'none',
    },
    neuronSize: {
      strategy: 'adaptive',
      minSize: 2,
      maxSize: 20,
    },
    canvas: {
      sizeStrategy: 'adaptive',
      aspectRatio: 'auto',
      maxWidth: 1400,
      maxHeight: 900,
    },
    interaction: {
      zoom: { enabled: true, min: 0.2, max: 10 },
      pan: { enabled: true },
      hover: 'details',
      click: 'focus',
    },
    rendering: {
      renderer: 'webgl',
      antialias: true,
      debug: false,
    },
  },

  // ===========================================================================
  // Architecture Only (very large networks)
  // ===========================================================================
  'architecture-only': {
    layout: {
      strategy: 'column',
      spacing: 'fixed',
    },
    defaultLayerConfig: {
      representation: 'collapsed',
      showLabel: true,
      showValues: false,
    },
    connections: {
      strategy: 'none',
    },
    neuronSize: {
      strategy: 'fixed',
      fixedSize: 60,
    },
    canvas: {
      sizeStrategy: 'fixed',
      width: 800,
      height: 300,
    },
    interaction: {
      zoom: { enabled: false },
      pan: { enabled: false },
      hover: 'details',
      click: 'expand',
    },
    rendering: {
      renderer: 'canvas2d',
      antialias: true,
      debug: false,
    },
  },

  // ===========================================================================
  // Interactive (full exploration)
  // ===========================================================================
  interactive: {
    layout: {
      strategy: 'column',
      spacing: 'fixed',
    },
    defaultLayerConfig: {
      representation: 'neurons',
      showLabel: true,
      showValues: true,
    },
    layerRules: [
      {
        threshold: 100,
        config: {
          representation: 'sampled',
          sampleCount: 30,
          showValues: false,
        },
      },
    ],
    connections: {
      strategy: 'on-hover',
      opacity: 0.6,
      opacityByWeight: true,
    },
    neuronSize: {
      strategy: 'fixed',
      fixedSize: 32,
    },
    canvas: {
      sizeStrategy: 'fill-container',
      maxWidth: 1600,
      maxHeight: 1000,
    },
    interaction: {
      zoom: { enabled: true, min: 0.1, max: 10, step: 0.1 },
      pan: { enabled: true, constrained: false },
      hover: 'connections',
      click: 'info',
    },
    lod: {
      enabled: true,
      levels: [
        {
          zoomRange: [0, 0.5],
          layerConfig: { representation: 'collapsed' },
          connectionConfig: { strategy: 'none' },
        },
        {
          zoomRange: [0.5, 1.5],
          layerConfig: { representation: 'bar' },
          connectionConfig: { strategy: 'none' },
        },
        {
          zoomRange: [1.5, 10],
          layerConfig: { representation: 'neurons' },
          connectionConfig: { strategy: 'on-hover' },
        },
      ],
    },
    rendering: {
      renderer: 'auto',
      antialias: true,
      debug: false,
    },
  },

  // ===========================================================================
  // Presentation (demos, slides)
  // ===========================================================================
  presentation: {
    layout: {
      strategy: 'column',
      spacing: 'fixed',
    },
    defaultLayerConfig: {
      representation: 'neurons',
      showLabel: true,
      showValues: true,
    },
    layerRules: [
      {
        threshold: 30,
        config: {
          representation: 'sampled',
          sampleCount: 15,
          showValues: true,
        },
      },
    ],
    connections: {
      strategy: 'strong',
      threshold: 0.2,
      opacity: 0.7,
      opacityByWeight: true,
    },
    neuronSize: {
      strategy: 'fixed',
      fixedSize: 44,
    },
    canvas: {
      sizeStrategy: 'fixed',
      width: 1000,
      height: 600,
    },
    interaction: {
      zoom: { enabled: false },
      pan: { enabled: false },
      hover: 'highlight',
      click: 'none',
    },
    rendering: {
      renderer: 'canvas2d',
      antialias: true,
      debug: false,
    },
  },

  // ===========================================================================
  // Debug (development)
  // ===========================================================================
  debug: {
    layout: {
      strategy: 'column',
      spacing: 'fixed',
    },
    defaultLayerConfig: {
      representation: 'neurons',
      showLabel: true,
      showValues: true,
    },
    connections: {
      strategy: 'all',
      opacity: 0.8,
      opacityByWeight: true,
    },
    neuronSize: {
      strategy: 'fixed',
      fixedSize: 40,
    },
    canvas: {
      sizeStrategy: 'fixed',
      width: 800,
      height: 500,
    },
    interaction: {
      zoom: { enabled: true, min: 0.1, max: 20 },
      pan: { enabled: true },
      hover: 'details',
      click: 'info',
    },
    rendering: {
      renderer: 'canvas2d',
      antialias: true,
      debug: true,
    },
  },
};

/**
 * Get a preset by name
 */
export function getPreset(name: PresetName): PresetConfig {
  return PRESETS[name];
}

/**
 * List all available preset names
 */
export function getPresetNames(): PresetName[] {
  return Object.keys(PRESETS) as PresetName[];
}
