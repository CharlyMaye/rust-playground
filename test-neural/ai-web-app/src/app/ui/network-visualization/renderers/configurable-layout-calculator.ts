import { Bounds, Connection, CssColor, Label, NetworkRenderData, Neuron } from './types';

import {
  LayerConfig,
  LayerRepresentation,
  VisualizationConfig,
} from '../config/visualization-config';

// ============================================================================
// Types
// ============================================================================

/**
 * Layer information extracted from activations
 */
export interface LayerInfo {
  readonly size: number;
  readonly activations: readonly number[];
  readonly activationFunction: string;
  readonly isOutput: boolean;
}

/**
 * Network architecture for layout calculation
 */
export interface NetworkArchitecture {
  readonly inputs: readonly number[];
  readonly layers: readonly LayerInfo[];
}

/**
 * Weight matrix for a layer
 */
export interface LayerWeights {
  readonly weights: number[] | number[][];
}

/**
 * Element representing a layer in the layout (not just neurons)
 */
export interface LayerElement {
  readonly type: LayerRepresentation;
  readonly layerIndex: number;
  readonly position: { x: number; y: number };
  readonly width: number;
  readonly height: number;
  readonly neurons?: readonly Neuron[];
  readonly gridData?: GridData;
  readonly barData?: BarData;
  readonly statsData?: StatsData;
  readonly featureMapsData?: FeatureMapsData;
}

export interface GridData {
  readonly rows: number;
  readonly cols: number;
  readonly cellSize: number;
  readonly values: readonly number[];
  readonly colors: readonly string[];
}

export interface BarData {
  readonly width: number;
  readonly height: number;
  readonly min: number;
  readonly max: number;
  readonly mean: number;
  readonly colorGradient: readonly string[];
}

export interface StatsData {
  readonly count: number;
  readonly min: number;
  readonly max: number;
  readonly mean: number;
  readonly std: number;
}

/**
 * Data for a multi-channel feature maps visualization (CNN layers).
 * Each channel is a 2D heatmap laid out in a grid.
 */
export interface FeatureMapsData {
  /** Number of channels */
  readonly channels: number;
  /** How many channels are actually displayed */
  readonly displayChannels: number;
  /** Rows in each individual feature map */
  readonly mapRows: number;
  /** Cols in each individual feature map */
  readonly mapCols: number;
  /** Cell size for each feature map pixel */
  readonly cellSize: number;
  /** Grid layout: columns of feature maps */
  readonly gridCols: number;
  /** Grid layout: rows of feature maps */
  readonly gridRows: number;
  /** Gap between feature maps in natural pixels */
  readonly gap: number;
  /** Per-channel grid data (colors + values) */
  readonly maps: readonly GridData[];
  /** Whether some channels were truncated */
  readonly truncated: boolean;
}

/**
 * Enhanced render data with layer elements
 */
export interface ConfigurableRenderData extends NetworkRenderData {
  readonly layerElements: readonly LayerElement[];
}

// ============================================================================
// Configurable Layout Calculator
// ============================================================================

/**
 * Configurable Network Layout Calculator
 *
 * Uses VisualizationConfig to determine how each layer should be represented.
 * Supports: neurons, sampled, bar, heatmap, histogram, stats, collapsed.
 */
export class ConfigurableLayoutCalculator {
  private readonly config: VisualizationConfig;

  // Calculated dimensions based on config
  private readonly baseDimensions = {
    neuronDiameter: 40,
    neuronPaddingY: 10,
    layerSpacing: 150,
    margin: 60,
    labelFontSize: 12,
    neuronFontSize: 12,
    labelOffsetY: 25,
  };

  constructor(config: VisualizationConfig) {
    this.config = config;
    this.applyConfigToDimensions();
  }

  private applyConfigToDimensions(): void {
    // Adjust dimensions based on neuron size strategy
    if (this.config.neuronSize.strategy === 'fixed' && this.config.neuronSize.fixedSize) {
      this.baseDimensions.neuronDiameter = this.config.neuronSize.fixedSize;
    }

    // Adjust based on layout spacing
    if (this.config.layout.spacing === 'adaptive') {
      // Will be calculated per-layer
    }
  }

  // ============================================================================
  // Main Entry Point
  // ============================================================================

  calculateLayout(
    architecture: NetworkArchitecture,
    weights: readonly LayerWeights[],
  ): ConfigurableRenderData {
    const layerSizes = this.getLayerSizes(architecture);
    const layerConfigs = this.resolveLayerConfigs(layerSizes);

    // Calculate bounds for each layer based on representation
    const layerBounds = this.calculateLayerBounds(layerSizes, layerConfigs);

    // Calculate total natural bounds
    const naturalBounds = this.calculateNaturalBounds(layerBounds);

    // Position layers
    const layerPositions = this.calculateLayerPositions(layerBounds, naturalBounds);

    // Build layer elements
    const layerElements = this.buildLayerElements(
      architecture,
      layerConfigs,
      layerPositions,
      layerBounds,
    );

    // Build neurons (for backward compatibility)
    const neurons = this.extractNeuronsFromElements(layerElements);

    // Build connections based on config
    const connections = this.buildConnections(weights, layerSizes, layerPositions, layerElements);

    // Build labels
    const labels = this.buildLabels(
      architecture,
      layerConfigs,
      layerPositions,
      naturalBounds.height,
      layerElements,
    );

    return {
      neurons,
      connections,
      labels,
      naturalBounds,
      layerElements,
    };
  }

  // ============================================================================
  // Layer Configuration Resolution
  // ============================================================================

  private getLayerSizes(architecture: NetworkArchitecture): readonly number[] {
    return [architecture.inputs.length, ...architecture.layers.map((l) => l.size)];
  }

  private resolveLayerConfigs(layerSizes: readonly number[]): readonly LayerConfig[] {
    return layerSizes.map((size, index) => {
      // Check for direct override
      const override = this.config.layerOverrides.get(index);
      if (override) {
        return { ...this.config.defaultLayerConfig, ...override };
      }

      // Check layer rules
      for (const rule of this.config.layerRules) {
        if (size >= rule.threshold) {
          return { ...this.config.defaultLayerConfig, ...rule.config };
        }
      }

      return this.config.defaultLayerConfig;
    });
  }

  // ============================================================================
  // Bounds Calculation
  // ============================================================================

  private calculateLayerBounds(
    layerSizes: readonly number[],
    layerConfigs: readonly LayerConfig[],
  ): readonly { width: number; height: number }[] {
    return layerSizes.map((size, index) => {
      const config = layerConfigs[index];
      return this.calculateSingleLayerBounds(size, config);
    });
  }

  private calculateSingleLayerBounds(
    size: number,
    config: LayerConfig,
  ): { width: number; height: number } {
    const { neuronDiameter, neuronPaddingY } = this.baseDimensions;

    switch (config.representation) {
      case 'neurons':
        return {
          width: neuronDiameter,
          height: size * neuronDiameter + (size - 1) * neuronPaddingY,
        };

      case 'sampled': {
        const sampleCount = config.sampleCount ?? Math.min(10, size);
        const displayCount = Math.min(sampleCount, size) + 1; // +1 for ellipsis
        return {
          width: neuronDiameter,
          height: displayCount * neuronDiameter + (displayCount - 1) * neuronPaddingY,
        };
      }

      case 'bar':
        return {
          width: neuronDiameter * 2,
          height: Math.min(200, size / 2),
        };

      case 'heatmap': {
        const shape = config.shape ?? this.inferShape(size);
        const [rows, cols] = shape;
        const cellSize = this.calculateHeatmapCellSize(rows, cols);
        return {
          width: cols * cellSize,
          height: rows * cellSize,
        };
      }

      case 'feature-maps': {
        const fmLayout = this.calculateFeatureMapsLayout(size, config);
        return {
          width: fmLayout.totalWidth,
          height: fmLayout.totalHeight,
        };
      }

      case 'histogram':
        return {
          width: 80,
          height: 60,
        };

      case 'stats':
        return {
          width: 80,
          height: 60,
        };

      case 'collapsed':
        return {
          width: 0,
          height: 0,
        };

      default:
        return {
          width: neuronDiameter,
          height: size * neuronDiameter + (size - 1) * neuronPaddingY,
        };
    }
  }

  private inferShape(size: number): readonly number[] {
    // Common image sizes
    if (size === 784) return [28, 28];
    if (size === 3072) return [32, 32, 3];
    if (size === 1024) return [32, 32];

    // Try to find a square-ish shape
    const sqrt = Math.sqrt(size);
    if (Number.isInteger(sqrt)) {
      return [sqrt, sqrt];
    }

    // Find closest factors
    for (let i = Math.floor(sqrt); i >= 1; i--) {
      if (size % i === 0) {
        return [i, size / i];
      }
    }

    return [1, size];
  }

  private calculateHeatmapCellSize(rows: number, cols: number): number {
    const maxDimension = Math.max(rows, cols);

    // Target size proportional to feature maps (not a tiny thumbnail)
    const targetMaxSize = 300;
    const cellSize = Math.max(4, Math.floor(targetMaxSize / maxDimension));

    return cellSize;
  }

  private calculateNaturalBounds(
    layerBounds: readonly { width: number; height: number }[],
  ): Bounds {
    const { layerSpacing, margin, labelOffsetY } = this.baseDimensions;

    // Filter out zero-size layers (collapsed)
    const visibleBounds = layerBounds.filter((b) => b.width > 0 && b.height > 0);
    if (visibleBounds.length === 0) {
      return { width: margin * 2, height: margin * 2 };
    }

    // Use smaller spacing for compact layouts
    const effectiveSpacing = visibleBounds.length === 1 ? 0 : layerSpacing / 2;
    const effectiveMargin = margin / 2;

    if (this.config.layout.strategy === 'row') {
      // Vertical stacking: layers go top-to-bottom
      const maxWidth = Math.max(...visibleBounds.map((b) => b.width));
      const totalHeight =
        visibleBounds.reduce((sum, b) => sum + b.height, 0) +
        (visibleBounds.length - 1) * effectiveSpacing;
      return {
        width: maxWidth + effectiveMargin * 2,
        height: totalHeight + effectiveMargin * 2 + labelOffsetY,
      };
    }

    // Horizontal column layout (default)
    const maxHeight = Math.max(...visibleBounds.map((b) => b.height));
    const totalWidth =
      visibleBounds.reduce((sum, b) => sum + b.width, 0) +
      (visibleBounds.length - 1) * effectiveSpacing;

    return {
      width: totalWidth + effectiveMargin * 2,
      height: maxHeight + effectiveMargin * 2 + labelOffsetY,
    };
  }

  // ============================================================================
  // Layer Positioning
  // ============================================================================

  private calculateLayerPositions(
    layerBounds: readonly { width: number; height: number }[],
    naturalBounds: Bounds,
  ): readonly { x: number; centerY: number }[] {
    const { layerSpacing, margin, labelOffsetY } = this.baseDimensions;

    // Filter indices of visible layers
    const visibleIndices = layerBounds
      .map((b, i) => ({ b, i }))
      .filter(({ b }) => b.width > 0 && b.height > 0);
    const effectiveSpacing = visibleIndices.length === 1 ? 0 : layerSpacing / 2;
    const effectiveMargin = margin / 2;

    if (this.config.layout.strategy === 'row') {
      // Vertical stacking: layers go top-to-bottom, centered horizontally
      const centerX = naturalBounds.width / 2;
      const positions: { x: number; centerY: number }[] = [];
      let y = effectiveMargin + labelOffsetY;

      for (let i = 0; i < layerBounds.length; i++) {
        const bounds = layerBounds[i];
        if (bounds.width === 0 && bounds.height === 0) {
          // Collapsed layer - position at same y as next visible
          positions.push({ x: centerX, centerY: y });
        } else {
          positions.push({
            x: centerX,
            centerY: y + bounds.height / 2,
          });
          y += bounds.height + effectiveSpacing;
        }
      }

      return positions;
    }

    // Horizontal column layout (default)
    const centerY = naturalBounds.height / 2;
    const positions: { x: number; centerY: number }[] = [];
    let x = effectiveMargin;

    for (const bounds of layerBounds) {
      if (bounds.width === 0 && bounds.height === 0) {
        positions.push({ x, centerY });
      } else {
        positions.push({
          x: x + bounds.width / 2,
          centerY,
        });
        x += bounds.width + effectiveSpacing;
      }
    }

    return positions;
  }

  // ============================================================================
  // Layer Element Building
  // ============================================================================

  private buildLayerElements(
    architecture: NetworkArchitecture,
    layerConfigs: readonly LayerConfig[],
    layerPositions: readonly { x: number; centerY: number }[],
    layerBounds: readonly { width: number; height: number }[],
  ): readonly LayerElement[] {
    const elements: LayerElement[] = [];
    const allActivations = this.getAllActivations(architecture);

    for (let i = 0; i < layerConfigs.length; i++) {
      const config = layerConfigs[i];
      const position = layerPositions[i];
      const bounds = layerBounds[i];
      const activations = allActivations[i];
      const isInput = i === 0;
      const isOutput = i === layerConfigs.length - 1;

      const element = this.buildSingleLayerElement(
        i,
        config,
        position,
        bounds,
        activations,
        isInput,
        isOutput,
        architecture,
      );

      elements.push(element);
    }

    return elements;
  }

  private getAllActivations(architecture: NetworkArchitecture): readonly (readonly number[])[] {
    return [architecture.inputs, ...architecture.layers.map((l) => l.activations)];
  }

  private buildSingleLayerElement(
    layerIndex: number,
    config: LayerConfig,
    position: { x: number; centerY: number },
    bounds: { width: number; height: number },
    activations: readonly number[],
    isInput: boolean,
    isOutput: boolean,
    architecture: NetworkArchitecture,
  ): LayerElement {
    const baseElement = {
      type: config.representation,
      layerIndex,
      position: { x: position.x, y: position.centerY },
      width: bounds.width,
      height: bounds.height,
    };

    switch (config.representation) {
      case 'neurons':
        return {
          ...baseElement,
          neurons: this.buildNeuronsForLayer(
            layerIndex,
            position,
            bounds,
            activations,
            isInput,
            isOutput,
            architecture,
            config,
          ),
        };

      case 'sampled':
        return {
          ...baseElement,
          neurons: this.buildSampledNeuronsForLayer(
            layerIndex,
            position,
            activations,
            config,
            isInput,
            isOutput,
            architecture,
          ),
        };

      case 'heatmap':
        return {
          ...baseElement,
          gridData: this.buildGridData(activations, config),
        };

      case 'feature-maps':
        return {
          ...baseElement,
          featureMapsData: this.buildFeatureMapsData(activations, config),
        };

      case 'bar':
        return {
          ...baseElement,
          barData: this.buildBarData(activations, bounds),
        };

      case 'stats':
        return {
          ...baseElement,
          statsData: this.buildStatsData(activations),
        };

      case 'collapsed':
        // Collapsed layers are placeholder elements - no visual rendering
        return baseElement;

      default:
        return baseElement;
    }
  }

  // ============================================================================
  // Neuron Building
  // ============================================================================

  private buildNeuronsForLayer(
    layerIndex: number,
    position: { x: number; centerY: number },
    bounds: { width: number; height: number },
    activations: readonly number[],
    isInput: boolean,
    isOutput: boolean,
    architecture: NetworkArchitecture,
    config: LayerConfig,
  ): readonly Neuron[] {
    const { neuronDiameter, neuronPaddingY } = this.baseDimensions;
    const neurons: Neuron[] = [];
    const radius = neuronDiameter / 2;

    const startY = position.centerY - bounds.height / 2 + radius;

    for (let i = 0; i < activations.length; i++) {
      const y = startY + i * (neuronDiameter + neuronPaddingY);
      const value = activations[i];

      let fill: CssColor;
      let label: string | undefined;
      let labelPosition: { x: number; y: number } | undefined;
      let labelAlign: 'left' | 'right' | 'center' | undefined;

      if (isInput) {
        fill = this.getInputColor(value);
        label = this.getInputLabel(i);
        labelPosition = { x: position.x - radius - 10, y };
        labelAlign = 'right';
      } else if (isOutput) {
        const activationFunc = architecture.layers[layerIndex - 1]?.activationFunction ?? 'linear';
        fill = this.getNeuronColor(value, activationFunc, true);
        label = activations.length > 1 ? `Out ${i}` : 'Out';
        labelPosition = { x: position.x + radius + 10, y };
        labelAlign = 'left';
      } else {
        const activationFunc = architecture.layers[layerIndex - 1]?.activationFunction ?? 'relu';
        fill = this.getNeuronColor(value, activationFunc, false);
      }

      const neuronRadius = isOutput ? radius * 1.2 : radius;
      const fontSize = isOutput ? 12 * 1.2 : 12;

      neurons.push({
        position: { x: position.x, y },
        radius: neuronRadius,
        activation: value,
        value: config.showValues !== false ? value.toFixed(2) : '',
        fill,
        stroke: 'var(--nn-stroke)',
        strokeWidth: isOutput ? 3 : 2,
        fontSize,
        fontWeight: isOutput ? 'bold' : 'normal',
        label,
        labelPosition,
        labelAlign,
      });
    }

    return neurons;
  }

  private buildSampledNeuronsForLayer(
    layerIndex: number,
    position: { x: number; centerY: number },
    activations: readonly number[],
    config: LayerConfig,
    isInput: boolean,
    isOutput: boolean,
    architecture: NetworkArchitecture,
  ): readonly Neuron[] {
    const { neuronDiameter, neuronPaddingY } = this.baseDimensions;
    const sampleCount = config.sampleCount ?? Math.min(10, activations.length);
    const radius = neuronDiameter / 2;

    // Sample indices evenly distributed
    const indices: number[] = [];
    for (let i = 0; i < sampleCount && i < activations.length; i++) {
      const idx = Math.floor((i * activations.length) / sampleCount);
      indices.push(idx);
    }

    const displayCount = indices.length + 1; // +1 for ellipsis
    const totalHeight = displayCount * neuronDiameter + (displayCount - 1) * neuronPaddingY;
    const startY = position.centerY - totalHeight / 2 + radius;

    const neurons: Neuron[] = [];

    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      const y = startY + i * (neuronDiameter + neuronPaddingY);
      const value = activations[idx];

      neurons.push({
        position: { x: position.x, y },
        radius,
        activation: value,
        value: value.toFixed(2),
        fill: isInput ? this.getInputColor(value) : this.getNeuronColor(value, 'relu', false),
        stroke: 'var(--nn-stroke)',
        strokeWidth: 2,
        fontSize: 12,
        fontWeight: 'normal',
      });
    }

    // Add ellipsis neuron
    const ellipsisY = startY + indices.length * (neuronDiameter + neuronPaddingY);
    neurons.push({
      position: { x: position.x, y: ellipsisY },
      radius: radius * 0.6,
      activation: 0,
      value: `+${activations.length - indices.length}`,
      fill: 'var(--nn-neutral)',
      stroke: 'var(--nn-stroke)',
      strokeWidth: 1,
      fontSize: 10,
      fontWeight: 'normal',
    });

    return neurons;
  }

  // ============================================================================
  // Alternative Representation Building
  // ============================================================================

  private buildGridData(activations: readonly number[], config: LayerConfig): GridData {
    const shape = config.shape ?? this.inferShape(activations.length);
    const [rows, cols] = shape;
    const cellSize = this.calculateHeatmapCellSize(rows, cols);

    const colors = activations.map((v) => this.valueToHeatmapColor(v, config.colorScheme));

    return {
      rows,
      cols,
      cellSize,
      values: activations,
      colors,
    };
  }

  // ============================================================================
  // Feature Maps Building (multi-channel heatmaps)
  // ============================================================================

  private calculateFeatureMapsLayout(
    size: number,
    config: LayerConfig,
  ): { totalWidth: number; totalHeight: number } {
    const channels = config.channels ?? 1;
    const maxChannels = config.maxChannels ?? 32;
    const displayChannels = Math.min(channels, maxChannels);
    const shape = config.shape ?? this.inferShape(Math.floor(size / channels));
    const [mapRows, mapCols] = shape;

    const cellSize = this.calculateFeatureMapCellSize(mapRows, mapCols, displayChannels);
    const gridCols = Math.min(displayChannels, Math.ceil(Math.sqrt(displayChannels)));
    const gridRows = Math.ceil(displayChannels / gridCols);
    const gap = Math.max(4, cellSize * 2);

    const totalWidth = gridCols * mapCols * cellSize + (gridCols - 1) * gap;
    const totalHeight = gridRows * mapRows * cellSize + (gridRows - 1) * gap;

    return { totalWidth, totalHeight };
  }

  private calculateFeatureMapCellSize(
    mapRows: number,
    mapCols: number,
    displayChannels: number,
  ): number {
    const gridCols = Math.min(displayChannels, Math.ceil(Math.sqrt(displayChannels)));
    const targetMaxWidth = 800;
    const maxDim = Math.max(mapRows, mapCols);
    const cellSize = Math.max(3, Math.floor(targetMaxWidth / (gridCols * maxDim)));
    return Math.min(cellSize, 20);
  }

  private buildFeatureMapsData(
    activations: readonly number[],
    config: LayerConfig,
  ): FeatureMapsData {
    const channels = config.channels ?? 1;
    const maxChannels = config.maxChannels ?? 32;
    const displayChannels = Math.min(channels, maxChannels);
    const shape = config.shape ?? this.inferShape(Math.floor(activations.length / channels));
    const [mapRows, mapCols] = shape;
    const channelSize = mapRows * mapCols;

    const cellSize = this.calculateFeatureMapCellSize(mapRows, mapCols, displayChannels);
    const gridCols = Math.min(displayChannels, Math.ceil(Math.sqrt(displayChannels)));
    const gridRows = Math.ceil(displayChannels / gridCols);
    const gap = Math.max(4, cellSize * 2);

    const maps: GridData[] = [];
    for (let c = 0; c < displayChannels; c++) {
      const offset = c * channelSize;
      const channelValues = activations.slice(offset, offset + channelSize);

      // Per-channel min/max normalization
      let min = Infinity;
      let max = -Infinity;
      for (const v of channelValues) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const range = max - min || 1;

      const normalized = channelValues.map((v) => (v - min) / range);
      const colors = normalized.map((v) =>
        this.valueToHeatmapColor(v, config.colorScheme ?? 'grayscale'),
      );

      maps.push({
        rows: mapRows,
        cols: mapCols,
        cellSize,
        values: channelValues,
        colors,
      });
    }

    return {
      channels,
      displayChannels,
      mapRows,
      mapCols,
      cellSize,
      gridCols,
      gridRows,
      gap,
      maps,
      truncated: channels > maxChannels,
    };
  }

  private valueToHeatmapColor(
    value: number,
    colorScheme: 'default' | 'grayscale' | 'viridis' | 'coolwarm' = 'default',
  ): string {
    const normalized = Math.max(0, Math.min(1, value));

    switch (colorScheme) {
      case 'grayscale':
        const gray = Math.round(normalized * 255);
        return `rgb(${gray},${gray},${gray})`;

      case 'viridis':
        return this.viridisColor(normalized);

      case 'coolwarm':
        return this.coolwarmColor(normalized);

      default:
        const intensity = Math.round(normalized * 255);
        return `rgb(${intensity},${intensity},${intensity})`;
    }
  }

  private viridisColor(t: number): string {
    // Simplified viridis approximation
    const r = Math.round((0.267 + t * 0.329) * 255);
    const g = Math.round((0.004 + t * 0.873) * 255);
    const b = Math.round((0.329 + t * (-0.329 + t * 0.267)) * 255);
    return `rgb(${Math.min(255, r)},${Math.min(255, g)},${Math.min(255, b)})`;
  }

  private coolwarmColor(t: number): string {
    // Blue (0) to White (0.5) to Red (1)
    if (t < 0.5) {
      const p = t * 2;
      return `rgb(${Math.round(p * 255)},${Math.round(p * 255)},255)`;
    } else {
      const p = (t - 0.5) * 2;
      return `rgb(255,${Math.round((1 - p) * 255)},${Math.round((1 - p) * 255)})`;
    }
  }

  private buildBarData(
    activations: readonly number[],
    bounds: { width: number; height: number },
  ): BarData {
    const min = Math.min(...activations);
    const max = Math.max(...activations);
    const mean = activations.reduce((a, b) => a + b, 0) / activations.length;

    // Create gradient colors based on distribution
    const sortedValues = [...activations].sort((a, b) => a - b);
    const colorGradient = sortedValues.map((v) =>
      this.valueToHeatmapColor((v - min) / (max - min || 1), 'viridis'),
    );

    return {
      width: bounds.width,
      height: bounds.height,
      min,
      max,
      mean,
      colorGradient,
    };
  }

  private buildStatsData(activations: readonly number[]): StatsData {
    const count = activations.length;
    const min = Math.min(...activations);
    const max = Math.max(...activations);
    const mean = activations.reduce((a, b) => a + b, 0) / count;

    const variance = activations.reduce((sum, v) => sum + (v - mean) ** 2, 0) / count;
    const std = Math.sqrt(variance);

    return { count, min, max, mean, std };
  }

  // ============================================================================
  // Connection Building
  // ============================================================================

  private buildConnections(
    weights: readonly LayerWeights[],
    layerSizes: readonly number[],
    layerPositions: readonly { x: number; centerY: number }[],
    layerElements: readonly LayerElement[],
  ): readonly Connection[] {
    const { strategy, threshold, maxCount } = this.config.connections;

    if (strategy === 'none') {
      return [];
    }

    const connections: Connection[] = [];

    for (let layerIndex = 0; layerIndex < weights.length; layerIndex++) {
      const layer = weights[layerIndex];
      const fromElement = layerElements[layerIndex];
      const toElement = layerElements[layerIndex + 1];

      // Skip connections for non-neuron representations
      if (fromElement.type !== 'neurons' && fromElement.type !== 'sampled') {
        continue;
      }
      if (toElement.type !== 'neurons' && toElement.type !== 'sampled') {
        continue;
      }

      const fromNeurons = fromElement.neurons ?? [];
      const toNeurons = toElement.neurons ?? [];

      const fromSize = fromNeurons.length;
      const toSize = toNeurons.length;

      const flatWeights = this.flattenWeights(
        layer.weights,
        layerSizes[layerIndex],
        layerSizes[layerIndex + 1],
      );
      const absWeights = flatWeights.map(Math.abs);
      const maxWeight = Math.max(...absWeights, 0.001);

      // Build connection list with filtering
      for (let from = 0; from < fromSize; from++) {
        for (let to = 0; to < toSize; to++) {
          const weightIndex = from * layerSizes[layerIndex + 1] + to;
          const weight = flatWeights[weightIndex] ?? 0;
          const normalizedWeight = Math.abs(weight) / maxWeight;

          // Apply strategy filters
          if (strategy === 'strong' && Math.abs(weight) < (threshold ?? 0.1)) {
            continue;
          }

          connections.push({
            from: fromNeurons[from].position,
            to: toNeurons[to].position,
            weight,
            color: weight >= 0 ? 'var(--nn-positive)' : 'var(--nn-negative)',
            opacity: (this.config.connections.opacity ?? 0.1) + normalizedWeight * 0.5,
            strokeWidth: (this.config.connections.strokeWidth ?? 1) + normalizedWeight * 1.5,
          });
        }
      }
    }

    // Apply sampling if needed
    if (strategy === 'sampled' && connections.length > (maxCount ?? 1000)) {
      return this.sampleConnections(connections, maxCount ?? 1000);
    }

    return connections;
  }

  private sampleConnections(
    connections: readonly Connection[],
    maxCount: number,
  ): readonly Connection[] {
    if (connections.length <= maxCount) {
      return connections;
    }

    // Sample by weight - keep stronger connections more likely
    const sorted = [...connections].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

    // Keep top 30% by weight, sample the rest
    const topCount = Math.floor(maxCount * 0.3);
    const sampleCount = maxCount - topCount;

    const top = sorted.slice(0, topCount);
    const rest = sorted.slice(topCount);

    // Random sample from rest
    const sampled: Connection[] = [];
    const step = rest.length / sampleCount;
    for (let i = 0; i < sampleCount && i * step < rest.length; i++) {
      sampled.push(rest[Math.floor(i * step)]);
    }

    return [...top, ...sampled];
  }

  private flattenWeights(
    weights: number[] | number[][],
    fromSize: number,
    toSize: number,
  ): number[] {
    if (!Array.isArray(weights[0])) {
      return weights as number[];
    }
    return (weights as number[][]).flat();
  }

  // ============================================================================
  // Label Building
  // ============================================================================

  private buildLabels(
    architecture: NetworkArchitecture,
    layerConfigs: readonly LayerConfig[],
    layerPositions: readonly { x: number; centerY: number }[],
    totalHeight: number,
    layerElements: readonly LayerElement[],
  ): readonly Label[] {
    const { margin, labelFontSize, labelOffsetY } = this.baseDimensions;
    const isRow = this.config.layout.strategy === 'row';
    const labels: Label[] = [];

    if (isRow) {
      // Row layout: labels above each layer (skip collapsed layers)
      const inputConfig = layerConfigs[0];
      if (inputConfig?.representation !== 'collapsed') {
        const inputElement = layerElements[0];
        const inputTop = layerPositions[0].centerY - (inputElement?.height ?? 0) / 2;
        labels.push({
          position: { x: layerPositions[0].x, y: inputTop - labelOffsetY },
          text: 'Input',
          color: 'var(--nn-neutral)',
          fontSize: labelFontSize,
          align: 'center',
        });
      }

      for (let i = 0; i < architecture.layers.length; i++) {
        const layer = architecture.layers[i];
        const config = layerConfigs[i + 1];

        // Skip labels for collapsed layers
        if (config?.representation === 'collapsed') {
          continue;
        }

        let text: string;
        if (config?.representation === 'feature-maps' || config?.representation === 'heatmap') {
          text = layer.activationFunction;
        } else if (layer.isOutput) {
          text = `Output (${layer.activationFunction})`;
        } else {
          text = `Hidden ${i + 1} (${layer.activationFunction})`;
        }

        const element = layerElements[i + 1];
        const elementTop = layerPositions[i + 1].centerY - (element?.height ?? 0) / 2;

        labels.push({
          position: { x: layerPositions[i + 1].x, y: elementTop - labelOffsetY },
          text,
          color: 'var(--nn-neutral)',
          fontSize: labelFontSize,
          align: 'center',
        });
      }
    } else {
      // Column layout: labels below each layer (abbreviated for space)
      const labelY = totalHeight - margin + labelOffsetY / 2;

      labels.push({
        position: { x: layerPositions[0].x, y: labelY },
        text: 'Input',
        color: 'var(--nn-neutral)',
        fontSize: labelFontSize,
        align: 'center',
      });

      const hiddenCount = architecture.layers.filter((l) => !l.isOutput).length;

      for (let i = 0; i < architecture.layers.length; i++) {
        const layer = architecture.layers[i];
        const config = layerConfigs[i + 1];

        let text: string;
        if (config?.representation === 'feature-maps' || config?.representation === 'heatmap') {
          text = layer.activationFunction;
        } else if (layer.isOutput) {
          // Abbreviated output label
          const shortActivation = this.abbreviateActivation(layer.activationFunction);
          text = `Out(${shortActivation})`;
        } else {
          // Abbreviated hidden label - just number if multiple hidden layers
          const shortActivation = this.abbreviateActivation(layer.activationFunction);
          if (hiddenCount === 1) {
            text = `Hidden(${shortActivation})`;
          } else {
            text = `H${i + 1}(${shortActivation})`;
          }
        }

        labels.push({
          position: { x: layerPositions[i + 1].x, y: labelY },
          text,
          color: 'var(--nn-neutral)',
          fontSize: labelFontSize,
          align: 'center',
        });
      }
    }

    return labels;
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  /**
   * Abbreviate activation function name for compact label display
   */
  private abbreviateActivation(activation: string): string {
    const lower = activation.toLowerCase();
    switch (lower) {
      case 'sigmoid':
        return 'σ';
      case 'tanh':
        return 'tanh';
      case 'relu':
        return 'ReLU';
      case 'softmax':
        return 'SM';
      case 'linear':
        return 'Lin';
      case 'leakyrelu':
        return 'LReLU';
      default:
        // Return first 4 chars for unknown activations
        return activation.slice(0, 4);
    }
  }

  private extractNeuronsFromElements(elements: readonly LayerElement[]): readonly Neuron[] {
    const neurons: Neuron[] = [];

    for (const element of elements) {
      if (element.neurons) {
        neurons.push(...element.neurons);
      }
    }

    return neurons;
  }

  private getInputLabel(index: number): string {
    const labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    return labels[index] || `I${index}`;
  }

  private getInputColor(value: number): CssColor {
    const normalized = Math.max(0, Math.min(1, value));
    return normalized > 0.5 ? 'var(--nn-positive)' : 'var(--nn-neutral)';
  }

  private getNeuronColor(value: number, activationFunction: string, isOutput: boolean): CssColor {
    if (isOutput) {
      const isSoftmax = activationFunction.toLowerCase() === 'softmax';
      const threshold = isSoftmax ? 0.33 : 0.5;
      return value > threshold ? 'var(--nn-positive)' : 'var(--nn-neutral)';
    }

    const func = activationFunction.toLowerCase();
    let normalized: number;

    if (func === 'tanh') {
      normalized = (value + 1) / 2;
    } else if (func === 'sigmoid' || func === 'softmax') {
      normalized = value;
    } else if (func === 'relu') {
      normalized = Math.min(value, 1);
    } else {
      normalized = (value + 1) / 2;
    }

    const r = Math.round(normalized * 34 + (1 - normalized) * 239);
    const g = Math.round(normalized * 197 + (1 - normalized) * 68);
    const b = Math.round(normalized * 94 + (1 - normalized) * 68);
    return `rgb(${r},${g},${b})`;
  }
}
