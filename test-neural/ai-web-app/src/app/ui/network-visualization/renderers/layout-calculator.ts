import {
  Bounds,
  Connection,
  ContentDimensions,
  CssColor,
  DEFAULT_CONTENT_DIMENSIONS,
  Label,
  NetworkRenderData,
  Neuron,
} from './types';

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
 * Network Layout Calculator
 *
 * Calculates positions in NATURAL coordinates (content-first approach).
 * The natural size is determined by readability constraints:
 * - Fixed neuron diameter
 * - Fixed padding between neurons
 * - Fixed spacing between layers
 *
 * The renderer then scales this to fit the display canvas.
 */
export class NetworkLayoutCalculator {
  private readonly dimensions: ContentDimensions;

  constructor(dimensions: Partial<ContentDimensions> = {}) {
    this.dimensions = { ...DEFAULT_CONTENT_DIMENSIONS, ...dimensions };
  }

  /**
   * Calculate complete render data in natural coordinates.
   * Returns data with naturalBounds for scaling.
   */
  calculateLayout(
    architecture: NetworkArchitecture,
    weights: readonly LayerWeights[],
  ): NetworkRenderData {
    const layerSizes = this.getLayerSizes(architecture);
    const naturalBounds = this.calculateNaturalBounds(layerSizes);
    const layerX = this.calculateLayerXPositions(layerSizes.length, naturalBounds.width);
    const layerY = this.calculateAllLayerYPositions(layerSizes, naturalBounds.height);

    return {
      connections: this.buildConnections(weights, layerSizes, layerX, layerY),
      neurons: this.buildNeurons(architecture, layerX, layerY),
      labels: this.buildLabels(architecture, layerX, naturalBounds.height),
      naturalBounds,
    };
  }

  /**
   * Update content dimensions
   */
  updateDimensions(dimensions: Partial<ContentDimensions>): void {
    Object.assign(this.dimensions, dimensions);
  }

  // ============================================================================
  // Natural Bounds Calculation
  // ============================================================================

  /**
   * Calculate natural bounds based on network structure.
   * Size is determined by content, not canvas.
   */
  private calculateNaturalBounds(layerSizes: readonly number[]): Bounds {
    const { neuronDiameter, neuronPaddingY, layerPaddingX, margin, labelOffsetY } = this.dimensions;

    // Width: layers * spacing + margins
    const width = margin * 2 + (layerSizes.length - 1) * layerPaddingX;

    // Height: tallest layer determines height
    const maxNeurons = Math.max(...layerSizes);
    const neuronsHeight = maxNeurons * neuronDiameter + (maxNeurons - 1) * neuronPaddingY;
    const height = margin * 2 + neuronsHeight + labelOffsetY;

    return { width, height };
  }

  private getLayerSizes(architecture: NetworkArchitecture): readonly number[] {
    return [architecture.inputs.length, ...architecture.layers.map((layer) => layer.size)];
  }

  private calculateLayerXPositions(layerCount: number, totalWidth: number): readonly number[] {
    const { margin, layerPaddingX } = this.dimensions;
    const positions: number[] = [];

    for (let i = 0; i < layerCount; i++) {
      positions.push(margin + i * layerPaddingX);
    }

    return positions;
  }

  private calculateAllLayerYPositions(
    layerSizes: readonly number[],
    totalHeight: number,
  ): number[][] {
    return layerSizes.map((size) => [...this.calculateNeuronYPositions(size, totalHeight)]);
  }

  private calculateNeuronYPositions(count: number, totalHeight: number): readonly number[] {
    const { neuronDiameter, neuronPaddingY, margin, labelOffsetY } = this.dimensions;
    const availableHeight = totalHeight - margin * 2 - labelOffsetY;

    // Calculate total height needed for this layer
    const layerHeight = count * neuronDiameter + (count - 1) * neuronPaddingY;

    // Center vertically
    const startY = margin + (availableHeight - layerHeight) / 2 + neuronDiameter / 2;
    const step = neuronDiameter + neuronPaddingY;

    const positions: number[] = [];
    for (let i = 0; i < count; i++) {
      positions.push(startY + i * step);
    }

    return positions;
  }

  // ============================================================================
  // Connection Building
  // ============================================================================

  private buildConnections(
    weights: readonly LayerWeights[],
    layerSizes: readonly number[],
    layerX: readonly number[],
    layerY: readonly number[][],
  ): readonly Connection[] {
    const connections: Connection[] = [];

    for (let layerIndex = 0; layerIndex < weights.length; layerIndex++) {
      const layer = weights[layerIndex];
      const fromSize = layerSizes[layerIndex];
      const toSize = layerSizes[layerIndex + 1];
      const fromX = layerX[layerIndex];
      const toX = layerX[layerIndex + 1];
      const fromY = layerY[layerIndex];
      const toY = layerY[layerIndex + 1];

      connections.push(
        ...this.buildConnectionsBetweenLayers(layer, fromSize, toSize, fromX, toX, fromY, toY),
      );
    }

    return connections;
  }

  private buildConnectionsBetweenLayers(
    layer: LayerWeights,
    fromSize: number,
    toSize: number,
    fromX: number,
    toX: number,
    fromY: readonly number[],
    toY: readonly number[],
  ): readonly Connection[] {
    const connections: Connection[] = [];
    const flatWeights = this.flattenWeights(layer.weights, fromSize, toSize);

    // Find max weight for normalization
    const absWeights = flatWeights.map(Math.abs);
    const maxWeight = Math.max(...absWeights, 0.001);

    for (let from = 0; from < fromSize; from++) {
      for (let to = 0; to < toSize; to++) {
        const weightIndex = from * toSize + to;
        const weight = flatWeights[weightIndex] ?? 0;
        const normalizedWeight = Math.abs(weight) / maxWeight;

        connections.push({
          from: { x: fromX, y: fromY[from] },
          to: { x: toX, y: toY[to] },
          weight,
          color: weight >= 0 ? 'var(--nn-positive)' : 'var(--nn-negative)',
          opacity: 0.1 + normalizedWeight * 0.5,
          strokeWidth: 1 + normalizedWeight * 1.5,
        });
      }
    }

    return connections;
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
  // Neuron Building
  // ============================================================================

  private buildNeurons(
    architecture: NetworkArchitecture,
    layerX: readonly number[],
    layerY: readonly number[][],
  ): readonly Neuron[] {
    const neurons: Neuron[] = [];
    const { neuronDiameter, neuronFontSize } = this.dimensions;
    const radius = neuronDiameter / 2;

    // Input layer
    const inputX = layerX[0];
    const inputY = layerY[0];
    for (let i = 0; i < architecture.inputs.length; i++) {
      const value = architecture.inputs[i];
      neurons.push({
        position: { x: inputX, y: inputY[i] },
        radius,
        activation: value,
        value: value.toFixed(2),
        fill: this.getInputColor(value),
        stroke: 'var(--nn-stroke)',
        strokeWidth: 2,
        fontSize: neuronFontSize,
        fontWeight: 'normal',
        label: this.getInputLabel(i),
        labelPosition: { x: inputX - radius - 10, y: inputY[i] },
        labelAlign: 'right',
      });
    }

    // Hidden and output layers
    for (let layerIndex = 0; layerIndex < architecture.layers.length; layerIndex++) {
      const layer = architecture.layers[layerIndex];
      const x = layerX[layerIndex + 1];
      const yPositions = layerY[layerIndex + 1];
      const isOutput = layer.isOutput;

      for (let neuronIndex = 0; neuronIndex < layer.activations.length; neuronIndex++) {
        const activation = layer.activations[neuronIndex];
        const neuronRadius = isOutput ? radius * 1.2 : radius;
        const fontSize = isOutput ? neuronFontSize * 1.2 : neuronFontSize;

        const fill = this.getNeuronColor(activation, layer.activationFunction, isOutput);

        const neuron: Neuron = {
          position: { x, y: yPositions[neuronIndex] },
          radius: neuronRadius,
          activation,
          value: activation.toFixed(2),
          fill,
          stroke: 'var(--nn-stroke)',
          strokeWidth: isOutput ? 3 : 2,
          fontSize,
          fontWeight: isOutput ? 'bold' : 'normal',
        };

        // Add label for output neurons
        if (isOutput) {
          neurons.push({
            ...neuron,
            label: layer.activations.length > 1 ? `Out ${neuronIndex}` : 'Out',
            labelPosition: { x: x + neuronRadius + 10, y: yPositions[neuronIndex] },
            labelAlign: 'left',
          });
        } else {
          neurons.push(neuron);
        }
      }
    }

    return neurons;
  }

  // ============================================================================
  // Label Building
  // ============================================================================

  private buildLabels(
    architecture: NetworkArchitecture,
    layerX: readonly number[],
    totalHeight: number,
  ): readonly Label[] {
    const { margin, labelFontSize, labelOffsetY } = this.dimensions;
    const labelY = totalHeight - margin + labelOffsetY / 2;
    const labels: Label[] = [];

    // Input layer label
    labels.push({
      position: { x: layerX[0], y: labelY },
      text: 'Input',
      color: 'var(--nn-neutral)',
      fontSize: labelFontSize,
      align: 'center',
    });

    // Hidden and output layer labels
    for (let i = 0; i < architecture.layers.length; i++) {
      const layer = architecture.layers[i];
      const text = layer.isOutput
        ? `Output (${layer.activationFunction})`
        : `Hidden ${i + 1} (${layer.activationFunction})`;

      labels.push({
        position: { x: layerX[i + 1], y: labelY },
        text,
        color: 'var(--nn-neutral)',
        fontSize: labelFontSize,
        align: 'center',
      });
    }

    return labels;
  }

  // ============================================================================
  // Color Utilities
  // ============================================================================

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

    // Interpolate between red (negative) and green (positive)
    const r = Math.round(normalized * 34 + (1 - normalized) * 239);
    const g = Math.round(normalized * 197 + (1 - normalized) * 68);
    const b = Math.round(normalized * 94 + (1 - normalized) * 68);
    return `rgb(${r},${g},${b})`;
  }
}
