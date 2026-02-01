import { Connection, CssColor, Label, NetworkRenderData, Neuron } from './types';

/**
 * Layer information extracted from activations
 */
export interface LayerInfo {
  /** Number of neurons in this layer */
  size: number;
  /** Activation values */
  activations: number[];
  /** Activation function name */
  activationFunction: string;
  /** Whether this is the output layer */
  isOutput: boolean;
}

/**
 * Network architecture for layout calculation
 */
export interface NetworkArchitecture {
  /** Input values */
  inputs: number[];
  /** Hidden and output layers */
  layers: LayerInfo[];
}

/**
 * Weight matrix for a layer
 */
export interface LayerWeights {
  /** Weight matrix (can be flat array or 2D array) */
  weights: number[] | number[][];
}

/**
 * Layout configuration
 */
export interface LayoutConfig {
  /** Canvas width */
  width: number;
  /** Canvas height */
  height: number;
  /** Margin from edges */
  margin: number;
  /** Vertical spacing config */
  verticalMargin: number;
  verticalPadding: number;
  /** Neuron radii */
  neuronRadius: {
    input: number;
    hidden: number;
    output: number;
  };
  /** Font sizes */
  fontSize: {
    input: number;
    hidden: number;
    output: number;
    label: number;
    layerLabel: number;
  };
  /** Y position for layer labels */
  labelY: number;
}

/**
 * Default layout configuration matching current design
 */
export const DEFAULT_LAYOUT_CONFIG: LayoutConfig = {
  width: 500,
  height: 280,
  margin: 60,
  verticalMargin: 30,
  verticalPadding: 40,
  neuronRadius: {
    input: 20,
    hidden: 16,
    output: 25,
  },
  fontSize: {
    input: 14,
    hidden: 9,
    output: 16,
    label: 11,
    layerLabel: 10,
  },
  labelY: 270,
};

/**
 * Network Layout Calculator
 *
 * Responsible for calculating positions of neurons, connections, and labels
 * based on network architecture and weights. This logic is separated from
 * rendering so it can be used by any renderer implementation.
 */
export class NetworkLayoutCalculator {
  private config: LayoutConfig;

  constructor(config: Partial<LayoutConfig> = {}) {
    this.config = { ...DEFAULT_LAYOUT_CONFIG, ...config };
  }

  /**
   * Calculate complete render data from architecture and weights
   */
  calculateLayout(architecture: NetworkArchitecture, weights: LayerWeights[]): NetworkRenderData {
    const layerSizes = this.getLayerSizes(architecture);
    const layerX = this.calculateLayerXPositions(layerSizes.length);
    const layerY = this.calculateAllLayerYPositions(layerSizes);

    return {
      connections: this.buildConnections(weights, layerSizes, layerX, layerY),
      neurons: this.buildNeurons(architecture, layerX, layerY),
      labels: this.buildLabels(architecture, layerX),
    };
  }

  /**
   * Update layout configuration
   */
  updateConfig(config: Partial<LayoutConfig>): void {
    this.config = { ...this.config, ...config };
  }

  // ============================================================================
  // Private: Layer Size Calculations
  // ============================================================================

  private getLayerSizes(architecture: NetworkArchitecture): number[] {
    const inputCount = architecture.inputs.length;
    const layerCounts = architecture.layers.map((layer) => layer.size);
    return [inputCount, ...layerCounts];
  }

  private calculateLayerXPositions(layerCount: number): number[] {
    const spacing = (this.config.width - 2 * this.config.margin) / (layerCount - 1);
    const positions: number[] = [];
    for (let i = 0; i < layerCount; i++) {
      positions.push(this.config.margin + i * spacing);
    }
    return positions;
  }

  private calculateAllLayerYPositions(layerSizes: number[]): number[][] {
    return layerSizes.map((size) => this.calculateNeuronYPositions(size));
  }

  private calculateNeuronYPositions(count: number): number[] {
    const available =
      this.config.height - 2 * this.config.verticalMargin - this.config.verticalPadding;
    const spacing = count > 1 ? available / (count - 1) : 0;
    const positions: number[] = [];
    const startY =
      this.config.verticalMargin +
      (this.config.height -
        2 * this.config.verticalMargin -
        this.config.verticalPadding -
        spacing * (count - 1)) /
        2;

    for (let i = 0; i < count; i++) {
      positions.push(startY + i * spacing);
    }
    return positions;
  }

  // ============================================================================
  // Private: Connection Building
  // ============================================================================

  private buildConnections(
    weights: LayerWeights[],
    layerSizes: number[],
    layerX: number[],
    layerY: number[][],
  ): Connection[] {
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
    fromY: number[],
    toY: number[],
  ): Connection[] {
    const connections: Connection[] = [];
    const isNestedArray = Array.isArray(layer.weights[0]);

    for (let i = 0; i < toSize; i++) {
      for (let j = 0; j < fromSize; j++) {
        let weight: number;

        if (isNestedArray) {
          weight = (layer.weights as number[][])[i][j];
        } else {
          weight = (layer.weights as number[])[i * fromSize + j];
        }

        if (weight === undefined || isNaN(weight)) {
          continue;
        }

        const absWeight = Math.abs(weight);
        const opacity = Math.min(absWeight / 2, 0.9) + 0.3;
        const strokeWidth = Math.min(absWeight * 2, 2.5) + 0.8;
        const color = weight > 0 ? 'var(--nn-positive)' : 'var(--nn-negative)';

        connections.push({
          from: { x: fromX, y: fromY[j] },
          to: { x: toX, y: toY[i] },
          weight,
          color,
          opacity,
          strokeWidth,
        });
      }
    }

    return connections;
  }

  // ============================================================================
  // Private: Neuron Building
  // ============================================================================

  private buildNeurons(
    architecture: NetworkArchitecture,
    layerX: number[],
    layerY: number[][],
  ): Neuron[] {
    const neurons: Neuron[] = [];

    // Input neurons
    neurons.push(...this.buildInputNeurons(architecture.inputs, layerX[0], layerY[0]));

    // Hidden and output neurons
    neurons.push(...this.buildHiddenAndOutputNeurons(architecture, layerX, layerY));

    return neurons;
  }

  private buildInputNeurons(inputs: number[], x: number, yPositions: number[]): Neuron[] {
    const neurons: Neuron[] = [];
    const threshold = 0.5;

    for (let i = 0; i < inputs.length; i++) {
      const value = inputs[i];
      const fill = value > threshold ? 'var(--nn-positive)' : 'var(--nn-neutral)';

      neurons.push({
        position: { x, y: yPositions[i] },
        radius: this.config.neuronRadius.input,
        activation: value,
        value: value.toFixed(1),
        fill,
        stroke: 'var(--nn-stroke)',
        strokeWidth: 2,
        label: this.getInputLabel(i),
        labelPosition: { x: x - 35, y: yPositions[i] },
        labelAlign: 'center',
        fontSize: this.config.fontSize.input,
        fontWeight: 'bold',
      });
    }

    return neurons;
  }

  private buildHiddenAndOutputNeurons(
    architecture: NetworkArchitecture,
    layerX: number[],
    layerY: number[][],
  ): Neuron[] {
    const neurons: Neuron[] = [];

    for (let layerIndex = 0; layerIndex < architecture.layers.length; layerIndex++) {
      const layer = architecture.layers[layerIndex];
      const x = layerX[layerIndex + 1];
      const yPositions = layerY[layerIndex + 1];
      const isOutput = layer.isOutput;
      const isSoftmax = layer.activationFunction.toLowerCase() === 'softmax';

      for (let neuronIndex = 0; neuronIndex < layer.activations.length; neuronIndex++) {
        const activation = layer.activations[neuronIndex];
        const radius = isOutput ? this.config.neuronRadius.output : this.config.neuronRadius.hidden;
        const fontSize = isOutput ? this.config.fontSize.output : this.config.fontSize.hidden;

        let fill: CssColor;
        if (isOutput) {
          const threshold = isSoftmax ? 0.33 : 0.5;
          fill = activation > threshold ? 'var(--nn-positive)' : 'var(--nn-neutral)';
        } else {
          fill = this.getNeuronColor(activation, layer.activationFunction);
        }

        const neuron: Neuron = {
          position: { x, y: yPositions[neuronIndex] },
          radius,
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
          neuron.label = layer.activations.length > 1 ? `Out ${neuronIndex}` : 'Out';
          neuron.labelPosition = { x: x + 40, y: yPositions[neuronIndex] };
          neuron.labelAlign = 'left';
        }

        neurons.push(neuron);
      }
    }

    return neurons;
  }

  // ============================================================================
  // Private: Label Building
  // ============================================================================

  private buildLabels(architecture: NetworkArchitecture, layerX: number[]): Label[] {
    const labels: Label[] = [];

    // Input layer label
    labels.push({
      position: { x: layerX[0], y: this.config.labelY },
      text: 'Input',
      color: 'var(--nn-neutral)',
      fontSize: this.config.fontSize.layerLabel,
      align: 'center',
    });

    // Hidden and output layer labels
    for (let i = 0; i < architecture.layers.length; i++) {
      const layer = architecture.layers[i];
      const isOutput = layer.isOutput;
      const text = isOutput
        ? `Output (${layer.activationFunction})`
        : `Hidden ${i + 1} (${layer.activationFunction})`;

      labels.push({
        position: { x: layerX[i + 1], y: this.config.labelY },
        text,
        color: 'var(--nn-neutral)',
        fontSize: this.config.fontSize.layerLabel,
        align: 'center',
      });
    }

    return labels;
  }

  // ============================================================================
  // Private: Utilities
  // ============================================================================

  private getInputLabel(index: number): string {
    const labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    return labels[index] || `I${index}`;
  }

  private getNeuronColor(value: number, activationFunction: string): string {
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
