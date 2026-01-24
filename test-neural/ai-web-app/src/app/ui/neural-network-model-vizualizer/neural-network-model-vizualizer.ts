import { Component, computed, input } from '@angular/core';
import { Activation, NeuralNetworkLayers } from '@cma/wasm/shared';

// ============================================================================
// Types pour le rendu SVG déclaratif
// ============================================================================

/** Représente une connexion (ligne) entre deux neurones */
export interface SvgConnection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  strokeWidth: number;
  opacity: number;
}

/** Représente un neurone (cercle + texte) */
export interface SvgNeuron {
  cx: number;
  cy: number;
  radius: number;
  fill: string;
  strokeWidth: number;
  value: string;
  label?: string;
  labelX?: number;
  labelAnchor?: 'start' | 'middle' | 'end';
  fontSize: string;
  fontWeight: string;
  textOffsetY: number;
}

/** Représente un label de couche */
export interface SvgLayerLabel {
  x: number;
  y: number;
  text: string;
}

/** Détails d'activation pour une couche */
export interface ActivationDetail {
  layerIndex: number;
  functionName: string;
  values: string;
}

/** Structure complète pour le rendu SVG */
export interface NetworkVisualizationData {
  connections: SvgConnection[];
  neurons: SvgNeuron[];
  layerLabels: SvgLayerLabel[];
  activationDetails: ActivationDetail[];
  outputSummary: string;
}

// ============================================================================
// Configuration
// ============================================================================

const SVG_CONFIG = {
  width: 500,
  height: 280,
  margin: 60,
  neuronRadius: {
    input: 20,
    hidden: 16,
    output: 25,
  },
  fontSize: {
    hidden: '9',
    output: '16',
    label: '11',
    layerLabel: '10',
    input: '14',
  },
  labelY: 270,
  verticalMargin: 30,
  verticalPadding: 40,
} as const;

const COLORS = {
  positive: '#22c55e',
  negative: '#ef4444',
  neutral: '#64748b',
  stroke: 'white',
  label: '#94a3b8',
} as const;

const ACTIVATION_THRESHOLD = {
  binary: 0.5,
  softmax: 0.33,
} as const;

// ============================================================================
// Composant
// ============================================================================

@Component({
  selector: 'app-neural-network-model-vizualizer',
  imports: [],
  templateUrl: './neural-network-model-vizualizer.html',
  styleUrl: './neural-network-model-vizualizer.scss',
  host: {
    class: 'card',
  },
})
export class NeuralNetworkModelVizualizer {
  public readonly activations = input<Activation<unknown, unknown> | null>();
  public readonly weights = input<NeuralNetworkLayers | undefined>();

  /** Données calculées pour le rendu SVG déclaratif */
  public readonly visualizationData = computed<NetworkVisualizationData | null>(() => {
    const activations = this.activations();
    const weights = this.weights();
    if (!activations || !weights) {
      return null;
    }
    return this._buildVisualizationData(activations, weights);
  });

  // Configuration exposée pour le template
  protected readonly config = SVG_CONFIG;
  protected readonly colors = COLORS;

  // ==========================================================================
  // Méthodes de construction des données de visualisation
  // ==========================================================================

  private _buildVisualizationData(
    activations: Activation<unknown, unknown>,
    weights: NeuralNetworkLayers,
  ): NetworkVisualizationData {
    const layerSizes = this._calculateLayerSizes(activations);
    const layerX = this._calculateLayerXPositions(layerSizes.length);
    const layerYPositions = this._calculateAllLayerYPositions(layerSizes);

    return {
      connections: this._buildAllConnections(weights, layerSizes, layerX, layerYPositions),
      neurons: this._buildAllNeurons(activations, layerX, layerYPositions),
      layerLabels: this._buildLayerLabels(activations, layerX),
      activationDetails: this._buildActivationDetails(activations),
      outputSummary: this._buildOutputSummary(activations),
    };
  }

  private _calculateLayerSizes(activations: Activation<unknown, unknown>): number[] {
    const inputCount = activations.inputs.length;
    const hiddenCounts = activations.layers.map((layer) => layer.activation.length);
    return [inputCount, ...hiddenCounts];
  }

  private _calculateLayerXPositions(layerCount: number): number[] {
    const spacing = (SVG_CONFIG.width - 2 * SVG_CONFIG.margin) / (layerCount - 1);
    const positions: number[] = [];
    for (let i = 0; i < layerCount; i++) {
      positions.push(SVG_CONFIG.margin + i * spacing);
    }
    return positions;
  }

  private _calculateAllLayerYPositions(layerSizes: number[]): number[][] {
    return layerSizes.map((size) => this._getNeuronYPositions(size, SVG_CONFIG.height));
  }

  // ==========================================================================
  // Construction des connexions
  // ==========================================================================

  private _buildAllConnections(
    weights: NeuralNetworkLayers,
    layerSizes: number[],
    layerX: number[],
    layerYPositions: number[][],
  ): SvgConnection[] {
    const connections: SvgConnection[] = [];

    // Connexions entre input et première couche cachée
    connections.push(
      ...this._buildConnectionsBetweenLayers(
        weights.layers[0],
        layerSizes[0],
        layerSizes[1],
        layerX[0],
        layerX[1],
        layerYPositions[0],
        layerYPositions[1],
      ),
    );

    // Connexions entre couches cachées et vers output
    for (let i = 1; i < weights.layers.length; i++) {
      connections.push(
        ...this._buildConnectionsBetweenLayers(
          weights.layers[i],
          layerSizes[i],
          layerSizes[i + 1],
          layerX[i],
          layerX[i + 1],
          layerYPositions[i],
          layerYPositions[i + 1],
        ),
      );
    }

    return connections;
  }

  private _buildConnectionsBetweenLayers(
    layer: { weights: number[] | number[][] },
    fromSize: number,
    toSize: number,
    fromX: number,
    toX: number,
    fromY: number[],
    toY: number[],
  ): SvgConnection[] {
    const connections: SvgConnection[] = [];
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
        const color = weight > 0 ? COLORS.positive : COLORS.negative;

        connections.push({
          x1: fromX,
          y1: fromY[j],
          x2: toX,
          y2: toY[i],
          color,
          strokeWidth,
          opacity,
        });
      }
    }

    return connections;
  }

  // ==========================================================================
  // Construction des neurones
  // ==========================================================================

  private _buildAllNeurons(
    activations: Activation<unknown, unknown>,
    layerX: number[],
    layerYPositions: number[][],
  ): SvgNeuron[] {
    const neurons: SvgNeuron[] = [];

    // Neurones d'entrée
    neurons.push(...this._buildInputNeurons(activations, layerX[0], layerYPositions[0]));

    // Couches cachées et de sortie
    neurons.push(...this._buildHiddenAndOutputNeurons(activations, layerX, layerYPositions));

    return neurons;
  }

  private _buildInputNeurons(
    activations: Activation<unknown, unknown>,
    x: number,
    yPositions: number[],
  ): SvgNeuron[] {
    return activations.inputs.map((inputValue: unknown, i: number) => {
      const numericValue = inputValue as number;
      return {
        cx: x,
        cy: yPositions[i],
        radius: SVG_CONFIG.neuronRadius.input,
        fill: numericValue > ACTIVATION_THRESHOLD.binary ? COLORS.positive : COLORS.neutral,
        strokeWidth: 2,
        value: numericValue.toFixed(1),
        label: this._getInputLabel(i),
        labelX: x - 35,
        labelAnchor: 'middle' as const,
        fontSize: SVG_CONFIG.fontSize.input,
        fontWeight: 'bold',
        textOffsetY: 5,
      };
    });
  }

  private _buildHiddenAndOutputNeurons(
    activations: Activation<unknown, unknown>,
    layerX: number[],
    layerYPositions: number[][],
  ): SvgNeuron[] {
    const neurons: SvgNeuron[] = [];
    const outputValues = activations.output as number[];

    activations.layers.forEach((layer, layerIndex) => {
      const x = layerX[layerIndex + 1];
      const yPositions = layerYPositions[layerIndex + 1];
      const isOutputLayer = layerIndex === activations.layers.length - 1;
      const isSoftmax = layer.function.toLowerCase() === 'softmax';
      const layerValues = isOutputLayer ? outputValues : layer.activation;

      layerValues.forEach((activationValue: number, neuronIndex: number) => {
        let color: string;
        let radius: number;
        let fontSize: string;

        if (isOutputLayer) {
          radius = SVG_CONFIG.neuronRadius.output;
          fontSize = SVG_CONFIG.fontSize.output;
          color = isSoftmax
            ? activationValue > ACTIVATION_THRESHOLD.softmax
              ? COLORS.positive
              : COLORS.neutral
            : activationValue > ACTIVATION_THRESHOLD.binary
              ? COLORS.positive
              : COLORS.negative;
        } else {
          radius = SVG_CONFIG.neuronRadius.hidden;
          fontSize = SVG_CONFIG.fontSize.hidden;
          color = this._getNeuronColor(activationValue, layer.function);
        }

        const neuron: SvgNeuron = {
          cx: x,
          cy: yPositions[neuronIndex],
          radius,
          fill: color,
          strokeWidth: isOutputLayer ? 3 : 2,
          value: activationValue.toFixed(2),
          fontSize,
          fontWeight: isOutputLayer ? 'bold' : 'normal',
          textOffsetY: isOutputLayer ? 6 : 4,
        };

        if (isOutputLayer) {
          neuron.label = outputValues.length > 1 ? `Out ${neuronIndex}` : 'Out';
          neuron.labelX = x + 40;
          neuron.labelAnchor = 'start';
        }

        neurons.push(neuron);
      });
    });

    return neurons;
  }

  // ==========================================================================
  // Construction des labels
  // ==========================================================================

  private _buildLayerLabels(
    activations: Activation<unknown, unknown>,
    layerX: number[],
  ): SvgLayerLabel[] {
    const labels: SvgLayerLabel[] = [{ x: layerX[0], y: SVG_CONFIG.labelY, text: 'Input' }];

    activations.layers.forEach((layer, i) => {
      const isOutputLayer = i === activations.layers.length - 1;
      const text = isOutputLayer
        ? `Output (${layer.function})`
        : `Hidden ${i + 1} (${layer.function})`;

      labels.push({
        x: layerX[i + 1],
        y: SVG_CONFIG.labelY,
        text,
      });
    });

    return labels;
  }

  // ==========================================================================
  // Construction des détails d'activation
  // ==========================================================================

  private _buildActivationDetails(activations: Activation<unknown, unknown>): ActivationDetail[] {
    return activations.layers.map((layer, i) => ({
      layerIndex: i + 1,
      functionName: layer.function,
      values: '[' + layer.activation.map((v: number) => v.toFixed(3)).join(', ') + ']',
    }));
  }

  private _buildOutputSummary(activations: Activation<unknown, unknown>): string {
    const outputValues = activations.output as number[];

    if (outputValues.length === 1) {
      const value = outputValues[0];
      const binary = value > ACTIVATION_THRESHOLD.binary ? '1' : '0';
      return `${value.toFixed(6)} → ${binary}`;
    } else {
      const formatted = '[' + outputValues.map((v: number) => v.toFixed(4)).join(', ') + ']';
      const maxIndex = outputValues.indexOf(Math.max(...outputValues));
      return `${formatted} → Class ${maxIndex}`;
    }
  }

  // ==========================================================================
  // Fonctions utilitaires
  // ==========================================================================

  private _getInputLabel(index: number): string {
    const labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    return labels[index] || `I${index}`;
  }

  private _getNeuronColor(value: number, activationFunction: string): string {
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

  private _getNeuronYPositions(count: number, height: number): number[] {
    const available = height - 2 * SVG_CONFIG.verticalMargin - SVG_CONFIG.verticalPadding;
    const spacing = count > 1 ? available / (count - 1) : 0;
    const positions: number[] = [];
    const startY =
      SVG_CONFIG.verticalMargin +
      (height -
        2 * SVG_CONFIG.verticalMargin -
        SVG_CONFIG.verticalPadding -
        spacing * (count - 1)) /
        2;
    for (let i = 0; i < count; i++) {
      positions.push(startY + i * spacing);
    }
    return positions;
  }
}
