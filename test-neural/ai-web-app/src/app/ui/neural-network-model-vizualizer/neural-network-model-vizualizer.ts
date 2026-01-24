import { Component, effect, input } from '@angular/core';
import { Activation, NeuralNetworkLayers } from '@cma/wasm/shared';

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';

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

  constructor() {
    effect(() => {
      const activations = this.activations();
      const weights = this.weights();
      if (!activations || !weights) {
        return;
      }
      setTimeout(() => {
        this._updateNetworkViz(activations, weights);
      }, 0);
    });
  }

  private _updateNetworkViz(
    activations: Activation<unknown, unknown>,
    weights: NeuralNetworkLayers,
  ): void {
    if (!activations || !weights) {
      return;
    }

    const svg = document.getElementById('networkViz');
    if (!svg) {
      return;
    }

    // Calculate dynamic structure
    const layerSizes = this._calculateLayerSizes(activations);
    const layerX = this._calculateLayerXPositions(layerSizes.length);
    const layerYPositions = this._calculateAllLayerYPositions(layerSizes);

    // Clear previous content
    this._clearSvg(svg);

    // Draw all connections between layers
    this._drawAllConnections(svg, weights, layerSizes, layerX, layerYPositions);

    // Draw all neuron layers
    this._drawInputLayer(svg, activations, layerX[0], layerYPositions[0]);
    this._drawHiddenAndOutputLayers(svg, activations, layerX, layerYPositions);

    // Draw layer labels
    this._drawLayerLabels(svg, activations, layerX);

    // Update activation details
    this._updateActivationDetails(activations);
  }

  private _calculateLayerSizes(activations: Activation<unknown, unknown>): number[] {
    const inputCount = activations.inputs.length;
    const hiddenCounts = activations.layers.map((layer) => layer.activation.length);
    // Ne pas ajouter outputCount car la dernière couche de activations.layers EST la sortie
    return [inputCount, ...hiddenCounts];
  }

  private _calculateLayerXPositions(layerCount: number): number[] {
    const spacing = (SVG_CONFIG.width - 2 * SVG_CONFIG.margin) / (layerCount - 1);
    const positions = [];
    for (let i = 0; i < layerCount; i++) {
      positions.push(SVG_CONFIG.margin + i * spacing);
    }
    return positions;
  }

  private _calculateAllLayerYPositions(layerSizes: number[]): number[][] {
    return layerSizes.map((size) => this._getNeuronYPositions(size, SVG_CONFIG.height));
  }

  private _clearSvg(svg: HTMLElement): void {
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }
  }

  private _drawAllConnections(
    svg: HTMLElement,
    weights: NeuralNetworkLayers,
    layerSizes: number[],
    layerX: number[],
    layerYPositions: number[][],
  ): void {
    // Draw connections between input and first hidden layer
    this._drawConnectionsBetweenLayers(
      svg,
      weights.layers[0],
      layerSizes[0],
      layerSizes[1],
      layerX[0],
      layerX[1],
      layerYPositions[0],
      layerYPositions[1],
    );

    // Draw connections between hidden layers and to output
    for (let i = 1; i < weights.layers.length; i++) {
      this._drawConnectionsBetweenLayers(
        svg,
        weights.layers[i],
        layerSizes[i],
        layerSizes[i + 1],
        layerX[i],
        layerX[i + 1],
        layerYPositions[i],
        layerYPositions[i + 1],
      );
    }
  }

  private _drawConnectionsBetweenLayers(
    svg: HTMLElement,
    layer: { weights: number[] | number[][] },
    fromSize: number,
    toSize: number,
    fromX: number,
    toX: number,
    fromY: number[],
    toY: number[],
  ): void {
    // Check if weights is an array of arrays or a flat array
    const isNestedArray = Array.isArray(layer.weights[0]);

    for (let i = 0; i < toSize; i++) {
      for (let j = 0; j < fromSize; j++) {
        let weight: number;

        if (isNestedArray) {
          // Weights are stored as array per output neuron
          weight = (layer.weights as number[][])[i][j];
        } else {
          // Weights are stored as flat array
          weight = (layer.weights as number[])[i * fromSize + j];
        }

        if (weight === undefined || isNaN(weight)) {
          continue;
        }

        const absWeight = Math.abs(weight);
        // Ajuster l'opacité pour rendre les poids plus visibles (min 0.3, max 0.9)
        const opacity = Math.min(absWeight / 2, 0.9) + 0.3;
        // Ajuster l'épaisseur pour rendre les poids plus visibles (min 0.8, max 3)
        const strokeWidth = Math.min(absWeight * 2, 2.5) + 0.8;
        const color = weight > 0 ? COLORS.positive : COLORS.negative;

        const line = document.createElementNS(SVG_NAMESPACE, 'line');
        line.setAttribute('x1', fromX.toString());
        line.setAttribute('y1', fromY[j].toString());
        line.setAttribute('x2', toX.toString());
        line.setAttribute('y2', toY[i].toString());
        line.setAttribute('stroke', color);
        line.setAttribute('stroke-width', strokeWidth.toString());
        line.setAttribute('stroke-opacity', opacity.toString());
        svg.appendChild(line);
      }
    }
  }

  private _drawInputLayer(
    svg: HTMLElement,
    activations: Activation<unknown, unknown>,
    x: number,
    yPositions: number[],
  ): void {
    activations.inputs.forEach((inputValue: unknown, i: number) => {
      const numericValue = inputValue as number;

      // Draw circle
      const circle = document.createElementNS(SVG_NAMESPACE, 'circle');
      circle.setAttribute('cx', x.toString());
      circle.setAttribute('cy', yPositions[i].toString());
      circle.setAttribute('r', SVG_CONFIG.neuronRadius.input.toString());
      circle.setAttribute(
        'fill',
        numericValue > ACTIVATION_THRESHOLD.binary ? COLORS.positive : COLORS.neutral,
      );
      circle.setAttribute('stroke', COLORS.stroke);
      circle.setAttribute('stroke-width', '2');
      svg.appendChild(circle);

      // Draw value
      const valueText = document.createElementNS(SVG_NAMESPACE, 'text');
      valueText.setAttribute('x', x.toString());
      valueText.setAttribute('y', (yPositions[i] + 5).toString());
      valueText.setAttribute('text-anchor', 'middle');
      valueText.setAttribute('fill', COLORS.stroke);
      valueText.setAttribute('font-weight', 'bold');
      valueText.setAttribute('font-size', SVG_CONFIG.fontSize.input);
      valueText.textContent = numericValue.toFixed(1);
      svg.appendChild(valueText);

      // Draw label
      const label = this._getInputLabel(i);
      const labelText = document.createElementNS(SVG_NAMESPACE, 'text');
      labelText.setAttribute('x', (x - 35).toString());
      labelText.setAttribute('y', (yPositions[i] + 5).toString());
      labelText.setAttribute('text-anchor', 'middle');
      labelText.setAttribute('fill', COLORS.label);
      labelText.setAttribute('font-size', SVG_CONFIG.fontSize.label);
      labelText.textContent = label;
      svg.appendChild(labelText);
    });
  }

  private _drawHiddenAndOutputLayers(
    svg: HTMLElement,
    activations: Activation<unknown, unknown>,
    layerX: number[],
    layerYPositions: number[][],
  ): void {
    const outputValues = activations.output as number[];

    activations.layers.forEach((layer, layerIndex) => {
      const x = layerX[layerIndex + 1]; // +1 because input is at index 0
      const yPositions = layerYPositions[layerIndex + 1];
      const isOutputLayer = layerIndex === activations.layers.length - 1;
      const isSoftmax = layer.function.toLowerCase() === 'softmax';

      // Pour la couche de sortie, utiliser activations.output au lieu de layer.activation
      const layerValues = isOutputLayer ? outputValues : layer.activation;

      layerValues.forEach((activationValue: number, neuronIndex: number) => {
        let color: string;
        let radius: number;
        let fontSize: string;

        if (isOutputLayer) {
          // Style pour la couche de sortie (grands cercles)
          radius = SVG_CONFIG.neuronRadius.output;
          fontSize = SVG_CONFIG.fontSize.output;

          if (isSoftmax) {
            color =
              activationValue > ACTIVATION_THRESHOLD.softmax ? COLORS.positive : COLORS.neutral;
          } else {
            color =
              activationValue > ACTIVATION_THRESHOLD.binary ? COLORS.positive : COLORS.negative;
          }
        } else {
          // Style pour les couches cachées (petits cercles)
          radius = SVG_CONFIG.neuronRadius.hidden;
          fontSize = SVG_CONFIG.fontSize.hidden;
          color = this._getNeuronColor(activationValue, layer.function);
        }

        // Draw circle
        const circle = document.createElementNS(SVG_NAMESPACE, 'circle');
        circle.setAttribute('cx', x.toString());
        circle.setAttribute('cy', yPositions[neuronIndex].toString());
        circle.setAttribute('r', radius.toString());
        circle.setAttribute('fill', color);
        circle.setAttribute('stroke', COLORS.stroke);
        circle.setAttribute('stroke-width', isOutputLayer ? '3' : '2');
        svg.appendChild(circle);

        // Draw value
        const valueText = document.createElementNS(SVG_NAMESPACE, 'text');
        valueText.setAttribute('x', x.toString());
        valueText.setAttribute('y', (yPositions[neuronIndex] + (isOutputLayer ? 6 : 4)).toString());
        valueText.setAttribute('text-anchor', 'middle');
        valueText.setAttribute('fill', COLORS.stroke);
        valueText.setAttribute('font-size', fontSize);
        if (isOutputLayer) {
          valueText.setAttribute('font-weight', 'bold');
        }
        valueText.textContent = activationValue.toFixed(2);
        svg.appendChild(valueText);

        // Draw label for output neurons
        if (isOutputLayer) {
          const label = outputValues.length > 1 ? `Out ${neuronIndex}` : 'Out';
          const labelText = document.createElementNS(SVG_NAMESPACE, 'text');
          labelText.setAttribute('x', (x + 40).toString());
          labelText.setAttribute('y', (yPositions[neuronIndex] + 5).toString());
          labelText.setAttribute('text-anchor', 'start');
          labelText.setAttribute('fill', COLORS.label);
          labelText.setAttribute('font-size', SVG_CONFIG.fontSize.label);
          labelText.textContent = label;
          svg.appendChild(labelText);
        }
      });
    });
  }

  private _drawLayerLabels(
    svg: HTMLElement,
    activations: Activation<unknown, unknown>,
    layerX: number[],
  ): void {
    // Input label
    const inputLabel = document.createElementNS(SVG_NAMESPACE, 'text');
    inputLabel.setAttribute('x', layerX[0].toString());
    inputLabel.setAttribute('y', SVG_CONFIG.labelY.toString());
    inputLabel.setAttribute('text-anchor', 'middle');
    inputLabel.setAttribute('fill', COLORS.neutral);
    inputLabel.setAttribute('font-size', SVG_CONFIG.fontSize.layerLabel);
    inputLabel.textContent = 'Input';
    svg.appendChild(inputLabel);

    // Hidden layers labels
    activations.layers.forEach((layer, i) => {
      const label = document.createElementNS(SVG_NAMESPACE, 'text');
      label.setAttribute('x', layerX[i + 1].toString());
      label.setAttribute('y', SVG_CONFIG.labelY.toString());
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('fill', COLORS.neutral);
      label.setAttribute('font-size', SVG_CONFIG.fontSize.layerLabel);

      const isOutputLayer = i === activations.layers.length - 1;
      if (isOutputLayer) {
        label.textContent = `Output (${layer.function})`;
      } else {
        label.textContent = `Hidden ${i + 1} (${layer.function})`;
      }
      svg.appendChild(label);
    });
  }

  private _updateActivationDetails(activations: Activation<unknown, unknown>): void {
    const detailsEl = document.getElementById('activationDetails');
    if (!detailsEl) {
      return;
    }
    detailsEl.textContent = '';

    // Display all hidden layers
    activations.layers.forEach((layer, i) => {
      const strong = document.createElement('strong');
      strong.textContent = `Layer ${i + 1} activations (${layer.function}): `;
      detailsEl.appendChild(strong);
      detailsEl.appendChild(
        document.createTextNode(
          '[' + layer.activation.map((v: number) => v.toFixed(3)).join(', ') + ']',
        ),
      );
      detailsEl.appendChild(document.createElement('br'));
    });

    // Display output
    const outputValues = activations.output as number[];
    const strong2 = document.createElement('strong');
    strong2.textContent = 'Output: ';
    detailsEl.appendChild(strong2);

    if (outputValues.length === 1) {
      detailsEl.appendChild(document.createTextNode(outputValues[0].toFixed(6) + ' → '));
      const strong3 = document.createElement('strong');
      strong3.textContent = outputValues[0] > ACTIVATION_THRESHOLD.binary ? '1' : '0';
      detailsEl.appendChild(strong3);
    } else {
      detailsEl.appendChild(
        document.createTextNode(
          '[' + outputValues.map((v: number) => v.toFixed(4)).join(', ') + ']',
        ),
      );
      const maxIndex = outputValues.indexOf(Math.max(...outputValues));
      const strong3 = document.createElement('strong');
      strong3.textContent = ` → Class ${maxIndex}`;
      detailsEl.appendChild(strong3);
    }
  }

  private _getInputLabel(index: number): string {
    const labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    return labels[index] || `I${index}`;
  }

  private _getNeuronColor(value: number, activationFunction: string): string {
    const func = activationFunction.toLowerCase();
    let normalized: number;

    if (func === 'tanh') {
      // Tanh output is -1 to 1
      normalized = (value + 1) / 2;
    } else if (func === 'sigmoid' || func === 'softmax') {
      // Sigmoid and Softmax output is 0 to 1
      normalized = value;
    } else if (func === 'relu') {
      // ReLU output is 0 to +inf, cap at 1
      normalized = Math.min(value, 1);
    } else {
      // Default normalization
      normalized = (value + 1) / 2;
    }

    // Gradient from red/blue to green
    const r = Math.round(normalized * 34 + (1 - normalized) * 239);
    const g = Math.round(normalized * 197 + (1 - normalized) * 68);
    const b = Math.round(normalized * 94 + (1 - normalized) * 68);
    return `rgb(${r},${g},${b})`;
  }

  private _getNeuronYPositions(count: number, height: number): number[] {
    const available = height - 2 * SVG_CONFIG.verticalMargin - SVG_CONFIG.verticalPadding;
    const spacing = count > 1 ? available / (count - 1) : 0;
    const positions = [];
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
