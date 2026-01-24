import { Component, effect, input } from '@angular/core';
import { Activation, NeuralNetworkLayers } from '@cma/wasm/shared';

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
      console.log('NeuralNetworkModelVizualizer - activations changed:', activations);
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
    console.log(
      'Updating network visualization with activations:',
      activations,
      'and weights:',
      weights,
    );

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

    const NS = 'http://www.w3.org/2000/svg';

    // Draw all connections between layers
    this._drawAllConnections(svg, NS, weights, layerSizes, layerX, layerYPositions);

    // Draw all neuron layers
    this._drawInputLayer(svg, NS, activations, layerX[0], layerYPositions[0]);
    this._drawHiddenAndOutputLayers(svg, NS, activations, layerX, layerYPositions);

    // Draw layer labels
    this._drawLayerLabels(svg, NS, activations, layerX);

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
    const svgWidth = 500;
    const margin = 60;
    const spacing = (svgWidth - 2 * margin) / (layerCount - 1);
    const positions = [];
    for (let i = 0; i < layerCount; i++) {
      positions.push(margin + i * spacing);
    }
    return positions;
  }

  private _calculateAllLayerYPositions(layerSizes: number[]): number[][] {
    return layerSizes.map((size) => this._getNeuronYPositions(size, 280));
  }

  private _clearSvg(svg: HTMLElement): void {
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }
  }

  private _drawAllConnections(
    svg: HTMLElement,
    NS: string,
    weights: NeuralNetworkLayers,
    layerSizes: number[],
    layerX: number[],
    layerYPositions: number[][],
  ): void {
    // Draw connections between input and first hidden layer
    this._drawConnectionsBetweenLayers(
      svg,
      NS,
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
        NS,
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
    NS: string,
    layer: any,
    fromSize: number,
    toSize: number,
    fromX: number,
    toX: number,
    fromY: number[],
    toY: number[],
  ): void {
    const colors = {
      positive: '#22c55e',
      negative: '#ef4444',
    };

    // Check if weights is an array of arrays or a flat array
    const isNestedArray = Array.isArray(layer.weights[0]);

    for (let i = 0; i < toSize; i++) {
      for (let j = 0; j < fromSize; j++) {
        let w: number;

        if (isNestedArray) {
          // Weights are stored as array per output neuron
          w = layer.weights[i][j];
        } else {
          // Weights are stored as flat array
          w = layer.weights[i * fromSize + j];
        }

        if (w === undefined || isNaN(w)) {
          console.warn(`Invalid weight at [${i}][${j}]:`, w);
          continue;
        }

        const absWeight = Math.abs(w);
        // Ajuster l'opacité pour rendre les poids plus visibles (min 0.3, max 0.9)
        const opacity = Math.min(absWeight / 2, 0.9) + 0.3;
        // Ajuster l'épaisseur pour rendre les poids plus visibles (min 0.8, max 3)
        const strokeWidth = Math.min(absWeight * 2, 2.5) + 0.8;
        const color = w > 0 ? colors.positive : colors.negative;

        const line = document.createElementNS(NS, 'line');
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
    NS: string,
    activations: Activation<unknown, unknown>,
    x: number,
    yPositions: number[],
  ): void {
    const colors = {
      positive: '#22c55e',
      neutral: '#64748b',
    };

    activations.inputs.forEach((val: unknown, i: number) => {
      const numVal = val as number;
      const intensity = numVal;

      // Draw circle
      const circle = document.createElementNS(NS, 'circle');
      circle.setAttribute('cx', x.toString());
      circle.setAttribute('cy', yPositions[i].toString());
      circle.setAttribute('r', '20');
      circle.setAttribute('fill', intensity > 0.5 ? colors.positive : colors.neutral);
      circle.setAttribute('stroke', 'white');
      circle.setAttribute('stroke-width', '2');
      svg.appendChild(circle);

      // Draw value
      const valueText = document.createElementNS(NS, 'text');
      valueText.setAttribute('x', x.toString());
      valueText.setAttribute('y', (yPositions[i] + 5).toString());
      valueText.setAttribute('text-anchor', 'middle');
      valueText.setAttribute('fill', 'white');
      valueText.setAttribute('font-weight', 'bold');
      valueText.setAttribute('font-size', '14');
      valueText.textContent = numVal.toFixed(1);
      svg.appendChild(valueText);

      // Draw label
      const label = this._getInputLabel(i);
      const labelText = document.createElementNS(NS, 'text');
      labelText.setAttribute('x', (x - 35).toString());
      labelText.setAttribute('y', (yPositions[i] + 5).toString());
      labelText.setAttribute('text-anchor', 'middle');
      labelText.setAttribute('fill', '#94a3b8');
      labelText.setAttribute('font-size', '11');
      labelText.textContent = label;
      svg.appendChild(labelText);
    });
  }

  private _drawHiddenAndOutputLayers(
    svg: HTMLElement,
    NS: string,
    activations: Activation<unknown, unknown>,
    layerX: number[],
    layerYPositions: number[][],
  ): void {
    const outputVal = activations.output as number[];
    const colors = {
      positive: '#22c55e',
      negative: '#ef4444',
      neutral: '#64748b',
    };

    activations.layers.forEach((layer, layerIndex) => {
      const x = layerX[layerIndex + 1]; // +1 because input is at index 0
      const yPositions = layerYPositions[layerIndex + 1];
      const isOutputLayer = layerIndex === activations.layers.length - 1;
      const isSoftmax = layer.function.toLowerCase() === 'softmax';

      layer.activation.forEach((val: number, neuronIndex: number) => {
        let color: string;
        let radius: string;
        let fontSize: string;

        if (isOutputLayer) {
          // Style pour la couche de sortie (grands cercles)
          radius = '25';
          fontSize = '16';

          if (isSoftmax) {
            color = val > 0.33 ? colors.positive : colors.neutral;
          } else {
            color = val > 0.5 ? colors.positive : colors.negative;
          }
        } else {
          // Style pour les couches cachées (petits cercles)
          radius = '16';
          fontSize = '9';
          color = this._getNeuronColor(val, layer.function);
        }

        // Draw circle
        const circle = document.createElementNS(NS, 'circle');
        circle.setAttribute('cx', x.toString());
        circle.setAttribute('cy', yPositions[neuronIndex].toString());
        circle.setAttribute('r', radius);
        circle.setAttribute('fill', color);
        circle.setAttribute('stroke', 'white');
        circle.setAttribute('stroke-width', isOutputLayer ? '3' : '2');
        svg.appendChild(circle);

        // Draw value
        const valueText = document.createElementNS(NS, 'text');
        valueText.setAttribute('x', x.toString());
        valueText.setAttribute('y', (yPositions[neuronIndex] + (isOutputLayer ? 6 : 4)).toString());
        valueText.setAttribute('text-anchor', 'middle');
        valueText.setAttribute('fill', 'white');
        valueText.setAttribute('font-size', fontSize);
        if (isOutputLayer) {
          valueText.setAttribute('font-weight', 'bold');
        }
        valueText.textContent = val.toFixed(2);
        svg.appendChild(valueText);

        // Draw label for output neurons
        if (isOutputLayer) {
          const label = outputVal.length > 1 ? `Out ${neuronIndex}` : 'Out';
          const labelText = document.createElementNS(NS, 'text');
          labelText.setAttribute('x', (x + 40).toString());
          labelText.setAttribute('y', (yPositions[neuronIndex] + 5).toString());
          labelText.setAttribute('text-anchor', 'start');
          labelText.setAttribute('fill', '#94a3b8');
          labelText.setAttribute('font-size', '11');
          labelText.textContent = label;
          svg.appendChild(labelText);
        }
      });
    });
  }

  private _drawLayerLabels(
    svg: HTMLElement,
    NS: string,
    activations: Activation<unknown, unknown>,
    layerX: number[],
  ): void {
    // Input label
    const inputLabel = document.createElementNS(NS, 'text');
    inputLabel.setAttribute('x', layerX[0].toString());
    inputLabel.setAttribute('y', '270');
    inputLabel.setAttribute('text-anchor', 'middle');
    inputLabel.setAttribute('fill', '#64748b');
    inputLabel.setAttribute('font-size', '10');
    inputLabel.textContent = 'Input';
    svg.appendChild(inputLabel);

    // Hidden layers labels
    activations.layers.forEach((layer, i) => {
      const label = document.createElementNS(NS, 'text');
      label.setAttribute('x', layerX[i + 1].toString());
      label.setAttribute('y', '270');
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('fill', '#64748b');
      label.setAttribute('font-size', '10');

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
    const outputVal = activations.output as number[];
    const strong2 = document.createElement('strong');
    strong2.textContent = 'Output: ';
    detailsEl.appendChild(strong2);

    if (outputVal.length === 1) {
      detailsEl.appendChild(document.createTextNode(outputVal[0].toFixed(6) + ' → '));
      const strong3 = document.createElement('strong');
      strong3.textContent = outputVal[0] > 0.5 ? '1' : '0';
      detailsEl.appendChild(strong3);
    } else {
      detailsEl.appendChild(
        document.createTextNode('[' + outputVal.map((v: number) => v.toFixed(4)).join(', ') + ']'),
      );
      const maxIndex = outputVal.indexOf(Math.max(...outputVal));
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
    const margin = 30;
    const available = height - 2 * margin - 40;
    const spacing = count > 1 ? available / (count - 1) : 0;
    const positions = [];
    const startY = margin + (height - 2 * margin - 40 - spacing * (count - 1)) / 2;
    for (let i = 0; i < count; i++) {
      positions.push(startY + i * spacing);
    }
    return positions;
  }
}
