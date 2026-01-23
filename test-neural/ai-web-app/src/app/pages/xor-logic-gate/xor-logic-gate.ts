import { ChangeDetectorRef, Component, computed, effect, inject, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';
import { PageFooter } from '../../ui/page-footer/page-footer';
import { NavigationBack } from '../../ui/navigation-back/navigation-back';
import { Loader } from '../../ui/loader/loader';
import { WasmFacade } from '@cma/wasm/shared/wasm';
import { ModelInfoComponent } from '../../ui/model-info/model-info';
import { DecimalPipe } from '@angular/common';
import { XorNetwork } from '@cma/wasm/xor_wasm/neural_wasm_xor';
import { NeuralNetworkLayers } from '@cma/wasm/shared/model-info';

type NetworkPrediction = {
    confidence: number,
    prediction: number,
    probabilities: [
        number,
        number
    ],
    raw: number
}

@Component({
  selector: 'app-xor-logic-gate',
  imports: [DecimalPipe, PageTitle, Loader, NavigationBack, ModelInfoComponent, PageFooter],
  templateUrl: './xor-logic-gate.html',
  styleUrl: './xor-logic-gate.scss',
})
export class XorLogicGate {
  private readonly changeDetector = inject(ChangeDetectorRef);  
  private readonly wasmService = inject(WasmFacade);
  public readonly xorIsLoading = this.wasmService.xorWasmResource.isLoading;
  public readonly xorNetwork = this.wasmService.xorNetwork;
  public readonly xorModelInfo = this.wasmService.xorModelInfo;
  public readonly xorArchitecture = this.wasmService.xorArchitecture;
  public readonly xorWeights = this.wasmService.xorWeights;
  public readonly xorTestAll = this.wasmService.xorTestAll;
  
  public readonly pageTitleOptions: Signal<PageTitleOptions> = signal({
    title: 'Neural XOR',
    subtitle: 'WebAssembly Neural Network Demo',
    icon: '🧠',
  });

  public readonly inputA = signal(0);
  public readonly inputB = signal(0);
  public readonly output = computed(() => {
    const network = this.xorNetwork();
    if (!network) {
      return null;
    }
    const inputA = this.inputA();
    const inputB = this.inputB();
    const prediction = network.predict(inputA, inputB);
    const output = JSON.parse(prediction) as NetworkPrediction;
    console.log('XOR Prediction:', output);
    return output;
  });
  public readonly activations = computed(() => {
    const network = this.xorNetwork();
    if (!network) {
      return null;
    }
    const inputA = this.inputA();
    const inputB = this.inputB();
    const acts = network.get_activations(inputA, inputB);
    return acts;
  })
  public readonly predictionDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    return output.prediction;
  });
  public readonly confidenceDisplay = computed(() => {
    const output = this.output();
    if (!output) {
      return 'N/A';
    }
    return (output.confidence * 100).toFixed(1) + '% confidence';
  });

  constructor() {
    effect(() => {
      const isLoading = this.xorIsLoading();
      if (isLoading) {
        return;
      }
      const network = this.xorNetwork();
      if (!network) {
        return;
      }
      const weights = this.xorWeights();
      if (!weights) {
        return;
      }
      const inputA = this.inputA();
      const inputB = this.inputB();
      console.log('Updating network viz for inputs:', inputA, inputB);
      setTimeout(() => {
        this._updateNetworkViz(network, weights, inputA, inputB);
      }, 0);
    });
    
  }
  public toggleInput(type: 'A' | 'B'): void {
    if (type === 'A') {
      this.inputA.set(this.inputA() === 0 ? 1 : 0);
    } else {
      this.inputB.set(this.inputB() === 0 ? 1 : 0);
    }
    console.log(this.xorWeights());
  }

  // TODO - create component !!!!
  private _updateNetworkViz(network: XorNetwork, weights: NeuralNetworkLayers, inputA: number, inputB: number): void {
    if (!network) {
      return;
    }
    const activations = JSON.parse(network.get_activations(inputA, inputB));
    
    
    const svg = document.getElementById('networkViz');
    if (!svg) {
      return;
    }
    const NS = 'http://www.w3.org/2000/svg';

    const layerSizes = [2, ...weights.layers.map((l: any) => l.shape[0])];
    const layerX = [60, 250, 440];
    const colors = {
        positive: '#22c55e',
        negative: '#ef4444',
        neutral: '#64748b'
    };
    
    // Clear previous content
    while (svg.firstChild) {
        svg.removeChild(svg.firstChild);
    }
    
    // Helper to create SVG elements safely
    function createLine(x1: string, y1: string, x2: string, y2: string, stroke: string, strokeWidth: string, strokeOpacity: string) {
        const line = document.createElementNS(NS, 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', stroke);
        line.setAttribute('stroke-width', strokeWidth);
        line.setAttribute('stroke-opacity', strokeOpacity);
        return line;
    }
    
    function createCircle(cx: string, cy: string, r: string, fill: string, stroke: string, strokeWidth: string) {
        const circle = document.createElementNS(NS, 'circle');
        circle.setAttribute('cx', cx);
        circle.setAttribute('cy', cy);
        circle.setAttribute('r', r);
        circle.setAttribute('fill', fill);
        circle.setAttribute('stroke', stroke);
        circle.setAttribute('stroke-width', strokeWidth);
        return circle;
    }
    
    function createText(x: string, y: string, content: string, options: { anchor?: string; fill?: string; fontSize?: string; fontWeight?: string } = {}) {
        const text = document.createElementNS(NS, 'text');
        text.setAttribute('x', x);
        text.setAttribute('y', y);
        text.setAttribute('text-anchor', options.anchor || 'middle');
        text.setAttribute('fill', options.fill || 'white');
        if (options.fontSize) text.setAttribute('font-size', options.fontSize);
        if (options.fontWeight) text.setAttribute('font-weight', options.fontWeight);
        text.textContent = content;
        return text;
    }
    
    // Calculate positions
    const hiddenY = this._getNeuronYPositions(layerSizes[1], 280);
    const inputY = this._getNeuronYPositions(2, 280);
    const outputY = [140];
    
    // Draw connections: input → hidden
    for (let i = 0; i < layerSizes[1]; i++) {
        for (let j = 0; j < 2; j++) {
            const w = weights.layers[0].weights[i * 2 + j];
            const opacity = Math.min(Math.abs(w) / 5, 1);
            const color = w > 0 ? colors.positive : colors.negative;
            svg.appendChild(createLine(
                 layerX[0].toString(), inputY[j].toString(), layerX[1].toString(), hiddenY[i].toString(),
                color, (Math.abs(w) / 3 + 0.5).toString(), (opacity * 0.6).toString()
            ));
        }
    }
    
    // Draw connections: hidden → output
    for (let i = 0; i < layerSizes[1]; i++) {
        const w = weights.layers[1].weights[i];
        const opacity = Math.min(Math.abs(w) / 5, 1);
        const color = w > 0 ? colors.positive : colors.negative;
        svg.appendChild(createLine(
            layerX[1].toString(), hiddenY[i].toString(), layerX[2].toString(), outputY[0].toString(),
            color, (Math.abs(w) / 3 + 0.5).toString(), (opacity * 0.6).toString()
        ));
    }
    
    // Draw neurons - Input layer
    activations.inputs.forEach((val: number, i: number) => {
        const intensity = val;
        svg.appendChild(createCircle(
            layerX[0].toString(), inputY[i].toString(), "20",
            intensity > 0.5 ? colors.positive : colors.neutral, 'white', "2"
        ));
        svg.appendChild(createText(
            layerX[0].toString(), (inputY[i] + 5).toString(), val.toFixed(0),
            { fontWeight: 'bold', fontSize: '14' }
        ));
        svg.appendChild(createText(
            (layerX[0] - 35).toString(), (inputY[i] + 5).toString(), i === 0 ? 'A' : 'B',
            { fill: '#94a3b8', fontSize: '11' }
        ));
    });
    
    // Draw neurons - Hidden layer
    const hiddenActivations = activations.layers[0].activation;
    hiddenActivations.forEach((val: number, i: number) => {
        const normalized = (val + 1) / 2; // tanh output is -1 to 1
        const r = Math.round(normalized * 34 + (1 - normalized) * 239);
        const g = Math.round(normalized * 197 + (1 - normalized) * 68);
        const b = Math.round(normalized * 94 + (1 - normalized) * 68);
        svg.appendChild(createCircle(
            layerX[1].toString(), hiddenY[i].toString(), "16",
            `rgb(${r},${g},${b})`, 'white', "2"
        ));
        svg.appendChild(createText(
            layerX[1].toString(), (hiddenY[i] + 4).toString(), val.toFixed(2),
            { fontSize: '9' }
        ));
    });
    
    // Draw neurons - Output layer
    const outputVal = activations.output;
    const outColor = outputVal > 0.5 ? colors.positive : colors.negative;
    svg.appendChild(createCircle(layerX[2].toString(), outputY[0].toString(), "25", outColor, 'white', "3"));
    svg.appendChild(createText(
        layerX[2].toString(), (outputY[0] + 6).toString(), outputVal.toFixed(2),
        { fontWeight: 'bold', fontSize: '16' }
    ));
    svg.appendChild(createText(
        (layerX[2] + 40).toString(), (outputY[0] + 5).toString(), 'Out',
        { anchor: 'start', fill: '#94a3b8', fontSize: '11' }
    ));
    
    // Layer labels
    svg.appendChild(createText(layerX[0].toString(), "270", 'Input', { fill: '#64748b', fontSize: '10' }));
    svg.appendChild(createText(layerX[1].toString(), "270", 'Hidden (Tanh)', { fill: '#64748b', fontSize: '10' }));
    svg.appendChild(createText(layerX[2].toString(), "270", 'Output (Sigmoid)', { fill: '#64748b', fontSize: '10' }));
    
    // Update details (safe - using textContent for dynamic values)
    const detailsEl = document.getElementById('activationDetails');
    if (!detailsEl) {
      return;
    }
    detailsEl.textContent = '';
    
    const strong1 = document.createElement('strong');
    strong1.textContent = 'Hidden layer activations: ';
    detailsEl.appendChild(strong1);
    detailsEl.appendChild(document.createTextNode('[' + hiddenActivations.map((v: number) => v.toFixed(3)).join(', ') + ']'));
    detailsEl.appendChild(document.createElement('br'));
    
    const strong2 = document.createElement('strong');
    strong2.textContent = 'Output: ';
    detailsEl.appendChild(strong2);
    detailsEl.appendChild(document.createTextNode(outputVal.toFixed(6) + ' → '));
    
    const strong3 = document.createElement('strong');
    strong3.textContent = outputVal > 0.5 ? '1' : '0';
    detailsEl.appendChild(strong3);
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
