import {
  Connection,
  DEFAULT_RENDER_CONFIG,
  INetworkRenderer,
  Label,
  NetworkRenderData,
  Neuron,
  RenderConfig,
  RendererPreference,
  Viewport,
} from './types';

import {
  BarData,
  ConfigurableRenderData,
  GridData,
  LayerElement,
  StatsData,
} from './configurable-layout-calculator';

/**
 * Configurable Canvas 2D renderer for neural network visualization.
 *
 * Extends the basic Canvas2DRenderer to support:
 * - Heatmap layer representation
 * - Bar layer representation
 * - Sampled neurons
 * - Stats display
 * - Collapsed layers
 *
 * Content-First Architecture:
 * - Receives data in natural coordinates
 * - Applies single scale transformation
 * - All elements scale uniformly
 */
export class ConfigurableCanvas2DRenderer implements INetworkRenderer {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly canvas: HTMLCanvasElement;
  private config: RenderConfig;
  private dpr: number;

  constructor(canvas: HTMLCanvasElement, config: Partial<RenderConfig> = {}) {
    this.canvas = canvas;
    this.config = { ...DEFAULT_RENDER_CONFIG, ...config };
    this.dpr = window.devicePixelRatio || 1;

    const ctx = canvas.getContext('2d', {
      alpha: true,
      desynchronized: true,
    });

    if (!ctx) {
      throw new Error('Failed to get 2D rendering context');
    }

    this.ctx = ctx;
    this.setupCanvas();
  }

  // ============================================================================
  // Canvas Setup
  // ============================================================================

  private setupCanvas(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * this.dpr;
    this.canvas.height = rect.height * this.dpr;
  }

  private resolveColor(color: string): string {
    if (color.startsWith('var(')) {
      const varName = color.match(/var\((--[^,)]+)/)?.[1];
      if (varName) {
        const resolved = getComputedStyle(document.documentElement)
          .getPropertyValue(varName)
          .trim();
        return resolved || color;
      }
    }
    return color;
  }

  // ============================================================================
  // Connection Rendering
  // ============================================================================

  private renderConnections(connections: readonly Connection[]): void {
    const maxConnections = this.config.maxConnections;
    const toRender = maxConnections ? connections.slice(0, maxConnections) : connections;

    // Batch by color/opacity for performance
    const batches = new Map<string, Connection[]>();
    for (const conn of toRender) {
      const key = `${conn.color}-${conn.opacity.toFixed(2)}`;
      if (!batches.has(key)) batches.set(key, []);
      batches.get(key)!.push(conn);
    }

    for (const batch of batches.values()) {
      if (batch.length === 0) continue;

      const first = batch[0];
      this.ctx.strokeStyle = this.resolveColor(first.color);
      this.ctx.globalAlpha = first.opacity;
      this.ctx.lineCap = 'round';

      this.ctx.beginPath();
      for (const conn of batch) {
        this.ctx.lineWidth = conn.strokeWidth;
        this.ctx.moveTo(conn.from.x, conn.from.y);
        this.ctx.lineTo(conn.to.x, conn.to.y);
      }
      this.ctx.stroke();
    }

    this.ctx.globalAlpha = 1;
  }

  // ============================================================================
  // Neuron Rendering
  // ============================================================================

  private renderNeurons(neurons: readonly Neuron[]): void {
    for (const neuron of neurons) {
      // Draw circle
      this.ctx.fillStyle = this.resolveColor(neuron.fill);
      this.ctx.strokeStyle = this.resolveColor(neuron.stroke);
      this.ctx.lineWidth = neuron.strokeWidth;

      this.ctx.beginPath();
      this.ctx.arc(neuron.position.x, neuron.position.y, neuron.radius, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.stroke();

      // Draw value text if not empty
      if (neuron.value) {
        this.ctx.fillStyle = this.resolveColor(neuron.stroke);
        this.ctx.font = `${neuron.fontWeight} ${neuron.fontSize}px 'Segoe UI', system-ui, sans-serif`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(neuron.value, neuron.position.x, neuron.position.y);
      }

      // Draw neuron label if present
      if (neuron.label && neuron.labelPosition) {
        this.ctx.fillStyle = this.resolveColor('var(--nn-neutral)');
        this.ctx.font = `normal 11px 'Segoe UI', system-ui, sans-serif`;
        this.ctx.textAlign = neuron.labelAlign || 'center';
        this.ctx.fillText(neuron.label, neuron.labelPosition.x, neuron.labelPosition.y);
      }
    }
  }

  // ============================================================================
  // Layer Element Rendering (New representations)
  // ============================================================================

  private renderLayerElements(elements: readonly LayerElement[]): void {
    for (const element of elements) {
      switch (element.type) {
        case 'neurons':
        case 'sampled':
          // Neurons are rendered separately via renderNeurons
          break;

        case 'heatmap':
          if (element.gridData) {
            this.renderHeatmap(element, element.gridData);
          }
          break;

        case 'bar':
          if (element.barData) {
            this.renderBar(element, element.barData);
          }
          break;

        case 'stats':
        case 'collapsed':
          if (element.statsData) {
            this.renderStats(element, element.statsData);
          }
          break;
      }
    }
  }

  private renderHeatmap(element: LayerElement, gridData: GridData): void {
    const { rows, cols, cellSize, colors } = gridData;
    const startX = element.position.x - (cols * cellSize) / 2;
    const startY = element.position.y - (rows * cellSize) / 2;

    // Draw border
    this.ctx.strokeStyle = this.resolveColor('var(--nn-stroke)');
    this.ctx.lineWidth = 1;
    this.ctx.strokeRect(startX, startY, cols * cellSize, rows * cellSize);

    // Draw cells
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const idx = row * cols + col;
        const color = colors[idx] ?? '#333';
        const x = startX + col * cellSize;
        const y = startY + row * cellSize;

        this.ctx.fillStyle = color;
        this.ctx.fillRect(x, y, cellSize, cellSize);
      }
    }

    // Draw grid lines if cells are large enough
    if (cellSize >= 4) {
      this.ctx.strokeStyle = 'rgba(255,255,255,0.1)';
      this.ctx.lineWidth = 0.5;

      for (let row = 0; row <= rows; row++) {
        this.ctx.beginPath();
        this.ctx.moveTo(startX, startY + row * cellSize);
        this.ctx.lineTo(startX + cols * cellSize, startY + row * cellSize);
        this.ctx.stroke();
      }

      for (let col = 0; col <= cols; col++) {
        this.ctx.beginPath();
        this.ctx.moveTo(startX + col * cellSize, startY);
        this.ctx.lineTo(startX + col * cellSize, startY + rows * cellSize);
        this.ctx.stroke();
      }
    }
  }

  private renderBar(element: LayerElement, barData: BarData): void {
    const { width, height, min, max, mean, colorGradient } = barData;
    const startX = element.position.x - width / 2;
    const startY = element.position.y - height / 2;

    // Draw background
    this.ctx.fillStyle = '#1a1a2e';
    this.ctx.fillRect(startX, startY, width, height);

    // Draw gradient representing distribution
    if (colorGradient.length > 0) {
      const bandWidth = width / Math.min(colorGradient.length, 50);
      const step = Math.max(1, Math.floor(colorGradient.length / 50));

      for (let i = 0; i < colorGradient.length; i += step) {
        const x = startX + (i / colorGradient.length) * width;
        this.ctx.fillStyle = colorGradient[i];
        this.ctx.fillRect(x, startY, bandWidth + 1, height);
      }
    }

    // Draw border
    this.ctx.strokeStyle = this.resolveColor('var(--nn-stroke)');
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(startX, startY, width, height);

    // Draw mean indicator
    const meanPosition = (mean - min) / (max - min || 1);
    const meanX = startX + meanPosition * width;
    this.ctx.strokeStyle = 'white';
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.moveTo(meanX, startY);
    this.ctx.lineTo(meanX, startY + height);
    this.ctx.stroke();

    // Draw stats text
    this.ctx.fillStyle = 'white';
    this.ctx.font = '10px monospace';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'bottom';
    this.ctx.fillText(`μ=${mean.toFixed(2)}`, element.position.x, startY - 2);
  }

  private renderStats(element: LayerElement, statsData: StatsData): void {
    const { count, min, max, mean, std } = statsData;
    const { width, height } = element;
    const startX = element.position.x - width / 2;
    const startY = element.position.y - height / 2;

    // Draw rounded rectangle background
    this.ctx.fillStyle = '#1a1a2e';
    this.ctx.strokeStyle = this.resolveColor('var(--nn-stroke)');
    this.ctx.lineWidth = 2;

    this.roundRect(startX, startY, width, height, 8);
    this.ctx.fill();
    this.ctx.stroke();

    // Draw count badge
    this.ctx.fillStyle = 'var(--nn-positive)';
    this.ctx.fillStyle = this.resolveColor('var(--nn-positive)');
    this.ctx.font = 'bold 14px monospace';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText(`n=${count}`, element.position.x, element.position.y - 10);

    // Draw stats
    this.ctx.fillStyle = 'white';
    this.ctx.font = '10px monospace';
    this.ctx.fillText(`μ=${mean.toFixed(2)}`, element.position.x, element.position.y + 8);
    this.ctx.fillText(`σ=${std.toFixed(2)}`, element.position.x, element.position.y + 20);
  }

  private roundRect(x: number, y: number, w: number, h: number, r: number): void {
    this.ctx.beginPath();
    this.ctx.moveTo(x + r, y);
    this.ctx.lineTo(x + w - r, y);
    this.ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    this.ctx.lineTo(x + w, y + h - r);
    this.ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    this.ctx.lineTo(x + r, y + h);
    this.ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    this.ctx.lineTo(x, y + r);
    this.ctx.quadraticCurveTo(x, y, x + r, y);
    this.ctx.closePath();
  }

  // ============================================================================
  // Label Rendering
  // ============================================================================

  private renderLabels(labels: readonly Label[]): void {
    this.ctx.textBaseline = 'middle';

    for (const label of labels) {
      this.ctx.fillStyle = this.resolveColor(label.color);
      this.ctx.font = `normal ${label.fontSize}px 'Segoe UI', system-ui, sans-serif`;
      this.ctx.textAlign = label.align;
      this.ctx.fillText(label.text, label.position.x, label.position.y);
    }
  }

  // ============================================================================
  // Debug Rendering
  // ============================================================================

  private renderDebugInfo(data: NetworkRenderData, viewport: Viewport): void {
    this.ctx.fillStyle = this.resolveColor('var(--nn-neutral)');
    this.ctx.font = '10px monospace';
    this.ctx.textAlign = 'left';
    this.ctx.textBaseline = 'top';

    const configurableData = data as ConfigurableRenderData;
    const layerTypes = configurableData.layerElements?.map((e) => e.type).join(', ') ?? 'neurons';

    const info = [
      `Renderer: Canvas2D (Configurable)`,
      `Natural: ${data.naturalBounds.width.toFixed(0)}×${data.naturalBounds.height.toFixed(0)}`,
      `Scale: ${viewport.scale.toFixed(3)}`,
      `Connections: ${data.connections.length}`,
      `Neurons: ${data.neurons.length}`,
      `Layers: ${layerTypes}`,
    ];

    info.forEach((line, i) => {
      this.ctx.fillText(line, 10, 10 + i * 12);
    });
  }

  // ============================================================================
  // INetworkRenderer Implementation
  // ============================================================================

  render(data: NetworkRenderData, viewport: Viewport): void {
    this.clear();

    // Apply device pixel ratio and viewport transformation
    this.ctx.save();
    this.ctx.scale(this.dpr, this.dpr);
    this.ctx.translate(viewport.offsetX, viewport.offsetY);
    this.ctx.scale(viewport.scale, viewport.scale);

    // Render in natural coordinates (scale applied via transformation)
    this.renderConnections(data.connections);

    // Render layer elements (heatmaps, bars, stats)
    const configurableData = data as ConfigurableRenderData;
    if (configurableData.layerElements) {
      this.renderLayerElements(configurableData.layerElements);
    }

    // Render neurons (from neuron-based elements)
    this.renderNeurons(data.neurons);
    this.renderLabels(data.labels);

    this.ctx.restore();

    // Debug info in screen space
    if (this.config.debug) {
      this.ctx.save();
      this.ctx.scale(this.dpr, this.dpr);
      this.renderDebugInfo(data, viewport);
      this.ctx.restore();
    }
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  resize(width: number, height: number): void {
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.canvas.width = width * this.dpr;
    this.canvas.height = height * this.dpr;
  }

  updateConfig(config: Partial<RenderConfig>): void {
    this.config = { ...this.config, ...config };
  }

  destroy(): void {
    this.clear();
  }

  getType(): RendererPreference {
    return 'canvas2d';
  }
}
