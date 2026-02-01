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

/**
 * Canvas 2D renderer for neural network visualization.
 *
 * Content-First Architecture:
 * - Receives data in natural coordinates
 * - Applies single scale transformation
 * - All elements scale uniformly
 *
 * Features:
 * - Efficient batch rendering
 * - Respects CSS custom properties for theming
 * - Optimized for 100K+ elements
 */
export class Canvas2DRenderer implements INetworkRenderer {
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

  /**
   * Setup canvas with device pixel ratio for crisp rendering
   */
  private setupCanvas(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * this.dpr;
    this.canvas.height = rect.height * this.dpr;
  }

  /**
   * Resolve CSS variable colors
   */
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
  // Rendering Methods
  // ============================================================================

  /**
   * Render connections (lines between neurons)
   */
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

  /**
   * Render neurons (circles with values)
   */
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

      // Draw value text
      this.ctx.fillStyle = this.resolveColor(neuron.stroke);
      this.ctx.font = `${neuron.fontWeight} ${neuron.fontSize}px 'Segoe UI', system-ui, sans-serif`;
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(neuron.value, neuron.position.x, neuron.position.y);

      // Draw neuron label if present
      if (neuron.label && neuron.labelPosition) {
        this.ctx.fillStyle = this.resolveColor('var(--nn-neutral)');
        this.ctx.font = `normal 11px 'Segoe UI', system-ui, sans-serif`;
        this.ctx.textAlign = neuron.labelAlign || 'center';
        this.ctx.fillText(neuron.label, neuron.labelPosition.x, neuron.labelPosition.y);
      }
    }
  }

  /**
   * Render layer labels
   */
  private renderLabels(labels: readonly Label[]): void {
    this.ctx.textBaseline = 'middle';

    for (const label of labels) {
      this.ctx.fillStyle = this.resolveColor(label.color);
      this.ctx.font = `normal ${label.fontSize}px 'Segoe UI', system-ui, sans-serif`;
      this.ctx.textAlign = label.align;
      this.ctx.fillText(label.text, label.position.x, label.position.y);
    }
  }

  /**
   * Render debug information
   */
  private renderDebugInfo(data: NetworkRenderData, viewport: Viewport): void {
    this.ctx.fillStyle = this.resolveColor('var(--nn-neutral)');
    this.ctx.font = '10px monospace';
    this.ctx.textAlign = 'left';
    this.ctx.textBaseline = 'top';

    const info = [
      `Renderer: Canvas2D`,
      `Natural: ${data.naturalBounds.width.toFixed(0)}×${data.naturalBounds.height.toFixed(0)}`,
      `Scale: ${viewport.scale.toFixed(3)}`,
      `Connections: ${data.connections.length}`,
      `Neurons: ${data.neurons.length}`,
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
