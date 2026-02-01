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
 * This renderer uses the native Canvas 2D API for high-performance rendering
 * of neural networks with thousands of connections and neurons.
 *
 * Features:
 * - Efficient batch rendering
 * - Respects CSS custom properties for theming
 * - Supports viewport transformations
 * - Optimized for 100K+ elements
 */
export class Canvas2DRenderer implements INetworkRenderer {
  private ctx: CanvasRenderingContext2D;
  private canvas: HTMLCanvasElement;
  private config: RenderConfig;
  private viewport: Viewport;
  private dpr: number;

  constructor(canvas: HTMLCanvasElement, config: Partial<RenderConfig> = {}) {
    this.canvas = canvas;
    this.config = { ...DEFAULT_RENDER_CONFIG, ...config };
    this.dpr = window.devicePixelRatio || 1;

    // Initialize context
    const ctx = canvas.getContext('2d', {
      alpha: true,
      desynchronized: true, // Hint for better performance
    });

    if (!ctx) {
      throw new Error('Failed to get 2D rendering context');
    }

    this.ctx = ctx;

    // Initialize viewport
    this.viewport = {
      x: 0,
      y: 0,
      width: canvas.width,
      height: canvas.height,
      scale: 1,
    };

    // Apply device pixel ratio for sharp rendering
    this.applyDevicePixelRatio();
  }

  /**
   * Apply device pixel ratio for crisp rendering on high-DPI displays
   */
  private applyDevicePixelRatio(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * this.dpr;
    this.canvas.height = rect.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);
  }

  /**
   * Get CSS custom property color from document
   */
  private getCssColor(variable: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
  }

  /**
   * Resolve CSS color (handles var() and direct colors)
   */
  private resolveColor(color: string): string {
    // If it's a CSS variable, resolve it
    if (color.startsWith('var(')) {
      const varName = color.match(/var\((--[^,)]+)/)?.[1];
      if (varName) {
        return this.getCssColor(varName) || color;
      }
    }
    return color;
  }

  /**
   * Apply viewport transformation
   */
  private applyViewportTransform(): void {
    this.ctx.save();
    this.ctx.translate(-this.viewport.x, -this.viewport.y);
    this.ctx.scale(this.viewport.scale, this.viewport.scale);
  }

  /**
   * Restore viewport transformation
   */
  private restoreViewportTransform(): void {
    this.ctx.restore();
  }

  /**
   * Render all connections (lines between neurons)
   */
  private renderConnections(connections: Connection[]): void {
    const maxConnections = this.config.maxConnections;
    const connectionsToRender = maxConnections ? connections.slice(0, maxConnections) : connections;

    // Batch rendering by color for performance
    const connectionsByColor = new Map<string, Connection[]>();

    for (const conn of connectionsToRender) {
      const key = `${conn.color}-${conn.opacity.toFixed(2)}`;
      if (!connectionsByColor.has(key)) {
        connectionsByColor.set(key, []);
      }
      connectionsByColor.get(key)!.push(conn);
    }

    // Render each batch
    for (const [_, batch] of connectionsByColor) {
      if (batch.length === 0) continue;

      const firstConn = batch[0];
      this.ctx.strokeStyle = this.resolveColor(firstConn.color);
      this.ctx.globalAlpha = firstConn.opacity;
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
   * Render all neurons (circles with values)
   */
  private renderNeurons(neurons: Neuron[]): void {
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

      // Draw label if present
      if (neuron.label && neuron.labelPosition) {
        const labelColor = this.getCssColor('--nn-label') || this.resolveColor(neuron.stroke);
        this.ctx.fillStyle = labelColor;
        this.ctx.font = `normal 11px 'Segoe UI', system-ui, sans-serif`;
        this.ctx.textAlign = neuron.labelAlign || 'center';
        this.ctx.fillText(neuron.label, neuron.labelPosition.x, neuron.labelPosition.y);
      }
    }
  }

  /**
   * Render all text labels
   */
  private renderLabels(labels: Label[]): void {
    this.ctx.textBaseline = 'middle';

    for (const label of labels) {
      this.ctx.fillStyle = this.resolveColor(label.color);
      this.ctx.font = `normal ${label.fontSize}px 'Segoe UI', system-ui, sans-serif`;
      this.ctx.textAlign = label.align;
      this.ctx.fillText(label.text, label.position.x, label.position.y);
    }
  }

  // ============================================================================
  // INetworkRenderer Implementation
  // ============================================================================

  render(data: NetworkRenderData): void {
    // Clear canvas
    this.clear();

    // Apply viewport transformation
    this.applyViewportTransform();

    // Render in order: connections first (background), then neurons, then labels
    this.renderConnections(data.connections);
    this.renderNeurons(data.neurons);
    this.renderLabels(data.labels);

    // Restore transformation
    this.restoreViewportTransform();

    // Debug info
    if (this.config.debug) {
      this.renderDebugInfo(data);
    }
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width / this.dpr, this.canvas.height / this.dpr);
  }

  resize(width: number, height: number): void {
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.canvas.width = width * this.dpr;
    this.canvas.height = height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);

    this.viewport.width = width;
    this.viewport.height = height;
  }

  setViewport(viewport: Viewport): void {
    this.viewport = { ...viewport };
  }

  updateConfig(config: Partial<RenderConfig>): void {
    this.config = { ...this.config, ...config };
  }

  destroy(): void {
    // Clean up resources
    this.clear();
    // Remove any event listeners if added in the future
  }

  getType(): RendererPreference {
    return 'canvas2d';
  }

  // ============================================================================
  // Debug Utilities
  // ============================================================================

  private renderDebugInfo(data: NetworkRenderData): void {
    this.ctx.fillStyle = this.getCssColor('--muted') || '#94a3b8';
    this.ctx.font = '10px monospace';
    this.ctx.textAlign = 'left';
    this.ctx.textBaseline = 'top';

    const info = [
      `Renderer: Canvas2D`,
      `Connections: ${data.connections.length}`,
      `Neurons: ${data.neurons.length}`,
      `Labels: ${data.labels.length}`,
      `DPR: ${this.dpr}`,
    ];

    info.forEach((line, i) => {
      this.ctx.fillText(line, 10, 10 + i * 12);
    });
  }
}
