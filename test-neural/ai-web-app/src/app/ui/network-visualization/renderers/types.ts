/**
 * Core types and interfaces for neural network rendering.
 * These types are renderer-agnostic and can be used by Canvas2D, WebGL, or WebGPU implementations.
 */

// ============================================================================
// Geometric Types
// ============================================================================

/**
 * 2D point in rendering space
 */
export interface Point {
  x: number;
  y: number;
}

/**
 * RGBA color representation
 */
export interface Color {
  r: number;
  g: number;
  b: number;
  a: number;
}

/**
 * CSS color string (for renderer convenience)
 */
export type CssColor = string;

/**
 * Viewport configuration
 */
export interface Viewport {
  x: number;
  y: number;
  width: number;
  height: number;
  scale: number;
}

// ============================================================================
// Render Data Types
// ============================================================================

/**
 * Connection between two neurons
 */
export interface Connection {
  /** Starting point */
  from: Point;
  /** Ending point */
  to: Point;
  /** Connection weight (affects visual style) */
  weight: number;
  /** Line color */
  color: CssColor;
  /** Line opacity (0-1) */
  opacity: number;
  /** Line width in pixels */
  strokeWidth: number;
}

/**
 * Neuron (node) in the network
 */
export interface Neuron {
  /** Center position */
  position: Point;
  /** Circle radius in pixels */
  radius: number;
  /** Activation value (affects color) */
  activation: number;
  /** Display value text */
  value: string;
  /** Fill color */
  fill: CssColor;
  /** Border color */
  stroke: CssColor;
  /** Border width */
  strokeWidth: number;
  /** Optional label (e.g., 'A', 'B', 'Out') */
  label?: string;
  /** Label position if present */
  labelPosition?: Point;
  /** Label alignment */
  labelAlign?: 'left' | 'center' | 'right';
  /** Font size for value text */
  fontSize: number;
  /** Font weight */
  fontWeight: 'normal' | 'bold';
}

/**
 * Text label for layers or other annotations
 */
export interface Label {
  /** Text position */
  position: Point;
  /** Text content */
  text: string;
  /** Text color */
  color: CssColor;
  /** Font size */
  fontSize: number;
  /** Text alignment */
  align: 'left' | 'center' | 'right';
}

/**
 * Complete network rendering data
 */
export interface NetworkRenderData {
  /** All connections between neurons */
  connections: Connection[];
  /** All neurons across all layers */
  neurons: Neuron[];
  /** Text labels (layer names, etc.) */
  labels: Label[];
}

// ============================================================================
// Renderer Configuration
// ============================================================================

/**
 * Renderer performance preferences
 */
export type RendererPreference = 'webgpu' | 'webgl' | 'canvas2d' | 'svg';

/**
 * Renderer configuration options
 */
export interface RenderConfig {
  /** Enable antialiasing (may impact performance) */
  antialias: boolean;
  /** Power preference for GPU selection */
  powerPreference: 'low-power' | 'high-performance' | 'default';
  /** Maximum connections to render (for performance) */
  maxConnections?: number;
  /** Level of detail */
  lodLevel: 'low' | 'medium' | 'high';
  /** Enable debug mode (shows FPS, etc.) */
  debug?: boolean;
}

/**
 * Default render configuration
 */
export const DEFAULT_RENDER_CONFIG: RenderConfig = {
  antialias: true,
  powerPreference: 'default',
  lodLevel: 'high',
  debug: false,
};

// ============================================================================
// Renderer Interface
// ============================================================================

/**
 * Abstract interface for neural network renderers.
 * All concrete renderers (Canvas2D, WebGL, WebGPU) must implement this interface.
 */
export interface INetworkRenderer {
  /**
   * Render the network with the provided data
   * @param data Network rendering data
   */
  render(data: NetworkRenderData): void;

  /**
   * Clear the rendering surface
   */
  clear(): void;

  /**
   * Resize the rendering surface
   * @param width New width in pixels
   * @param height New height in pixels
   */
  resize(width: number, height: number): void;

  /**
   * Set the viewport (for zooming/panning)
   * @param viewport Viewport configuration
   */
  setViewport(viewport: Viewport): void;

  /**
   * Update renderer configuration
   * @param config Partial configuration to update
   */
  updateConfig(config: Partial<RenderConfig>): void;

  /**
   * Clean up resources and destroy the renderer
   */
  destroy(): void;

  /**
   * Get the renderer type name
   */
  getType(): RendererPreference;
}
