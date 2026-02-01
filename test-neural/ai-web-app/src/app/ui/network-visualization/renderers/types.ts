/**
 * Core types and interfaces for neural network rendering.
 *
 * Architecture: Content-First Rendering
 * =====================================
 * 1. Calculate natural size based on readability constraints
 * 2. Draw in virtual coordinate space (natural pixels)
 * 3. Scale to fit display canvas
 *
 * This approach ensures:
 * - Consistent spacing between neurons regardless of canvas size
 * - Readable text at any scale
 * - Identical output between Canvas2D and WebGL
 */

// ============================================================================
// Geometric Types
// ============================================================================

/**
 * 2D point in rendering space (natural coordinates)
 */
export interface Point {
  readonly x: number;
  readonly y: number;
}

/**
 * Bounding box dimensions
 */
export interface Bounds {
  readonly width: number;
  readonly height: number;
}

/**
 * RGBA color representation (0-1 range)
 */
export interface Color {
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly a: number;
}

/**
 * CSS color string (for renderer convenience)
 */
export type CssColor = string;

// ============================================================================
// Content Dimensions (Natural Size)
// ============================================================================

/**
 * Fixed dimensions for readable content.
 * These define the "natural" size of elements before scaling.
 */
export interface ContentDimensions {
  /** Neuron circle diameter in natural pixels */
  readonly neuronDiameter: number;
  /** Vertical padding between neurons */
  readonly neuronPaddingY: number;
  /** Horizontal padding between layers */
  readonly layerPaddingX: number;
  /** Margin around the entire network */
  readonly margin: number;
  /** Font size for neuron values */
  readonly neuronFontSize: number;
  /** Font size for layer labels */
  readonly labelFontSize: number;
  /** Label offset from bottom of content */
  readonly labelOffsetY: number;
}

/**
 * Default content dimensions for optimal readability
 */
export const DEFAULT_CONTENT_DIMENSIONS: ContentDimensions = {
  neuronDiameter: 40,
  neuronPaddingY: 15,
  layerPaddingX: 120,
  margin: 60,
  neuronFontSize: 14,
  labelFontSize: 12,
  labelOffsetY: 30,
};

// ============================================================================
// Render Data Types (Natural Coordinates)
// ============================================================================

/**
 * Connection between two neurons in natural coordinates
 */
export interface Connection {
  readonly from: Point;
  readonly to: Point;
  readonly weight: number;
  readonly color: CssColor;
  readonly opacity: number;
  readonly strokeWidth: number;
}

/**
 * Neuron in natural coordinates
 */
export interface Neuron {
  readonly position: Point;
  readonly radius: number;
  readonly activation: number;
  readonly value: string;
  readonly fill: CssColor;
  readonly stroke: CssColor;
  readonly strokeWidth: number;
  readonly fontSize: number;
  readonly fontWeight: 'normal' | 'bold';
  readonly label?: string;
  readonly labelPosition?: Point;
  readonly labelAlign?: 'left' | 'center' | 'right';
}

/**
 * Text label in natural coordinates
 */
export interface Label {
  readonly position: Point;
  readonly text: string;
  readonly color: CssColor;
  readonly fontSize: number;
  readonly align: 'left' | 'center' | 'right';
}

/**
 * Complete network data in natural coordinates.
 * These coordinates are independent of canvas size.
 */
export interface NetworkRenderData {
  readonly connections: readonly Connection[];
  readonly neurons: readonly Neuron[];
  readonly labels: readonly Label[];
  /** Natural bounds of the content (before scaling) */
  readonly naturalBounds: Bounds;
}

// ============================================================================
// Viewport (Scaling)
// ============================================================================

/**
 * Display viewport configuration.
 * Defines how natural coordinates map to display pixels.
 */
export interface Viewport {
  /** Display width in CSS pixels */
  readonly width: number;
  /** Display height in CSS pixels */
  readonly height: number;
  /** Scale factor (natural → display) */
  readonly scale: number;
  /** Offset X for centering */
  readonly offsetX: number;
  /** Offset Y for centering */
  readonly offsetY: number;
}

/**
 * Calculate viewport to fit natural content in display area.
 * Centers the content and applies uniform scaling.
 */
export function calculateViewport(
  naturalBounds: Bounds,
  displayWidth: number,
  displayHeight: number,
): Viewport {
  const scaleX = displayWidth / naturalBounds.width;
  const scaleY = displayHeight / naturalBounds.height;
  const scale = Math.min(scaleX, scaleY);

  // Center the content
  const scaledWidth = naturalBounds.width * scale;
  const scaledHeight = naturalBounds.height * scale;
  const offsetX = (displayWidth - scaledWidth) / 2;
  const offsetY = (displayHeight - scaledHeight) / 2;

  return { width: displayWidth, height: displayHeight, scale, offsetX, offsetY };
}

// ============================================================================
// Renderer Configuration
// ============================================================================

/**
 * Renderer type preference
 */
export type RendererPreference = 'webgpu' | 'webgl' | 'canvas2d' | 'svg';

/**
 * Renderer configuration options
 */
export interface RenderConfig {
  readonly antialias: boolean;
  readonly powerPreference: 'low-power' | 'high-performance' | 'default';
  readonly maxConnections?: number;
  readonly debug: boolean;
}

/**
 * Default render configuration
 */
export const DEFAULT_RENDER_CONFIG: RenderConfig = {
  antialias: true,
  powerPreference: 'default',
  debug: false,
};

// ============================================================================
// Renderer Interface
// ============================================================================

/**
 * Network renderer interface.
 * All renderers (Canvas2D, WebGL, WebGPU) must implement this contract.
 */
export interface INetworkRenderer {
  /**
   * Render network data to the canvas.
   * @param data Network data in natural coordinates
   * @param viewport Viewport with scale and offset
   */
  render(data: NetworkRenderData, viewport: Viewport): void;

  /** Clear the canvas */
  clear(): void;

  /** Resize the canvas (CSS pixels) */
  resize(width: number, height: number): void;

  /** Update configuration */
  updateConfig(config: Partial<RenderConfig>): void;

  /** Clean up resources */
  destroy(): void;

  /** Get renderer type */
  getType(): RendererPreference;
}
