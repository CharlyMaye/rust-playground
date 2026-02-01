import { ConfigurableCanvas2DRenderer } from './configurable-canvas2d-renderer';
import { ConfigurableWebGLRenderer } from './configurable-webgl-renderer';
import { INetworkRenderer, RenderConfig, RendererPreference } from './types';

/**
 * Renderer factory for creating the appropriate renderer based on
 * browser capabilities and user preferences.
 *
 * Supports progressive enhancement:
 * 1. Try WebGPU (if available and requested)
 * 2. Fall back to WebGL (if available and requested)
 * 3. Fall back to Canvas 2D (always available)
 */
export class RendererFactory {
  /**
   * Check if WebGPU is available
   */
  static isWebGPUAvailable(): boolean {
    return 'gpu' in navigator;
  }

  /**
   * Check if WebGL is available
   */
  static isWebGLAvailable(): boolean {
    try {
      const canvas = document.createElement('canvas');
      return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
    } catch (e) {
      return false;
    }
  }

  /**
   * Check if Canvas 2D is available (should always be true)
   */
  static isCanvas2DAvailable(): boolean {
    try {
      const canvas = document.createElement('canvas');
      return !!canvas.getContext('2d');
    } catch (e) {
      return false;
    }
  }

  /**
   * Get supported renderer preferences in order of capability
   */
  static getSupportedRenderers(): RendererPreference[] {
    const supported: RendererPreference[] = [];

    if (this.isWebGPUAvailable()) {
      supported.push('webgpu');
    }
    if (this.isWebGLAvailable()) {
      supported.push('webgl');
    }
    if (this.isCanvas2DAvailable()) {
      supported.push('canvas2d');
    }

    return supported;
  }

  /**
   * Get the best available renderer preference
   */
  static getBestAvailableRenderer(): RendererPreference {
    const supported = this.getSupportedRenderers();
    return supported[0] || 'canvas2d';
  }

  /**
   * Create a renderer instance based on preferences
   *
   * @param canvas The canvas element to render to
   * @param preferences Ordered list of renderer preferences (tries in order)
   * @param config Renderer configuration
   * @returns Renderer instance
   */
  static create(
    canvas: HTMLCanvasElement,
    preferences: RendererPreference[] = ['canvas2d'],
    config?: Partial<RenderConfig>,
  ): INetworkRenderer {
    // Try each preference in order
    for (const pref of preferences) {
      try {
        switch (pref) {
          case 'webgpu':
            if (this.isWebGPUAvailable()) {
              // WebGPU renderer not yet implemented
              console.warn('WebGPU renderer not yet implemented, falling back');
              continue;
            }
            break;

          case 'webgl':
            if (this.isWebGLAvailable()) {
              return new ConfigurableWebGLRenderer(canvas, config);
            }
            break;

          case 'canvas2d':
            if (this.isCanvas2DAvailable()) {
              return new ConfigurableCanvas2DRenderer(canvas, config);
            }
            break;
        }
      } catch (error) {
        console.warn(`Failed to create ${pref} renderer:`, error);
        continue;
      }
    }

    // Last resort: try Canvas2D
    if (this.isCanvas2DAvailable()) {
      return new ConfigurableCanvas2DRenderer(canvas, config);
    }

    throw new Error('No suitable renderer available');
  }

  /**
   * Create a renderer with automatic detection
   * Uses the best available renderer based on browser capabilities
   */
  static createAuto(canvas: HTMLCanvasElement, config?: Partial<RenderConfig>): INetworkRenderer {
    const best = this.getBestAvailableRenderer();
    return this.create(canvas, [best], config);
  }
}
