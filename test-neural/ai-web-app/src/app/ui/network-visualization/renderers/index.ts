/**
 * Public API for the rendering system.
 * This barrel file exports all public types and classes.
 */

// Core types and interfaces
export * from './types';

// Renderers
export * from './configurable-canvas2d-renderer';
export * from './configurable-webgl-renderer';

// Layout calculator
export { ConfigurableLayoutCalculator } from './configurable-layout-calculator';
export type {
  BarData,
  ConfigurableRenderData,
  GridData,
  LayerElement,
  LayerInfo,
  LayerWeights,
  NetworkArchitecture,
  StatsData,
} from './configurable-layout-calculator';

// Factory
export * from './renderer-factory';
