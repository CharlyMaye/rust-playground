/**
 * Adapter utilities for converting existing WASM data structures
 * to the new NetworkArchitecture and LayerWeights formats.
 */

import { Activation, NeuralNetworkLayers } from '@cma/wasm/shared';
import { LayerWeights, NetworkArchitecture } from './renderers';

/**
 * Convert WASM Activation data to NetworkArchitecture format
 */
export function activationToArchitecture(
  activation: Activation<unknown, unknown>,
): NetworkArchitecture {
  return {
    inputs: activation.inputs as number[],
    layers: activation.layers.map((layer, index) => ({
      size: layer.activation.length,
      activations: layer.activation as number[],
      activationFunction: layer.function,
      isOutput: index === activation.layers.length - 1,
    })),
  };
}

/**
 * Convert WASM NeuralNetworkLayers to LayerWeights array
 */
export function neuralNetworkLayersToWeights(layers: NeuralNetworkLayers): LayerWeights[] {
  return layers.layers.map((layer) => ({
    weights: layer.weights,
  }));
}

/**
 * Example usage:
 *
 * ```typescript
 * // In your component:
 * import { activationToArchitecture, neuralNetworkLayersToWeights } from './adapter';
 *
 * const architecture = computed(() => {
 *   const activations = this.activations();
 *   if (!activations) return null;
 *   return activationToArchitecture(activations);
 * });
 *
 * const weights = computed(() => {
 *   const wts = this.weights();
 *   if (!wts) return null;
 *   return neuralNetworkLayersToWeights(wts);
 * });
 *
 * // Then use in template:
 * <app-network-visualization
 *   [architecture]="architecture()"
 *   [weights]="weights()"
 * />
 * ```
 */
