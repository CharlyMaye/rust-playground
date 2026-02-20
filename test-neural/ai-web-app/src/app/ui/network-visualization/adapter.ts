/**
 * Adapter utilities for converting existing WASM data structures
 * to the new NetworkArchitecture and LayerWeights formats.
 */

import { Activation, CnnActivationsResponse, NeuralNetworkLayers } from '@cma/wasm/shared';
import { NetworkVisualizationBuilder } from './config/visualization-builder';
import { VisualizationConfig } from './config/visualization-config';
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

/** One layer of CNN visualization (input or feature-maps). */
export interface CnnLayerViz {
  /** Human-readable label displayed above the canvas */
  label: string;
  architecture: NetworkArchitecture;
  weights: LayerWeights[];
  config: VisualizationConfig;
}

/**
 * Convert CNN activations to an array of per-layer visualization objects.
 *
 * Each CNN layer gets its own canvas. The first entry is the input
 * image (using the actual drawn pixel data), followed by one entry
 * per spatial CNN layer (feature-maps).
 *
 * @param response  CNN activations from WASM forward pass
 * @param inputData Flattened drawn pixel data (0-255) — shown as the input heatmap
 */
export function cnnActivationsToLayerVizArray(
  response: CnnActivationsResponse,
  inputData: number[],
): CnnLayerViz[] {
  const [, inH, inW] = response.input_shape;
  const result: CnnLayerViz[] = [];

  // ---- Input layer (the drawn image) ----
  // Normalise 0-255 → 0-1 for the grayscale color mapper
  const normalizedInput = inputData.map((v) => v / 255);
  const inputArch: NetworkArchitecture = {
    inputs: normalizedInput,
    layers: [],
  };
  const inputConfig = NetworkVisualizationBuilder.create()
    .withLayoutStrategy('row')
    .withSpacing('adaptive')
    .withConnections('none')
    .withRenderer('canvas2d')
    .withCanvasConfig({
      sizeStrategy: 'adaptive',
      aspectRatio: 'auto',
      maxWidth: 400,
      maxHeight: 400,
    })
    .forInputLayer({
      representation: 'heatmap',
      shape: [inH, inW],
      showValues: false,
      colorScheme: 'grayscale',
    })
    .build();

  result.push({
    label: `Input ${inH}×${inW}`,
    architecture: inputArch,
    weights: [],
    config: inputConfig,
  });

  // ---- CNN layers (feature maps) ----
  const spatialLayers = response.layers.filter(
    (l) =>
      l.shape.length === 3 &&
      l.shape[1] > 1 &&
      l.shape[2] > 1 &&
      !l.layer_type.toLowerCase().includes('activation') &&
      !l.layer_type.toLowerCase().includes('relu'),
  );

  for (const layer of spatialLayers) {
    const [channels, height, width] = layer.shape;
    const arch: NetworkArchitecture = {
      inputs: [0], // placeholder, not rendered (feature-maps start at layer 0)
      layers: [
        {
          size: channels * height * width,
          activations: layer.activations,
          activationFunction: `${layer.layer_type} ${channels}ch ${height}×${width}`,
          isOutput: true,
        },
      ],
    };

    const cfg = NetworkVisualizationBuilder.create()
      .withLayoutStrategy('row')
      .withSpacing('adaptive')
      .withConnections('none')
      .withRenderer('canvas2d')
      .withCanvasConfig({
        sizeStrategy: 'adaptive',
        aspectRatio: 'auto',
        maxWidth: 1200,
        maxHeight: 2000,
      })
      .withNeuronSizeConfig({ strategy: 'adaptive', minSize: 2, maxSize: 20 })
      .forInputLayer({ representation: 'collapsed', showValues: false })
      .forLayer(1, {
        representation: 'feature-maps',
        shape: [height, width],
        channels,
        maxChannels: 16,
        showLabel: false,
        showValues: false,
        colorScheme: 'grayscale',
      })
      .build();

    result.push({
      label: `${layer.layer_type} ${channels}ch ${height}×${width}`,
      architecture: arch,
      weights: [{ weights: [] }],
      config: cfg,
    });
  }

  return result;
}
