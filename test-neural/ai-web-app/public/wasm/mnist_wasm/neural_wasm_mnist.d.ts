/* tslint:disable */
/* eslint-disable */

/**
 * MNIST Neural Network exposed to JavaScript
 */
export class MnistNetwork {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Run inference and return all neuron activations for visualization
     */
    get_activations(pixels: Float64Array): string;
    /**
     * Get class names (digits 0-9)
     */
    get_class_names(): string;
    /**
     * Get class probabilities for 784 pixels
     */
    get_probabilities(pixels: Float64Array): string;
    /**
     * Get all weights and biases as JSON
     */
    get_weights(): string;
    /**
     * Get model info with accuracy and metadata
     */
    model_info(): string;
    /**
     * Create a new MNIST network by loading the embedded model
     */
    constructor();
    /**
     * Predict MNIST digit from pixel array
     * Accepts 784 pixels (28x28 image) or normalized values
     * Returns JSON with digit prediction (0-9), probabilities, and confidence
     */
    predict(pixels: Float64Array): string;
    /**
     * Test with sample MNIST digits
     * Returns results with digit predictions (0-9)
     */
    test_all(): string;
}

/**
 * Initialize the module
 */
export function main(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_mnistnetwork_free: (a: number, b: number) => void;
    readonly main: () => void;
    readonly mnistnetwork_get_activations: (a: number, b: number, c: number) => [number, number];
    readonly mnistnetwork_get_class_names: (a: number) => [number, number];
    readonly mnistnetwork_get_probabilities: (a: number, b: number, c: number) => [number, number];
    readonly mnistnetwork_get_weights: (a: number) => [number, number];
    readonly mnistnetwork_model_info: (a: number) => [number, number];
    readonly mnistnetwork_new: () => [number, number, number];
    readonly mnistnetwork_predict: (a: number, b: number, c: number) => [number, number];
    readonly mnistnetwork_test_all: (a: number) => [number, number];
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
