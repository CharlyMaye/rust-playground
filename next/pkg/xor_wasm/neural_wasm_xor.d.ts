/* tslint:disable */
/* eslint-disable */

/**
 * XOR Neural Network exposed to JavaScript
 */
export class XorNetwork {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Run inference and return all neuron activations for visualization
     */
    get_activations(x1: number, x2: number): string;
    /**
     * Get class names
     */
    get_class_names(): string;
    /**
     * Get class probabilities
     */
    get_probabilities(x1: number, x2: number): string;
    /**
     * Get all weights and biases as JSON
     */
    get_weights(): string;
    /**
     * Get model info with accuracy and metadata
     */
    model_info(): string;
    /**
     * Create a new XOR network by loading the embedded model
     */
    constructor();
    /**
     * Predict XOR result for two binary inputs
     * Returns JSON with prediction details
     */
    predict(x1: number, x2: number): string;
    /**
     * Test all XOR combinations and return results as JSON string
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
    readonly __wbg_xornetwork_free: (a: number, b: number) => void;
    readonly main: () => void;
    readonly xornetwork_get_activations: (a: number, b: number, c: number) => [number, number];
    readonly xornetwork_get_class_names: (a: number) => [number, number];
    readonly xornetwork_get_probabilities: (a: number, b: number, c: number) => [number, number];
    readonly xornetwork_get_weights: (a: number) => [number, number];
    readonly xornetwork_model_info: (a: number) => [number, number];
    readonly xornetwork_new: () => [number, number, number];
    readonly xornetwork_predict: (a: number, b: number, c: number) => [number, number];
    readonly xornetwork_test_all: (a: number) => [number, number];
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
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
