/* tslint:disable */
/* eslint-disable */

/**
 * XOR Neural Network exposed to JavaScript
 */
export class XorNetwork {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get confidence percentage (0-100)
     */
    confidence(x1: number, x2: number): number;
    /**
     * Run inference and return all neuron activations for visualization
     */
    get_activations(x1: number, x2: number): string;
    /**
     * Get all weights and biases as JSON
     */
    get_weights(): string;
    /**
     * Get model info
     */
    model_info(): string;
    /**
     * Create a new XOR network by loading the embedded model
     */
    constructor();
    /**
     * Predict XOR result for two binary inputs
     * Returns 0 or 1 (rounded prediction)
     */
    predict(x1: number, x2: number): number;
    /**
     * Get raw prediction value (0.0 to 1.0)
     */
    predict_raw(x1: number, x2: number): number;
    /**
     * Test all XOR combinations and return results as JSON string
     */
    test_all(): string;
}

/**
 * Initialize the module
 */
export function main(): void;

/**
 * Quick predict function (uses singleton - no parsing overhead)
 */
export function xor_predict(x1: number, x2: number): number;

/**
 * Quick raw predict function (uses singleton - no parsing overhead)
 */
export function xor_predict_raw(x1: number, x2: number): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_xornetwork_free: (a: number, b: number) => void;
    readonly main: () => void;
    readonly xor_predict: (a: number, b: number) => number;
    readonly xor_predict_raw: (a: number, b: number) => number;
    readonly xornetwork_confidence: (a: number, b: number, c: number) => number;
    readonly xornetwork_get_activations: (a: number, b: number, c: number) => [number, number];
    readonly xornetwork_get_weights: (a: number) => [number, number];
    readonly xornetwork_model_info: (a: number) => [number, number];
    readonly xornetwork_new: () => [number, number, number];
    readonly xornetwork_predict: (a: number, b: number, c: number) => number;
    readonly xornetwork_predict_raw: (a: number, b: number, c: number) => number;
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
