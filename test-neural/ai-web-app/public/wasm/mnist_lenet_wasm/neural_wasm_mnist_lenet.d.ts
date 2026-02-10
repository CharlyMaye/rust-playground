/* tslint:disable */
/* eslint-disable */

/**
 * LeNet-5 MNIST Neural Network exposed to JavaScript
 */
export class MnistLeNetNetwork {
    free(): void;
    [Symbol.dispose](): void;
    get_architecture(): string;
    get_class_names(): string;
    get_probabilities(pixels: Float32Array): string;
    get_weights(): string;
    model_info(): string;
    /**
     * Create a new LeNet-5 MNIST network by loading the embedded model
     */
    constructor();
    predict(pixels: Float32Array): string;
    test_all(): string;
}

export function main(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_mnistlenetnetwork_free: (a: number, b: number) => void;
    readonly mnistlenetnetwork_get_architecture: (a: number) => [number, number];
    readonly mnistlenetnetwork_get_class_names: (a: number) => [number, number];
    readonly mnistlenetnetwork_get_probabilities: (a: number, b: number, c: number) => [number, number];
    readonly mnistlenetnetwork_get_weights: (a: number) => [number, number];
    readonly mnistlenetnetwork_model_info: (a: number) => [number, number];
    readonly mnistlenetnetwork_new: () => [number, number, number];
    readonly mnistlenetnetwork_predict: (a: number, b: number, c: number) => [number, number];
    readonly mnistlenetnetwork_test_all: (a: number) => [number, number];
    readonly main: () => void;
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
