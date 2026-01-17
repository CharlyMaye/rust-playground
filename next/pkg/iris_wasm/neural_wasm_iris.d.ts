/* tslint:disable */
/* eslint-disable */

export class IrisClassifier {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get layer-by-layer activations for visualization
     */
    get_activations(sepal_length: number, sepal_width: number, petal_length: number, petal_width: number): string;
    get_class_names(): string;
    /**
     * Get class probabilities for a prediction
     */
    get_probabilities(sepal_length: number, sepal_width: number, petal_length: number, petal_width: number): string;
    get_weights(): string;
    model_info(): string;
    constructor();
    /**
     * Predict iris species from measurements
     * Parameters: sepal_length, sepal_width, petal_length, petal_width (in cm)
     */
    predict(sepal_length: number, sepal_width: number, petal_length: number, petal_width: number): string;
    /**
     * Test all samples from the dataset
     */
    test_all(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_irisclassifier_free: (a: number, b: number) => void;
    readonly irisclassifier_get_activations: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly irisclassifier_get_class_names: (a: number) => [number, number];
    readonly irisclassifier_get_probabilities: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly irisclassifier_get_weights: (a: number) => [number, number];
    readonly irisclassifier_model_info: (a: number) => [number, number];
    readonly irisclassifier_new: () => [number, number, number];
    readonly irisclassifier_predict: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly irisclassifier_test_all: (a: number) => [number, number];
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
