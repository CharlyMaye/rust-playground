/**
 * Web Worker for off-main-thread CNN activation computation.
 *
 * Receives { wasmJsUrl, wasmBinaryUrl, input } messages.
 * Dynamically loads the WASM module, runs get_cnn_activations,
 * and posts back the JSON result.
 */

/// <reference lib="webworker" />

interface WorkerRequest {
  wasmJsUrl: string;
  wasmBinaryUrl: string;
  input: Float32Array;
}

let cachedModule: any = null;
let cachedNetwork: any = null;
let cachedWasmJsUrl: string | null = null;

addEventListener('message', async ({ data }: MessageEvent<WorkerRequest>) => {
  try {
    const { wasmJsUrl, wasmBinaryUrl, input } = data;

    // Resolve relative URLs against the page origin (not the worker script URL)
    const base = self.location.origin + '/';
    const resolvedJsUrl = new URL(wasmJsUrl, base).href;
    const resolvedBinUrl = new URL(wasmBinaryUrl, base).href;

    // Reuse cached module if same WASM (common case: user draws multiple times)
    if (cachedWasmJsUrl !== resolvedJsUrl || !cachedNetwork) {
      // Dynamic import of the WASM JS glue
      const mod = await import(/* @vite-ignore */ resolvedJsUrl);
      cachedModule = mod;

      // Initialize WASM with explicit binary URL
      await mod.default(resolvedBinUrl);

      // Find the Network class (name varies per model)
      const NetworkClass = Object.values(mod).find(
        (v: any) => typeof v === 'function' && v.prototype && 'get_cnn_activations' in v.prototype,
      ) as (new () => any) | undefined;

      if (!NetworkClass) {
        postMessage({ error: 'No network class found in WASM module' });
        return;
      }

      cachedNetwork = new NetworkClass();
      cachedWasmJsUrl = wasmJsUrl;
    }

    const json: string = cachedNetwork.get_cnn_activations(input);
    const result = JSON.parse(json);
    postMessage(result);
  } catch (err: any) {
    postMessage({ error: err?.message ?? 'Worker error' });
  }
});
