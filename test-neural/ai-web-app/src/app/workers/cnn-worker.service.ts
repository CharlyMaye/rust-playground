import { Injectable, NgZone, signal } from '@angular/core';
import { CnnActivationsResponse } from '@cma/wasm/shared';

/**
 * Service that runs CNN activation computation in a Web Worker.
 *
 * Falls back to synchronous main-thread computation when
 * Worker or dynamic import is unavailable (e.g. some browsers, SSR).
 */
@Injectable({ providedIn: 'root' })
export class CnnWorkerService {
  private worker: Worker | null = null;
  private pendingResolve: ((result: CnnActivationsResponse | null) => void) | null = null;
  readonly isComputing = signal(false);

  constructor(private readonly ngZone: NgZone) {
    this.initWorker();
  }

  private initWorker(): void {
    if (typeof Worker === 'undefined') return;

    try {
      this.worker = new Worker(new URL('../workers/cnn-activations.worker', import.meta.url), {
        type: 'module',
      });

      this.worker.onmessage = ({ data }) => {
        this.ngZone.run(() => {
          this.isComputing.set(false);
          if (this.pendingResolve) {
            if (data && 'error' in data) {
              this.pendingResolve(null);
            } else {
              this.pendingResolve(data as CnnActivationsResponse);
            }
            this.pendingResolve = null;
          }
        });
      };

      this.worker.onerror = () => {
        this.ngZone.run(() => {
          this.isComputing.set(false);
          if (this.pendingResolve) {
            this.pendingResolve(null);
            this.pendingResolve = null;
          }
        });
      };
    } catch {
      this.worker = null;
    }
  }

  /**
   * Compute CNN activations off the main thread.
   *
   * @param wasmJsUrl   Absolute URL to the WASM JS glue module
   * @param wasmBinaryUrl Absolute URL to the .wasm binary
   * @param input       Flattened 28×28 pixel data (Float32Array)
   * @returns Promise resolving to the activations or null on error
   */
  compute(
    wasmJsUrl: string,
    wasmBinaryUrl: string,
    input: Float32Array,
  ): Promise<CnnActivationsResponse | null> {
    if (!this.worker) {
      // Fallback: no Worker support — caller should use synchronous path
      return Promise.resolve(null);
    }

    this.isComputing.set(true);

    return new Promise((resolve) => {
      this.pendingResolve = resolve;
      this.worker!.postMessage({ wasmJsUrl, wasmBinaryUrl, input }, [input.buffer]);
    });
  }

  /** Whether the worker is available */
  get available(): boolean {
    return this.worker !== null;
  }
}
