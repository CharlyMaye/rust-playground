// TypeScript declarations for AlexNet-Mini MNIST WASM stub

export interface InitOutput {
  memory: WebAssembly.Memory;
}

export declare class MnistAlexNetNetwork {
  constructor();
  predict(pixels: Float64Array): string;
  model_info(): string;
  get_architecture(): string;
  get_weights(): string;
  test_all(): string;
}

declare function init(input?: string | URL | Request | RequestInfo): Promise<InitOutput>;
export default init;
