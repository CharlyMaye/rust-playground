import { Component, effect, inject, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { WasmService } from '@cma/wasm/shared';
import init, { XorNetwork} from '@cma/wasm/xor_wasm/neural_wasm_xor.js';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  private readonly wasmService = inject(WasmService);
  public readonly xorInitOutput = this.wasmService.xorWasmResource.value.asReadonly();
  public readonly irisInitOutput = this.wasmService.irisWasmResource.value.asReadonly();

  constructor() {
    effect  (() => {
      console.log('WASM Resource loaded:', this.xorInitOutput());
    });
    effect  (() => {
      console.log('WASM Resource loaded:', this.irisInitOutput());
    });
  }
}
