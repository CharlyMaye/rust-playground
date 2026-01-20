import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import init, { XorNetwork} from '../wasm/xor_wasm/neural_wasm_xor.js';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {

  constructor() {
  fetch('/wasm/xor_wasm/neural_wasm_xor_bg.wasm')
    .then(r => console.log('status:', r.status))
    .then(() =>
      init('/wasm/xor_wasm/neural_wasm_xor_bg.wasm')
    )
    .then(() => {
      const xorNetwork = new XorNetwork();
      console.log('XOR Network initialized:', xorNetwork);
    })
    .catch(e => console.error(e));
  }
}
