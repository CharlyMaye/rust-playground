import { Component, input } from '@angular/core';

@Component({
  selector: 'app-loader',
  imports: [],
  templateUrl: './loader.html',
  styleUrl: './loader.scss',
  host: {
    class: 'card loading',
    role: 'status',
    'aria-live': 'polite',
  },
})
export class Loader {
  public readonly message = input<string>('');
}
