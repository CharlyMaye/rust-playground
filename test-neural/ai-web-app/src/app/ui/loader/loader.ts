import { Component, input } from '@angular/core';

@Component({
  selector: 'app-loader',
  imports: [],
  templateUrl: './loader.html',
  styleUrl: './loader.scss',
  host: {
    class: 'card loading',
  },
})
export class Loader {
  public readonly message = input<string>('');
}
