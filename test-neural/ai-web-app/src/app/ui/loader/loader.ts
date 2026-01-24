import { Component, input } from '@angular/core';

/**
 * Loading indicator component with optional message.
 * Displays a visual loading state with accessibility support.
 */
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
  /** Optional message to display during loading */
  public readonly message = input<string>('');
}
