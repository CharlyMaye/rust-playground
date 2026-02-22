import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/**
 * Loading indicator component with optional message.
 * Displays a visual loading state with accessibility support.
 */
@Component({
  selector: 'app-loader',
  imports: [],
  templateUrl: './loader.html',
  host: {
    class: 'card loading',
    role: 'status',
    'aria-live': 'polite',
  },
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Loader {
  /** Optional message to display during loading */
  public readonly message = input<string>('');
}
