import { ChangeDetectionStrategy, Component } from '@angular/core';

/**
 * MNIST digit classifier demo page.
 * Placeholder for future handwritten digit recognition feature.
 */
@Component({
  selector: 'app-empty',
  imports: [],
  templateUrl: './empty.html',
  styleUrl: './empty.scss',
  host: { class: 'page container' },
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Empty {}
