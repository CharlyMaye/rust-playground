import { ChangeDetectionStrategy, Component } from '@angular/core';

/**
 * About section component.
 * Displays information about the neural network demo application.
 */
@Component({
  selector: 'app-about',
  templateUrl: './about.html',
  styleUrl: './about.scss',
  host: {
    class: 'card',
  },
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class About {}
