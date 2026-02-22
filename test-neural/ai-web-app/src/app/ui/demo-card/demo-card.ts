import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { RouterLinkWithHref } from '@angular/router';
import { Loader } from '../loader/loader';

/**
 * Demo card component for displaying neural network demos on the home page.
 * Shows icon, name, description, badges and handles loading state.
 */
@Component({
  selector: 'app-demo-card',
  imports: [DecimalPipe, RouterLinkWithHref, Loader],
  templateUrl: './demo-card.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DemoCard {
  /** Route path for navigation */
  readonly route = input.required<string>();

  /** Icon emoji to display */
  readonly icon = input.required<string>();

  /** Demo name */
  readonly name = input<string>();

  /** Fallback name when name is not available */
  readonly fallbackName = input<string>('');

  /** Demo description */
  readonly description = input<string>();

  /** Fallback description when description is not available */
  readonly fallbackDescription = input<string>('');

  /** Whether the demo is currently loading */
  readonly isLoading = input<boolean>(false);

  /** Loading message to display */
  readonly loadingMessage = input<string>('Loading...');

  /** Input badge text (e.g., "2 inputs" or "28×28 image") */
  readonly inputBadge = input<string>();

  /** Output badge text (e.g., "1 output" or "10 classes") */
  readonly outputBadge = input<string>();

  /** Accuracy percentage (undefined = not trained) */
  readonly accuracy = input<number>();

  /** Whether to show "Not trained" badge when accuracy is undefined */
  readonly showNotTrained = input<boolean>(false);

  /** Custom badge text (e.g., "Coming soon") */
  readonly customBadge = input<string>();

  /** Custom badge type (warning, info, success) */
  readonly customBadgeType = input<'info' | 'success' | 'warning'>('warning');
}
