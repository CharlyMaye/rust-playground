import { ChangeDetectionStrategy, Component } from '@angular/core';

/**
 * Generic card container component.
 * Provides consistent styling for content sections.
 */
@Component({
  selector: 'app-card',
  imports: [],
  templateUrl: './card.html',
  host: {
    class: 'card',
  },
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Card {}
