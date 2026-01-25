import { Component } from '@angular/core';

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
})
export class Card {}
