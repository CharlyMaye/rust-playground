import { Component } from '@angular/core';

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
})
export class About {}
