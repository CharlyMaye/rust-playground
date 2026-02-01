import { Component } from '@angular/core';
import { ComingSoon as ComingSoonCmp } from 'src/app/ui/coming-soon/coming-soon';

/**
 * MNIST digit classifier demo page.
 * Placeholder for future handwritten digit recognition feature.
 */
@Component({
  selector: 'app-coming-soon-page',
  imports: [ComingSoonCmp],
  templateUrl: './coming-soon.html',
  styleUrl: './coming-soon.scss',
  host: { class: 'page container' },
})
export class ComingSoon {}
