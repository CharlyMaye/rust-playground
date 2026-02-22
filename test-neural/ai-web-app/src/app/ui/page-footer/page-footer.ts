import { ChangeDetectionStrategy, Component } from '@angular/core';

/**
 * Page footer component.
 * Displays consistent footer content across all pages.
 */
@Component({
  selector: 'app-page-footer',
  imports: [],
  templateUrl: './page-footer.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PageFooter {}
