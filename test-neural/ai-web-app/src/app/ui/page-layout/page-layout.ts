import { Component, input } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { NavigationBack } from '../navigation-back/navigation-back';
import { PageFooter } from '../page-footer/page-footer';
import { PageTitle, PageTitleOptions } from '../page-title/page-title';

/**
 * Standard page layout wrapper component.
 * Provides consistent structure with title, navigation, content area, and footer.
 */
@Component({
  selector: 'app-page-layout',
  imports: [RouterOutlet, PageTitle, NavigationBack, PageFooter],
  templateUrl: './page-layout.html',
  styleUrl: './page-layout.scss',
})
export class PageLayout {
  /** Configuration options for the page title */
  public readonly pageTitleOptions = input.required<PageTitleOptions>();
  /** Whether to display the back navigation button */
  public readonly showBackButton = input<boolean>(true);
}
