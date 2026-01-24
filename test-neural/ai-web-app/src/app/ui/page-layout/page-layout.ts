import { Component, input } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { NavigationBack } from '../navigation-back/navigation-back';
import { PageFooter } from '../page-footer/page-footer';
import { PageTitle, PageTitleOptions } from '../page-title/page-title';

@Component({
  selector: 'app-page-layout',
  imports: [RouterOutlet, PageTitle, NavigationBack, PageFooter],
  templateUrl: './page-layout.html',
  styleUrl: './page-layout.scss',
})
export class PageLayout {
  public readonly pageTitleOptions = input.required<PageTitleOptions>();
  public readonly showBackButton = input<boolean>(true);
}
