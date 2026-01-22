import { Component, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';
import { PageFooter } from '../../ui/page-footer/page-footer';
import { About } from '../../ui/about/about';
import { RouterLink, RouterLinkWithHref } from '@angular/router';
import { NavigationBack } from '../../ui/navigation-back/navigation-back';

@Component({
  selector: 'app-home',
  imports: [PageTitle, NavigationBack, About, PageFooter, RouterLinkWithHref],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home {
  public readonly pageTitleOptions: Signal<PageTitleOptions> = signal({
    title: 'Neural Networks',
    subtitle: 'Interactive WebAssembly Demos',
    icon: '🧠',
  });
}
