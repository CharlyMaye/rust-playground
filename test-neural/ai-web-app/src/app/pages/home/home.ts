import { Component, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';
import { PageFooter } from '../../ui/page-footer/page-footer';

@Component({
  selector: 'app-home',
  imports: [PageTitle, PageFooter],
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
