import { Component, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';
import { PageFooter } from '../../ui/page-footer/page-footer';
import { About } from '../../ui/about/about';
import { RouterLinkWithHref } from '@angular/router';


@Component({
  selector: 'app-home',
  imports: [PageTitle, About, PageFooter, RouterLinkWithHref],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home {
  public readonly pageTitleOptions: Signal<PageTitleOptions> = signal({
    title: 'Neural Networks',
    subtitle: 'Interactive WebAssembly Demos',
    icon: '🧠',
  });

  constructor() {
  }

  public ngOnInit() {
    this.loadModelAccuracies();
  }

  async loadModelAccuracies() {
  }
}
