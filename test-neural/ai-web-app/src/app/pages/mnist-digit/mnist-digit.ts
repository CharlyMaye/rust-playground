import { Component, signal, Signal } from '@angular/core';
import { NavigationBack } from '../../ui/navigation-back/navigation-back';
import { PageFooter } from '../../ui/page-footer/page-footer';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';

@Component({
  selector: 'app-mnist-digit',
  imports: [PageTitle, NavigationBack, PageFooter],
  templateUrl: './mnist-digit.html',
  styleUrl: './mnist-digit.scss',
})
export class MnistDigit {
  public readonly pageTitleOptions: Signal<PageTitleOptions> = signal({
    title: 'MNIST Digit Recognizer',
    subtitle: 'Handwritten Digit Classification',
    icon: '✍️',
  });
}
