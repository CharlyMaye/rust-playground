import { Component, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';
import { PageFooter } from '../../ui/page-footer/page-footer';

@Component({
  selector: 'app-iris-classifier',
  imports: [PageTitle, PageFooter],
  templateUrl: './iris-classifier.html',
  styleUrl: './iris-classifier.scss',
})
export class IrisClassifier {
  public readonly pageTitleOptions: Signal<PageTitleOptions> = signal({
    title: 'Iris Classifier',
    subtitle: 'Multi-class Neural Network Classification',
    icon: '🌸',
  });
}
