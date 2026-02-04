import { Routes } from '@angular/router';
import { PageTitleOptions } from '../../ui/page-title/page-title';
import { RouteOptions } from '../model';

export function getRoutes(options: RouteOptions): Routes {
  const pageTitleOptions: PageTitleOptions = {
    title: 'LeNet-5 CNN Digit Recognizer',
    subtitle: 'Convolutional Neural Network (1998)',
    icon: '🧠',
  };
  return [
    {
      path: '',
      loadComponent: () => import('../../ui/page-layout/page-layout').then((m) => m.PageLayout),
      data: { pageTitleOptions, showBackButton: options.showBackButton },
      children: [
        {
          path: '',
          loadComponent: () => import('./mnist-lenet').then((m) => m.MnistLeNet),
          children: [],
        },
      ],
    },
  ];
}
