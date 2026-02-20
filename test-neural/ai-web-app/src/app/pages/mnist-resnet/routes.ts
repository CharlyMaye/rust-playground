import { Routes } from '@angular/router';
import { PageTitleOptions } from '../../ui/page-title/page-title';
import { RouteOptions } from '../model';

export function getRoutes(options: RouteOptions): Routes {
  const pageTitleOptions: PageTitleOptions = {
    title: 'ResNet-Micro CNN Digit Recognizer',
    subtitle: 'Residual Network (2015)',
    icon: '🔗',
  };
  return [
    {
      path: '',
      loadComponent: () => import('../../ui/page-layout/page-layout').then((m) => m.PageLayout),
      data: { pageTitleOptions, showBackButton: options.showBackButton },
      children: [
        {
          path: '',
          loadComponent: () => import('./mnist-resnet').then((m) => m.MnistResNet),
          children: [],
        },
      ],
    },
  ];
}
