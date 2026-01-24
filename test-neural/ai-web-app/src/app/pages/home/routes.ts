import { Routes } from '@angular/router';
import { PageTitleOptions } from '../../ui/page-title/page-title';
import { RouteOptions } from '../model';

export function getRoutes(options: RouteOptions): Routes {
  const pageTitleOptions: PageTitleOptions = {
    title: 'Neural Networks',
    subtitle: 'Interactive WebAssembly Demos',
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
          loadComponent: () => import('./home').then((m) => m.Home),
          children: [],
        },
      ],
    },
  ];
}
