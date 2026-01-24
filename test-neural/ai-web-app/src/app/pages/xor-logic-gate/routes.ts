import { Routes } from '@angular/router';
import { PageTitleOptions } from '../../ui/page-title/page-title';
import { RouteOptions } from '../model';

export function getRoutes(options: RouteOptions): Routes {
  const pageTitleOptions: PageTitleOptions = {
    title: 'Neural XOR',
    subtitle: 'WebAssembly Neural Network Demo',
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
          loadComponent: () => import('./xor-logic-gate').then((m) => m.XorLogicGate),
          children: [],
        },
      ],
    },
  ];
}
