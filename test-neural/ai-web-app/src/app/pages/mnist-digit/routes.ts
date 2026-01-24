import { Routes } from '@angular/router';

export function getRoutes(): Routes {
  return [
    {
      path: '',
      loadComponent: () => import('./mnist-digit').then((m) => m.MnistDigit),
    },
  ];
}
