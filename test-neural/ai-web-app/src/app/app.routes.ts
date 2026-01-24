import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./ui/main-layout/main-layout').then((m) => m.MainLayout),
    children: [
      {
        path: 'home',
        loadChildren: () =>
          import('./pages/home').then((m) => m.getRoutes({ showBackButton: false })),
      },
      {
        path: 'xor-logic-gate',
        loadChildren: () =>
          import('./pages/xor-logic-gate').then((m) => m.getRoutes({ showBackButton: true })),
      },
      {
        path: 'mnist-digit',
        loadChildren: () =>
          import('./pages/mnist-digit').then((m) => m.getRoutes({ showBackButton: true })),
      },
      {
        path: 'iris-classifier',
        loadChildren: () =>
          import('./pages/iris-classifier').then((m) => m.getRoutes({ showBackButton: true })),
      },
      {
        path: '',
        redirectTo: 'home',
        pathMatch: 'full',
      },
    ],
  },
];
