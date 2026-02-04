import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: 'home',
    loadChildren: () => import('./pages/home').then((m) => m.getRoutes({ showBackButton: false })),
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
    path: 'mnist-lenet',
    loadChildren: () =>
      import('./pages/mnist-lenet').then((m) => m.getRoutes({ showBackButton: true })),
  },
  {
    path: 'mnist-resnet',
    loadChildren: () =>
      import('./pages/mnist-resnet').then((m) => m.getRoutes({ showBackButton: true })),
  },
  {
    path: 'iris-classifier',
    loadChildren: () =>
      import('./pages/iris-classifier').then((m) => m.getRoutes({ showBackButton: true })),
  },
  {
    path: 'coming-soon',
    loadChildren: () =>
      import('./pages/coming-soon').then((m) => m.getRoutes({ showBackButton: true })),
  },
  {
    path: 'empty',
    loadChildren: () => import('./pages/empty').then((m) => m.getRoutes({ showBackButton: true })),
  },
  {
    path: '',
    redirectTo: 'home',
    pathMatch: 'full',
  },
];
