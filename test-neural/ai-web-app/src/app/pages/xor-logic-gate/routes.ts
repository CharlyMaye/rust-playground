import { Routes } from '@angular/router';

export function getRoutes(): Routes {
  return [
    {
      path: '',
      loadComponent: () => import('./xor-logic-gate').then((m) => m.XorLogicGate),
    },
  ];
}
