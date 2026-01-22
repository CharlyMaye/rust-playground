import { Routes } from "@angular/router";

export function getRoutes(): Routes {
    return [
        {
            path: '',
            loadComponent: () => import('./home').then(m => m.Home)
        }
    ]
}
