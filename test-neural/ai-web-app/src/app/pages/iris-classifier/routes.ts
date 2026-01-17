import { Routes } from "@angular/router";

export function getRoutes(): Routes {
    return [
        {
            path: '',
            loadComponent: () => import('./iris-classifier').then(m => m.IrisClassifier)
        }
    ]
}
