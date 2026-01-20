import { ApplicationConfig, inject, Injectable, provideBrowserGlobalErrorListeners, provideEnvironmentInitializer } from '@angular/core';
import { PreloadAllModules, provideRouter, withComponentInputBinding, withHashLocation, withPreloading } from '@angular/router';

import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideRouter(routes, withHashLocation(), withComponentInputBinding(), withPreloading(PreloadAllModules))
  ]
};

