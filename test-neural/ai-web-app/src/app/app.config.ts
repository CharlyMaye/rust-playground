import { ApplicationConfig, provideBrowserGlobalErrorListeners } from '@angular/core';
import {
  PreloadAllModules,
  provideRouter,
  withComponentInputBinding,
  withHashLocation,
  withPreloading,
  withViewTransitions,
} from '@angular/router';

import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideRouter(
      routes,
      withHashLocation(),
      withViewTransitions({
        skipInitialTransition: true,
      }),
      withComponentInputBinding(),
      withPreloading(PreloadAllModules),
    ),
  ],
};
