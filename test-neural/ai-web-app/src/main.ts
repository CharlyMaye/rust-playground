import { bootstrapApplication } from '@angular/platform-browser';
import { App } from './app/app';
import { appConfig } from './app/app.config';

// Suppress benign ResizeObserver loop error
// This error occurs when resize observations can't be delivered in a single frame
// and is harmless. See: https://github.com/WICG/resize-observer/issues/38
const resizeObserverError = /ResizeObserver loop/;
window.addEventListener('error', (event) => {
  if (resizeObserverError.test(event.message)) {
    event.stopImmediatePropagation();
  }
});

bootstrapApplication(App, appConfig).catch((err) => console.error(err));
