import { Component, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';
import { PageFooter } from '../../ui/page-footer/page-footer';
import { About } from '../../ui/about/about';
import { RouterLink, RouterLinkWithHref } from '@angular/router';
import { NavigationBack } from '../../ui/navigation-back/navigation-back';
import * as xorModule from "neural-wasm-xor";
import * as irisModule from "neural-wasm-iris";

@Component({
  selector: 'app-home',
  imports: [PageTitle, NavigationBack, About, PageFooter, RouterLinkWithHref],
  templateUrl: './home.html',
  styleUrl: './home.scss',
})
export class Home {
  public readonly pageTitleOptions: Signal<PageTitleOptions> = signal({
    title: 'Neural Networks',
    subtitle: 'Interactive WebAssembly Demos',
    icon: '🧠',
  });

  constructor() {
  }

  public ngOnInit() {
    this.loadModelAccuracies();
  }

  async loadModelAccuracies() {
    const element = document.getElementById('xor-accuracy');
    if (!element) {
        console.error('XOR accuracy element not found');
        return;
    }
    const irisElement = document.getElementById('iris-accuracy');
    if (!irisElement) {
        console.error('Iris accuracy element not found');
        return;
    }
    try {
          // Load XOR accuracy
          // const xorModule = await import('./pkg/xor_wasm/neural_wasm_xor.js');
          await xorModule.default();
          const xorNet = new xorModule.XorNetwork();
          const xorInfo = JSON.parse(xorNet.model_info());
          element.textContent = 
              `${xorInfo.accuracy.toFixed(1)}% accuracy`;
      } catch (e) {
          console.error('Failed to load XOR accuracy:', e);
          element.textContent = 'N/A';
      }

      try {
          // Load Iris accuracy
          // const irisModule = await import('./pkg/iris_wasm/neural_wasm_iris.js');
          await irisModule.default();
          const irisClassifier = new irisModule.IrisClassifier();
          const irisInfo = JSON.parse(irisClassifier.model_info());
          irisElement.textContent = 
              `${irisInfo.accuracy.toFixed(1)}% accuracy`;
      } catch (e) {
          console.error('Failed to load Iris accuracy:', e);
          irisElement.textContent = 'N/A';
      }
  }
}
