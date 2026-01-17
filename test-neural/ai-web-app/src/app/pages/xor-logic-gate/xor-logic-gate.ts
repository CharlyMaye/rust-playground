import { Component, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';
import { PageFooter } from '../../ui/page-footer/page-footer';
import { NavigationBack } from '../../ui/navigation-back/navigation-back';

@Component({
  selector: 'app-xor-logic-gate',
  imports: [PageTitle, NavigationBack, PageFooter],
  templateUrl: './xor-logic-gate.html',
  styleUrl: './xor-logic-gate.scss',
})
export class XorLogicGate {
  public readonly pageTitleOptions: Signal<PageTitleOptions> = signal({
    title: 'Neural XOR',
    subtitle: 'WebAssembly Neural Network Demo',
    icon: '🧠',
  });
}
