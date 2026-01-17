import { Component, signal, Signal } from '@angular/core';
import { PageTitle, PageTitleOptions } from '../../ui/page-title/page-title';

@Component({
  selector: 'app-xor-logic-gate',
  imports: [PageTitle],
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
