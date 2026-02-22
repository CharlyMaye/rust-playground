import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'app-coming-soon',
  imports: [],
  templateUrl: './coming-soon.html',
  styleUrl: './coming-soon.scss',
  host: {
    class: 'card',
  },
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ComingSoon {}
