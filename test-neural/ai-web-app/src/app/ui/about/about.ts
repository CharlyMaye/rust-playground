import { Component } from '@angular/core';
import { Card } from '../card/card';

@Component({
  selector: 'app-about',
  imports: [Card],
  templateUrl: './about.html',
  styleUrl: './about.scss',
  host: {
    class: 'card',
  },
})
export class About {}
