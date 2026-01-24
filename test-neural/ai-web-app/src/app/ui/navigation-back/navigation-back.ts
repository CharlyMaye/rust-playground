import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

/**
 * Back navigation button component.
 * Provides a link to return to the home page.
 */
@Component({
  selector: 'app-navigation-back',
  imports: [RouterLink],
  templateUrl: './navigation-back.html',
  styleUrl: './navigation-back.scss',
})
export class NavigationBack {}
