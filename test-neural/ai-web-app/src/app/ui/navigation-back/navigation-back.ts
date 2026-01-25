import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

/**
 * Back navigation button component.
 * Provides a link to return to the home page.
 */
@Component({
  selector: 'nav',
  imports: [RouterLink],
  templateUrl: './navigation-back.html',
})
export class NavigationBack {}
