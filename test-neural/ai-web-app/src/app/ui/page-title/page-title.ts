import { Component, computed, input } from '@angular/core';

/**
 * Configuration options for the page title.
 */
export type PageTitleOptions = {
  /** Main title text */
  title: string;
  /** Optional subtitle text */
  subtitle?: string;
  /** Optional icon identifier */
  icon?: string;
};

/**
 * Page title component with optional subtitle and icon.
 * Used at the top of pages to provide context and branding.
 */
@Component({
  selector: 'app-page-title',
  imports: [],
  templateUrl: './page-title.html',
  styleUrl: './page-title.scss',
})
export class PageTitle {
  /** Title configuration options */
  public readonly options = input.required<PageTitleOptions>();

  /** Whether a subtitle is provided */
  public readonly hasSubtitle = computed(() => {
    return !!this.options().subtitle;
  });
  /** Whether an icon is provided */
  public readonly hasIcon = computed(() => {
    return !!this.options().icon;
  });
}
