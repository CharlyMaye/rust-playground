import { Component, computed, input } from '@angular/core';

export type PageTitleOptions = {
  title: string;
  subtitle?: string;
  icon?: string;
};

@Component({
  selector: 'app-page-title',
  imports: [],
  templateUrl: './page-title.html',
  styleUrl: './page-title.scss',
})
export class PageTitle {
  public readonly options = input.required<PageTitleOptions>();

  public readonly hasSubtitle = computed(() => {
    return !!this.options().subtitle;
  });
  public readonly hasIcon = computed(() => {
    return !!this.options().icon;
  });
}
