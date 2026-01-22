import { DecimalPipe } from '@angular/common';
import { Component, input } from '@angular/core';
import { ModelInfo } from '@cma/wasm/shared/model-info';

@Component({
  selector: 'app-model-info',
  imports: [DecimalPipe],
  templateUrl: './model-info.html',
  styleUrl: './model-info.scss'
})
export class ModelInfoComponent {
  public readonly modelInfo = input<ModelInfo | undefined>(undefined)
}
