import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { ModelInfo } from '@cma/wasm/shared';

/**
 * Displays neural network model metadata.
 * Shows model name, version, and training parameters.
 */
@Component({
  selector: 'app-model-info',
  imports: [DecimalPipe],
  templateUrl: './model-info.html',
  styleUrl: './model-info.scss',
  host: { class: 'card' },
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ModelInfoComponent {
  /** Model metadata to display */
  public readonly modelInfo = input<ModelInfo | undefined>(undefined);

  /** Activation functions used in the network (e.g., "ReLU → Softmax") */
  public readonly activations = input<string | undefined>(undefined);
}
