import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { WasmFacade } from '@cma/wasm/shared';
import { About } from '../../ui/about/about';
import { DemoCard } from '../../ui/demo-card/demo-card';

/** Demo card configuration */
interface DemoConfig {
  /** Route path */
  route: string;
  /** Icon emoji */
  icon: string;
  /** Demo name */
  name: string | undefined;
  /** Fallback name when not loaded */
  fallbackName: string;
  /** Demo description */
  description: string | undefined;
  /** Fallback description when not loaded */
  fallbackDescription: string;
  /** Whether loading */
  isLoading: boolean;
  /** Loading message */
  loadingMessage: string;
  /** Input badge text */
  inputBadge: string;
  /** Output badge text */
  outputBadge: string;
  /** Accuracy percentage */
  accuracy: number | undefined;
  /** Show "Not trained" badge when accuracy is undefined */
  showNotTrained: boolean;
  /** Custom badge text */
  customBadge?: string;
  /** Custom badge type */
  customBadgeType?: 'info' | 'success' | 'warning';
}

/**
 * Home page displaying available neural network demos.
 * Shows cards for XOR, Iris classifier, and MNIST demos.
 */
@Component({
  selector: 'app-home',
  imports: [About, DemoCard],
  templateUrl: './home.html',
  styleUrl: './home.scss',
  host: { class: 'page container' },
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Home {
  private readonly wasmService = inject(WasmFacade);

  /** All demos configuration for the grid */
  readonly demos = computed<DemoConfig[]>(() => [
    {
      route: '/xor-logic-gate',
      icon: '⊕',
      name: this.wasmService.xorModelInfo()?.name,
      fallbackName: 'XOR Gate',
      description: this.wasmService.xorModelInfo()?.description,
      fallbackDescription: 'Binary logic gate neural network',
      isLoading: this.wasmService.xorWasmResource.isLoading(),
      loadingMessage: 'Loading Neural Network...',
      inputBadge: `${this.wasmService.xorArchitecture()?.[0] ?? 2} inputs`,
      outputBadge: `${this.wasmService.xorArchitecture()?.slice(-1)?.[0] ?? 1} output`,
      accuracy: this.wasmService.xorModelInfo()?.accuracy,
      showNotTrained: false,
    },
    {
      route: '/iris-classifier',
      icon: '🌸',
      name: this.wasmService.irisModelInfo()?.name,
      fallbackName: 'Iris Classifier',
      description: this.wasmService.irisModelInfo()?.description,
      fallbackDescription: 'Flower species classification',
      isLoading: this.wasmService.irisWasmResource.isLoading(),
      loadingMessage: 'Loading Neural Network...',
      inputBadge: `${this.wasmService.irisArchitecture()?.[0] ?? 4} inputs`,
      outputBadge: `${this.wasmService.irisArchitecture()?.slice(-1)?.[0] ?? 3} classes`,
      accuracy: this.wasmService.irisModelInfo()?.accuracy,
      showNotTrained: false,
    },
    {
      route: '/mnist-digit',
      icon: '✍️',
      name: this.wasmService.mnistModelInfo()?.name,
      fallbackName: 'MNIST Digit',
      description: this.wasmService.mnistModelInfo()?.description,
      fallbackDescription: 'Handwritten digit recognition',
      isLoading: this.wasmService.mnistWasmResource.isLoading(),
      loadingMessage: 'Loading Neural Network...',
      inputBadge: `${this.wasmService.mnistArchitecture()?.[0] ?? 784} inputs`,
      outputBadge: `${this.wasmService.mnistArchitecture()?.slice(-1)?.[0] ?? 10} classes`,
      accuracy: this.wasmService.mnistModelInfo()?.accuracy,
      showNotTrained: false,
    },
    {
      route: '/mnist-lenet',
      icon: '🧠',
      name: this.wasmService.mnistLeNetModelInfo()?.name,
      fallbackName: 'LeNet-5',
      description: this.wasmService.mnistLeNetModelInfo()?.description,
      fallbackDescription: 'LeNet-5 CNN (LeCun et al., 1998)',
      isLoading: this.wasmService.mnistLeNetWasmResource.isLoading(),
      loadingMessage: 'Loading LeNet-5 CNN...',
      inputBadge: '28×28 image',
      outputBadge: '10 classes',
      accuracy: this.wasmService.mnistLeNetModelInfo()?.accuracy,
      showNotTrained: false,
    },
    {
      route: '/mnist-resnet',
      icon: '🔗',
      name: this.wasmService.mnistResNetModelInfo()?.name,
      fallbackName: 'ResNet',
      description: this.wasmService.mnistResNetModelInfo()?.description,
      fallbackDescription: 'ResNet CNN (He et al., 2015)',
      isLoading: this.wasmService.mnistResNetWasmResource.isLoading(),
      loadingMessage: 'Loading ResNet CNN...',
      inputBadge: '28×28 image',
      outputBadge: '10 classes',
      accuracy: this.wasmService.mnistResNetModelInfo()?.accuracy,
      showNotTrained: false,
    },
    {
      route: '/mnist-alexnet',
      icon: '🔥',
      name: this.wasmService.mnistAlexNetModelInfo()?.name,
      fallbackName: 'AlexNet-Mini',
      description: this.wasmService.mnistAlexNetModelInfo()?.description,
      fallbackDescription: 'AlexNet-style CNN (Krizhevsky et al., 2012)',
      isLoading: this.wasmService.mnistAlexNetWasmResource.isLoading(),
      loadingMessage: 'Loading AlexNet-Mini CNN...',
      inputBadge: '28×28 image',
      outputBadge: '10 classes',
      accuracy: this.wasmService.mnistAlexNetModelInfo()?.accuracy,
      showNotTrained: true,
    },
    {
      route: '/mnist-vgg',
      icon: '📚',
      name: this.wasmService.mnistVggModelInfo()?.name,
      fallbackName: 'VGG-Tiny',
      description: this.wasmService.mnistVggModelInfo()?.description,
      fallbackDescription: 'VGG-style CNN (Simonyan & Zisserman, 2014)',
      isLoading: this.wasmService.mnistVggWasmResource.isLoading(),
      loadingMessage: 'Loading VGG-Tiny CNN...',
      inputBadge: '28×28 image',
      outputBadge: '10 classes',
      accuracy: this.wasmService.mnistVggModelInfo()?.accuracy,
      showNotTrained: true,
    },
    {
      route: '/coming-soon',
      icon: '⏰',
      name: 'Coming Soon',
      fallbackName: 'Coming Soon',
      description: undefined,
      fallbackDescription: '',
      isLoading: false,
      loadingMessage: '',
      inputBadge: '',
      outputBadge: '',
      accuracy: undefined,
      showNotTrained: false,
      customBadge: 'Coming soon',
      customBadgeType: 'warning',
    },
  ]);
}
