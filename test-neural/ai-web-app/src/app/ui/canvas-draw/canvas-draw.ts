import { ChangeDetectionStrategy, Component, ElementRef, afterNextRender, input, output, viewChild } from '@angular/core';

@Component({
  selector: 'app-canvas-draw',
  imports: [],
  templateUrl: './canvas-draw.html',
  host: { class: 'canvas-draw' },
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CanvasDraw {
  public readonly gridSize = input<{ rows: number; cols: number }>({
    rows: 28,
    cols: 28,
  });

  public readonly dataChanged = output<number[][]>();

  /** Emitted only when the user lifts the mouse/finger (end of stroke).
   *  Use for expensive operations like CNN activation computation. */
  public readonly drawingComplete = output<number[][]>();

  private readonly canvasRef = viewChild.required<ElementRef<HTMLCanvasElement>>('canvas');
  private ctx: CanvasRenderingContext2D | null = null;
  private isDrawing = false;
  private grid: number[][] = [];
  private cellWidth = 0;
  private cellHeight = 0;

  constructor() {
    afterNextRender(() => {
      this.initCanvas();
    });
  }

  private initCanvas(): void {
    const canvas = this.canvasRef().nativeElement;
    this.ctx = canvas.getContext('2d');

    if (!this.ctx) return;

    const { rows, cols } = this.gridSize();
    this.cellWidth = canvas.width / cols;
    this.cellHeight = canvas.height / rows;

    // Initialize grid with zeros
    this.grid = Array(rows)
      .fill(0)
      .map(() => Array(cols).fill(0));

    this.drawGrid();
    this.setupEventListeners(canvas);
  }

  private drawGrid(): void {
    if (!this.ctx) return;

    const canvas = this.canvasRef().nativeElement;
    const { rows, cols } = this.gridSize();

    // Clear canvas
    this.ctx.fillStyle = 'white';
    this.ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid cells
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const intensity = this.grid[row][col];
        if (intensity > 0) {
          // Convert 0-255 to 0-1 for CSS rgba
          const alpha = intensity / 255;
          this.ctx.fillStyle = `rgba(0, 0, 0, ${alpha})`;
          this.ctx.fillRect(
            col * this.cellWidth,
            row * this.cellHeight,
            this.cellWidth,
            this.cellHeight,
          );
        }
      }
    }

    // Draw grid lines
    this.ctx.strokeStyle = '#e0e0e0';
    this.ctx.lineWidth = 0.5;

    // Vertical lines
    for (let i = 0; i <= cols; i++) {
      this.ctx.beginPath();
      this.ctx.moveTo(i * this.cellWidth, 0);
      this.ctx.lineTo(i * this.cellWidth, canvas.height);
      this.ctx.stroke();
    }

    // Horizontal lines
    for (let i = 0; i <= rows; i++) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, i * this.cellHeight);
      this.ctx.lineTo(canvas.width, i * this.cellHeight);
      this.ctx.stroke();
    }
  }

  private setupEventListeners(canvas: HTMLCanvasElement): void {
    // Mouse events
    canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
    canvas.addEventListener('mousemove', (e) => this.draw(e));
    canvas.addEventListener('mouseup', () => this.stopDrawing());
    canvas.addEventListener('mouseleave', () => this.stopDrawing());

    // Touch events for mobile - must be non-passive to allow preventDefault()
    canvas.addEventListener(
      'touchstart',
      (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
          clientX: touch.clientX,
          clientY: touch.clientY,
        });
        canvas.dispatchEvent(mouseEvent);
      },
      { passive: false },
    );

    canvas.addEventListener(
      'touchmove',
      (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
          clientX: touch.clientX,
          clientY: touch.clientY,
        });
        canvas.dispatchEvent(mouseEvent);
      },
      { passive: false },
    );

    canvas.addEventListener(
      'touchend',
      (e) => {
        e.preventDefault();
        const mouseEvent = new MouseEvent('mouseup', {});
        canvas.dispatchEvent(mouseEvent);
      },
      { passive: false },
    );
  }

  private startDrawing(event: MouseEvent): void {
    this.isDrawing = true;
    this.draw(event);
  }

  private draw(event: MouseEvent): void {
    if (!this.isDrawing) return;

    const canvas = this.canvasRef().nativeElement;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const col = Math.floor(x / this.cellWidth);
    const row = Math.floor(y / this.cellHeight);

    const { rows, cols } = this.gridSize();
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      this.grid[row][col] = 255; // Full intensity (0-255 like MNIST)
      this.drawGrid();
      // Emit a clone for zoneless change detection
      this.dataChanged.emit(this.grid.map((row) => [...row]));
    }
  }

  private stopDrawing(): void {
    if (this.isDrawing) {
      this.isDrawing = false;
      this.drawingComplete.emit(this.grid.map((row) => [...row]));
    }
  }

  public clear(): void {
    const { rows, cols } = this.gridSize();
    this.grid = Array(rows)
      .fill(0)
      .map(() => Array(cols).fill(0));
    this.drawGrid();
    // Emit a clone for zoneless change detection
    this.dataChanged.emit(this.grid.map((row) => [...row]));
  }

  public getGridData(): number[][] {
    return this.grid;
  }

  public setGridData(data: number[][]): void {
    this.grid = data;
    this.drawGrid();
  }
}
