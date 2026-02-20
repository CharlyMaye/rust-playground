import {
  Color,
  DEFAULT_RENDER_CONFIG,
  INetworkRenderer,
  NetworkRenderData,
  RenderConfig,
  RendererPreference,
  Viewport,
} from './types';

import {
  BarData,
  ConfigurableRenderData,
  FeatureMapsData,
  GridData,
  LayerElement,
} from './configurable-layout-calculator';

/**
 * Configurable WebGL-based neural network renderer for high-performance visualization.
 *
 * Extends the basic WebGLRenderer to support:
 * - Heatmap layer representation (rendered as quads)
 * - Bar layer representation
 * - Stats display
 *
 * Content-First Architecture:
 * - Receives data in natural coordinates
 * - Applies uniform transformation matrix for scaling
 * - All elements scale uniformly
 *
 * Text and complex UI elements rendered to offscreen canvas then composited.
 */
export class ConfigurableWebGLRenderer implements INetworkRenderer {
  private readonly gl: WebGLRenderingContext;
  private readonly canvas: HTMLCanvasElement;
  private config: RenderConfig;
  private dpr: number = 1;
  private displayWidth: number = 0;
  private displayHeight: number = 0;

  // Shader programs
  private lineProgram: WebGLProgram | null = null;
  private circleProgram: WebGLProgram | null = null;
  private quadProgram: WebGLProgram | null = null;
  private textureProgram: WebGLProgram | null = null;

  // Buffers
  private lineBuffer: WebGLBuffer | null = null;
  private circleBuffer: WebGLBuffer | null = null;
  private quadBuffer: WebGLBuffer | null = null;
  private heatmapBuffer: WebGLBuffer | null = null;

  // Text rendering
  private textCanvas: HTMLCanvasElement;
  private textCtx: CanvasRenderingContext2D;
  private textTexture: WebGLTexture | null = null;

  constructor(canvas: HTMLCanvasElement, config: Partial<RenderConfig> = {}) {
    this.canvas = canvas;
    this.config = { ...DEFAULT_RENDER_CONFIG, ...config };

    const gl = canvas.getContext('webgl', {
      antialias: this.config.antialias,
      alpha: true,
      preserveDrawingBuffer: false,
    });

    if (!gl) {
      throw new Error('WebGL not supported');
    }

    this.gl = gl;

    // Create offscreen canvas for text/stats rendering
    this.textCanvas = document.createElement('canvas');
    this.textCtx = this.textCanvas.getContext('2d')!;

    // Configure image smoothing for sharp text rendering
    this.textCtx.imageSmoothingEnabled = true;
    this.textCtx.imageSmoothingQuality = 'high';

    this.initializeWebGL();
    this.setupCanvas();
  }

  // ============================================================================
  // Initialization
  // ============================================================================

  private initializeWebGL(): void {
    const gl = this.gl;

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    this.lineProgram = this.createLineProgram();
    this.circleProgram = this.createCircleProgram();
    this.quadProgram = this.createQuadProgram();
    this.textureProgram = this.createTextureProgram();

    this.lineBuffer = gl.createBuffer();
    this.circleBuffer = gl.createBuffer();
    this.quadBuffer = gl.createBuffer();
    this.heatmapBuffer = gl.createBuffer();
    this.textTexture = gl.createTexture();

    this.setupFullscreenQuad();
  }

  private setupFullscreenQuad(): void {
    const gl = this.gl;
    // Fullscreen quad: position (x, y) + texcoord (u, v)
    const quadVertices = new Float32Array([-1, -1, 0, 1, 1, -1, 1, 1, -1, 1, 0, 0, 1, 1, 1, 0]);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);
  }

  private setupCanvas(): void {
    this.dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.displayWidth = rect.width;
    this.displayHeight = rect.height;
    this.canvas.width = rect.width * this.dpr;
    this.canvas.height = rect.height * this.dpr;
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    // Setup text canvas to match
    this.textCanvas.width = this.canvas.width;
    this.textCanvas.height = this.canvas.height;
  }

  // ============================================================================
  // Shader Programs
  // ============================================================================

  private createLineProgram(): WebGLProgram {
    const vertexSource = `
      attribute vec2 a_position;
      attribute vec4 a_color;
      
      uniform vec2 u_resolution;
      uniform vec2 u_offset;
      uniform float u_scale;
      
      varying vec4 v_color;
      
      void main() {
        vec2 scaled = a_position * u_scale + u_offset;
        vec2 clipSpace = (scaled / u_resolution) * 2.0 - 1.0;
        gl_Position = vec4(clipSpace * vec2(1, -1), 0, 1);
        v_color = a_color;
      }
    `;

    const fragmentSource = `
      precision mediump float;
      varying vec4 v_color;
      
      void main() {
        gl_FragColor = v_color;
      }
    `;

    return this.createProgram(vertexSource, fragmentSource);
  }

  private createCircleProgram(): WebGLProgram {
    const vertexSource = `
      attribute vec2 a_center;
      attribute float a_radius;
      attribute vec4 a_color;
      attribute vec2 a_offset;
      
      uniform vec2 u_resolution;
      uniform vec2 u_translate;
      uniform float u_scale;
      
      varying vec4 v_color;
      varying vec2 v_texCoord;
      
      void main() {
        vec2 scaledCenter = a_center * u_scale + u_translate;
        float scaledRadius = a_radius * u_scale;
        vec2 position = scaledCenter + a_offset * scaledRadius;
        
        vec2 clipSpace = (position / u_resolution) * 2.0 - 1.0;
        gl_Position = vec4(clipSpace * vec2(1, -1), 0, 1);
        v_color = a_color;
        v_texCoord = a_offset;
      }
    `;

    const fragmentSource = `
      precision mediump float;
      varying vec4 v_color;
      varying vec2 v_texCoord;
      
      void main() {
        float dist = length(v_texCoord);
        if (dist > 1.0) discard;
        
        float alpha = 1.0 - smoothstep(0.95, 1.0, dist);
        gl_FragColor = vec4(v_color.rgb, v_color.a * alpha);
      }
    `;

    return this.createProgram(vertexSource, fragmentSource);
  }

  /**
   * Quad program for rendering heatmap cells
   */
  private createQuadProgram(): WebGLProgram {
    const vertexSource = `
      attribute vec2 a_position;
      attribute vec4 a_color;
      
      uniform vec2 u_resolution;
      uniform vec2 u_offset;
      uniform float u_scale;
      
      varying vec4 v_color;
      
      void main() {
        vec2 scaled = a_position * u_scale + u_offset;
        vec2 clipSpace = (scaled / u_resolution) * 2.0 - 1.0;
        gl_Position = vec4(clipSpace * vec2(1, -1), 0, 1);
        v_color = a_color;
      }
    `;

    const fragmentSource = `
      precision mediump float;
      varying vec4 v_color;
      
      void main() {
        gl_FragColor = v_color;
      }
    `;

    return this.createProgram(vertexSource, fragmentSource);
  }

  private createTextureProgram(): WebGLProgram {
    const vertexSource = `
      attribute vec2 a_position;
      attribute vec2 a_texCoord;
      
      varying vec2 v_texCoord;
      
      void main() {
        gl_Position = vec4(a_position, 0, 1);
        v_texCoord = a_texCoord;
      }
    `;

    const fragmentSource = `
      precision mediump float;
      uniform sampler2D u_texture;
      varying vec2 v_texCoord;
      
      void main() {
        gl_FragColor = texture2D(u_texture, v_texCoord);
      }
    `;

    return this.createProgram(vertexSource, fragmentSource);
  }

  private createProgram(vertexSource: string, fragmentSource: string): WebGLProgram {
    const gl = this.gl;

    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentSource);

    const program = gl.createProgram()!;
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`Failed to link program: ${gl.getProgramInfoLog(program)}`);
    }

    return program;
  }

  private compileShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Failed to compile shader: ${info}`);
    }

    return shader;
  }

  // ============================================================================
  // Color Utilities
  // ============================================================================

  private resolveColor(color: string): string {
    if (color.startsWith('var(')) {
      const varName = color.match(/var\(([^)]+)\)/)?.[1];
      if (varName) {
        return getComputedStyle(this.canvas).getPropertyValue(varName).trim() || color;
      }
    }
    return color;
  }

  private parseColor(color: string): Color {
    const resolved = this.resolveColor(color);

    if (resolved.startsWith('#')) {
      const hex = resolved.slice(1);
      return {
        r: parseInt(hex.slice(0, 2), 16) / 255,
        g: parseInt(hex.slice(2, 4), 16) / 255,
        b: parseInt(hex.slice(4, 6), 16) / 255,
        a: 1,
      };
    }

    const match = resolved.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
    if (match) {
      return {
        r: parseInt(match[1]) / 255,
        g: parseInt(match[2]) / 255,
        b: parseInt(match[3]) / 255,
        a: match[4] ? parseFloat(match[4]) : 1,
      };
    }

    return { r: 1, g: 1, b: 1, a: 1 };
  }

  // ============================================================================
  // Connection Rendering
  // ============================================================================

  private renderConnections(data: NetworkRenderData, viewport: Viewport): void {
    if (!this.lineProgram || !this.lineBuffer) return;

    const gl = this.gl;
    gl.useProgram(this.lineProgram);

    const vertices: number[] = [];
    for (const conn of data.connections) {
      const color = this.parseColor(conn.color);
      vertices.push(conn.from.x, conn.from.y, color.r, color.g, color.b, conn.opacity);
      vertices.push(conn.to.x, conn.to.y, color.r, color.g, color.b, conn.opacity);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);

    const resLoc = gl.getUniformLocation(this.lineProgram, 'u_resolution');
    const offsetLoc = gl.getUniformLocation(this.lineProgram, 'u_offset');
    const scaleLoc = gl.getUniformLocation(this.lineProgram, 'u_scale');

    gl.uniform2f(resLoc, this.displayWidth, this.displayHeight);
    gl.uniform2f(offsetLoc, viewport.offsetX, viewport.offsetY);
    gl.uniform1f(scaleLoc, viewport.scale);

    const posLoc = gl.getAttribLocation(this.lineProgram, 'a_position');
    const colorLoc = gl.getAttribLocation(this.lineProgram, 'a_color');
    const stride = 6 * 4;

    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 2 * 4);

    gl.drawArrays(gl.LINES, 0, vertices.length / 6);
  }

  // ============================================================================
  // Neuron Rendering
  // ============================================================================

  private renderNeurons(data: NetworkRenderData, viewport: Viewport): void {
    if (!this.circleProgram || !this.circleBuffer) return;

    const gl = this.gl;
    gl.useProgram(this.circleProgram);

    const quadOffsets = [
      [-1, -1],
      [1, -1],
      [-1, 1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ];

    const vertices: number[] = [];
    for (const neuron of data.neurons) {
      const color = this.parseColor(neuron.fill);
      for (const [ox, oy] of quadOffsets) {
        vertices.push(
          neuron.position.x,
          neuron.position.y,
          neuron.radius,
          color.r,
          color.g,
          color.b,
          1.0,
          ox,
          oy,
        );
      }
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.circleBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);

    const resLoc = gl.getUniformLocation(this.circleProgram, 'u_resolution');
    const translateLoc = gl.getUniformLocation(this.circleProgram, 'u_translate');
    const scaleLoc = gl.getUniformLocation(this.circleProgram, 'u_scale');

    gl.uniform2f(resLoc, this.displayWidth, this.displayHeight);
    gl.uniform2f(translateLoc, viewport.offsetX, viewport.offsetY);
    gl.uniform1f(scaleLoc, viewport.scale);

    const centerLoc = gl.getAttribLocation(this.circleProgram, 'a_center');
    const radiusLoc = gl.getAttribLocation(this.circleProgram, 'a_radius');
    const colorLoc = gl.getAttribLocation(this.circleProgram, 'a_color');
    const offsetLoc = gl.getAttribLocation(this.circleProgram, 'a_offset');
    const stride = 9 * 4;

    gl.enableVertexAttribArray(centerLoc);
    gl.vertexAttribPointer(centerLoc, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(radiusLoc);
    gl.vertexAttribPointer(radiusLoc, 1, gl.FLOAT, false, stride, 2 * 4);
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 3 * 4);
    gl.enableVertexAttribArray(offsetLoc);
    gl.vertexAttribPointer(offsetLoc, 2, gl.FLOAT, false, stride, 7 * 4);

    gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 9);
  }

  // ============================================================================
  // Layer Element Rendering (Heatmaps, Bars)
  // ============================================================================

  private renderLayerElements(elements: readonly LayerElement[], viewport: Viewport): void {
    for (const element of elements) {
      switch (element.type) {
        case 'heatmap':
          if (element.gridData) {
            this.renderHeatmapWebGL(element, element.gridData, viewport);
          }
          break;

        case 'feature-maps':
          if (element.featureMapsData) {
            this.renderFeatureMapsWebGL(element, element.featureMapsData, viewport);
          }
          break;

        case 'bar':
          if (element.barData) {
            this.renderBarWebGL(element, element.barData, viewport);
          }
          break;

        // Stats and collapsed rendered via text canvas
      }
    }
  }

  private renderHeatmapWebGL(element: LayerElement, gridData: GridData, viewport: Viewport): void {
    if (!this.quadProgram || !this.heatmapBuffer) return;

    const gl = this.gl;
    gl.useProgram(this.quadProgram);

    const { rows, cols, cellSize, colors } = gridData;
    const startX = element.position.x - (cols * cellSize) / 2;
    const startY = element.position.y - (rows * cellSize) / 2;

    // Build quad vertices for each cell
    const vertices: number[] = [];

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const idx = row * cols + col;
        const colorStr = colors[idx] ?? 'rgb(50,50,50)';
        const color = this.parseColor(colorStr);

        const x = startX + col * cellSize;
        const y = startY + row * cellSize;

        // Two triangles for a quad
        // Triangle 1
        vertices.push(x, y, color.r, color.g, color.b, 1.0);
        vertices.push(x + cellSize, y, color.r, color.g, color.b, 1.0);
        vertices.push(x, y + cellSize, color.r, color.g, color.b, 1.0);
        // Triangle 2
        vertices.push(x + cellSize, y, color.r, color.g, color.b, 1.0);
        vertices.push(x + cellSize, y + cellSize, color.r, color.g, color.b, 1.0);
        vertices.push(x, y + cellSize, color.r, color.g, color.b, 1.0);
      }
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.heatmapBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);

    const resLoc = gl.getUniformLocation(this.quadProgram, 'u_resolution');
    const offsetLoc = gl.getUniformLocation(this.quadProgram, 'u_offset');
    const scaleLoc = gl.getUniformLocation(this.quadProgram, 'u_scale');

    gl.uniform2f(resLoc, this.displayWidth, this.displayHeight);
    gl.uniform2f(offsetLoc, viewport.offsetX, viewport.offsetY);
    gl.uniform1f(scaleLoc, viewport.scale);

    const posLoc = gl.getAttribLocation(this.quadProgram, 'a_position');
    const colorLoc = gl.getAttribLocation(this.quadProgram, 'a_color');
    const stride = 6 * 4;

    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 2 * 4);

    gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 6);
  }

  private renderFeatureMapsWebGL(
    element: LayerElement,
    fmData: FeatureMapsData,
    viewport: Viewport,
  ): void {
    if (!this.quadProgram || !this.heatmapBuffer) return;

    const gl = this.gl;
    gl.useProgram(this.quadProgram);

    const { mapRows, mapCols, cellSize, gridCols, gap, maps } = fmData;
    const mapWidth = mapCols * cellSize;
    const mapHeight = mapRows * cellSize;

    const totalWidth = element.width;
    const totalHeight = element.height;
    const originX = element.position.x - totalWidth / 2;
    const originY = element.position.y - totalHeight / 2;

    const vertices: number[] = [];

    for (let i = 0; i < maps.length; i++) {
      const col = i % gridCols;
      const row = Math.floor(i / gridCols);
      const mapX = originX + col * (mapWidth + gap);
      const mapY = originY + row * (mapHeight + gap);

      const gridData = maps[i];
      for (let r = 0; r < mapRows; r++) {
        for (let c = 0; c < mapCols; c++) {
          const idx = r * mapCols + c;
          const colorStr = gridData.colors[idx] ?? 'rgb(50,50,50)';
          const color = this.parseColor(colorStr);

          const x = mapX + c * cellSize;
          const y = mapY + r * cellSize;

          vertices.push(x, y, color.r, color.g, color.b, 1.0);
          vertices.push(x + cellSize, y, color.r, color.g, color.b, 1.0);
          vertices.push(x, y + cellSize, color.r, color.g, color.b, 1.0);
          vertices.push(x + cellSize, y, color.r, color.g, color.b, 1.0);
          vertices.push(x + cellSize, y + cellSize, color.r, color.g, color.b, 1.0);
          vertices.push(x, y + cellSize, color.r, color.g, color.b, 1.0);
        }
      }
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.heatmapBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);

    const resLoc = gl.getUniformLocation(this.quadProgram, 'u_resolution');
    const offsetLoc = gl.getUniformLocation(this.quadProgram, 'u_offset');
    const scaleLoc = gl.getUniformLocation(this.quadProgram, 'u_scale');

    gl.uniform2f(resLoc, this.displayWidth, this.displayHeight);
    gl.uniform2f(offsetLoc, viewport.offsetX, viewport.offsetY);
    gl.uniform1f(scaleLoc, viewport.scale);

    const posLoc = gl.getAttribLocation(this.quadProgram, 'a_position');
    const colorLoc = gl.getAttribLocation(this.quadProgram, 'a_color');
    const stride = 6 * 4;

    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 2 * 4);

    gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 6);
  }

  private renderBarWebGL(element: LayerElement, barData: BarData, viewport: Viewport): void {
    if (!this.quadProgram || !this.heatmapBuffer) return;

    const gl = this.gl;
    gl.useProgram(this.quadProgram);

    const { width, height, colorGradient } = barData;
    const startX = element.position.x - width / 2;
    const startY = element.position.y - height / 2;

    // Build vertical gradient bar
    const vertices: number[] = [];
    const bands = Math.min(colorGradient.length, 50);
    const bandWidth = width / bands;

    for (let i = 0; i < bands; i++) {
      const colorIdx = Math.floor((i / bands) * colorGradient.length);
      const color = this.parseColor(colorGradient[colorIdx] ?? 'rgb(50,50,50)');

      const x = startX + i * bandWidth;

      // Quad for this band
      vertices.push(x, startY, color.r, color.g, color.b, 1.0);
      vertices.push(x + bandWidth, startY, color.r, color.g, color.b, 1.0);
      vertices.push(x, startY + height, color.r, color.g, color.b, 1.0);
      vertices.push(x + bandWidth, startY, color.r, color.g, color.b, 1.0);
      vertices.push(x + bandWidth, startY + height, color.r, color.g, color.b, 1.0);
      vertices.push(x, startY + height, color.r, color.g, color.b, 1.0);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.heatmapBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);

    const resLoc = gl.getUniformLocation(this.quadProgram, 'u_resolution');
    const offsetLoc = gl.getUniformLocation(this.quadProgram, 'u_offset');
    const scaleLoc = gl.getUniformLocation(this.quadProgram, 'u_scale');

    gl.uniform2f(resLoc, this.displayWidth, this.displayHeight);
    gl.uniform2f(offsetLoc, viewport.offsetX, viewport.offsetY);
    gl.uniform1f(scaleLoc, viewport.scale);

    const posLoc = gl.getAttribLocation(this.quadProgram, 'a_position');
    const colorLoc = gl.getAttribLocation(this.quadProgram, 'a_color');
    const stride = 6 * 4;

    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 2 * 4);

    gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 6);
  }

  // ============================================================================
  // Text Rendering (2D Canvas → Texture)
  // ============================================================================

  private renderTextToCanvas(
    data: NetworkRenderData,
    viewport: Viewport,
    layerElements?: readonly LayerElement[],
  ): void {
    const ctx = this.textCtx;
    ctx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);

    ctx.save();
    ctx.scale(this.dpr, this.dpr);

    // Render layer labels
    for (const label of data.labels) {
      const screenX = label.position.x * viewport.scale + viewport.offsetX;
      const screenY = label.position.y * viewport.scale + viewport.offsetY;

      ctx.fillStyle = this.resolveColor(label.color);
      ctx.font = `normal ${label.fontSize}px 'Segoe UI', system-ui, sans-serif`;
      ctx.textAlign = label.align;
      ctx.textBaseline = 'middle';
      ctx.fillText(label.text, screenX, screenY);
    }

    // Render neuron values and labels
    for (const neuron of data.neurons) {
      const screenX = neuron.position.x * viewport.scale + viewport.offsetX;
      const screenY = neuron.position.y * viewport.scale + viewport.offsetY;
      const fontSize = neuron.fontSize * viewport.scale;

      // Neuron value
      if (neuron.value) {
        ctx.fillStyle = this.resolveColor(neuron.stroke);
        ctx.font = `${neuron.fontWeight} ${fontSize}px 'Segoe UI', system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(neuron.value, screenX, screenY);
      }

      // Neuron label
      if (neuron.label && neuron.labelPosition) {
        const labelX = neuron.labelPosition.x * viewport.scale + viewport.offsetX;
        const labelY = neuron.labelPosition.y * viewport.scale + viewport.offsetY;

        ctx.fillStyle = this.resolveColor('var(--nn-neutral)');
        ctx.font = `normal ${11 * viewport.scale}px 'Segoe UI', system-ui, sans-serif`;
        ctx.textAlign = neuron.labelAlign || 'center';
        ctx.fillText(neuron.label, labelX, labelY);
      }
    }

    // Render stats/bar labels for layer elements
    if (layerElements) {
      for (const element of layerElements) {
        if (element.type === 'stats' || element.type === 'collapsed') {
          this.renderStatsToCanvas(ctx, element, viewport);
        } else if (element.type === 'bar' && element.barData) {
          this.renderBarLabelToCanvas(ctx, element, element.barData, viewport);
        }
      }
    }

    ctx.restore();
  }

  private renderStatsToCanvas(
    ctx: CanvasRenderingContext2D,
    element: LayerElement,
    viewport: Viewport,
  ): void {
    if (!element.statsData) return;

    const { count, mean, std } = element.statsData;
    const screenX = element.position.x * viewport.scale + viewport.offsetX;
    const screenY = element.position.y * viewport.scale + viewport.offsetY;
    const width = element.width * viewport.scale;
    const height = element.height * viewport.scale;

    // Draw background
    ctx.fillStyle = '#1a1a2e';
    ctx.strokeStyle = this.resolveColor('var(--nn-stroke)');
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(screenX - width / 2, screenY - height / 2, width, height, 8 * viewport.scale);
    ctx.fill();
    ctx.stroke();

    // Draw count
    ctx.fillStyle = this.resolveColor('var(--nn-positive)');
    ctx.font = `bold ${14 * viewport.scale}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`n=${count}`, screenX, screenY - 10 * viewport.scale);

    // Draw stats
    ctx.fillStyle = 'white';
    ctx.font = `${10 * viewport.scale}px monospace`;
    ctx.fillText(`μ=${mean.toFixed(2)}`, screenX, screenY + 8 * viewport.scale);
    ctx.fillText(`σ=${std.toFixed(2)}`, screenX, screenY + 20 * viewport.scale);
  }

  private renderBarLabelToCanvas(
    ctx: CanvasRenderingContext2D,
    element: LayerElement,
    barData: BarData,
    viewport: Viewport,
  ): void {
    const screenX = element.position.x * viewport.scale + viewport.offsetX;
    const screenY = element.position.y * viewport.scale + viewport.offsetY;
    const height = element.height * viewport.scale;

    ctx.fillStyle = 'white';
    ctx.font = `${10 * viewport.scale}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`μ=${barData.mean.toFixed(2)}`, screenX, screenY - height / 2 - 2);
  }

  private compositeTextTexture(): void {
    if (!this.textureProgram || !this.quadBuffer || !this.textTexture) return;

    const gl = this.gl;

    gl.bindTexture(gl.TEXTURE_2D, this.textTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.textCanvas);

    gl.useProgram(this.textureProgram);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);

    const posLoc = gl.getAttribLocation(this.textureProgram, 'a_position');
    const texLoc = gl.getAttribLocation(this.textureProgram, 'a_texCoord');
    const stride = 4 * 4;

    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(texLoc);
    gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, stride, 2 * 4);

    const textureLoc = gl.getUniformLocation(this.textureProgram, 'u_texture');
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textTexture);
    gl.uniform1i(textureLoc, 0);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // ============================================================================
  // Debug Rendering
  // ============================================================================

  private renderDebugInfo(
    data: NetworkRenderData,
    viewport: Viewport,
    layerElements?: readonly LayerElement[],
  ): void {
    const ctx = this.textCtx;

    ctx.save();
    ctx.scale(this.dpr, this.dpr);

    ctx.fillStyle = this.resolveColor('var(--nn-neutral)');
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';

    const layerTypes = layerElements?.map((e) => e.type).join(', ') ?? 'neurons';

    const info = [
      `Renderer: WebGL (Configurable)`,
      `Natural: ${data.naturalBounds.width.toFixed(0)}×${data.naturalBounds.height.toFixed(0)}`,
      `Scale: ${viewport.scale.toFixed(3)}`,
      `Connections: ${data.connections.length}`,
      `Neurons: ${data.neurons.length}`,
      `Layers: ${layerTypes}`,
    ];

    info.forEach((line, i) => {
      ctx.fillText(line, 10, 10 + i * 12);
    });

    ctx.restore();
  }

  // ============================================================================
  // INetworkRenderer Implementation
  // ============================================================================

  render(data: NetworkRenderData, viewport: Viewport): void {
    this.clear();

    const configurableData = data as ConfigurableRenderData;
    const layerElements = configurableData.layerElements;

    // Render WebGL content
    this.renderConnections(data, viewport);

    // Render layer elements (heatmaps, bars)
    if (layerElements) {
      this.renderLayerElements(layerElements, viewport);
    }

    // Render neurons
    this.renderNeurons(data, viewport);

    // Render text to canvas
    this.renderTextToCanvas(data, viewport, layerElements);

    if (this.config.debug) {
      this.renderDebugInfo(data, viewport, layerElements);
    }

    // Composite text as texture
    this.compositeTextTexture();
  }

  clear(): void {
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    this.textCtx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
  }

  resize(width: number, height: number): void {
    // Update DPR in case of browser zoom
    this.dpr = window.devicePixelRatio || 1;

    this.displayWidth = width;
    this.displayHeight = height;
    this.canvas.width = width * this.dpr;
    this.canvas.height = height * this.dpr;
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    this.textCanvas.width = this.canvas.width;
    this.textCanvas.height = this.canvas.height;
  }

  updateConfig(config: Partial<RenderConfig>): void {
    this.config = { ...this.config, ...config };
  }

  destroy(): void {
    const gl = this.gl;

    if (this.lineProgram) gl.deleteProgram(this.lineProgram);
    if (this.circleProgram) gl.deleteProgram(this.circleProgram);
    if (this.quadProgram) gl.deleteProgram(this.quadProgram);
    if (this.textureProgram) gl.deleteProgram(this.textureProgram);
    if (this.lineBuffer) gl.deleteBuffer(this.lineBuffer);
    if (this.circleBuffer) gl.deleteBuffer(this.circleBuffer);
    if (this.quadBuffer) gl.deleteBuffer(this.quadBuffer);
    if (this.heatmapBuffer) gl.deleteBuffer(this.heatmapBuffer);
    if (this.textTexture) gl.deleteTexture(this.textTexture);

    this.lineProgram = null;
    this.circleProgram = null;
    this.quadProgram = null;
    this.textureProgram = null;
    this.lineBuffer = null;
    this.circleBuffer = null;
    this.quadBuffer = null;
    this.heatmapBuffer = null;
    this.textTexture = null;

    const loseContext = gl.getExtension('WEBGL_lose_context');
    if (loseContext) loseContext.loseContext();
  }

  getType(): RendererPreference {
    return 'webgl';
  }
}
