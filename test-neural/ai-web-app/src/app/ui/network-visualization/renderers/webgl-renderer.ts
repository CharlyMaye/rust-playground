import {
  Color,
  DEFAULT_RENDER_CONFIG,
  INetworkRenderer,
  NetworkRenderData,
  RenderConfig,
  RendererPreference,
  Viewport,
} from './types';

/**
 * WebGL-based neural network renderer for high-performance visualization.
 *
 * Content-First Architecture:
 * - Receives data in natural coordinates
 * - Applies uniform transformation matrix for scaling
 * - All elements (neurons, connections, labels) scale uniformly
 *
 * Text Rendering:
 * - Text is rendered to an offscreen canvas
 * - Then uploaded as a texture and composited on the WebGL canvas
 * - Single canvas output, no overlay needed
 */
export class WebGLRenderer implements INetworkRenderer {
  private readonly gl: WebGLRenderingContext;
  private readonly canvas: HTMLCanvasElement;
  private config: RenderConfig;
  private dpr: number = 1;
  private displayWidth: number = 0;
  private displayHeight: number = 0;

  // Shader programs
  private lineProgram: WebGLProgram | null = null;
  private circleProgram: WebGLProgram | null = null;
  private textureProgram: WebGLProgram | null = null;

  // Buffers
  private lineBuffer: WebGLBuffer | null = null;
  private circleBuffer: WebGLBuffer | null = null;
  private quadBuffer: WebGLBuffer | null = null;

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

    // Create offscreen canvas for text rendering
    this.textCanvas = document.createElement('canvas');
    this.textCtx = this.textCanvas.getContext('2d')!;

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
    this.textureProgram = this.createTextureProgram();

    this.lineBuffer = gl.createBuffer();
    this.circleBuffer = gl.createBuffer();
    this.quadBuffer = gl.createBuffer();
    this.textTexture = gl.createTexture();

    // Setup fullscreen quad for texture rendering
    this.setupQuadBuffer();
  }

  private setupQuadBuffer(): void {
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
        // Apply scale and offset, then convert to clip space
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
        // Scale center and radius, apply offset
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
        
        // Anti-aliasing
        float alpha = 1.0 - smoothstep(0.95, 1.0, dist);
        gl_FragColor = vec4(v_color.rgb, v_color.a * alpha);
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
  // Rendering (Natural Coordinates)
  // ============================================================================

  private renderConnections(data: NetworkRenderData, viewport: Viewport): void {
    if (!this.lineProgram || !this.lineBuffer) return;

    const gl = this.gl;
    gl.useProgram(this.lineProgram);

    // Prepare vertex data (position + color)
    const vertices: number[] = [];
    for (const conn of data.connections) {
      const color = this.parseColor(conn.color);
      vertices.push(conn.from.x, conn.from.y, color.r, color.g, color.b, conn.opacity);
      vertices.push(conn.to.x, conn.to.y, color.r, color.g, color.b, conn.opacity);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);

    // Set uniforms (transformation)
    const resLoc = gl.getUniformLocation(this.lineProgram, 'u_resolution');
    const offsetLoc = gl.getUniformLocation(this.lineProgram, 'u_offset');
    const scaleLoc = gl.getUniformLocation(this.lineProgram, 'u_scale');

    gl.uniform2f(resLoc, this.displayWidth, this.displayHeight);
    gl.uniform2f(offsetLoc, viewport.offsetX, viewport.offsetY);
    gl.uniform1f(scaleLoc, viewport.scale);

    // Set attributes
    const posLoc = gl.getAttribLocation(this.lineProgram, 'a_position');
    const colorLoc = gl.getAttribLocation(this.lineProgram, 'a_color');
    const stride = 6 * 4;

    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, stride, 2 * 4);

    gl.drawArrays(gl.LINES, 0, vertices.length / 6);
  }

  private renderNeurons(data: NetworkRenderData, viewport: Viewport): void {
    if (!this.circleProgram || !this.circleBuffer) return;

    const gl = this.gl;
    gl.useProgram(this.circleProgram);

    // Quad offsets for circle rendering
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

    // Set uniforms
    const resLoc = gl.getUniformLocation(this.circleProgram, 'u_resolution');
    const translateLoc = gl.getUniformLocation(this.circleProgram, 'u_translate');
    const scaleLoc = gl.getUniformLocation(this.circleProgram, 'u_scale');

    gl.uniform2f(resLoc, this.displayWidth, this.displayHeight);
    gl.uniform2f(translateLoc, viewport.offsetX, viewport.offsetY);
    gl.uniform1f(scaleLoc, viewport.scale);

    // Set attributes
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

  /**
   * Render labels to offscreen 2D canvas, then composite as WebGL texture.
   */
  private renderLabels(data: NetworkRenderData, viewport: Viewport): void {
    // Step 1: Render text to 2D canvas
    this.renderTextToCanvas(data, viewport);

    // Step 2: Upload canvas as texture and draw as fullscreen quad
    this.compositeTextTexture();
  }

  private renderTextToCanvas(data: NetworkRenderData, viewport: Viewport): void {
    const ctx = this.textCtx;
    ctx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);

    // Apply DPR scaling
    ctx.save();
    ctx.scale(this.dpr, this.dpr);

    // Render layer labels
    for (const label of data.labels) {
      const screenX = label.position.x * viewport.scale + viewport.offsetX;
      const screenY = label.position.y * viewport.scale + viewport.offsetY;
      const fontSize = label.fontSize * viewport.scale;

      ctx.fillStyle = this.resolveColor(label.color);
      ctx.font = `normal ${fontSize}px 'Segoe UI', system-ui, sans-serif`;
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
      ctx.fillStyle = this.resolveColor(neuron.stroke);
      ctx.font = `${neuron.fontWeight} ${fontSize}px 'Segoe UI', system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(neuron.value, screenX, screenY);

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

    ctx.restore();
  }

  /**
   * Upload text canvas as texture and render as fullscreen quad.
   */
  private compositeTextTexture(): void {
    if (!this.textureProgram || !this.quadBuffer || !this.textTexture) return;

    const gl = this.gl;

    // Upload text canvas as texture
    gl.bindTexture(gl.TEXTURE_2D, this.textTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.textCanvas);

    // Draw fullscreen quad with texture
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

  private renderDebugInfo(data: NetworkRenderData, viewport: Viewport): void {
    const ctx = this.textCtx;

    // Debug info is rendered before compositing
    ctx.save();
    ctx.scale(this.dpr, this.dpr);

    ctx.fillStyle = this.resolveColor('var(--nn-neutral)');
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';

    const info = [
      `Renderer: WebGL`,
      `Natural: ${data.naturalBounds.width.toFixed(0)}×${data.naturalBounds.height.toFixed(0)}`,
      `Scale: ${viewport.scale.toFixed(3)}`,
      `Connections: ${data.connections.length}`,
      `Neurons: ${data.neurons.length}`,
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

    if (this.config.debug) {
      console.log('[WebGL] Rendering:', {
        natural: data.naturalBounds,
        viewport,
        connections: data.connections.length,
        neurons: data.neurons.length,
      });
    }

    // Render WebGL content (connections and neuron circles)
    this.renderConnections(data, viewport);
    this.renderNeurons(data, viewport);

    // Render text to canvas (includes debug info if enabled)
    this.renderTextToCanvas(data, viewport);

    if (this.config.debug) {
      this.renderDebugInfo(data, viewport);
    }

    // Composite text as texture on top of WebGL
    this.compositeTextTexture();
  }

  clear(): void {
    this.gl.clearColor(0, 0, 0, 0);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
    this.textCtx.clearRect(0, 0, this.textCanvas.width, this.textCanvas.height);
  }

  resize(width: number, height: number): void {
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
    if (this.textureProgram) gl.deleteProgram(this.textureProgram);
    if (this.lineBuffer) gl.deleteBuffer(this.lineBuffer);
    if (this.circleBuffer) gl.deleteBuffer(this.circleBuffer);
    if (this.quadBuffer) gl.deleteBuffer(this.quadBuffer);
    if (this.textTexture) gl.deleteTexture(this.textTexture);

    this.lineProgram = null;
    this.circleProgram = null;
    this.textureProgram = null;
    this.lineBuffer = null;
    this.circleBuffer = null;
    this.quadBuffer = null;
    this.textTexture = null;

    const loseContext = gl.getExtension('WEBGL_lose_context');
    if (loseContext) loseContext.loseContext();
  }

  getType(): RendererPreference {
    return 'webgl';
  }
}
