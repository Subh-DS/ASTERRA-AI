const VERT = `#version 300 es
void main(){
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
  gl_Position = vec4(p, 0.0, 1.0);
}`

const FRAG = `#version 300 es
precision highp float;
out vec4 outColor;

uniform vec2 uRes;
uniform float uTime;
uniform vec2 uMouse;
uniform float uMouseIn;
uniform vec4 uLift;
uniform sampler2D uHeights;
uniform float uHasHeights;
uniform float uRealBlend;
uniform float uDim;
uniform float uAspect;
uniform vec3 uCDim;
uniform vec3 uCPaper;
uniform vec3 uCElev;
uniform vec3 uCSig;
uniform float uABoost;

vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 mod289(vec4 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(
      i.z + vec4(0.0, i1.z, i2.z, 1.0))
    + i.y + vec4(0.0, i1.y, i2.y, 1.0))
    + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

float fbm(vec3 p){
  float f = 0.0;
  float a = 0.5;
  for(int i = 0; i < 5; i++){
    f += a * snoise(p);
    p *= 2.02;
    a *= 0.5;
  }
  return f;
}

float realH(vec2 uv){
  if(uHasHeights < 0.5) return 0.0;
  return texture(uHeights, uv).r;
}

void main(){
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 q = vec2(uv.x * uAspect, uv.y);

  vec2 mUv = uMouse / uRes;
  vec2 toM = mUv - uv;
  toM.x *= uAspect;
  float md = length(toM);
  float pull = exp(-md * md * 16.0) * uMouseIn;
  q += normalize(toM + 1e-5) * pull * -0.09;

  float t = uTime * 0.05;
  float n = fbm(vec3(q * 2.1, t)) * 0.5 + 0.5;

  float lift = 0.0;
  float liftMask = 0.0;
  if(uLift.z > 0.001){
    vec2 lc = uv - uLift.xy;
    lc.x *= uAspect;
    float ld = length(lc);
    float bump = exp(-ld * ld * 7.0);
    float ringR = uLift.w;
    float ring = exp(-pow((ld - ringR) / 0.05, 2.0)) * smoothstep(ringR + 0.25, ringR, ld);
    lift = (bump * 0.30 + ring * 0.55) * uLift.z;
    liftMask = clamp(bump * 1.4 + ring * 1.2, 0.0, 1.0) * uLift.z;
  }

  float h = n + lift;
  float rb = clamp(uRealBlend, 0.0, 1.0);
  if(rb > 0.001){
    float rh = realH(uv);
    if(rb > 0.995){
      h = rh;
    } else {
      h = mix(h, mix(n, rh, 0.92) + lift * rb, rb);
    }
  }

  float lv = h * 15.0;
  float fw = fwidth(lv) + 1e-4;
  float dLine = abs(fract(lv) - 0.5);
  float line = 1.0 - min((dLine / fw) * 0.9, 1.0);

  float mv = lv / 5.0;
  float mfw = fw * 1.8 + 1e-4;
  float dMajor = abs(fract(mv) - 0.5);
  float major = 1.0 - min((dMajor / mfw) * 0.75, 1.0);

  vec3 cDim = uCDim;
  vec3 cPaper = uCPaper;
  vec3 cElev = uCElev;
  vec3 cSig = uCSig;

  vec3 col = uCDim;
  float alpha = line * (0.20 + pull * 0.45);
  alpha += major * (0.22 + pull * 0.35);
  col = mix(col, uCPaper, major * 0.55);

  col = mix(col, uCElev, clamp(liftMask, 0.0, 0.85));
  col = mix(col, uCSig, rb * (0.35 + 0.45 * line));
  alpha += rb * 0.10;

  alpha *= uDim * uABoost;
  outColor = vec4(col * uDim, clamp(alpha, 0.0, 1.0));
}`

function compile(gl, type, src) {
  const s = gl.createShader(type)
  gl.shaderSource(s, src)
  gl.compileShader(s)
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    // Info log can be null on some drivers — never throw Error(null).
    throw new Error(gl.getShaderInfoLog(s) || `contour-field shader compile failed (type=${type})`)
  }
  return s
}

const PALETTES = {
  dark: {
    dim: [0.604, 0.655, 0.612],
    paper: [0.929, 0.906, 0.839],
    elev: [0.91, 0.639, 0.239],
    sig: [0.31, 0.851, 0.769],
    boost: 1,
  },
  light: {
    dim: [0.42, 0.388, 0.318],
    paper: [0.106, 0.161, 0.129],
    elev: [0.702, 0.435, 0.071],
    sig: [0.039, 0.478, 0.408],
    boost: 1.25,
  },
}

export class ContourField {
  constructor(canvas, { tier = 'high', reducedMotion = false, interactive = true } = {}) {
    this.canvas = canvas
    this.reducedMotion = reducedMotion
    this.interactive = interactive
    this.tierMode = tier === 'low' ? 'static-frame' : tier === 'medium' ? 'live-halfres' : 'live-full'
    this.gl = canvas.getContext('webgl2', { alpha: true, antialias: false, premultipliedAlpha: false })
    if (!this.gl) {
      this.dead = true
      return
    }
    const gl = this.gl
    this.prog = gl.createProgram()
    gl.attachShader(this.prog, compile(gl, gl.VERTEX_SHADER, VERT))
    gl.attachShader(this.prog, compile(gl, gl.FRAGMENT_SHADER, FRAG))
    gl.linkProgram(this.prog)
    if (!gl.getProgramParameter(this.prog, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(this.prog) || 'contour-field program link failed')
    }
    gl.useProgram(this.prog)
    this.u = {}
    for (const name of [
      'uRes',
      'uTime',
      'uMouse',
      'uMouseIn',
      'uLift',
      'uHeights',
      'uHasHeights',
      'uRealBlend',
      'uDim',
      'uAspect',
      'uCDim',
      'uCPaper',
      'uCElev',
      'uCSig',
      'uABoost',
    ]) {
      this.u[name] = gl.getUniformLocation(this.prog, name)
    }
    this.theme = document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'
    this.heightTex = null
    this.hasHeights = 0
    this.realBlend = 0
    this.realTarget = 0
    this.dim = 1
    this.dimTarget = 1
    this.mouse = [-999, -999]
    this.mouseIn = 0
    this.mouseSmooth = [-999, -999]
    this.lift = { x: 0.5, y: 0.5, amount: 0, radius: 0.1, peak: 0 }
    this.start = performance.now()
    this.raf = null
    this.staticRendered = false
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
    this.resize()
    this.loop = this.loop.bind(this)
    this.onMove = this.onMove.bind(this)
    window.addEventListener('pointermove', this.onMove, { passive: true })
    window.addEventListener('pointerleave', () => (this.mouseIn = 0))
    if (this.tierMode !== 'static-frame' && !this.reducedMotion) {
      this.raf = requestAnimationFrame(this.loop)
    } else {
      this.renderFrame(0.5)
    }
  }

  onMove(e) {
    if (this.tierMode === 'static-frame' || !this.interactive) return
    const r = this.canvas.getBoundingClientRect()
    const dpr = this.canvas.width / Math.max(r.width, 1)
    this.mouse[0] = (e.clientX - r.left) * dpr
    this.mouse[1] = (r.height - (e.clientY - r.top)) * dpr
    this.mouseIn = 1
  }

  resize() {
    if (!this.gl) return
    const rect = this.canvas.getBoundingClientRect()
    const scale =
      this.tierMode === 'live-full'
        ? Math.min(window.devicePixelRatio || 1, 2)
        : this.tierMode === 'live-halfres'
          ? 0.66
          : 0.5
    const w = Math.max(2, Math.round(rect.width * scale))
    const h = Math.max(2, Math.round(rect.height * scale))
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w
      this.canvas.height = h
      this.gl.viewport(0, 0, w, h)
    }
    this.aspect = rect.width / Math.max(rect.height, 1)
    this.staticRendered = false
    if (this.tierMode === 'static-frame') this.renderFrame(0.5)
  }

  setDim(v) {
    this.dimTarget = v
  }

  setRealGrid(data, w, h, blend = 1) {
    if (!this.gl) return
    const gl = this.gl
    const bytes = new Uint8Array(w * h)
    let mn = Infinity
    let mx = -Infinity
    for (let i = 0; i < data.length; i++) {
      const v = data[i]
      if (v < mn) mn = v
      if (v > mx) mx = v
    }
    const span = mx - mn || 1
    for (let i = 0; i < data.length; i++) bytes[i] = ((data[i] - mn) / span) * 255
    if (this.heightTex) gl.deleteTexture(this.heightTex)
    this.heightTex = gl.createTexture()
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.heightTex)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, bytes)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    this.hasHeights = 1
    this.realTarget = blend
  }

  clearReal() {
    this.hasHeights = 0
    this.realTarget = 0
    this.realBlend = 0
    this.staticRendered = false
    if (this.tierMode === 'static-frame') this.renderFrame(0.5)
  }

  triggerLift(clientX, clientY) {
    if (!this.gl) return
    const r = this.canvas.getBoundingClientRect()
    this.lift.x = (clientX - r.left) / Math.max(r.width, 1)
    this.lift.y = 1 - (clientY - r.top) / Math.max(r.height, 1)
    this.lift.amount = 1
    this.lift.radius = 0.02
    this.lift.peak = performance.now()
    if (this.tierMode === 'static-frame') this.renderFrame(0.5)
  }

  setTheme(theme) {
    this.theme = theme === 'light' ? 'light' : 'dark'
    if (this.tierMode === 'static-frame') this.renderFrame(0.5)
  }

  renderFrame(timeSec) {
    const gl = this.gl
    if (!gl) return
    gl.useProgram(this.prog)
    gl.uniform2f(this.u.uRes, this.canvas.width, this.canvas.height)
    gl.uniform1f(this.u.uTime, timeSec)
    gl.uniform2f(this.u.uMouse, this.mouseSmooth[0], this.mouseSmooth[1])
    gl.uniform1f(this.u.uMouseIn, this.mouseIn)
    gl.uniform4f(this.u.uLift, this.lift.x, this.lift.y, this.lift.amount, this.lift.radius)
    gl.uniform1f(this.u.uHasHeights, this.hasHeights)
    gl.uniform1f(this.u.uRealBlend, this.realBlend)
    gl.uniform1f(this.u.uDim, this.dim)
    gl.uniform1f(this.u.uAspect, this.aspect || 1)
    const pal = PALETTES[this.theme] || PALETTES.dark
    gl.uniform3f(this.u.uCDim, pal.dim[0], pal.dim[1], pal.dim[2])
    gl.uniform3f(this.u.uCPaper, pal.paper[0], pal.paper[1], pal.paper[2])
    gl.uniform3f(this.u.uCElev, pal.elev[0], pal.elev[1], pal.elev[2])
    gl.uniform3f(this.u.uCSig, pal.sig[0], pal.sig[1], pal.sig[2])
    gl.uniform1f(this.u.uABoost, pal.boost)
    gl.uniform1i(this.u.uHeights, 0)
    gl.clearColor(0, 0, 0, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    gl.drawArrays(gl.TRIANGLES, 0, 3)
  }

  loop() {
    this.raf = requestAnimationFrame(this.loop)
    if (document.hidden) return
    const now = performance.now()
    this.dim += (this.dimTarget - this.dim) * 0.06
    this.realBlend += (this.realTarget - this.realBlend) * 0.04
    const k = this.mouseSmooth[0] < -900 ? 1 : 0.08
    this.mouseSmooth[0] += (this.mouse[0] - this.mouseSmooth[0]) * k
    this.mouseSmooth[1] += (this.mouse[1] - this.mouseSmooth[1]) * k
    if (this.lift.amount > 0.001) {
      const age = (now - this.lift.peak) / 1000
      this.lift.radius = Math.min(0.55, 0.02 + age * 0.28)
      this.lift.amount = Math.max(0, 1 - age * 0.55)
    }
    if (this.tierMode === 'live-full' && !this.reducedMotion) this.mouseIn = this.mouseIn
    else this.mouseIn = 0
    this.renderFrame(this.reducedMotion ? 12 : (now - this.start) / 1000)
  }

  dispose() {
    if (this.raf) cancelAnimationFrame(this.raf)
    window.removeEventListener('pointermove', this.onMove)
    if (this.heightTex && this.gl) this.gl.deleteTexture(this.heightTex)
    if (this.gl) {
      this.gl.getExtension('WEBGL_lose_context')?.loseContext()
    }
  }
}
