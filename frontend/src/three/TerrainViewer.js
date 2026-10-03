import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { FlyControls } from './flyControls'
import { formatScale, niceScale } from '../utils/scaleBar'

const SKY_VERT = `
varying vec3 vDir;
void main(){
  vDir = normalize(position);
  vec4 p = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * p;
}`

const SKY_FRAG = `
varying vec3 vDir;
uniform vec3 uTop;
uniform vec3 uHorizon;
uniform vec3 uSunDir;
uniform vec3 uSunTint;
void main(){
  float t = smoothstep(-0.08, 0.55, vDir.y);
  vec3 col = mix(uHorizon, uTop, t);
  float glow = pow(max(dot(normalize(vDir), normalize(uSunDir)), 0.0), 18.0);
  col += uSunTint * glow * 0.5;
  gl_FragColor = vec4(col, 1.0);
}`

const SLOPE_VERT = `
varying vec3 vNormalW;
void main(){
  vNormalW = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`

const SLOPE_FRAG = `
varying vec3 vNormalW;
uniform vec3 uSunDir;
vec3 ramp(float t){
  t = clamp(t, 0.0, 1.0);
  vec3 c1 = vec3(0.310, 0.851, 0.769);
  vec3 c2 = vec3(0.929, 0.906, 0.839);
  vec3 c3 = vec3(0.910, 0.639, 0.239);
  vec3 c4 = vec3(0.886, 0.329, 0.227);
  if(t < 0.33) return mix(c1, c2, t / 0.33);
  if(t < 0.66) return mix(c2, c3, (t - 0.33) / 0.33);
  return mix(c3, c4, (t - 0.66) / 0.34);
}
void main(){
  float slope = 1.0 - clamp(abs(vNormalW.y), 0.0, 1.0);
  float shade = 0.45 + 0.55 * max(dot(normalize(vNormalW), normalize(uSunDir)), 0.0);
  gl_FragColor = vec4(ramp(pow(slope, 0.7)) * shade, 1.0);
}`

const ELEV_VERT = `
varying float vH;
varying vec3 vNormalW;
varying vec2 vUvW;
void main(){
  vH = (modelMatrix * vec4(position, 1.0)).y;
  vNormalW = normalize(mat3(modelMatrix) * normal);
  vUvW = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`

const ELEV_FRAG = `
varying float vH;
varying vec3 vNormalW;
varying vec2 vUvW;
uniform float uHMin;
uniform float uHMax;
uniform vec3 uSunDir;
uniform sampler2D uMap;
uniform float uHasMap;
uniform float uMixMap;
vec3 hramp(float t){
  t = clamp(t, 0.0, 1.0);
  vec3 c1 = vec3(0.075, 0.216, 0.224);
  vec3 c2 = vec3(0.180, 0.420, 0.380);
  vec3 c3 = vec3(0.550, 0.560, 0.420);
  vec3 c4 = vec3(0.720, 0.480, 0.300);
  vec3 c5 = vec3(0.920, 0.900, 0.860);
  if(t < 0.25) return mix(c1, c2, t / 0.25);
  if(t < 0.50) return mix(c2, c3, (t - 0.25) / 0.25);
  if(t < 0.75) return mix(c3, c4, (t - 0.50) / 0.25);
  return mix(c4, c5, (t - 0.75) / 0.25);
}
void main(){
  float t = (vH - uHMin) / max(uHMax - uHMin, 1e-6);
  vec3 tint = hramp(t);
  float shade = 0.55 + 0.45 * max(dot(normalize(vNormalW), normalize(uSunDir)), 0.0);
  vec3 col = tint;
  if (uHasMap > 0.5) {
    vec3 tex = texture2D(uMap, vUvW).rgb;
    col = mix(tint, tex, uMixMap);
  }
  gl_FragColor = vec4(col * shade, 1.0);
}`

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
}

function easeInOutBack(t, s = 1.1) {
  const c = s
  const inOut = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
  return inOut + (t > 0.55 ? Math.sin(((t - 0.55) * Math.PI) / 0.45) * 0.035 * c : 0)
}

const SCENE_THEMES = {
  dark: {
    bg: '#11150f',
    fog: '#1c1f18',
    skyTop: '#0d1411',
    skyHorizon: '#3f4c39',
    sunTint: '#e8a33d',
    hemiSky: '#31473c',
    hemiGround: '#0b120e',
    sunIntensity: 2.5,
    exposure: 1.12,
    fillColor: '#8fa89b',
    fillIntensity: 0.35,
  },
  light: {
    bg: '#dfe3d2',
    fog: '#cfd8cc',
    skyTop: '#a9c4cf',
    skyHorizon: '#ece5cf',
    sunTint: '#f7dfae',
    hemiSky: '#e6ecdd',
    hemiGround: '#97906f',
    sunIntensity: 1.55,
    exposure: 0.95,
    fillColor: '#fff4e0',
    fillIntensity: 0.3,
  },
}

function demoRegions(mask, width, height, threshold = 0.5, cell = 12, limit = 32) {
  if (!mask || !width || !height) return []
  const cols = Math.ceil(width / cell)
  const rows = Math.ceil(height / cell)
  const active = new Uint8Array(cols * rows)
  for (let gy = 0; gy < rows; gy++) {
    for (let gx = 0; gx < cols; gx++) {
      let total = 0
      let count = 0
      for (let y = gy * cell; y < Math.min(height, (gy + 1) * cell); y++) {
        for (let x = gx * cell; x < Math.min(width, (gx + 1) * cell); x++) {
          total += Number(mask[y * width + x]) || 0
          count++
        }
      }
      active[gy * cols + gx] = total / Math.max(1, count) >= threshold ? 1 : 0
    }
  }
  const visited = new Uint8Array(active.length)
  const regions = []
  for (let start = 0; start < active.length && regions.length < limit; start++) {
    if (!active[start] || visited[start]) continue
    const queue = [start]
    visited[start] = 1
    let minX = start % cols, maxX = minX, minY = Math.floor(start / cols), maxY = minY
    while (queue.length) {
      const index = queue.pop()
      const gx = index % cols
      const gy = Math.floor(index / cols)
      minX = Math.min(minX, gx); maxX = Math.max(maxX, gx)
      minY = Math.min(minY, gy); maxY = Math.max(maxY, gy)
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = gx + dx, ny = gy + dy
        if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue
        const next = ny * cols + nx
        if (active[next] && !visited[next]) {
          visited[next] = 1
          queue.push(next)
        }
      }
    }
    const x0 = minX * cell, y0 = minY * cell
    const x1 = Math.min(width - 1, (maxX + 1) * cell)
    const y1 = Math.min(height - 1, (maxY + 1) * cell)
    if ((x1 - x0) * (y1 - y0) >= cell * cell * 2) {
      regions.push({ polygon: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], area: (x1 - x0) * (y1 - y0) })
    }
  }
  return regions.sort((a, b) => b.area - a.area)
}

function demoEnvironmentFromMasks(dsm, minH) {
  const masks = dsm?.masks
  const width = Number(dsm?.width) || 0
  const height = Number(dsm?.height) || 0
  if (!masks || !width || !height) return null
  const water = demoRegions(masks.water, width, height, 0.52, 12, 12)
  const vegetation = demoRegions(masks.vegetation, width, height, 0.50, 14, 24)
  const buildings = demoRegions(masks.building, width, height, 0.58, 10, 80)
  return {
    roads: [],
    water: water.map((item, i) => ({ id: -1000 - i, polygon: item.polygon, ground_elevation: minH, class: 'water', source: 'demo-mask' })),
    landcover: vegetation.map((item, i) => ({ id: -2000 - i, polygon: item.polygon, ground_elevation: minH, class: 'forest', source: 'demo-mask' })),
    trees: vegetation.slice(0, 80).map((item, i) => {
      const xs = item.polygon.map((p) => p[0])
      const ys = item.polygon.map((p) => p[1])
      return {
        id: -3000 - i,
        point: [(Math.min(...xs) + Math.max(...xs)) * 0.5, (Math.min(...ys) + Math.max(...ys)) * 0.5],
        ground_elevation: minH,
        height: 5 + (i % 5),
        canopy_radius: 1.8 + (i % 3) * 0.45,
        source: 'demo-mask',
      }
    }),
    demoBuildings: buildings.map((item, i) => ({
      id: -4000 - i,
      polygon: item.polygon,
      ground_elevation: minH,
      height: Math.max(3, Math.min(18, 3 + Math.sqrt(item.area) * 0.12)),
      roof_elevation: minH + Math.max(3, Math.min(18, 3 + Math.sqrt(item.area) * 0.12)),
      area_px: item.area,
      source: 'demo-mask',
    })),
  }
}

export class TerrainViewer {
  constructor(container, opts = {}) {
    this.container = container
    this.tier = opts.tier || 'high'
    this.reducedMotion = !!opts.reducedMotion
    this.caps = opts.caps
    this.revealPlayedFor = opts.revealedForJobId ?? null
    this.onProbe = opts.onProbe || (() => {})
    this.onMeasure = opts.onMeasure || (() => {})
    this.onPick = opts.onPick || (() => {})
    this.onLoad = opts.onLoad || (() => {})
    this.onRevealProgress = opts.onRevealProgress || (() => {})
    this.onFpsDrop = opts.onFpsDrop || (() => {})
    this.onFlyLock = opts.onFlyLock || (() => {})
    this.onTourChange = opts.onTourChange || (() => {})
    this.tour = null

    this.renderer = new THREE.WebGLRenderer({ antialias: this.tier !== 'low', preserveDrawingBuffer: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this.caps.dprCap))
    this.renderer.shadowMap.enabled = this.caps.shadows
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.05
    this.renderer.domElement.className = 'main'
    container.appendChild(this.renderer.domElement)
    this.dom = this.renderer.domElement

    this.scene = new THREE.Scene()
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 50000)

    this.orbit = new OrbitControls(this.camera, this.dom)
    this.orbit.enableDamping = true
    this.orbit.dampingFactor = 0.08
    this.orbit.maxPolarAngle = Math.PI * 0.495
    this.orbit.enabled = false

    this.fly = new FlyControls(this.camera, this.dom)
    this.fly.enabled = false
    this.fly.onLockChange = (locked) => this.onFlyLock(locked)

    this.mesh = null
    this.structGroup = null
    this.layerGroups = {}
    this.groundPlane = null
    this.baseHeights = null
    this.baseValid = null
    this.gridW = 0
    this.gridH = 0
    this.zScale = 1.0
    this.mode = 'orbit'
    this.transitioning = false
    this.measurePts = []
    this.measureGroup = null
    this.probeRay = new THREE.Raycaster()
    this.probePending = false
    this.lastProbeAt = 0
    this.downInfo = null
    this.frameMonitor = null
    this.disposed = false
    this.contextLost = false

    // Active owned resources for safe disposal.
    this._ownedTextures = new Set()
    this._ownedMaterials = new Set()
    this._ownedGeometries = new Set()
    this._ownedRenderTargets = new Set()
    this._zAnim = 0
    this._revealRaf = 0
    this._loadToken = 0

    this._buildEnvironment()

    this.ro = new ResizeObserver(() => this._resize())
    this.ro.observe(container)
    this._resize()

    this.dom.addEventListener('pointermove', (e) => this._onPointerMove(e))
    this.dom.addEventListener('pointerdown', (e) => (this.downInfo = { x: e.clientX, y: e.clientY, t: performance.now() }))
    this.dom.addEventListener('pointerup', (e) => this._onPointerUp(e))
    this._onContextLost = (e) => {
      e.preventDefault()
      this.contextLost = true
      if (this._revealRaf) {
        cancelAnimationFrame(this._revealRaf)
        this._revealRaf = 0
      }
      if (this._zAnim) {
        cancelAnimationFrame(this._zAnim)
        this._zAnim = 0
      }
      if (this.frameMonitor) clearInterval(this.frameMonitor)
      this.frameMonitor = null
      this.transitioning = false
    }
    this._onContextRestored = () => {
      this.contextLost = false
      try {
        this.renderer.resetState?.()
      } catch {
        /* renderer may not be ready yet */
      }
      // Re-upload the terrain geometry so the displaced Y values are visible.
      if (this.mesh) {
        this.mesh.geometry.attributes.position.needsUpdate = true
        this.mesh.geometry.computeVertexNormals?.()
      }
    }
    this.dom.addEventListener('webglcontextlost', this._onContextLost, false)
    this.dom.addEventListener('webglcontextrestored', this._onContextRestored, false)
    window.addEventListener('keydown', this._onKey)

    this.clock = new THREE.Clock()
    this._loop()
  }

  _buildEnvironment() {
    const caps = this.caps
    this.themeName = document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'
    const th = SCENE_THEMES[this.themeName]
    this.scene.background = new THREE.Color(th.bg)
    this.renderer.toneMappingExposure = th.exposure
    // Gradient sky dome on every tier — one static draw call, so even the
    // lowest tier sits in atmosphere instead of a black void. Fog stays
    // tier-gated for fill-rate reasons.
    if (caps.fog === 'full') {
      this.scene.fog = new THREE.FogExp2(new THREE.Color(th.fog), 0.00022)
    } else if (caps.fog === 'flat') {
      this.scene.fog = new THREE.FogExp2(new THREE.Color(th.fog), 0.00028)
    } else {
      this.scene.fog = null
    }
    {
      const sky = new THREE.Mesh(
        new THREE.SphereGeometry(18000, 32, 20),
        new THREE.ShaderMaterial({
          vertexShader: SKY_VERT,
          fragmentShader: SKY_FRAG,
          side: THREE.BackSide,
          depthWrite: false,
          uniforms: {
            uTop: { value: new THREE.Color(th.skyTop) },
            uHorizon: { value: new THREE.Color(th.skyHorizon) },
            uSunDir: { value: new THREE.Vector3(-0.55, 0.28, 0.4).normalize() },
            uSunTint: { value: new THREE.Color(th.sunTint) },
          },
        }),
      )
      this.skyMesh = sky
      this._ownedGeometries.add(sky.geometry)
      this._ownedMaterials.add(sky.material)
      this.scene.add(sky)
    }

    this.sun = new THREE.DirectionalLight('#ffe7c4', th.sunIntensity)
    this.scene.add(this.sun)
    this.hemi = new THREE.HemisphereLight(th.hemiSky, th.hemiGround, 0.5)
    this.scene.add(this.hemi)
    // Faint cool counter-fill so shadow sides hold depth instead of
    // crushing to black. Kept dim so the terrain stays the focus.
    this.fill = new THREE.DirectionalLight(th.fillColor, th.fillIntensity)
    this.fill.position.set(140, 50, 160)
    this.scene.add(this.fill)
    if (!this.caps.shadows) {
      this.hemi.intensity = 0.75
      this.sun.intensity = 1.6
    }
    this._applyHillshade(true)
  }

  /** Phase 11: grid→world mapping for hazard overlays. */
  getHeightMapping() {
    const m = this.gridMapping || {}
    return { minH: m.minH ?? 0, maxH: m.maxH ?? 1, gw: m.gw ?? 0, gh: m.gh ?? 0, zScale: this.zScale ?? 1, sx: m.sx ?? 1, sy: m.sy ?? 1 }
  }

  setTheme(name) {
    const th = SCENE_THEMES[name === 'light' ? 'light' : 'dark']
    this.themeName = name === 'light' ? 'light' : 'dark'
    this.scene.background = new THREE.Color(th.bg)
    if (this.scene.fog) this.scene.fog.color = new THREE.Color(th.fog)
    if (this.groundPlane) this.groundPlane.material.color = new THREE.Color(th.fog)
    if (this.skyMesh) {
      const u = this.skyMesh.material.uniforms
      u.uTop.value = new THREE.Color(th.skyTop)
      u.uHorizon.value = new THREE.Color(th.skyHorizon)
      u.uSunTint.value = new THREE.Color(th.sunTint)
    }
    this.renderer.toneMappingExposure = th.exposure
    // Lights were previously left at their boot-theme values on toggle, so a
    // mid-session theme switch kept the old lighting. Re-apply themed lights.
    if (this.sun) {
      this.sun.intensity = th.sunIntensity
      if (this.hemi) {
        this.hemi.color = new THREE.Color(th.hemiSky)
        this.hemi.groundColor = new THREE.Color(th.hemiGround)
        this.hemi.intensity = 0.5
      }
      if (this.caps && this.caps.shadows) {
        this._applyHillshade(this.hillshadeOn ?? true)
      } else {
        if (this.hemi) this.hemi.intensity = 0.75
        this.sun.intensity = 1.6
      }
      if (this.fill) {
        this.fill.color = new THREE.Color(th.fillColor)
        this.fill.intensity = th.fillIntensity
      }
    }
  }

  _applyHillshade(on) {
    this.hillshadeOn = !!on
    if (this.caps.shadows) {
      this.sun.castShadow = on
      this.hemi.intensity = on ? 0.5 : 0.95
      this.sun.intensity = on ? 2.5 : 1.3
    }
  }

  _resize() {
    if (this.disposed) return
    const w = this.container.clientWidth || 1
    const h = this.container.clientHeight || 1
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    if (this.skyMesh) this.skyMesh.position.copy(this.camera.position)
  }

  load(dsm) {
    if (this.disposed || this.contextLost) return
    // Tokenize the load so a stale finish (e.g. mid-disposal) cannot mutate
    // the viewer after a new DSM has been kicked off.
    this._loadToken += 1
    const myToken = this._loadToken
    this.stopTour(true)

    const { heights, width: gw, height: gh, textureSrc, textureCanvas, normalSrc } = dsm
    const jId = dsm.id || dsm.jobId || null
    this.gridW = gw
    this.gridH = gh

    // Dispose the previous terrain before registering resources for the new
    // one. Disposing after creation also disposes the new material and leaves
    // the mesh without a usable material.
    this._disposePriorResources()

    // Validate and sanitize height data. Cells that are NaN, non-finite, or
    // the -9999 nodata marker are INVALID (masked), never zero elevation:
    // they are filled with the valid-grid minimum for interpolation safety
    // and later cut out of the mesh index as true holes (see below).
    const NODATA = -9999
    const rawValid = new Uint8Array(heights.length)
    const sanitizedHeights = new Float32Array(heights.length)
    let minH = Infinity, maxH = -Infinity
    let invalidCells = 0
    for (let i = 0; i < heights.length; i++) {
      const v = heights[i]
      const ok = Number.isFinite(v) && v !== NODATA
      rawValid[i] = ok ? 1 : 0
      if (!ok) {
        invalidCells++
        continue
      }
      sanitizedHeights[i] = v
      if (v < minH) minH = v
      if (v > maxH) maxH = v
    }
    if (minH === Infinity) {
      minH = 0
      maxH = 1
    }
    for (let i = 0; i < heights.length; i++) {
      if (!rawValid[i]) sanitizedHeights[i] = minH
    }
    if (myToken !== this._loadToken) return
    // Forensic P0-2: world units for X/Z. Backend reports ground sampling
    // distance in result.metadata.pixel_size_m; when known the plane is
    // metric (X/Z meters like Y), otherwise honest pixel units.
    let sx = 1
    let sy = 1
    const ps = dsm.pixelSizeM
    if (Array.isArray(ps) && ps.length >= 2) {
      sx = Number(ps[0])
      sy = Number(ps[1])
    } else if (typeof ps === 'number') {
      sx = sy = ps
    }
    if (!Number.isFinite(sx) || sx <= 0) sx = 1
    if (!Number.isFinite(sy) || sy <= 0) sy = 1
    this.sx = sx
    this.sy = sy
    // Phase 11: expose the grid→world mapping for hazard overlays (additive).
    this.gridMapping = { minH, maxH, gw, gh, sx, sy }

    const segCap = this.caps.meshSegments
    const k = Math.min(1, segCap / Math.max(gw, gh))
    const segX = Math.max(32, Math.round((gw - 1) * k))
    const segY = Math.max(32, Math.round((gh - 1) * k))

    const geo = new THREE.PlaneGeometry((gw - 1) * sx, (gh - 1) * sy, segX, segY)
    geo.rotateX(-Math.PI / 2)
    const pos = geo.attributes.position
    const nVerts = pos.count
    this.vertexCount = nVerts

    const sampler = this._makeSampler(sanitizedHeights, gw, gh)
    this.samplerFn = sampler
    this.R = Math.max((gw - 1) * sx, (gh - 1) * sy) * 0.95 || Math.max(gw, gh) * 0.95
    if (this.caps.fog === 'full') this.scene.fog.density = 0.85 / (this.R * 4.6)

    const spanH = (maxH - minH) || 1.0

    this.baseHeights = new Float32Array(nVerts)
    this.baseValid = new Uint8Array(nVerts)
    let minBH = Infinity, maxBH = -Infinity
    let invalidVerts = 0
    for (let i = 0; i < pos.count; i++) {
      const wx = pos.getX(i)
      const wz = pos.getZ(i)
      const hv = sampler(wx / sx + (gw - 1) / 2, wz / sy + (gh - 1) / 2)
      // Nearest source cell decides validity: bilinear blends are for valid
      // terrain only; masked cells must not leak zero-pits into the surface.
      const gx = Math.min(gw - 1, Math.max(0, Math.round(wx / sx + (gw - 1) / 2)))
      const gy = Math.min(gh - 1, Math.max(0, Math.round(wz / sy + (gh - 1) / 2)))
      const valid = rawValid[gy * gw + gx] === 1
      this.baseValid[i] = valid ? 1 : 0
      if (!valid) {
        invalidVerts++
        this.baseHeights[i] = 0
        pos.setY(i, 0)
        continue
      }
      const hvSafe = isNaN(hv) || !isFinite(hv) ? minH : hv
      const normalizedH = hvSafe - minH
      this.baseHeights[i] = normalizedH
      if (normalizedH < minBH) minBH = normalizedH
      if (normalizedH > maxBH) maxBH = normalizedH
      pos.setY(i, 0)
    }
    if (myToken !== this._loadToken) return
    // Cut masked cells out of the mesh: drop every triangle touching an
    // invalid vertex, leaving a true hole instead of a zero-elevation pit.
    // All other systems (reveal, z-scale, probe, measure) are untouched —
    // invalid vertices keep finite baseHeights so their math stays safe.
    if (invalidVerts > 0 && geo.getIndex()) {
      const srcIdx = geo.getIndex().array
      const kept = new (srcIdx instanceof Uint32Array ? Uint32Array : Uint16Array)(srcIdx.length)
      let n = 0
      for (let t = 0; t < srcIdx.length; t += 3) {
        if (this.baseValid[srcIdx[t]] && this.baseValid[srcIdx[t + 1]] && this.baseValid[srcIdx[t + 2]]) {
          kept[n++] = srcIdx[t]
          kept[n++] = srcIdx[t + 1]
          kept[n++] = srcIdx[t + 2]
        }
      }
      geo.setIndex(new THREE.BufferAttribute(kept.slice(0, n), 1))
    }
    geo.computeVertexNormals()

    // Prioritize high-resolution user image texture
    let tex
    if (textureSrc) {
      tex = new THREE.TextureLoader().load(textureSrc)
    } else if (textureCanvas && textureCanvas instanceof HTMLCanvasElement) {
      tex = new THREE.CanvasTexture(textureCanvas)
    } else {
      tex = new THREE.Texture()
    }
    tex.colorSpace = THREE.SRGBColorSpace
    tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy?.() || 4
    this._ownedTextures.add(tex)

    let normalMap = null
    if (normalSrc) {
      normalMap = new THREE.TextureLoader().load(normalSrc)
      normalMap.anisotropy = this.renderer.capabilities.getMaxAnisotropy?.() || 4
      this._ownedTextures.add(normalMap)
    }

    this.matTexture = new THREE.MeshStandardMaterial({
      map: tex,
      normalMap: normalMap,
      normalScale: new THREE.Vector2(1.8, 1.8),
      roughness: 0.68,
      metalness: 0.04,
    })
    this.matSlope = new THREE.ShaderMaterial({
      vertexShader: SLOPE_VERT,
      fragmentShader: SLOPE_FRAG,
      uniforms: { uSunDir: { value: this.sun.position.clone().normalize() } },
    })
    const elevUniforms = () => ({
      uHMin: { value: 0 },
      uHMax: { value: Math.max(maxBH - minBH, 1e-6) },
      uSunDir: { value: this.sun.position.clone().normalize() },
      uMap: { value: tex },
      uHasMap: { value: 1 },
      uMixMap: { value: 0 },
    })
    this.matElev = new THREE.ShaderMaterial({
      vertexShader: ELEV_VERT,
      fragmentShader: ELEV_FRAG,
      uniforms: elevUniforms(),
    })
    this.matHybrid = new THREE.ShaderMaterial({
      vertexShader: ELEV_VERT,
      fragmentShader: ELEV_FRAG,
      uniforms: { ...elevUniforms(), uMixMap: { value: 0.55 } },
    })
    this.matWire = new THREE.MeshBasicMaterial({ wireframe: true, color: '#4fd9c4', transparent: true, opacity: 0.35 })
    this._ownedMaterials.add(this.matTexture)
    this._ownedMaterials.add(this.matSlope)
    this._ownedMaterials.add(this.matElev)
    this._ownedMaterials.add(this.matHybrid)
    this._ownedMaterials.add(this.matWire)
    this.colorMode = this.colorMode || 'rgb'
    this.slopeOn = !!this.slopeOn
    this._applyMaterial()
    // World-size + unit honesty for the HUD scale bar (display-only).
    const _meta = dsm.metadata || null
    const _metric = _meta ? !!_meta.is_metric : !!dsm.crs
    this.worldInfo = {
      sizeX: (gw - 1) * sx,
      sizeZ: (gh - 1) * sy,
      unit: _meta && dsm.offline ? 'demo' : _metric ? 'm' : 'rel',
      peakH: maxBH,
    }

    this.mesh = new THREE.Mesh(geo, this.matCurrent)
    this._ownedGeometries.add(geo)
    this.mesh.receiveShadow = this.caps.shadows
    this.mesh.castShadow = this.caps.shadows
    this.scene.add(this.mesh)

    // Believable context from real scene data only: a grounding plane that
    // fills the void beneath the tile edges, plus massing blocks extruded
    // from measured building footprints (never invented).
    this._buildGroundPlane(spanH)
    this._buildStructures(dsm, { minH, sx, sy, gw, gh, tex })

    this.sun.position.set(-(gw - 1) * sx * 0.55, (gh - 1) * sy * 0.6, (gh - 1) * sy * 0.62)
    if (this.caps.shadows) {
      this.sun.castShadow = true
      const cam = this.sun.shadow.camera
      const ext = Math.max((gw - 1) * sx, (gh - 1) * sy) * 0.75
      cam.left = -ext
      cam.right = ext
      cam.top = ext
      cam.bottom = -ext
      cam.near = Math.max(1, this.R * 0.001)
      cam.far = Math.max((gw - 1) * sx, (gh - 1) * sy) * 3
      this.sun.shadow.mapSize.set(2048, 2048)
      this.sun.shadow.bias = -0.0001
      this.sun.shadow.normalBias = 0.02
      cam.updateProjectionMatrix()
    }

    const peakH = Math.max(maxBH, this.structurePeakH || 0) * Math.max(0.0001, this.zScale)
    const spanScaled = spanH * Math.max(0.0001, this.zScale)
    const halfR = Math.max((gw - 1) * sx, (gh - 1) * sy) * 0.5
    const boundRadius = Math.sqrt(halfR * halfR + spanScaled * spanScaled * 0.25)
    const fov = THREE.MathUtils.degToRad(this.camera.fov)
    const aspect = (this.container.clientWidth || 1) / Math.max(1, this.container.clientHeight || 1)
    const tanV = Math.tan(fov * 0.5)
    const tanH = tanV * aspect
    const tanFov = Math.min(tanV, tanH)
    const distFromTarget = (boundRadius * 0.99) / Math.max(0.001, tanFov)
    const heightClearance = peakH * 0.5
    this.orbitTarget = new THREE.Vector3(0, peakH * 0.42, 0)
    this.orbitHome = new THREE.Vector3().setFromSphericalCoords(
      distFromTarget,
      Math.PI * 0.32,
      -Math.PI * 0.36,
    ).add(this.orbitTarget)
    if (this.orbitHome.y < peakH + heightClearance) {
      this.orbitHome.y = peakH + heightClearance
    }
    this.orbit.target.copy(this.orbitTarget)

    this.measurePts = []
    this._updateMeasureLine()
    this.onMeasure(null)

    this.applyZScaleImmediate(this.zScale, 1)
    this.camera.position.copy(this.orbitHome)
    this.camera.lookAt(this.orbitTarget)
    this.orbit.update()

    if (myToken !== this._loadToken) return
    const firstReveal = !this.revealedForJob(jId)
    if (firstReveal && !this.reducedMotion && this.caps.reveal !== 'instant') {
      this._runReveal(jId)
    } else {
      this.revealedForJob(jId)
      this.onRevealProgress(this.vertexCount, true)
      this._settle()
    }
    this._startFrameMonitor()
  }

  _buildGroundPlane(spanH) {
    this._removeContext()
    const span = Number.isFinite(spanH) && spanH > 0 ? spanH : 1
    const geo = new THREE.CircleGeometry(this.R * 2.4, 48)
    geo.rotateX(-Math.PI / 2)
    // Tinted to the theme fog color (not hard black): at grazing angles the
    // disc used to read as a black wall behind the tile. Still catches
    // shadows and fills the void beneath tile edges.
    const fogColor = SCENE_THEMES[this.themeName]?.fog ?? '#141714'
    const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(fogColor), roughness: 1.0, metalness: 0.0 })
    const plane = new THREE.Mesh(geo, mat)
    plane.position.y = -Math.max(1.5, span * 0.03)
    plane.receiveShadow = this.caps.shadows
    this._ownedGeometries.add(geo)
    this._ownedMaterials.add(mat)
    this.groundPlane = plane
    this.scene.add(plane)
  }

  _buildStructures(dsm, { minH, sx, sy, gw, gh, tex }) {
    this._removeStructures()
    const mappedEnvironment = dsm?.environment || {}
    const hasMappedEnvironment = ['roads', 'water', 'landcover', 'trees'].some((key) => (mappedEnvironment[key] || []).length)
    const demoEnvironment = !hasMappedEnvironment && dsm?.offline ? demoEnvironmentFromMasks(dsm, minH) : null
    const environment = demoEnvironment || mappedEnvironment
    const list = [
      ...(Array.isArray(dsm?.buildings) ? dsm.buildings : []),
      ...(demoEnvironment?.demoBuildings || []),
    ]
    const ox = ((gw - 1) * sx) / 2
    const oz = ((gh - 1) * sy) / 2
    const groups = {
      BUILDINGS: new THREE.Group(),
      WATER: new THREE.Group(),
      VEGETATION: new THREE.Group(),
      TREES: new THREE.Group(),
      ROADS: new THREE.Group(),
    }
    this.structurePeakH = 0

    const shapeFor = (polygon) => {
      if (!Array.isArray(polygon) || polygon.length < 3) return null
      const shape = new THREE.Shape()
      let count = 0
      polygon.forEach(([col, row]) => {
        const wx = Number(col) * sx - ox
        const wz = -(Number(row) * sy - oz)
        if (!Number.isFinite(wx) || !Number.isFinite(wz)) return
        if (!count) shape.moveTo(wx, wz)
        else shape.lineTo(wx, wz)
        count += 1
      })
      return count >= 3 ? shape : null
    }

    const yFor = (feature, offset = 0.05) => {
      const ground = Number(feature?.ground_elevation)
      return (Number.isFinite(ground) ? ground : minH) - minH + offset
    }

    const addPolygon = (group, feature, material, offset = 0.05) => {
      const shape = shapeFor(feature?.polygon)
      if (!shape) return null
      let geometry
      try {
        geometry = new THREE.ShapeGeometry(shape)
        geometry.rotateX(-Math.PI / 2)
      } catch {
        return null
      }
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.y = yFor(feature, offset)
      mesh.userData.featureId = feature?.id ?? null
      group.add(mesh)
      this._ownedGeometries.add(geometry)
      return mesh
    }

    // Only cleaned footprints with usable elevations are rendered. The
    // backend is responsible for confidence/relief gates; the frontend must
    // never invent a structure from an absent record.
    const usable = list.filter((b) => {
      if (!Array.isArray(b.polygon) || b.polygon.length < 3) return false
      const roof = Number(b.roof_elevation)
      const gnd = Number(b.ground_elevation)
      const h = Number(b.height)
      return Number.isFinite(roof) || (Number.isFinite(gnd) && Number.isFinite(h) && h > 0)
    })
    usable.sort((a, b) => (Number(b.area_m2) || Number(b.area_px) || 0) - (Number(a.area_m2) || Number(a.area_px) || 0))
    let added = 0
    for (const b of usable) {
      if (added >= 150) break
      const roof = Number(b.roof_elevation)
      const gnd = Number(b.ground_elevation)
      const h = Number(b.height)
      const ground = (Number.isFinite(gnd) ? gnd : minH) - minH
      const top = (Number.isFinite(roof) ? roof : gnd + Math.max(h, 0.5)) - minH
      if (!(top > ground)) continue
      const shape = shapeFor(b.polygon)
      if (!shape) continue
      let geo
      try {
        geo = new THREE.ExtrudeGeometry(shape, { depth: Math.max(top - ground, 0.5), bevelEnabled: false })
      } catch {
        continue
      }
      geo.rotateX(-Math.PI / 2)
      const heightRatio = Math.min(1, Math.max(0, (top - ground) / 30))
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color().setHSL(0.10, 0.16, 0.49 + heightRatio * 0.12),
        roughness: 0.78,
        metalness: 0.02,
      })
      if (tex) {
        mat.map = tex
        mat.color.set(0xffffff)
        const pos = geo.attributes.position
        const uv = geo.attributes.uv
        if (pos && uv) {
          for (let vi = 0; vi < pos.count; vi++) {
            const u = (pos.getX(vi) + ox) / ((gw - 1) * sx)
            const v = (pos.getZ(vi) + oz) / ((gh - 1) * sy)
            uv.setXY(vi, u, v)
          }
          uv.needsUpdate = true
        }
      }
      this._ownedMaterials.add(mat)
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.y = ground
      mesh.userData.buildingId = b.id ?? null
      mesh.castShadow = this.caps.shadows
      mesh.receiveShadow = this.caps.shadows
      this._ownedGeometries.add(geo)
      groups.BUILDINGS.add(mesh)
      this.structurePeakH = Math.max(this.structurePeakH, top)
      added++
    }

    const waterMat = new THREE.MeshStandardMaterial({
      color: '#278fc0', roughness: 0.16, metalness: 0.2,
      transparent: true, opacity: 0.82, depthWrite: false,
    })
    const vegetationMat = new THREE.MeshStandardMaterial({
      color: '#3f9b55', roughness: 0.94, metalness: 0,
      transparent: true, opacity: 0.48, depthWrite: false,
    })
    this._ownedMaterials.add(waterMat)
    this._ownedMaterials.add(vegetationMat)
    for (const feature of environment.water || []) addPolygon(groups.WATER, feature, waterMat, 0.18)
    for (const feature of environment.landcover || []) addPolygon(groups.VEGETATION, feature, vegetationMat, 0.10)

    const roadMat = new THREE.MeshStandardMaterial({ color: '#68665f', roughness: 0.98, metalness: 0 })
    this._ownedMaterials.add(roadMat)
    for (const road of environment.roads || []) {
      const path = road?.path || []
      if (path.length < 2) continue
      const width = Math.max(0.8, Number(road.width_m) || 4)
      const vertices = []
      for (let i = 0; i < path.length; i++) {
        const prev = path[Math.max(0, i - 1)]
        const next = path[Math.min(path.length - 1, i + 1)]
        const dx = (Number(next[0]) - Number(prev[0])) * sx
        const dz = (Number(next[1]) - Number(prev[1])) * sy
        const len = Math.hypot(dx, dz) || 1
        const nx = -dz / len * width * 0.5
        const nz = dx / len * width * 0.5
        const x = Number(path[i][0]) * sx - ox
        const z = -(Number(path[i][1]) * sy - oz)
        vertices.push(x - nx, yFor(road, 0.08), z - nz, x + nx, yFor(road, 0.08), z + nz)
      }
      const indices = []
      for (let i = 0; i < path.length - 1; i++) {
        const a = i * 2, b = i * 2 + 1, c = i * 2 + 2, d = i * 2 + 3
        indices.push(a, c, b, b, c, d)
      }
      const geometry = new THREE.BufferGeometry()
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3))
      geometry.setIndex(indices)
      geometry.computeVertexNormals()
      const mesh = new THREE.Mesh(geometry, roadMat)
      groups.ROADS.add(mesh)
      this._ownedGeometries.add(geometry)
    }

    for (const tree of environment.trees || []) {
      const point = tree?.point || []
      if (point.length !== 2) continue
      const x = Number(point[0]) * sx - ox
      const z = -(Number(point[1]) * sy - oz)
      const height = Math.max(2.5, Number(tree.height) || 8)
      const radius = Math.max(1.0, Number(tree.canopy_radius) || height * 0.3)
      const treeGroup = new THREE.Group()
      const trunkMaterial = new THREE.MeshStandardMaterial({ color: '#6c4d32', roughness: 1 })
      const canopyMaterial = new THREE.MeshStandardMaterial({ color: '#2f8b4b', roughness: 0.92 })
      this._ownedMaterials.add(trunkMaterial)
      this._ownedMaterials.add(canopyMaterial)
      const trunk = new THREE.Mesh(new THREE.CylinderGeometry(Math.max(0.16, radius * 0.12), Math.max(0.22, radius * 0.15), height * 0.45, 7), trunkMaterial)
      const canopy = new THREE.Mesh(new THREE.ConeGeometry(radius, height * 0.72, 8), canopyMaterial)
      trunk.position.y = height * 0.225
      canopy.position.y = height * 0.64
      treeGroup.add(trunk, canopy)
      treeGroup.position.set(x, yFor(tree, 0), z)
      treeGroup.userData.featureId = tree.id ?? null
      groups.TREES.add(treeGroup)
      this._ownedGeometries.add(trunk.geometry)
      this._ownedGeometries.add(canopy.geometry)
      this.structurePeakH = Math.max(this.structurePeakH, treeGroup.position.y + height)
    }

    for (const [name, group] of Object.entries(groups)) {
      if (!group.children.length) continue
      group.scale.y = this.zScale
      this.layerGroups[name] = group
      this.scene.add(group)
    }
    this.structGroup = this.layerGroups.BUILDINGS || null
    this.onLoad({ ok: true, groups: this.layerNames(), structureCount: added })
  }

  _removeStructures() {
    for (const group of Object.values(this.layerGroups || {})) this.scene.remove(group)
    this.layerGroups = {}
    this.structGroup = null
  }

  layerNames() {
    const terrain = this.mesh ? [{ id: 'TERRAIN', label: 'Terrain', count: 1 }] : []
    return terrain.concat(Object.entries(this.layerGroups).map(([id, group]) => ({
      id,
      label: ({ BUILDINGS: 'Buildings', WATER: 'Water', VEGETATION: 'Vegetation', TREES: 'Trees', ROADS: 'Roads' })[id] || id,
      count: group.children.length,
    })))
  }

  setLayerVisible(group, visible) {
    if (group === 'TERRAIN' && this.mesh) this.mesh.visible = !!visible
    else if (this.layerGroups[group]) this.layerGroups[group].visible = !!visible
  }

  _removeContext() {
    this._removeStructures()
    if (this.groundPlane) {
      this.scene.remove(this.groundPlane)
      this.groundPlane = null
    }
  }

  _disposePriorResources() {
    if (this.mesh) {
      this.scene.remove(this.mesh)
    }
    this._removeContext()
    // Free anything that was registered as owned by THIS viewer instance.
    // We do not free the newly-created matTexture / its maps because they
    // were just registered into _ownedTextures/_ownedMaterials and are
    // currently in use.
    for (const t of this._ownedTextures) {
      try { t.dispose?.() } catch { /* shared */ }
    }
    this._ownedTextures.clear()

    for (const m of this._ownedMaterials) {
      try { m.dispose?.() } catch { /* shared */ }
    }
    this._ownedMaterials.clear()

    for (const g of this._ownedGeometries) {
      try { g.dispose?.() } catch { /* shared */ }
    }
    this._ownedGeometries.clear()

    for (const rt of this._ownedRenderTargets) {
      try { rt.dispose?.() } catch { /* shared */ }
    }
    this._ownedRenderTargets.clear()

    this.matTexture = null
    this.matSlope = null
    this.matElev = null
    this.matHybrid = null
    this.matWire = null
    this.matCurrent = null
    this.structurePeakH = 0
    this.mesh = null
  }

  _generateNormalMap(heights, gw, gh) {
    const normalMapData = new Uint8Array(gw * gh * 4)
    const scale = 1.0 / Math.max(gw, gh) * 100.0

    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        const i = y * gw + x

        const xl = x > 0 ? x - 1 : 0
        const xr = x < gw - 1 ? x + 1 : gw - 1
        const yb = y > 0 ? y - 1 : 0
        const yt = y < gh - 1 ? y + 1 : gh - 1

        const hl = heights[y * gw + xl]
        const hr = heights[y * gw + xr]
        const hb = heights[yb * gw + x]
        const ht = heights[yt * gw + x]

        const dzdx = (hr - hl) * 0.5 * scale
        const dzdy = (ht - hb) * 0.5 * scale

        const nx = -dzdx
        const ny = -dzdy
        const nz = 1.0

        const len = Math.sqrt(nx * nx + ny * ny + nz * nz)
        const invLen = 1.0 / (len || 1)

        const idx = (y * gw + x) * 4
        normalMapData[idx] = Math.round((nx * invLen * 0.5 + 0.5) * 255)
        normalMapData[idx + 1] = Math.round((ny * invLen * 0.5 + 0.5) * 255)
        normalMapData[idx + 2] = Math.round((nz * invLen * 0.5 + 0.5) * 255)
        normalMapData[idx + 3] = 255
      }
    }

    const normalMap = new THREE.DataTexture(normalMapData, gw, gh, THREE.RGBAFormat, THREE.UnsignedByteType)
    normalMap.wrapS = THREE.ClampToEdgeWrapping
    normalMap.wrapT = THREE.ClampToEdgeWrapping
    normalMap.anisotropy = this.renderer.capabilities.getMaxAnisotropy?.() || 4
    normalMap.needsUpdate = true
    return normalMap
  }

  revealedForJob(jobId) {
    if (!jobId) return false
    if (this.revealPlayedFor === jobId) return true
    this.revealPlayedFor = jobId
    return false
  }

  _makeSampler(heights, gw, gh) {
    return (u, v) => {
      const x = Math.min(gw - 1, Math.max(0, u))
      const y = Math.min(gh - 1, Math.max(0, v))
      const x0 = Math.floor(x)
      const y0 = Math.floor(y)
      const x1 = Math.min(gw - 1, x0 + 1)
      const y1 = Math.min(gh - 1, y0 + 1)
      const fx = x - x0
      const fy = y - y0
      const h00 = heights[y0 * gw + x0]
      const h10 = heights[y0 * gw + x1]
      const h01 = heights[y1 * gw + x0]
      const h11 = heights[y1 * gw + x1]

      const h00v = isNaN(h00) ? 0 : h00
      const h10v = isNaN(h10) ? 0 : h10
      const h01v = isNaN(h01) ? 0 : h01
      const h11v = isNaN(h11) ? 0 : h11

      const result = (h00v * (1 - fx) + h10v * fx) * (1 - fy) + (h01v * (1 - fx) + h11v * fx) * fy
      return isNaN(result) ? 0 : result
    }
  }

  _runReveal(jobId) {
    this.revealPlayedFor = jobId
    this.transitioning = true
    this.orbit.enabled = false
    this.fly.enabled = false

    const dur = this.caps.reveal === 'full' ? 3200 : 1800
    const camMove = this.caps.reveal === 'full'
    const startPos = camMove
      ? this.orbitHome.clone().add(new THREE.Vector3(0, this.R * 0.25, 0))
      : this.orbitHome.clone()
    const endPos = this.orbitHome.clone()
    // Terrain emerges from darkness: exposure ramps with reveal progress.
    const baseExposure = this.renderer.toneMappingExposure
    this.renderer.toneMappingExposure = baseExposure * 0.04
    const t0 = performance.now()

    const finalize = (reason) => {
      if (this.disposed) return
      this.applyZScaleImmediate(this.zScale, 1, true)
      this.renderer.toneMappingExposure = baseExposure
      this.camera.position.copy(this.orbitHome)
      this.camera.lookAt(this.orbitTarget)
      this.transitioning = false
      this._settle()
      this.onRevealProgress(this.vertexCount, true)
      this._revealRaf = 0
    }

    // frame is declared in the outer scope so the closure can mutate it.
    let frame = 0
    const tick = () => {
      if (this.disposed) {
        this._revealRaf = 0
        return
      }
      if (this.contextLost) {
        // Pause until restore; do not finalize yet.
        this._revealRaf = requestAnimationFrame(tick)
        return
      }
      if (document.hidden) {
        finalize('tab hidden')
        return
      }
      const raw = Math.min(1, (performance.now() - t0) / dur)
      const k = easeInOutCubic(raw)
      const disp = easeInOutBack(raw)
      frame++
      const needNormals = frame % 6 === 0 || raw >= 1
      this.applyZScaleImmediate(this.zScale, disp, needNormals)
      this.renderer.toneMappingExposure = baseExposure * (0.04 + 0.96 * k)
      if (camMove) {
        this.camera.position.lerpVectors(startPos, endPos, k)
        this.camera.lookAt(this.orbitTarget)
      }
      this.onRevealProgress(Math.floor(k * this.vertexCount), false)
      if (raw < 1) {
        this._revealRaf = requestAnimationFrame(tick)
      } else {
        finalize('done')
      }
    }
    try {
      if (camMove) {
        this.camera.position.copy(startPos)
        this.camera.lookAt(this.orbitTarget)
      }
      this._revealRaf = requestAnimationFrame(tick)
    } catch (exc) {
      console.error('[TerrainViewer] reveal start failed:', exc)
      finalize('exception')
    }
  }

  _settle() {
    if (this.mode === 'orbit') {
      this.orbit.enabled = true
      this.orbit.update()
    }
  }

  applyZScaleImmediate(scale, dispFactor = 1, recomputeNormals = true) {
    if (this.disposed || !this.mesh) return
    const pos = this.mesh.geometry.attributes.position
    let minY = Infinity, maxY = -Infinity
    for (let i = 0; i < pos.count; i++) {
      const h = this.baseHeights[i]
      const y = (isNaN(h) || !isFinite(h) ? 0 : h) * scale * dispFactor
      pos.setY(i, y)
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
    pos.needsUpdate = true
    // Building massing lives in the same normalized-height space, so it
    // rides exaggeration and the reveal displacement exactly.
    for (const group of Object.values(this.layerGroups || {})) group.scale.y = scale * dispFactor
    if (recomputeNormals) this.mesh.geometry.computeVertexNormals()
  }

  setZScale(target, animate = true) {
    if (this.disposed) return
    const from = this.zScale
    const to = target
    this.zScale = to
    if (!animate || this.reducedMotion || this.tier === 'low') {
      this.applyZScaleImmediate(to, 1)
      return
    }
    const t0 = performance.now()
    const dur = 400
    if (this._zAnim) cancelAnimationFrame(this._zAnim)
    const stepFn = () => {
      if (this.disposed) {
        this._zAnim = 0
        return
      }
      if (this.contextLost) {
        this._zAnim = 0
        return
      }
      const raw = Math.min(1, (performance.now() - t0) / dur)
      const k = easeInOutCubic(raw)
      this.zScale = from + (to - from) * k
      this.applyZScaleImmediate(this.zScale, 1)
      if (raw < 1) this._zAnim = requestAnimationFrame(stepFn)
      else this._zAnim = 0
    }
    stepFn()
  }

  setMode(next) {
    if (this.disposed) return
    if (!this.mesh || next === this.mode || this.transitioning) return
    this.stopTour(true)
    const ms = this.caps.modeTransitionMs
    this.fly.releaseLock()
    if (ms <= 0 || this.reducedMotion) {
      this.mode = next
      this._enableMode(next)
      return
    }
    this.transitioning = true
    this.orbit.enabled = false
    this.fly.enabled = false
    const startPos = this.camera.position.clone()
    const startQ = this.camera.quaternion.clone()
    let endPos
    let endLook
    if (next === 'fly') {
      const dir = new THREE.Vector3().subVectors(this.orbit.target, startPos).normalize()
      endPos = this.orbit.target.clone().addScaledVector(dir, -this.R * 0.4)
      endPos.y = this.terrainHeightAt(endPos.x, endPos.z) + this.R * 0.15 + 25
      endLook = this.orbit.target.clone()
    } else {
      endPos = this.orbitHome.clone()
      endLook = this.orbitTarget.clone()
    }
    const t0 = performance.now()
    const stepFn = () => {
      if (this.disposed) return
      const raw = Math.min(1, (performance.now() - t0) / ms)
      const k = easeInOutCubic(raw)
      this.camera.position.lerpVectors(startPos, endPos, k)
      const m = new THREE.Matrix4().lookAt(this.camera.position, endLook, new THREE.Vector3(0, 1, 0))
      const q = new THREE.Quaternion().setFromRotationMatrix(m)
      this.camera.quaternion.slerpQuaternions(startQ, q, k)
      if (raw < 1) requestAnimationFrame(stepFn)
      else {
        this.mode = next
        this.transitioning = false
        this._enableMode(next)
      }
    }
    stepFn()
  }

  _enableMode(next) {
    if (next === 'orbit') {
      this.orbit.target.copy(this.orbitTarget)
      this.orbit.enabled = true
      this.orbit.update()
    } else {
      this.fly.speed = this.R * 0.045
      this.fly.enabled = true
      this.fly.velocity.set(0, 0, 0)
      this.fly.setFromCamera()
    }
  }

  terrainHeightAt(wx, wz) {
    if (this.disposed || !this.samplerFn) return 0
    const sx = this.sx || 1
    const sy = this.sy || 1
    const u = wx / sx + (this.gridW - 1) / 2
    const v = wz / sy + (this.gridH - 1) / 2
    return this.samplerFn(u, v)
  }

  _pick(e) {
    if (this.disposed || !this.mesh) return null
    const rect = this.dom.getBoundingClientRect()
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    )
    this.probeRay.setFromCamera(ndc, this.camera)
    const hits = this.probeRay.intersectObject(this.mesh)
    return hits.length ? hits[0] : null
  }

  _pickBuilding(e) {
    const group = this.layerGroups?.BUILDINGS
    if (this.disposed || !group) return null
    const rect = this.dom.getBoundingClientRect()
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    )
    this.probeRay.setFromCamera(ndc, this.camera)
    const hits = this.probeRay.intersectObject(group, true)
    return hits.find((hit) => hit.object.userData?.buildingId != null) || null
  }

  _onPointerMove(e) {
    if (this.disposed) return
    if (this.transitioning || !this.mesh) return
    if (this.mode === 'fly' && this.fly.locked) return
    const now = performance.now()
    if (now - this.lastProbeAt < 30) return
    this.lastProbeAt = now
    if (!this.probePending) {
      this.probePending = true
      requestAnimationFrame(() => {
        this.probePending = false
        if (this.disposed) return
        const hit = this._pick(e)
        if (hit) {
          this.onProbe({
            x: e.clientX,
            y: e.clientY,
            height: hit.point.y,
            ground: this.terrainHeightAt(hit.point.x, hit.point.z),
          })
        } else {
          this.onProbe(null)
        }
      })
    }
  }

  _onPointerUp(e) {
    if (this.disposed) return
    if (this.transitioning || !this.mesh) return
    if (this.mode === 'fly' && !this.fly.locked) {
      try {
        this.fly.requestLock()
      } catch {
        /* pointer lock requires gesture */
      }
      return
    }
    if (this.downInfo) {
      const moved = Math.hypot(e.clientX - this.downInfo.x, e.clientY - this.downInfo.y)
      const dt = performance.now() - this.downInfo.t
      this.downInfo = null
      if (moved > 5 || dt > 400) return
    }
    const buildingHit = this._pickBuilding(e)
    if (buildingHit) {
      this.onPick(buildingHit.object.userData.buildingId)
      return
    }
    this.onPick(null)
    const hit = this._pick(e)
    if (!hit) return
    if (this.mode === 'fly' && this.fly.locked) return
    this.measurePts.push({ x: hit.point.x, y: hit.point.y, z: hit.point.z })
    if (this.measurePts.length > 2) this.measurePts = [this.measurePts[2]]
    this._updateMeasureLine()
    if (this.measurePts.length === 2) {
      const [a, b] = this.measurePts
      const dh = Math.abs(a.y - b.y)
      const dist = Math.hypot(b.x - a.x, b.z - a.z)
      const midScreen = this._worldToScreen({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 + 4, z: (a.z + b.z) / 2 })
      this.onMeasure(midScreen ? { ...midScreen, dh, dist } : null)
    } else {
      this.onMeasure(null)
    }
  }

  _worldToScreen(p) {
    const v = new THREE.Vector3(p.x, p.y, p.z).project(this.camera)
    if (v.z > 1) return null
    const rect = this.dom.getBoundingClientRect()
    return {
      x: ((v.x + 1) / 2) * rect.width,
      y: ((1 - v.y) / 2) * rect.height,
    }
  }

  _updateMeasureLine() {
    if (this.measureGroup) {
      this.scene.remove(this.measureGroup)
      this.measureGroup.traverse((o) => o.geometry?.dispose())
      this.measureGroup = null
    }
    if (!this.measurePts.length) return
    this.measureGroup = new THREE.Group()
    for (const p of this.measurePts) {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(Math.max(2, this.R * 0.004), 10, 10),
        new THREE.MeshBasicMaterial({ color: '#4fd9c4' }),
      )
      m.position.set(p.x, p.y + this.R * 0.006, p.z)
      this.measureGroup.add(m)
    }
    if (this.measurePts.length === 2) {
      const [a, b] = this.measurePts
      const lift = this.R * 0.006
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(a.x, a.y + lift, a.z),
        new THREE.Vector3(b.x, b.y + lift, b.z),
      ])
      this.measureGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: '#4fd9c4' })))
    }
    this.scene.add(this.measureGroup)
  }

  clearMeasure() {
    this.measurePts = []
    this._updateMeasureLine()
    this.onMeasure(null)
  }

  _onKey = (e) => {
    if (e.key === 'Escape') this.clearMeasure()
  }

  _applyMaterial() {
    const base = { rgb: this.matTexture, elevation: this.matElev, hybrid: this.matHybrid, wire: this.matWire }[this.colorMode] || this.matTexture
    this.matCurrent = this.slopeOn && this.matSlope ? this.matSlope : base
    if (!this.mesh) return
    this.mesh.material = this.matCurrent
  }

  setColorMode(mode) {
    if (!['rgb', 'elevation', 'hybrid', 'wire'].includes(mode)) return
    this.colorMode = mode
    this._applyMaterial()
  }

  setSlopeView(on) {
    this.slopeOn = !!on
    if (!this.mesh) return
    this._applyMaterial()
  }

  setHillshade(on) {
    this._applyHillshade(on)
  }

  resetView() {
    if (this.disposed) return
    if (this.mode === 'fly') {
      this.setMode('orbit')
    } else {
      this.orbit.target.copy(this.orbitTarget)
      this.camera.position.copy(this.orbitHome)
      this.orbit.update()
    }
  }

  getOrientation() {
    if (this.mode === 'fly') {
      return { heading: this.fly.getHeading(), speed: this.fly.getSpeed() }
    }
    const az = this.orbit.getAzimuthalAngle()
    let deg = (-az * 180) / Math.PI + 180
    deg = ((deg % 360) + 360) % 360
    return { heading: deg, speed: 0 }
  }

  getScaleBar() {    // Display-only scale estimate from the current orbit framing.
    if (this.disposed || !this.mesh || !this.orbitTarget) return null
    const hPx = this.container.clientHeight || 1
    const dist = this.camera.position.distanceTo(this.orbit.target)
    const worldPerPx = (2 * dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) * 0.5)) / Math.max(1, hPx)
    const targetPx = 110
    const raw = worldPerPx * targetPx
    const val = niceScale(raw)
    const unit = this.worldInfo?.unit || 'rel'
    const label = formatScale(val, unit)
    return { label, px: Math.max(36, (val / Math.max(worldPerPx, 1e-9))) }
  }

  getCamRange() {
    // Honest camera-to-focus distance in world units (m when metric).
    if (this.disposed || !this.mesh || !this.orbitTarget) return null
    const dist = this.camera.position.distanceTo(this.orbit.target)
    const unit = this.worldInfo?.unit || 'rel'
    return { dist, label: formatScale(dist, unit) }
  }

  flyTo(pos, target, dur = 900) {
    if (this.disposed || !this.mesh) return
    if (this.reducedMotion || dur <= 0) {
      this.camera.position.copy(pos)
      this.orbit.target.copy(target)
      this.camera.lookAt(target)
      this.orbit.update()
      return
    }
    const p0 = this.camera.position.clone()
    const t0 = this.orbit.target.clone()
    const start = performance.now()
    const step = (now) => {
      if (this.disposed) return
      const k = Math.min(1, (now - start) / dur)
      const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2
      this.camera.position.lerpVectors(p0, pos, e)
      this.orbit.target.lerpVectors(t0, target, e)
      this.camera.lookAt(this.orbit.target)
      this.orbit.update()
      if (k < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }

  setPreset(name) {
    if (this.disposed || !this.mesh || !this.orbitTarget) return
    if (this.mode === 'fly') return
    const T = this.orbitTarget.clone()
    const d = Math.max(this.R * 1.1, 10)
    const h = Math.max(this.R * 0.9, 10)
    const P = {
      top: [new THREE.Vector3(T.x, T.y + d * 1.35, T.z + 0.001), T],
      north: [new THREE.Vector3(T.x, h * 0.55, T.z - d), T],
      south: [new THREE.Vector3(T.x, h * 0.55, T.z + d), T],
      east: [new THREE.Vector3(T.x + d, h * 0.55, T.z), T],
      west: [new THREE.Vector3(T.x - d, h * 0.55, T.z), T],
      iso: [this.orbitHome.clone(), T],
    }[name]
    if (!P) return
    this.stopTour()
    this.flyTo(P[0], P[1])
  }

  // ---- auto flythrough tour (display motion; data untouched) ----
  startTour(speed = 1) {
    if (this.disposed || !this.mesh || this.mode === 'fly') return
    this.stopTour(true)
    this.tour = { angle: Math.atan2(this.camera.position.x - this.orbitTarget.x, this.camera.position.z - this.orbitTarget.z), speed }
    this.orbit.enabled = false
    this.onTourChange?.(true)
  }

  pauseTour() {
    if (this.tour) {
      this.tour.paused = true
      this.onTourChange?.(true)
    }
  }

  resumeTour() {
    if (this.tour) {
      this.tour.paused = false
      this.onTourChange?.(true)
    }
  }

  stopTour(silent) {
    if (!this.tour) return
    this.tour = null
    if (this.mode === 'orbit' && !this.transitioning) {
      this.orbit.enabled = true
      this.orbit.update()
    }
    if (!silent) this.onTourChange?.(false)
  }

  setTourSpeed(speed) {
    if (this.tour) this.tour.speed = speed
  }

  _tourTick(dt) {
    const t = this.tour
    if (!t || t.paused || this.disposed || !this.mesh) return
    t.angle += dt * 0.14 * (t.speed || 1)
    const r = Math.max(this.R * 0.62, 20)
    const x = this.orbitTarget.x + Math.sin(t.angle) * r
    const z = this.orbitTarget.z + Math.cos(t.angle) * r
    const ground = this.terrainHeightAt(x, z)
    const y = Math.max(ground + Math.max(this.R * 0.16, 14), this.orbitTarget.y + 8)
    this.camera.position.set(x, y, z)
    this.camera.lookAt(this.orbitTarget)
  }

  screenshot() {
    this.renderer.render(this.scene, this.camera)
    return this.dom.toDataURL('image/png')
  }

  _startFrameMonitor() {
    if (this.frameMonitor) {
      clearInterval(this.frameMonitor)
      this.frameMonitor = null
    }
    if (this.tier === 'low') return
    let frames = 0
    let acc = 0
    let last = performance.now()
    const int = setInterval(() => {
      const now = performance.now()
      acc += now - last
      last = now
      frames++
      if (frames >= 300) {
        clearInterval(int)
      }
    }, 16)
    setTimeout(() => {
      clearInterval(int)
      if (this.disposed) return
      const elapsed = acc
      const n = frames
      if (n < 90) return
      const avg = elapsed / n
      if (this.transitioning) return
      if (this.mode !== 'orbit' && this.mode !== 'fly') return
      // Raise the threshold so a brief stall doesn't trigger demotion. Also
      // we do not demote during the first 8s of a new DSM because tile
      // upload / model warm-up can spike frame times temporarily.
      if (avg > 60) this.onFpsDrop(avg)
    }, 8200)
  }

  _loop() {
    if (this.disposed) return
    if (this.contextLost) {
      requestAnimationFrame(() => this._loop())
      return
    }
    requestAnimationFrame(() => this._loop())
    const dt = Math.min(this.clock.getDelta(), 0.1)
    if (this.tour) this._tourTick(dt)
    else if (this.mode === 'orbit' && this.orbit.enabled) this.orbit.update()
    if (this.mode === 'fly' && this.fly.enabled) this.fly.update(dt)
    if (this.skyMesh) this.skyMesh.position.copy(this.camera.position)
    this.renderer.render(this.scene, this.camera)
  }

  dispose() {
    this.disposed = true
    this.stopTour(true)
    if (this._revealRaf) {
      cancelAnimationFrame(this._revealRaf)
      this._revealRaf = 0
    }
    if (this._zAnim) {
      cancelAnimationFrame(this._zAnim)
      this._zAnim = 0
    }
    if (this.ro) {
      this.ro.disconnect()
      this.ro = null
    }
    if (this._onContextLost) this.dom.removeEventListener('webglcontextlost', this._onContextLost)
    if (this._onContextRestored) this.dom.removeEventListener('webglcontextrestored', this._onContextRestored)
    window.removeEventListener('keydown', this._onKey)
    if (this.frameMonitor) clearInterval(this.frameMonitor)
    this.frameMonitor = null
    this.fly.dispose()
    this.orbit.dispose()
    this._disposePriorResources()
    if (this.skyMesh && this.skyMesh.material) this.skyMesh.material.dispose()
    this.skyMesh = null
    this.renderer.dispose()
    this.dom.remove()
  }
}

function statsMax(arr) {
  let mx = -Infinity
  for (let i = 0; i < arr.length; i += 7) if (arr[i] > mx) mx = arr[i]
  return mx
}
