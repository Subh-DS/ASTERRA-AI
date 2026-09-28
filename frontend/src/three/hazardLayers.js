// Phase 11: hazard overlay layers for TerrainViewer (additive — the base
// scene, camera controls and reconstruction are never modified).
import * as THREE from 'three'

const WATER_COLORS = [
  [0.62, 0.85, 0.95], // 0–0.5 m
  [0.35, 0.70, 0.90], // 0.5–1 m
  [0.20, 0.52, 0.82], // 1–2 m
  [0.12, 0.36, 0.70], // 2–5 m
  [0.07, 0.24, 0.55], // >5 m
]
const SOIL = [0.45, 0.33, 0.22]
const SCAR = [0.36, 0.26, 0.17]
const DEPTH_EDGES = [0.5, 1.0, 2.0, 5.0, Infinity]

function depthColor(d) {
  for (let i = 0; i < DEPTH_EDGES.length; i++) if (d <= DEPTH_EDGES[i]) return WATER_COLORS[i]
  return WATER_COLORS[4]
}

// Deterministic PRNG (mulberry32) so debris looks identical every run.
function rng(seed) {
  let a = seed >>> 0
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export class HazardLayers {
  constructor(viewer) {
    this.v = viewer
    this.group = new THREE.Group()
    this.group.name = 'hazard-overlays'
    viewer.scene.add(this.group)
    this.layers = {} // name -> Object3D
    this.t = 0
    this.playing = false
    this.kind = null
    this._raf = 0
    this._bob = 0
    this.reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
  }

  mapping() {
    const m = this.v.getHeightMapping?.() || { minH: 0, gw: 0, gh: 0, zScale: 1, sx: 1, sy: 1 }
    return { sx: 1, sy: 1, ...m }
  }

  // grid cell (gc, gr) of a stride-s grid over the full HxW terrain grid
  cellXZ(gc, gr, stride, gw, gh) {
    const { sx, sy } = this.mapping()
    return [(gc * stride - (gw - 1) / 2) * sx, (gr * stride - (gh - 1) / 2) * sy]
  }

  yFor(elev) {
    const { minH, zScale } = this.mapping()
    return (elev - minH) * zScale
  }

  _track(name, obj) {
    this.layers[name] = obj
    this.group.add(obj)
    return obj
  }

  _patchMesh(mask, gh2, gw2, stride, gw, gh, yFn, color, opacity) {
    const pos = []
    const col = []
    const idx = []
    let vi = 0
    for (let gr = 0; gr < gh2; gr++) {
      for (let gc = 0; gc < gw2; gc++) {
        if (!mask[gr * gw2 + gc]) continue
        const [x0, z0] = this.cellXZ(gc, gr, stride, gw, gh)
        const [x1, z1] = this.cellXZ(gc + 1, gr + 1, stride, gw, gh)
        const c = Array.isArray(color[0]) ? null : color
        const quad = [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]
        for (const [x, z] of quad) {
          pos.push(x, yFn(gc, gr, x, z), z)
          const cc = c || color[gr * gw2 + gc]
          col.push(cc[0], cc[1], cc[2])
        }
        idx.push(vi, vi + 2, vi + 1, vi, vi + 3, vi + 2)
        vi += 4
      }
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3))
    g.setIndex(idx)
    g.computeVertexNormals()
    const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
      vertexColors: true, transparent: true, opacity, roughness: 0.55, metalness: 0.05,
    }))
    return m
  }

  // ---- coastal ----
  showCoastal(result, heights) {
    this.clear()
    this.kind = 'coastal'
    const g = result.grids
    const { minH, gw, gh } = this.mapping()
    if (!gw) return
    const W2 = g.grid_w
    const H2 = g.grid_h
    const s = g.grid_stride
    const mask = g._maskU8
    const depth = g._depthF32
    const wl = result.statistics.water_level_m
    const y = this.yFor(wl) + Math.max(0.02, (this.mapping().maxH - minH) * 0.002)
    const colors = new Array(W2 * H2)
    for (let i = 0; i < mask.length; i++) colors[i] = mask[i] ? depthColor(depth[i]) : [0, 0, 0]
    const water = this._patchMesh(mask, H2, W2, s, gw, gh, () => y, colors, 0.72)
    water.material.roughness = 0.18
    water.material.metalness = 0.25
    water.renderOrder = 5
    this._track('water', water)
    // shoreline: boundary cells → line segments
    const lp = []
    const at = (gr, gc) => (gr < 0 || gc < 0 || gr >= H2 || gc >= W2) ? 0 : mask[gr * W2 + gc]
    for (let gr = 0; gr < H2; gr++) {
      for (let gc = 0; gc < W2; gc++) {
        if (!at(gr, gc)) continue
        const [x0, z0] = this.cellXZ(gc, gr, s, gw, gh)
        const [x1, z1] = this.cellXZ(gc + 1, gr + 1, s, gw, gh)
        if (!at(gr - 1, gc)) lp.push(x0, y, z0, x1, y, z0)
        if (!at(gr + 1, gc)) lp.push(x0, y, z1, x1, y, z1)
        if (!at(gr, gc - 1)) lp.push(x0, y, z0, x0, y, z1)
        if (!at(gr, gc + 1)) lp.push(x1, y, z0, x1, y, z1)
      }
    }
    const lg = new THREE.BufferGeometry()
    lg.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3))
    const shore = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0xeaf6ff, transparent: true, opacity: 0.85 }))
    this._track('shoreline', shore)
    this._waterBaseY = y
    // Ambient wave motion over the stored base positions (subtle — the
    // flood extent itself always comes from the backend simulation).
    const wp = water.geometry.attributes.position
    this._waterBase = wp.array.slice()
    this.setT(1)
    this._startWaterAmbient()
  }

  // ---- landslide ----
  showLandslide(result, heights) {
    this.clear()
    this.kind = 'landslide'
    const g = result.grids
    const { gw, gh } = this.mapping()
    if (!gw) return
    const W2 = g.grid_w
    const H2 = g.grid_h
    const s = g.grid_stride
    const H = result.terrain_source.grid[0]
    const W = result.terrain_source.grid[1]
    const sample = (r, c) => {
      r = Math.min(H - 1, Math.max(0, Math.round(r)))
      c = Math.min(W - 1, Math.max(0, Math.round(c)))
      return heights[r * W + c]
    }
    const lift = 0.15
    const yPatch = (gc, gr) => this.yFor(sample(gr * s, gc * s)) + lift
    const scar = this._patchMesh(g._sourceU8, H2, W2, s, gw, gh, yPatch, SCAR, 0.92)
    this._track('scar', scar)
    const dep = this._patchMesh(g._depU8, H2, W2, s, gw, gh, yPatch, SOIL, 0.0)
    this._track('deposition', dep)
    // debris mass: source patch cloned, animated along keyframes
    const debris = this._patchMesh(g._sourceU8, H2, W2, s, gw, gh, yPatch, SOIL, 0.95)
    this._track('debris', debris)
    this._debrisBase = debris.geometry.attributes.position.array.slice()
    // particles along the path polyline
    const path = result.polyline_px || []
    const R = rng(11)
    const pn = Math.min(400, Math.max(60, path.length * 8))
    const pp = new Float32Array(pn * 3)
    this._seeds = []
    for (let i = 0; i < pn; i++) {
      const k = path.length ? path[Math.floor(R() * path.length)] : [W / 2, H / 2]
      const ox = (R() - 0.5) * 6
      const oz = (R() - 0.5) * 6
      this._seeds.push({ c: k[0] + ox, r: k[1] + oz, h: R() })
      pp[i * 3] = 0; pp[i * 3 + 1] = -9999; pp[i * 3 + 2] = 0
    }
    const pg = new THREE.BufferGeometry()
    pg.setAttribute('position', new THREE.BufferAttribute(pp, 3))
    const pts = new THREE.Points(pg, new THREE.PointsMaterial({
      color: 0x8a6f4d, size: 1.6, transparent: true, opacity: 0.0, sizeAttenuation: true,
    }))
    this._track('particles', pts)
    this._slide = { result, sample, s, gw, gh, H, W }
    this.setT(0)
  }

  _waterWave(t) {
    const w = this.layers.water
    const sh = this.layers.shoreline
    const base = this._waterBase
    if (!w || !base) return
    const arr = w.geometry.attributes.position.array
    const m = this.mapping()
    const span = Math.max(0.001, (m.maxH ?? 1) - (m.minH ?? 0))
    const amp = Math.max(0.04, span * (m.zScale || 1) * 0.004)
    for (let i = 0; i < arr.length; i += 3) {
      const x = base[i]
      const z = base[i + 2]
      arr[i + 1] =
        base[i + 1] +
        Math.sin(x * 0.08 + t * 1.6) * amp +
        Math.sin(z * 0.06 - t * 1.1) * amp
    }
    w.geometry.attributes.position.needsUpdate = true
    // Foam breathing on the shoreline + faint specular shimmer.
    if (sh) sh.material.opacity = 0.7 + 0.25 * (0.5 + 0.5 * Math.sin(t * 2.2))
    w.material.opacity = Math.min(0.85, Math.max(0.15, w.material.opacity + Math.sin(t * 1.6) * 0.002))
  }

  _startWaterAmbient() {
    this._stopWaterAmbient()
    if (this.reducedMotion || this.disposed) return
    if (!this.layers.water || !this._waterBase) return
    const t0 = performance.now()
    const tick = () => {
      if (this.disposed || this.kind !== 'coastal' || !this.layers.water) {
        this._waterRaf = 0
        return
      }
      if (!document.hidden && this.layers.water.visible !== false) {
        this._waterWave((performance.now() - t0) / 1000)
      }
      this._waterRaf = requestAnimationFrame(tick)
    }
    this._waterRaf = requestAnimationFrame(tick)
  }

  _stopWaterAmbient() {
    if (this._waterRaf) cancelAnimationFrame(this._waterRaf)
    this._waterRaf = 0
  }

  // ---- storm: rain + haze + dimmed light (pure visualization state) ----
  setStorm(on) {
    this.storm = !!on
    const v = this.v
    if (on && !this.layers.rain) {
      const n = 1200
      const range = Math.max(300, (v.R || 200) * 1.4)
      const cx = v.orbitTarget?.x || 0
      const cy = v.orbitTarget?.y || 0
      const cz = v.orbitTarget?.z || 0
      const pos = new Float32Array(n * 3)
      const R = rng(77)
      for (let i = 0; i < n; i++) {
        pos[i * 3] = cx + (R() - 0.5) * range
        pos[i * 3 + 1] = cy + R() * range * 0.7
        pos[i * 3 + 2] = cz + (R() - 0.5) * range
      }
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
      const pts = new THREE.Points(g, new THREE.PointsMaterial({
        color: 0x9fb8c8, size: 1.4, transparent: true, opacity: 0.5, sizeAttenuation: true,
      }))
      this._stormRange = range
      this._track('rain', pts)
    }
    if (this.layers.rain) this.layers.rain.visible = on
    const fog = v.scene.fog
    if (fog) {
      if (on && this._fogBase == null) this._fogBase = fog.density
      if (this._fogBase != null) fog.density = on ? this._fogBase * 2.4 : this._fogBase
    }
    if (on && this._expBase == null) this._expBase = v.renderer.toneMappingExposure
    if (this._expBase != null) {
      v.renderer.toneMappingExposure = on ? this._expBase * 0.8 : this._expBase
    }
    if (v.sun) {
      if (on && this._sunBase == null) this._sunBase = v.sun.intensity
      if (this._sunBase != null) v.sun.intensity = on ? this._sunBase * 0.65 : this._sunBase
    }
    if (on && !this.reducedMotion) this._startStormLoop()
    else this._stopStormLoop()
  }

  _startStormLoop() {
    if (this._stormRaf) return
    let last = performance.now()
    const tick = () => {
      if (this.disposed || !this.storm || !this.layers.rain) {
        this._stormRaf = 0
        return
      }
      if (!document.hidden) {
        const now = performance.now()
        const dt = Math.min(0.05, (now - last) / 1000)
        last = now
        const rain = this.layers.rain
        const arr = rain.geometry.attributes.position.array
        const top = (this.v.orbitTarget?.y || 0) + this._stormRange * 0.7
        const bottom = (this.v.orbitTarget?.y || 0) - this._stormRange * 0.1
        const fall = this._stormRange * 1.6 * dt
        const drift = this._stormRange * 0.12 * dt
        for (let i = 0; i < arr.length; i += 3) {
          arr[i + 1] -= fall
          arr[i] += drift
          if (arr[i + 1] < bottom) arr[i + 1] = top
        }
        rain.geometry.attributes.position.needsUpdate = true
      }
      this._stormRaf = requestAnimationFrame(tick)
    }
    this._stormRaf = requestAnimationFrame(tick)
  }

  _stopStormLoop() {
    if (this._stormRaf) cancelAnimationFrame(this._stormRaf)
    this._stormRaf = 0
  }

  // ---- earthquake: restrained procedural shake (visualization only —
  // there is no seismic forecast model behind this) ----
  startQuake({ intensity = 0.6, aftershocks = true } = {}) {
    this.stopQuake()
    if (this.reducedMotion) return
    const v = this.v
    if (!v.camera) return
    const R = v.R || 200
    // Peak camera offset stays tiny relative to scene size: readable, never nauseating.
    const peak = Math.max(0.2, R * 0.004) * Math.min(1, Math.max(0.1, intensity))
    const dustN = this.reducedMotion ? 0 : 220
    let dust = null
    if (dustN > 0) {
      try {
        dust = this._makeDust(dustN)
        if (dust) this._track('quakedust', dust)
      } catch { dust = null }
    }
    // Schedule: main shock + two decaying aftershocks.
    const shocks = aftershocks
      ? [{ at: 0, dur: 3.6, mag: 1.0 }, { at: 5.2, dur: 1.8, mag: 0.4 }, { at: 9.0, dur: 1.4, mag: 0.25 }]
      : [{ at: 0, dur: 3.6, mag: 1.0 }]
    this._quake = {
      peak, shocks, t0: performance.now(), last: new THREE.Vector3(),
      meshBase: v.mesh ? v.mesh.position.clone() : null,
      structBase: v.structGroup ? v.structGroup.position.clone() : null,
      dust,
    }
    const tick = () => {
      const q = this._quake
      if (!q || this.disposed) {
        this._quakeRaf = 0
        return
      }
      const t = (performance.now() - q.t0) / 1000
      // Compose as a delta so user orbiting mid-quake is never overridden.
      v.camera.position.sub(q.last)
      if (v.mesh && q.meshBase) v.mesh.position.sub(q.lastMesh || new THREE.Vector3())
      if (v.structGroup && q.structBase) v.structGroup.position.sub(q.lastStruct || new THREE.Vector3())
      q.last.set(0, 0, 0)
      q.lastMesh = new THREE.Vector3()
      q.lastStruct = new THREE.Vector3()
      let alive = false
      for (const s of q.shocks) {
        const st = t - s.at
        if (st < 0 || st > s.dur + 1.2) continue
        alive = true
        if (st <= s.dur) {
          const env = Math.exp(-2.2 * (st / s.dur)) * s.mag * q.peak
          const f = 9 + 3 * Math.sin(st * 1.7)
          q.last.set(
            env * Math.sin(2 * Math.PI * f * st),
            env * 0.45 * Math.sin(2 * Math.PI * (f * 1.31) * st + 1.3),
            env * Math.sin(2 * Math.PI * (f * 0.77) * st + 2.1),
          )
          if (v.mesh && q.meshBase) q.lastMesh.copy(q.last).multiplyScalar(0.35)
          if (v.structGroup && q.structBase) q.lastStruct.copy(q.last).multiplyScalar(1.4)
        }
      }
      v.camera.position.add(q.last)
      if (v.mesh && q.meshBase) v.mesh.position.add(q.lastMesh)
      if (v.structGroup && q.structBase) v.structGroup.position.add(q.lastStruct)
      if (q.dust) this._updateQuakeDust(q, t)
      if (!alive && t > q.shocks[q.shocks.length - 1].at + q.shocks[q.shocks.length - 1].dur + 1.2) {
        this.stopQuake()
        return
      }
      this._quakeRaf = requestAnimationFrame(tick)
    }
    this._quakeRaf = requestAnimationFrame(tick)
  }

  _makeDust(n) {
    const v = this.v
    const m = this.mapping()
    if (!m.gw) return null
    const R = rng(913)
    const pos = new Float32Array(n * 3)
    const seeds = []
    for (let i = 0; i < n; i++) {
      const gx = R() * (m.gw - 1)
      const gz = R() * (m.gh - 1)
      const x = (gx - (m.gw - 1) / 2) * m.sx
      const z = (gz - (m.gh - 1) / 2) * m.sy
      seeds.push({ x, z, h: R(), r: 2 + R() * 6 })
      pos[i * 3] = x
      pos[i * 3 + 1] = -9999
      pos[i * 3 + 2] = z
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    const pts = new THREE.Points(g, new THREE.PointsMaterial({
      color: 0xb8a888, size: 2.2, transparent: true, opacity: 0.0, sizeAttenuation: true,
    }))
    pts.userData.seeds = seeds
    return pts
  }

  _updateQuakeDust(q, t) {
    const pts = q.dust
    if (!pts) return
    const active = t < 11
    pts.material.opacity = active ? 0.55 : 0
    pts.visible = active
    if (!active) return
    const arr = pts.geometry.attributes.position.array
    const seeds = pts.userData.seeds
    for (let i = 0; i < seeds.length; i++) {
      const sd = seeds[i]
      const cyc = ((t * 0.5 + sd.h) % 1 + 1) % 1
      arr[i * 3] = sd.x + Math.sin((t + sd.h * 9) * 2.1) * 1.5
      arr[i * 3 + 1] = this.yFor(this.mapping().minH) + cyc * sd.r + 1
      arr[i * 3 + 2] = sd.z + Math.cos((t + sd.h * 7) * 1.7) * 1.5
    }
    pts.geometry.attributes.position.needsUpdate = true
  }

  stopQuake() {
    const q = this._quake
    if (this._quakeRaf) cancelAnimationFrame(this._quakeRaf)
    this._quakeRaf = 0
    if (q) {
      try {
        const v = this.v
        v.camera.position.sub(q.last)
        if (v.mesh && q.meshBase) {
          v.mesh.position.sub(q.lastMesh || new THREE.Vector3())
        }
        if (v.structGroup && q.structBase) {
          v.structGroup.position.sub(q.lastStruct || new THREE.Vector3())
        }
        if (q.dust) {
          this.group.remove(q.dust)
          q.dust.geometry?.dispose?.()
          q.dust.material?.dispose?.()
          delete this.layers.quakedust
        }
      } catch { /* restore best-effort */ }
    }
    this._quake = null
  }

  // ---- evacuation: A* routes over the real DSM (planning visualization) ----
  showEvacuation(data) {
    for (const k of ['evac-a', 'evac-b', 'evac-zones', 'evac-start']) {
      const o = this.layers[k]
      if (o) {
        this.group.remove(o)
        o.geometry?.dispose?.()
        if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((mm) => mm.dispose?.())
        delete this.layers[k]
      }
    }
    if (!data) return
    const mkLine = (pts, color) => {
      if (!pts || pts.length < 2) return null
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.Float32BufferAttribute(pts.flatMap((p) => [p.x, p.y, p.z]), 3))
      return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }))
    }
    // NOTE: routes are baked at the current exaggeration (like the hazard
    // overlays); the XZ corridor — the planning-relevant part — is exact
    // regardless. Recalculate after changing vertical exaggeration.
    const a = data.primary ? mkLine(data.primary.pts, 0x4fd9c4) : null
    if (a) {
      a.renderOrder = 6
      this._track('evac-a', a)
    }
    const b = data.alternative ? mkLine(data.alternative.pts, 0xe8a33d) : null
    if (b) {
      b.renderOrder = 6
      this._track('evac-b', b)
    }
    if (Array.isArray(data.zones) && data.zones.length) {
      const zg = new THREE.Group()
      for (const z of data.zones) {
        const disc = new THREE.Mesh(
          new THREE.CircleGeometry(Math.max(z.r, 1), 28),
          new THREE.MeshBasicMaterial({ color: 0x4fd9c4, transparent: true, opacity: 0.35, side: THREE.DoubleSide }),
        )
        disc.rotation.x = -Math.PI / 2
        disc.position.set(z.x, z.y, z.z)
        zg.add(disc)
      }
      this._track('evac-zones', zg)
    }
    if (data.start) {
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(1.6, 12, 10),
        new THREE.MeshBasicMaterial({ color: 0xe8a33d }),
      )
      s.position.set(data.start.x, data.start.y + 1, data.start.z)
      this._track('evac-start', s)
    }
  }

  gridXY(r, c) {
    const { gw, gh, sx, sy } = this.mapping()
    return [(c - (gw - 1) / 2) * sx, (r - (gh - 1) / 2) * sy]
  }

  setT(t) {
    this.t = Math.min(1, Math.max(0, t))
    if (this.kind === 'coastal') {
      const w = this.layers.water
      const sh = this.layers.shoreline
      if (w) {
        // fill preview: water rises from low terrain to the scenario level
        const span = Math.max(0.001, this._waterBaseY - this.yFor(this.mapping().minH))
        w.position.y = -span * (1 - this.t)
        w.material.opacity = 0.15 + 0.57 * this.t
        w.visible = this.t > 0.001
      }
      if (sh) {
        sh.position.y = w ? w.position.y : 0
        sh.visible = !!w?.visible
      }
    } else if (this.kind === 'landslide' && this._slide) {
      const { result, sample } = this._slide
      const kf = result.keyframes
      const f = this.t * (kf.length - 1)
      const i0 = Math.min(kf.length - 2, Math.floor(f))
      const fr = f - i0
      const A = kf[i0]
      const B = kf[i0 + 1]
      const r = A.row + (B.row - A.row) * fr
      const c = A.col + (B.col - A.col) * fr
      const src = result.statistics.source_center_rc
      const [x0] = this.gridXY(src[0], src[1])
      const [, z0] = this.gridXY(src[0], src[1])
      const [x1] = this.gridXY(r, c)
      const [, z1] = this.gridXY(r, c)
      const deb = this.layers.debris
      if (deb) {
        const sc = 1 + ((result.keyframes.at(-1)?.radius_px || 10) / 10 - 1) * this.t * 0.4
        deb.position.set(x1 - x0 * sc, 0, z1 - z0 * sc)
        deb.scale.set(sc, 1, sc)
        deb.visible = this.t < 0.999
      }
      const dep = this.layers.deposition
      if (dep) {
        dep.material.opacity = 0.9 * Math.max(0, (this.t - 0.45) / 0.55)
        dep.visible = this.t > 0.45
      }
      const scar = this.layers.scar
      if (scar) scar.material.opacity = 0.35 + 0.57 * Math.min(1, this.t * 3)
      const pts = this.layers.particles
      if (pts) {
        const show = this.t > 0.05 && this.t < 0.95
        pts.material.opacity = show ? 0.75 : 0
        if (show) {
          const arr = pts.geometry.attributes.position.array
          for (let i = 0; i < this._seeds.length; i++) {
            const sd = this._seeds[i]
            const pr = sd.r + (r - src[0]) * sd.h
            const pc = sd.c + (c - src[1]) * sd.h
            const m = this.mapping()
            const [x, z] = [(pc - (m.gw - 1) / 2) * m.sx, (pr - (m.gh - 1) / 2) * m.sy]
            arr[i * 3] = x
            arr[i * 3 + 1] = this.yFor(sample(pr, pc)) + 1 + sd.h * 3
            arr[i * 3 + 2] = z
          }
          pts.geometry.attributes.position.needsUpdate = true
        }
      }
    }
  }

  play() {
    if (this.playing || this.reducedMotion) { if (this.reducedMotion) this.setT(1); return }
    this.playing = true
    this._t0 = performance.now()
    this._from = this.t
    const dur = this.kind === 'coastal' ? 3500 : 6000
    const step = (now) => {
      if (!this.playing) return
      const t = this._from + (now - this._t0) / dur
      if (t >= 1) { this.setT(1); this.pause(); return }
      this.setT(t)
      this._raf = requestAnimationFrame(step)
    }
    this._raf = requestAnimationFrame(step)
  }

  pause() {
    this.playing = false
    cancelAnimationFrame(this._raf)
  }

  reset() {
    this.pause()
    this.setT(0)
  }

  setLayerVisible(name, v) {
    if (this.layers[name]) this.layers[name].visible = v
  }

  setGroupVisible(v) {
    this.group.visible = v
  }

  focusOn(x, y, z) {
    const o = this.v.orbit
    if (!o) return
    if (this.reducedMotion) { o.target.set(x, y, z); return }
    const t0 = o.target.clone()
    const t1 = new THREE.Vector3(x, y, z)
    const c0 = this.v.camera.position.clone()
    const dir = c0.clone().sub(t0)
    const c1 = t1.clone().add(dir.multiplyScalar(0.55))
    const start = performance.now()
    const step = (now) => {
      const k = Math.min(1, (now - start) / 900)
      const e = k * k * (3 - 2 * k)
      o.target.lerpVectors(t0, t1, e)
      this.v.camera.position.lerpVectors(c0, c1, e)
      if (k < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }

  // world position of a terrain-grid cell (for camera focus)
  worldOf(r, c, heights, W) {
    const { gw, gh, sx, sy } = this.mapping()
    return [(c - (gw - 1) / 2) * sx, this.yFor(heights[r * W + c]), (r - (gh - 1) / 2) * sy]
  }

  _restoreAtmosphere() {
    const v = this.v
    try {
      if (v.scene.fog && this._fogBase != null) v.scene.fog.density = this._fogBase
      if (this._expBase != null) v.renderer.toneMappingExposure = this._expBase
      if (v.sun && this._sunBase != null) v.sun.intensity = this._sunBase
    } catch { /* viewer gone */ }
    this._fogBase = null
    this._expBase = null
    this._sunBase = null
  }

  clear() {
    this.pause()
    this.stopQuake()
    this._stopWaterAmbient()
    this._stopStormLoop()
    this._restoreAtmosphere()
    this._waterBase = null
    for (const k of Object.keys(this.layers)) {
      const o = this.layers[k]
      this.group.remove(o)
      o.traverse?.((c) => {
        c.geometry?.dispose?.()
        if (c.material && c !== o) (Array.isArray(c.material) ? c.material : [c.material]).forEach((m) => m.dispose?.())
      })
      o.geometry?.dispose?.()
      if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose?.())
    }
    this.layers = {}
    this._slide = null
    this.kind = null
    this.t = 0
  }

  dispose() {
    this.clear()
    this.v.scene.remove(this.group)
  }
}
