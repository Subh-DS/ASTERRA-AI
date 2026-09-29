/**
 * Phase 9: GLB structure viewer — loads model.glb (TERRAIN + BUILDING_* groups)
 * as the canonical structure-aware scene. GLB geometry is Y-up and metric;
 * display exaggeration is applied to the viewer root only.
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

function groupOf(name) {
  if (name === 'TERRAIN') return 'TERRAIN'
  if (name.startsWith('BUILDING_') || name.startsWith('ROOF_BUILDING_') || name.startsWith('SLOPED_ROOF_')) return 'BUILDINGS'
  if (name.startsWith('ROAD')) return 'ROADS'
  if (name.startsWith('VEGETATION') || name.startsWith('LANDCOVER')) return 'VEGETATION'
  if (name.startsWith('WATER')) return 'WATER'
  if (name.startsWith('INFRA')) return 'INFRASTRUCTURE'
  if (name.startsWith('SEMANTIC_WIREFRAME_') || name.startsWith('SEGMENTATION_')) return 'SEGMENTATION'
  return 'OTHER'
}

export const LAYER_LABELS = {
  TERRAIN: 'Terrain',
  BUILDINGS: 'Buildings',
  ROADS: 'Roads',
  VEGETATION: 'Vegetation',
  WATER: 'Water',
  INFRASTRUCTURE: 'Infrastructure',
  SEGMENTATION: 'Segmentation diagnostics',
}

export class StructureViewer {
  constructor(container, opts = {}) {
    this.container = container
    this.onPick = opts.onPick || (() => {})
    this.onLoad = opts.onLoad || (() => {})
    this.disposed = false
    this.displayScale = Number.isFinite(opts.displayScale) ? opts.displayScale : 1.75
    this.legacyZUp = !!opts.legacyZUp
    this.gridW = Number(opts.gridW) || 0
    this.gridH = Number(opts.gridH) || 0
    const pixelSize = Array.isArray(opts.pixelSizeM) ? opts.pixelSizeM : [opts.pixelSizeM, opts.pixelSizeM]
    this.sx = Number.isFinite(Number(pixelSize[0])) && Number(pixelSize[0]) > 0 ? Number(pixelSize[0]) : 1
    this.sy = Number.isFinite(Number(pixelSize[1])) && Number(pixelSize[1]) > 0 ? Number(pixelSize[1]) : 1
    const alignment = opts.sceneAlignment || {}
    const meshGrid = alignment.mesh_grid || {}
    this.meshW = Number(meshGrid.width) || this.gridW
    this.meshH = Number(meshGrid.height) || this.gridH
    const meshResolution = Array.isArray(meshGrid.resolution_m) ? meshGrid.resolution_m : [this.sx, this.sy]
    this.meshSx = Number(meshResolution[0]) > 0 ? Number(meshResolution[0]) : this.sx
    this.meshSy = Number(meshResolution[1]) > 0 ? Number(meshResolution[1]) : this.sy
    this.minH = Number.isFinite(Number(opts.verticalOrigin)) ? Number(opts.verticalOrigin) : 0
    this.maxH = Number.isFinite(Number(opts.maxH)) ? Number(opts.maxH) : this.minH + 1
    this.mode = 'orbit'
    this.tour = null
    this.onTourChange = opts.onTourChange || (() => {})

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.12
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap
    this.renderer.domElement.className = 'main struct'
    container.appendChild(this.renderer.domElement)

    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color('#11150f')
    this.scene.fog = new THREE.Fog('#11150f', 500, 4000)
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100000)
    this.orbit = new OrbitControls(this.camera, this.renderer.domElement)
    this.orbit.enableDamping = true
    this.orbit.dampingFactor = 0.08
    this.orbit.maxPolarAngle = Math.PI * 0.495

    this.scene.add(new THREE.HemisphereLight(0xe6ecdd, 0x1a241e, 1.05))
    this.sun = new THREE.DirectionalLight(0xffe7c4, 2.2)
    this.sun.castShadow = true
    this.sun.shadow.mapSize.set(1024, 1024)
    this.sun.shadow.bias = -0.00035
    this.sun.shadow.normalBias = 0.04
    this.scene.add(this.sun)
    this.scene.add(this.sun.target)
    this.fill = new THREE.DirectionalLight(0x7d9fc0, 0.55)
    this.scene.add(this.fill)

    this.root = null
    this.groups = {}
    this.downInfo = null
    this.ro = new ResizeObserver(() => this._resize())
    this.ro.observe(container)
    this._resize()

    this.renderer.domElement.addEventListener('pointerdown', (e) => {
      this.downInfo = { x: e.clientX, y: e.clientY }
    })
    this.renderer.domElement.addEventListener('pointerup', (e) => this._pick(e))
    this._loop()
  }

  _resize() {
    if (this.disposed) return
    const w = this.container.clientWidth || 1
    const h = this.container.clientHeight || 1
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  load(url, onDone) {
    new GLTFLoader().load(
      url,
      (gltf) => {
        if (this.disposed) return
        if (this.root) {
          this.scene.remove(this.root)
          this.root = null
        }
        this.root = gltf.scene
        // Jobs made before the Y-up export contract used raster Z-up meshes.
        // Keep them viewable while all new GLBs use the canonical axis.
        if (this.legacyZUp) this.root.rotation.x = -Math.PI / 2
        if (this.legacyZUp) {
          this.root.position.set(-((this.gridW - 1) * this.sx) / 2, 0, ((this.gridH - 1) * this.sy) / 2)
        } else {
          // Center the actual exported mesh extent. Source rasters may be
          // downsampled (for example 533x576 -> 474x512), so centering from
          // source dimensions shifts buildings and context layers.
          const localBox = new THREE.Box3().setFromObject(this.root)
          const localCenter = localBox.getCenter(new THREE.Vector3())
          this.root.position.set(-localCenter.x, 0, -localCenter.z)
        }
        this._applyDisplayScale()
        this.groups = {}
        this.root.traverse((o) => {
          if (o.isMesh) {
            const g = groupOf(o.name || '')
            o.userData.group = g
            o.castShadow = g === 'BUILDINGS' || g === 'INFRASTRUCTURE'
            o.receiveShadow = true
            const m = /^(?:SLOPED_ROOF_|ROOF_)?BUILDING_(\d+)_/.exec(o.name || '')
            if (m) o.userData.buildingId = parseInt(m[1], 10)
            const materials = Array.isArray(o.material) ? o.material : [o.material]
            for (const material of materials) {
              if (!material) continue
              if ('roughness' in material) material.roughness = g === 'BUILDINGS' ? 0.82 : 0.94
              if ('metalness' in material) material.metalness = g === 'BUILDINGS' ? 0.03 : 0
              if (!material.map && material.color) {
                if (o.name?.startsWith('ROOF_BUILDING_') || o.name?.startsWith('SLOPED_ROOF_')) {
                  material.color.lerp(new THREE.Color('#d2b98d'), 0.18)
                } else if (g === 'BUILDINGS') {
                  material.color.lerp(new THREE.Color('#d7d0bd'), 0.12)
                }
              }
            }
            this.groups[g] = this.groups[g] || []
            this.groups[g].push(o)
          }
        })
        for (const object of this.groups.SEGMENTATION || []) object.visible = false
        this.scene.add(this.root)
        // Fit camera to measured bounds (never hard-coded).
        const box = new THREE.Box3().setFromObject(this.root)
        const center = box.getCenter(new THREE.Vector3())
        const sphere = box.getBoundingSphere(new THREE.Sphere())
        this.R = sphere.radius * 1.7
        this.camera.near = Math.max(0.05, sphere.radius / 1500)
        this.camera.far = Math.max(2000, sphere.radius * 16)
        this.camera.updateProjectionMatrix()
        this.orbit.minDistance = Math.max(1, sphere.radius * 0.08)
        this.orbit.maxDistance = Math.max(100, sphere.radius * 12)
        const fov = THREE.MathUtils.degToRad(this.camera.fov)
        const dist = (sphere.radius * 1.15) / Math.max(0.05, Math.tan(fov / 2))
        this.orbit.target.copy(center)
        this.camera.position.set(
          center.x + dist * 0.55, center.y + dist * 0.62, center.z + dist * 0.55,
        )
        this.sun.position.set(center.x - sphere.radius, center.y + sphere.radius * 1.4, center.z + sphere.radius)
        this.sun.target.position.copy(center)
        this.fill.position.set(center.x + sphere.radius, center.y + sphere.radius * 0.6, center.z - sphere.radius)
        this._setAtmosphere(sphere.radius)
        this.camera.lookAt(center)
        this.onLoad({ ok: true, groups: this.layerNames(), radius: sphere.radius, displayScale: this.displayScale, meshExtent: { width: this.meshW, height: this.meshH, resolution: [this.meshSx, this.meshSy] } })
        onDone?.(null)
      },
      undefined,
      (err) => {
        this.onLoad({ ok: false, error: String(err?.message || err) })
        onDone?.(err)
      },
    )
  }

  layerNames() {
    return Object.keys(this.groups)
      .filter((g) => g !== 'OTHER')
      .map((g) => ({ id: g, label: LAYER_LABELS[g] || g, count: this.groups[g].length }))
  }

  setLayerVisible(group, visible) {
    for (const o of this.groups[group] || []) o.visible = visible
  }

  setDisplayScale(value) {
    const scale = Number(value)
    if (!Number.isFinite(scale) || scale <= 0) return
    this.displayScale = scale
    if (!this.root) return
    this._applyDisplayScale()
    const box = new THREE.Box3().setFromObject(this.root)
    const center = box.getCenter(new THREE.Vector3())
    const sphere = box.getBoundingSphere(new THREE.Sphere())
    this.R = sphere.radius * 1.7
    this.camera.near = Math.max(0.05, sphere.radius / 1500)
    this.camera.far = Math.max(2000, sphere.radius * 16)
    this.camera.updateProjectionMatrix()
    this.orbit.minDistance = Math.max(1, sphere.radius * 0.08)
    this.orbit.maxDistance = Math.max(100, sphere.radius * 12)
    this._setAtmosphere(sphere.radius)
    this.sun.position.set(center.x - sphere.radius, center.y + sphere.radius * 1.4, center.z + sphere.radius)
    this.sun.target.position.copy(center)
    this.fill.position.set(center.x + sphere.radius, center.y + sphere.radius * 0.6, center.z - sphere.radius)
    const fov = THREE.MathUtils.degToRad(this.camera.fov)
    const dist = (sphere.radius * 1.15) / Math.max(0.05, Math.tan(fov / 2))
    this.orbit.target.copy(center)
    this.camera.position.set(center.x + dist * 0.55, center.y + dist * 0.62, center.z + dist * 0.55)
    this.camera.lookAt(center)
    this.orbit.update()
  }

  _applyDisplayScale() {
    if (!this.root) return
    // Legacy Z-up meshes are rotated at the root, so their local Z axis is
    // the world elevation axis. New exports are already Y-up.
    if (this.legacyZUp) this.root.scale.set(1, 1, this.displayScale)
    else this.root.scale.set(1, this.displayScale, 1)
  }

  getHeightMapping() {
    return { minH: this.minH, maxH: this.maxH, gw: this.gridW, gh: this.gridH, meshW: this.meshW, meshH: this.meshH, zScale: this.displayScale, sx: this.meshSx, sy: this.meshSy }
  }

  worldOf(r, c, heights, W) {
    const h = Number(heights?.[r * W + c])
    const sourceH = Math.max(1, this.gridH - 1)
    const sourceW = Math.max(1, (W || this.gridW) - 1)
    const meshCol = (Number(c) / sourceW) * Math.max(1, this.meshW - 1)
    const meshRow = (Number(r) / sourceH) * Math.max(1, this.meshH - 1)
    return [
      (meshCol - (this.meshW - 1) / 2) * this.meshSx,
      (Number.isFinite(h) ? h - this.minH : 0) * this.displayScale,
      -(meshRow - (this.meshH - 1) / 2) * this.meshSy,
    ]
  }

  focusOn(x, y, z) {
    this.orbit.target.set(x, y, z)
    this.camera.lookAt(this.orbit.target)
    this.orbit.update()
  }

  setMode(next) {
    this.mode = next === 'fly' ? 'fly' : 'orbit'
    // The structure scene intentionally keeps OrbitControls as its stable
    // navigation primitive; the camera dock still exposes the shared mode
    // contract without inventing a second flight implementation.
    this.orbit.enabled = true
  }

  getOrientation() {
    const az = this.orbit.getAzimuthalAngle()
    let heading = (-az * 180) / Math.PI + 180
    heading = ((heading % 360) + 360) % 360
    return { heading, speed: 0 }
  }

  getScaleBar() {
    if (!this.root || !this.R) return null
    const hPx = this.container.clientHeight || 1
    const dist = this.camera.position.distanceTo(this.orbit.target)
    const worldPerPx = (2 * dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) * 0.5)) / Math.max(1, hPx)
    const raw = worldPerPx * 110
    const value = Math.max(1, Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-6)))))
    return { label: `${value >= 100 ? Math.round(value) : value} m`, px: Math.max(36, value / Math.max(worldPerPx, 1e-9)) }
  }

  getCamRange() {
    if (!this.root) return null
    return { dist: this.camera.position.distanceTo(this.orbit.target), label: `${this.camera.position.distanceTo(this.orbit.target).toFixed(1)} m` }
  }

  resetView() {
    if (!this.root) return
    this._fitCamera()
  }

  setPreset(name) {
    if (!this.root) return
    const t = this.orbit.target.clone()
    const d = Math.max(this.R * 1.1, 10)
    const positions = {
      top: new THREE.Vector3(t.x, t.y + d * 1.35, t.z + 0.001),
      north: new THREE.Vector3(t.x, t.y + d * 0.55, t.z - d),
      iso: new THREE.Vector3(t.x + d * 0.55, t.y + d * 0.62, t.z + d * 0.55),
    }
    const position = positions[name] || positions.iso
    this.camera.position.copy(position)
    this.camera.lookAt(t)
    this.orbit.update()
  }

  startTour(speed = 1) {
    if (!this.root) return
    this.stopTour(true)
    this.tour = { angle: Math.atan2(this.camera.position.x - this.orbit.target.x, this.camera.position.z - this.orbit.target.z), speed, paused: false }
    this.orbit.enabled = false
    this.onTourChange(true)
  }

  pauseTour() { if (this.tour) { this.tour.paused = true; this.onTourChange(true) } }
  resumeTour() { if (this.tour) { this.tour.paused = false; this.onTourChange(true) } }
  stopTour(silent = false) {
    if (!this.tour) return
    this.tour = null
    this.orbit.enabled = true
    if (!silent) this.onTourChange(false)
  }
  setTourSpeed(speed) { if (this.tour) this.tour.speed = speed }

  setTheme(name) {
    const color = name === 'light' ? '#dfe3d2' : '#11150f'
    this.scene.background = new THREE.Color(color)
    if (this.scene.fog) this.scene.fog.color.set(color)
  }
  setColorMode() {}
  setSlopeView() {}
  setHillshade(on) { if (this.sun) this.sun.intensity = on ? 2.2 : 1.1 }
  clearMeasure() {}

  _fitCamera() {
    if (!this.root) return
    const box = new THREE.Box3().setFromObject(this.root)
    const center = box.getCenter(new THREE.Vector3())
    const sphere = box.getBoundingSphere(new THREE.Sphere())
    this.camera.near = Math.max(0.05, sphere.radius / 1500)
    this.camera.far = Math.max(2000, sphere.radius * 16)
    this.camera.updateProjectionMatrix()
    this.orbit.minDistance = Math.max(1, sphere.radius * 0.08)
    this.orbit.maxDistance = Math.max(100, sphere.radius * 12)
    this._setAtmosphere(sphere.radius)
    const fov = THREE.MathUtils.degToRad(this.camera.fov)
    const dist = (sphere.radius * 1.15) / Math.max(0.05, Math.tan(fov / 2))
    this.R = sphere.radius * 1.7
    this.orbit.target.copy(center)
    this.camera.position.set(center.x + dist * 0.55, center.y + dist * 0.62, center.z + dist * 0.55)
    this.camera.lookAt(center)
    this.orbit.update()
  }

  _setAtmosphere(radius) {
    const extent = Math.max(20, Number(radius) || 20)
    if (this.scene.fog) {
      this.scene.fog.near = extent * 1.4
      this.scene.fog.far = extent * 8
    }
  }

  _pick(e) {
    if (!this.downInfo || !this.root) return
    const dx = e.clientX - this.downInfo.x
    const dy = e.clientY - this.downInfo.y
    this.downInfo = null
    if (dx * dx + dy * dy > 25) return // was a drag, not a click
    const rect = this.renderer.domElement.getBoundingClientRect()
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    )
    const ray = new THREE.Raycaster()
    ray.setFromCamera(ndc, this.camera)
    const hits = ray.intersectObjects(this.root.children, true)
    const b = hits.find((h) => h.object.userData.buildingId != null)
    this.onPick(b ? b.object.userData.buildingId : null)
  }

  screenshot() {
    this.renderer.render(this.scene, this.camera)
    return this.renderer.domElement.toDataURL('image/png')
  }

  _loop() {
    if (this.disposed) return
    requestAnimationFrame(() => this._loop())
    if (this.tour && !this.tour.paused && this.root) {
      this.tour.angle += 0.14 * (this.tour.speed || 1) * 0.016
      const radius = Math.max(this.R * 0.62, 20)
      const x = this.orbit.target.x + Math.sin(this.tour.angle) * radius
      const z = this.orbit.target.z + Math.cos(this.tour.angle) * radius
      this.camera.position.set(x, this.orbit.target.y + radius * 0.55, z)
      this.camera.lookAt(this.orbit.target)
    } else {
      this.orbit.update()
    }
    this.renderer.render(this.scene, this.camera)
  }

  dispose() {
    this.disposed = true
    this.ro.disconnect()
    this.root?.traverse((o) => {
      if (!o.isMesh) return
      o.geometry?.dispose()
      const m = o.material
      if (Array.isArray(m)) m.forEach((x) => x.dispose?.())
      else m?.dispose?.()
      m?.map?.dispose?.()
    })
    this.renderer.dispose()
    this.container.innerHTML = ''
  }
}
