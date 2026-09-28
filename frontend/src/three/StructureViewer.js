/**
 * Phase 9: GLB structure viewer — loads model.glb (TERRAIN + BUILDING_* groups)
 * alongside the existing DSM TerrainViewer (which stays the default/fast path).
 * Metric geometry is shown as-is; z-exaggeration remains a Terrain-view setting.
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

function groupOf(name) {
  if (name === 'TERRAIN') return 'TERRAIN'
  if (name.startsWith('ROOF_BUILDING_')) return 'BUILDINGS'
  if (name.startsWith('BUILDING_')) return 'BUILDINGS'
  if (name.startsWith('ROAD')) return 'ROADS'
  if (name.startsWith('TREE_') || name.startsWith('VEGETATION_TREE')) return 'TREES'
  if (name.startsWith('VEGETATION')) return 'VEGETATION'
  if (name.startsWith('WATER')) return 'WATER'
  if (name.startsWith('INFRA')) return 'INFRASTRUCTURE'
  return 'OTHER'
}

export const LAYER_LABELS = {
  TERRAIN: 'Terrain',
  BUILDINGS: 'Buildings',
  ROADS: 'Roads',
  VEGETATION: 'Vegetation',
  TREES: 'Trees',
  WATER: 'Water',
  INFRASTRUCTURE: 'Infrastructure',
}

export class StructureViewer {
  constructor(container, opts = {}) {
    this.container = container
    this.onPick = opts.onPick || (() => {})
    this.onLoad = opts.onLoad || (() => {})
    this.disposed = false

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.domElement.className = 'main struct'
    container.appendChild(this.renderer.domElement)

    this.scene = new THREE.Scene()
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100000)
    this.orbit = new OrbitControls(this.camera, this.renderer.domElement)
    this.orbit.enableDamping = true
    this.orbit.dampingFactor = 0.08
    this.orbit.maxPolarAngle = Math.PI * 0.495

    this.scene.add(new THREE.HemisphereLight(0xe6ecdd, 0x1a241e, 0.9))
    this.sun = new THREE.DirectionalLight(0xffe7c4, 2.2)
    this.scene.add(this.sun)
    this.scene.add(this.sun.target)

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
        this.groups = {}
        this.root.traverse((o) => {
          if (o.isMesh) {
            const g = groupOf(o.name || '')
            o.userData.group = g
            const m = /^BUILDING_(\d+)_/.exec(o.name || '')
            if (m) o.userData.buildingId = parseInt(m[1], 10)
            this.groups[g] = this.groups[g] || []
            this.groups[g].push(o)
          }
        })
        this.scene.add(this.root)
        // Fit camera to measured bounds (never hard-coded).
        const box = new THREE.Box3().setFromObject(this.root)
        const center = box.getCenter(new THREE.Vector3())
        const sphere = box.getBoundingSphere(new THREE.Sphere())
        const fov = THREE.MathUtils.degToRad(this.camera.fov)
        const dist = (sphere.radius * 1.15) / Math.max(0.05, Math.tan(fov / 2))
        this.orbit.target.copy(center)
        this.camera.position.set(
          center.x + dist * 0.55, center.y + dist * 0.62, center.z + dist * 0.55,
        )
        this.sun.position.set(center.x - sphere.radius, center.y + sphere.radius * 1.4, center.z + sphere.radius)
        this.sun.target.position.copy(center)
        this.camera.lookAt(center)
        this.onLoad({ ok: true, groups: this.layerNames(), radius: sphere.radius })
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
    this.orbit.update()
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
