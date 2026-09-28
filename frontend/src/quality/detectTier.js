const LEVELS = ['low', 'medium', 'high']

function webgl2Available() {
  try {
    const c = document.createElement('canvas')
    return !!c.getContext('webgl2')
  } catch {
    return false
  }
}

async function benchmarkGPU() {
  const {
    WebGLRenderer,
    PerspectiveCamera,
    Scene,
    PlaneGeometry,
    Mesh,
    MeshStandardMaterial,
    DirectionalLight,
    AmbientLight,
  } = await import('three')
  const canvas = document.createElement('canvas')
  canvas.width = 320
  canvas.height = 240
  let renderer
  try {
    renderer = new WebGLRenderer({ canvas, antialias: true })
  } catch {
    return Number.POSITIVE_INFINITY
  }
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = 2
  const scene = new Scene()
  const cam = new PerspectiveCamera(50, 320 / 240, 0.1, 100)
  cam.position.set(0, 2.4, 3)
  cam.lookAt(0, 0, 0)
  const geo = new PlaneGeometry(4, 4, 300, 300)
  const mat = new MeshStandardMaterial({ roughness: 0.9 })
  const pos = geo.attributes.position
  for (let i = 0; i < pos.count; i++) {
    pos.setZ(i, Math.sin(pos.getX(i) * 6) * Math.cos(pos.getY(i) * 5) * 0.15)
  }
  const mesh = new Mesh(geo, mat)
  mesh.rotation.x = -Math.PI / 2
  mesh.castShadow = true
  mesh.receiveShadow = true
  scene.add(mesh)
  const sun = new DirectionalLight(0xffffff, 2.4)
  sun.position.set(3, 6, 2)
  sun.castShadow = true
  sun.shadow.mapSize.set(1024, 1024)
  scene.add(sun)
  scene.add(new AmbientLight(0xffffff, 0.4))

  const frames = 24
  const t0 = performance.now()
  for (let i = 0; i < frames; i++) {
    sun.position.set(Math.cos(i / frames * 6.28) * 4, 6, Math.sin(i / frames * 6.28) * 4)
    renderer.render(scene, cam)
  }
  const avg = (performance.now() - t0) / frames
  geo.dispose()
  mat.dispose()
  renderer.dispose()
  return avg
}

export async function detectQualityTier() {
  if (!webgl2Available()) {
    return { level: 'low', reason: 'no-webgl2', auto: true }
  }
  const cores = navigator.hardwareConcurrency || 4
  const mem = navigator.deviceMemory || 8
  let frameMs
  try {
    frameMs = await benchmarkGPU()
  } catch {
    return { level: 'low', reason: 'benchmark-failed', auto: true }
  }
  let level
  if (frameMs <= 20 && cores >= 6 && mem >= 4) level = 'high'
  else if (frameMs <= 42 || (cores >= 8 && frameMs <= 60)) level = 'medium'
  else level = 'low'
  return { level, reason: `frame ${frameMs.toFixed(1)}ms · ${cores}c · ${mem}gb`, auto: true }
}

export function resolveTier(autoLevel, override) {
  if (override && LEVELS.includes(override)) return override
  return LEVELS.includes(autoLevel) ? autoLevel : 'medium'
}

export const TIER_CAPS = {
  high: {
    shadows: true,
    fog: 'full',
    skyGradient: true,
    reveal: 'full',
    meshSegments: 512,
    contourMode: 'live-full',
    modeTransitionMs: 600,
    idleMotion: true,
    dprCap: 2,
  },
  medium: {
    shadows: false,
    fog: 'flat',
    skyGradient: false,
    reveal: 'displacement-only',
    meshSegments: 288,
    contourMode: 'live-halfres',
    modeTransitionMs: 300,
    idleMotion: true,
    dprCap: 1.5,
  },
  low: {
    shadows: false,
    fog: 'none',
    skyGradient: false,
    reveal: 'instant',
    meshSegments: 160,
    contourMode: 'static-frame',
    modeTransitionMs: 0,
    idleMotion: false,
    dprCap: 1,
  },
}
