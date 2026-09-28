import * as THREE from 'three'
import { fbm2d } from '../utils/sampleScene'

// Cinematic ASTERRA hero terrain — shaded ridge mesh with a natural faceted finish.
// Same API as before: initHeroTerrain(canvas, { reducedMotion }) -> { setTheme, setRunning, dispose }
export function initHeroTerrain(canvas, { reducedMotion }) {
  const isLight = document.documentElement.dataset.theme === 'light'
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75))
  const scene = new THREE.Scene()
  scene.fog = new THREE.Fog(isLight ? 0xccd2bd : 0x05090c, isLight ? 520 : 420, 1200)

  const cam = new THREE.PerspectiveCamera(42, 1, 1, 4000)
  cam.position.set(0, 148, 268)
  cam.lookAt(0, 6, 0)

  // lights for cinematic shading (rim kept restrained so it models form, not haze)
  const sun = new THREE.DirectionalLight(isLight ? 0xfff2d8 : 0xffe0ae, isLight ? 1.45 : 2.1)
  sun.position.set(-140, 180, 80)
  scene.add(sun)
  const rim = new THREE.DirectionalLight(0x4fd9c4, isLight ? 0.5 : 1.0)
  rim.position.set(160, 60, -120)
  scene.add(rim)
  const amb = new THREE.AmbientLight(isLight ? 0xbcc8c2 : 0x2a4a42, isLight ? 1.0 : 1.35)
  scene.add(amb)

  const SEG = 150
  const geo = new THREE.PlaneGeometry(460, 460, SEG, SEG)
  geo.rotateX(-Math.PI / 2)
  const pos = geo.attributes.position
  const colors = new Float32Array(pos.count * 3)
  const tmp = new THREE.Color()
  let minH = Infinity
  let maxH = -Infinity
  const hs = new Float32Array(pos.count)
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i) / 92
    const z = pos.getZ(i) / 92
    const ridge = 1 - Math.abs(fbm2d(x * 1.4 + 4.2, z * 1.4 - 2.7, 4))
    const base = fbm2d(x + 4.2, z - 2.7, 5) * 26 + fbm2d(x * 3, z * 3, 3) * 6
    const h = base + ridge * ridge * 22
    hs[i] = h
    if (h < minH) minH = h
    if (h > maxH) maxH = h
  }
  // Natural hypsometric tint + slope/altitude detail (rock faces, valley
  // vegetation, high snow) with fine grain, so the massif reads like real
  // earth rather than a stylized ramp. Re-runs on theme toggle.
  function paint(light) {
    const cLow = new THREE.Color(light ? '#5c6f60' : '#1c332b')
    const cMid = new THREE.Color(light ? '#8a9678' : '#3d5c46')
    const cHigh = new THREE.Color(light ? '#c4b28e' : '#8a7350')
    const cPeak = new THREE.Color(light ? '#dfc89e' : '#d9a45b')
    const cRock = new THREE.Color(light ? '#8a8272' : '#453e35')
    const cVeg = new THREE.Color(light ? '#6f8560' : '#2e4a35')
    const cSnow = new THREE.Color(light ? '#f6f1e2' : '#e8e2d4')
    for (let i = 0; i < pos.count; i++) {
      const t = (hs[i] - minH) / Math.max(1e-5, maxH - minH)
      if (t < 0.35) tmp.copy(cLow).lerp(cMid, t / 0.35)
      else if (t < 0.7) tmp.copy(cMid).lerp(cHigh, (t - 0.35) / 0.35)
      else tmp.copy(cHigh).lerp(cPeak, (t - 0.7) / 0.3)
      colors[i * 3] = tmp.r
      colors[i * 3 + 1] = tmp.g
      colors[i * 3 + 2] = tmp.b
    }
    geo.computeVertexNormals()
    // Detail pass driven by real surface orientation + altitude.
    const nrm = geo.attributes.normal
    const rock = new THREE.Color()
    for (let i = 0; i < pos.count; i++) {
      const nx = pos.getX(i) / 92
      const nz = pos.getZ(i) / 92
      const t = (hs[i] - minH) / Math.max(1e-5, maxH - minH)
      const slope = 1 - nrm.getY(i) // 0 = flat, →1 = cliff
      tmp.setRGB(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2])
      // Exposed rock on steeps.
      const rocky = Math.min(1, Math.max(0, (slope - 0.12) / 0.28))
      if (rocky > 0) {
        rock.copy(cRock)
        tmp.lerp(rock, rocky * 0.65)
      }
      // Vegetation hugging low, gentle ground with patchy variation.
      const patch = fbm2d(nx * 6 + 11, nz * 6 - 7, 3) * 0.5 + 0.5
      const veg = t < 0.42 && slope < 0.22 ? (1 - t / 0.42) * (1 - slope / 0.22) * (0.35 + patch * 0.5) : 0
      if (veg > 0) {
        rock.copy(cVeg)
        tmp.lerp(rock, Math.min(0.6, veg))
      }
      // Snow resting on high, gentle shoulders.
      const snow = t > 0.78 && slope < 0.3 ? Math.min(1, (t - 0.78) / 0.12) * (1 - slope / 0.3) : 0
      if (snow > 0) {
        rock.copy(cSnow)
        tmp.lerp(rock, Math.min(0.85, snow))
      }
      // Fine grain so slopes don't look airbrushed.
      const grain = 0.93 + (fbm2d(nx * 16, nz * 16, 2) * 0.5 + 0.5) * 0.14
      colors[i * 3] = tmp.r * grain
      colors[i * 3 + 1] = tmp.g * grain
      colors[i * 3 + 2] = tmp.b * grain
    }
    geo.attributes.color.needsUpdate = true
  }
  for (let i = 0; i < pos.count; i++) pos.setY(i, hs[i])
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  paint(isLight)

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 1.0,
    metalness: 0.0,
    flatShading: false,
    transparent: true,
    opacity: 0.98,
  })
  const mesh = new THREE.Mesh(geo, mat)
  scene.add(mesh)

  let raf = null
  let running = !reducedMotion
  function resize() {
    const r = canvas.getBoundingClientRect()
    const w = Math.max(2, r.width)
    const h = Math.max(2, r.height)
    renderer.setSize(w, h, false)
    cam.aspect = w / Math.max(h, 1)
    cam.updateProjectionMatrix()
  }
  const ro = new ResizeObserver(resize)
  ro.observe(canvas)
  resize()

  function loop(t) {
    raf = requestAnimationFrame(loop)
    if (document.hidden || !running) return
    const s = t * 0.001
    mesh.rotation.y = s * 0.028
    mesh.position.y = Math.sin(s * 0.32) * 2.2
    cam.position.y = 148 + Math.sin(s * 0.18) * 7
    cam.lookAt(0, 6, 0)
    renderer.render(scene, cam)
  }
  renderer.render(scene, cam)
  if (running) raf = requestAnimationFrame(loop)

  return {
    setTheme(theme) {
      const light = theme === 'light'
      mat.opacity = 0.96
      paint(light)
      scene.fog.color.set(light ? 0xccd2bd : 0x05090c)
      sun.color.set(light ? 0xfff2d8 : 0xffe0ae)
      sun.intensity = light ? 1.45 : 2.1
      rim.intensity = light ? 0.5 : 1.0
      amb.color.set(light ? 0xbcc8c2 : 0x2a4a42)
      amb.intensity = light ? 1.0 : 1.35
    },
    setRunning(v) {
      running = v
      if (!v && raf) {
        cancelAnimationFrame(raf)
        raf = null
      } else if (v && !raf) {
        raf = requestAnimationFrame(loop)
      }
    },
    dispose() {
      if (raf) cancelAnimationFrame(raf)
      ro.disconnect()
      geo.dispose()
      mat.dispose()
      renderer.dispose()
    },
  }
}
