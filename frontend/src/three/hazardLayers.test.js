// Regression tests for the dynamic water system (Phase 31).
// Grids below are synthetic test fixtures for rendering logic ONLY — they
// never represent real flood data. Backend truth is covered by
// backend/tests/test_p11_hazards.py and utils/hazard.test.js.
import { beforeAll, describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { HazardLayers } from './hazardLayers'

beforeAll(() => {
  if (typeof globalThis.window === 'undefined') globalThis.window = {}
  if (!globalThis.window.matchMedia) {
    globalThis.window.matchMedia = () => ({ matches: false })
  }
  if (typeof globalThis.requestAnimationFrame === 'undefined') {
    globalThis.requestAnimationFrame = () => 0
  }
  if (typeof globalThis.cancelAnimationFrame === 'undefined') {
    globalThis.cancelAnimationFrame = () => {}
  }
})

function mockViewer() {
  return {
    scene: { add() {}, remove() {} },
    renderer: { toneMappingExposure: 1.1 },
    sun: { intensity: 2.5 },
    orbitTarget: { x: 0, y: 5, z: 0 },
    R: 200,
    getHeightMapping: () => ({ minH: 40, maxH: 52, gw: 8, gh: 8, zScale: 1, sx: 10, sy: 10 }),
  }
}

// 4x4 stride-2 mask over an 8x8 grid: three flooded cells, depths 0.3/1.5/6.
function coastalResult() {
  const mask = new Uint8Array(16)
  mask[5] = 1
  mask[6] = 1
  mask[10] = 1
  const depth = new Float32Array(16)
  depth[5] = 0.3
  depth[6] = 1.5
  depth[10] = 6.0
  return {
    simulation_type: 'coastal_inundation',
    grids: { grid_w: 4, grid_h: 4, grid_stride: 2, _maskU8: mask, _depthF32: depth },
    statistics: { water_level_m: 46 },
  }
}

describe('coastal water mesh', () => {
  it('builds water + shoreline layers from the simulation mask', () => {
    const ctrl = new HazardLayers(mockViewer())
    ctrl.showCoastal(coastalResult(), null)
    expect(ctrl.layers.water).toBeTruthy()
    expect(ctrl.layers.shoreline).toBeTruthy()
    expect(ctrl.layers.water.geometry.attributes.position.count).toBeGreaterThan(0)
    ctrl.dispose()
  })

  it('rises with timeline progress and hides at t=0', () => {
    const ctrl = new HazardLayers(mockViewer())
    ctrl.showCoastal(coastalResult(), null)
    ctrl.setT(0)
    expect(ctrl.layers.water.visible).toBe(false)
    ctrl.setT(0.5)
    expect(ctrl.layers.water.visible).toBe(true)
    expect(ctrl.layers.water.material.opacity).toBeGreaterThan(0.15)
    expect(ctrl.layers.water.material.opacity).toBeLessThanOrEqual(0.85)
    ctrl.dispose()
  })
})

describe('ambient wave motion', () => {
  it('displaces vertices over time and breathes the shoreline', () => {
    const ctrl = new HazardLayers(mockViewer())
    ctrl.showCoastal(coastalResult(), null)
    ctrl.setT(1)
    const arr = ctrl.layers.water.geometry.attributes.position.array
    const before = Array.from(arr)
    ctrl._waterWave(2.5)
    const moved = arr.some((v, i) => Math.abs(v - before[i]) > 1e-9)
    expect(moved).toBe(true)
    const shore = ctrl.layers.shoreline.material.opacity
    expect(shore).toBeGreaterThanOrEqual(0.7)
    expect(shore).toBeLessThanOrEqual(0.95)
    expect(ctrl.layers.water.material.opacity).toBeGreaterThanOrEqual(0.15)
    expect(ctrl.layers.water.material.opacity).toBeLessThanOrEqual(0.85)
    ctrl.dispose()
  })
})

describe('storm visualization state', () => {
  it('creates rain, dims light, and restores everything on clear', () => {
    const v = mockViewer()
    const ctrl = new HazardLayers(v)
    ctrl.setStorm(true)
    expect(ctrl.layers.rain).toBeTruthy()
    expect(ctrl.layers.rain.visible).toBe(true)
    expect(v.renderer.toneMappingExposure).toBeLessThan(1.1)
    expect(v.sun.intensity).toBeLessThan(2.5)
    ctrl.setStorm(false)
    expect(ctrl.layers.rain.visible).toBe(false)
    expect(v.renderer.toneMappingExposure).toBeCloseTo(1.1)
    expect(v.sun.intensity).toBeCloseTo(2.5)
    ctrl.dispose()
  })
})

describe('evacuation route rendering', () => {
  const evacData = {
    start: { x: 0, y: 5, z: 0 },
    primary: { pts: [{ x: 0, y: 5, z: 0 }, { x: 10, y: 6, z: 0 }, { x: 20, y: 7, z: 10 }] },
    alternative: { pts: [{ x: 0, y: 5, z: 0 }, { x: -10, y: 6, z: 5 }] },
    zones: [{ x: 20, y: 7, z: 10, r: 8 }],
  }

  it('builds route lines, safe zones and start marker', () => {
    const ctrl = new HazardLayers(mockViewer())
    ctrl.showEvacuation(evacData)
    expect(ctrl.layers['evac-a']).toBeTruthy()
    expect(ctrl.layers['evac-b']).toBeTruthy()
    expect(ctrl.layers['evac-zones']).toBeTruthy()
    expect(ctrl.layers['evac-start']).toBeTruthy()
    ctrl.dispose()
  })

  it('clears routes with null and on dispose', () => {
    const ctrl = new HazardLayers(mockViewer())
    ctrl.showEvacuation(evacData)
    ctrl.showEvacuation(null)
    expect(ctrl.layers['evac-a']).toBeUndefined()
    expect(ctrl.layers['evac-zones']).toBeUndefined()
    ctrl.showEvacuation(evacData)
    ctrl.clear()
    expect(Object.keys(ctrl.layers).length).toBe(0)
    ctrl.dispose()
  })
})

describe('earthquake procedural shake', () => {
  function quakeViewer() {
    const v = mockViewer()
    v.camera = { position: new THREE.Vector3(10, 20, 30) }
    v.mesh = { position: new THREE.Vector3(0, 0, 0) }
    v.structGroup = { position: new THREE.Vector3(0, 0, 0) }
    return v
  }

  it('captures transform bases, creates dust, and preserves user motion on stop', () => {
    const v = quakeViewer()
    const ctrl = new HazardLayers(v)
    ctrl.startQuake({ intensity: 0.8, aftershocks: true })
    expect(ctrl._quake).toBeTruthy()
    expect(ctrl._quake.meshBase.toArray()).toEqual([0, 0, 0])
    expect(ctrl.layers.quakedust).toBeTruthy()
    // user orbits mid-quake; stopping must keep their movement, only drop dust+state
    v.camera.position.set(10.5, 20.2, 29.7)
    ctrl.stopQuake()
    expect(v.camera.position.toArray()).toEqual([10.5, 20.2, 29.7])
    expect(ctrl._quake).toBeNull()
    expect(ctrl.layers.quakedust).toBeUndefined()
    ctrl.dispose()
  })

  it('clears quake state with the scene', () => {
    const v = quakeViewer()
    const ctrl = new HazardLayers(v)
    ctrl.startQuake({ intensity: 0.5, aftershocks: false })
    ctrl.clear()
    expect(ctrl._quake).toBeNull()
    expect(Object.keys(ctrl.layers).length).toBe(0)
    ctrl.dispose()
  })
})
