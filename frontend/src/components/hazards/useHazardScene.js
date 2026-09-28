// Phase 11: binds the canonical backend result to the Three.js scene.
// Single HazardLayers instance per viewer; everything else is visualization.
import { useEffect, useRef } from 'react'
import { HazardLayers } from '../../three/hazardLayers'
import { useApp } from '../../store/useAppStore'
import { decodeGrids } from '../../utils/hazard'

function focusFor(result, ctrl, heights) {
  try {
    const g = result.grids
    const H = result.terrain_source.grid[0]
    const W = result.terrain_source.grid[1]
    if (result.simulation_type === 'coastal_inundation' && g._depthF32) {
      let bi = 0
      for (let i = 1; i < g._depthF32.length; i++) if (g._depthF32[i] > g._depthF32[bi]) bi = i
      if (g._depthF32[bi] <= 0) return
      const r = Math.floor(bi / g.grid_w) * g.grid_stride
      const c = (bi % g.grid_w) * g.grid_stride
      const [x, y, z] = ctrl.worldOf(Math.min(H - 1, r), Math.min(W - 1, c), heights, W)
      ctrl.focusOn(x, y, z)
    } else if (result.simulation_type === 'landslide') {
      const [r, c] = result.statistics.source_center_rc
      const [x, y, z] = ctrl.worldOf(r, c, heights, W)
      ctrl.focusOn(x, y, z)
    }
  } catch { /* camera assist is best-effort */ }
}

export function useHazardScene() {
  const viewerApi = useApp((s) => s.viewerApi)
  const dsm = useApp((s) => s.dsm)
  const result = useApp((s) => s.hazard.result)
  const layers = useApp((s) => s.hazard.layers)
  const animT = useApp((s) => s.hazard.animT)
  const showSimulated = useApp((s) => s.hazard.showSimulated)
  const storm = useApp((s) => s.hazard.storm)
  const quake = useApp((s) => s.hazard.quake)
  const evac = useApp((s) => s.hazard.evac)
  const ctrlRef = useRef(null)

  useEffect(() => {
    if (!viewerApi || !dsm) return
    const ctrl = new HazardLayers(viewerApi)
    ctrlRef.current = ctrl
    ctrl.setStorm?.(useApp.getState().hazard.storm)
    return () => { ctrl.dispose(); ctrlRef.current = null }
  }, [viewerApi, dsm?.id])

  useEffect(() => {
    const ctrl = ctrlRef.current
    if (!ctrl || !result || !dsm?.heights) return
    decodeGrids(result)
    const H = result.terrain_source.grid[0]
    const W = result.terrain_source.grid[1]
    const heights = (dsm.width === W && dsm.height === H) ? dsm.heights : null
    try {
      if (result.simulation_type === 'coastal_inundation') {
        ctrl.showCoastal(result, heights)
      } else {
        if (!heights) return
        ctrl.showLandslide(result, heights)
      }
    } catch (err) {
      // A 3D-visualization bug must never take down the viewer: the
      // canonical statistics above stay intact and usable.
      console.error('[hazards] overlay failed:', err)
      useApp.getState().setHazard({ error: `3D overlay failed — ${err.message}. Statistics above are unaffected.` })
      return
    }
    for (const [k, v] of Object.entries(layers)) ctrl.setLayerVisible(k, v)
    ctrl.setGroupVisible(showSimulated)
    ctrl.setT(animT)
    if (heights) focusFor(result, ctrl, heights)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result])

  useEffect(() => {
    const ctrl = ctrlRef.current
    if (!ctrl) return
    for (const [k, v] of Object.entries(layers)) ctrl.setLayerVisible(k, v)
  }, [layers])

  useEffect(() => {
    if (!result) ctrlRef.current?.clear()
  }, [result])
  useEffect(() => { ctrlRef.current?.setGroupVisible(showSimulated) }, [showSimulated])
  useEffect(() => { ctrlRef.current?.setT(animT) }, [animT])
  useEffect(() => { ctrlRef.current?.setStorm?.(storm) }, [storm])
  useEffect(() => {
    const ctrl = ctrlRef.current
    if (!ctrl) return
    if (quake) ctrl.startQuake?.({ intensity: quake.intensity ?? 0.6, aftershocks: quake.aftershocks !== false })
    else ctrl.stopQuake?.()
  }, [quake])
  useEffect(() => {
    const ctrl = ctrlRef.current
    if (!ctrl) return
    if (evac) ctrl.showEvacuation?.(evac)
    else if (typeof ctrl.showEvacuation === 'function' && !useApp.getState().hazard.result) {
      // No simulation context left — drop any orphan route overlays.
      ctrl.showEvacuation(null)
    }
  }, [evac])

  return ctrlRef
}
