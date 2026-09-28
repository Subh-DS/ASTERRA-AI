import { useEffect, useMemo, useRef, useState } from 'react'
import { useApp } from '../../store/useAppStore'
import { dsmModeInfo } from '../../utils/dsmMode'
import { astar, downsampleGrid, maskCentroid, nearestFree, pickGoals, slopeGrid } from '../../utils/evac'

const N = 64

/** Paint a backend stride-mask onto the N×N evac grid. */
function rasterizeMask(grids, mask, W, H) {
  const blocked = new Uint8Array(N * N)
  if (!mask || !grids) return blocked
  const gw2 = grids.grid_w
  const gh2 = grids.grid_h
  const s = grids.grid_stride
  for (let gr = 0; gr < gh2; gr++) {
    for (let gc = 0; gc < gw2; gc++) {
      if (!mask[gr * gw2 + gc]) continue
      const r0 = Math.floor(((gr * s) / H) * N)
      const r1 = Math.min(N - 1, Math.floor((((gr + 1) * s) / H) * N))
      const c0 = Math.floor(((gc * s) / W) * N)
      const c1 = Math.min(N - 1, Math.floor((((gc + 1) * s) / W) * N))
      for (let y = r0; y <= r1; y++) {
        for (let x = c0; x <= c1; x++) blocked[y * N + x] = 1
      }
    }
  }
  return blocked
}

function fmtLen(m, metric) {
  if (!Number.isFinite(m)) return '—'
  if (metric) return m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`
  return `${m.toFixed(1)} rel`
}

/**
 * Evacuation routing (Phase 36). A* runs on the REAL downsampled DSM with
 * backend simulation masks as blocked cells. Routes, zones and statistics
 * are planning visualizations — always labeled SIMULATED.
 */
export default function EvacuationScenario() {
  const dsm = useApp((s) => s.dsm)
  const result = useApp((s) => s.hazard.result)
  const animT = useApp((s) => s.hazard.animT)
  const quake = useApp((s) => s.hazard.quake)
  const evac = useApp((s) => s.hazard.evac)
  const evacRequest = useApp((s) => s.hazard.evacRequest)
  const setHazard = useApp((s) => s.setHazard)
  const viewerApi = useApp((s) => s.viewerApi)
  const [computing, setComputing] = useState(false)
  const [note, setNote] = useState(null)

  const coastal = result?.simulation_type === 'coastal_inundation'
  const slide = result?.simulation_type === 'landslide'

  // Forget stale routes when the underlying simulation changes.
  useEffect(() => {
    setHazard({ evac: null })
    setNote(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result])

  // The mission timeline can request a recompute as the ROUTE CHECK event.
  const firstRequest = useRef(true)
  useEffect(() => {
    if (firstRequest.current) {
      firstRequest.current = false
      return
    }
    if (evacRequest > 0) compute()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evacRequest])

  function compute() {
    setNote(null)
    if (!dsm?.heights || !result) return
    setComputing(true)
    try {
      const W = dsm.width
      const H = dsm.height
      const map = viewerApi?.getHeightMapping?.() || {}
      const sx = Number.isFinite(map.sx) && map.sx > 0 ? map.sx : 1
      const sy = Number.isFinite(map.sy) && map.sy > 0 ? map.sy : 1
      const minH = Number.isFinite(map.minH) ? map.minH : 0
      const zScale = Number.isFinite(map.zScale) && map.zScale > 0 ? map.zScale : 1
      const metric = dsmModeInfo(dsm).isMetric
      const g = result.grids || {}

      const blocked = new Uint8Array(N * N)
      if (coastal && g._maskU8) {
        const b = rasterizeMask(g, g._maskU8, W, H)
        for (let i = 0; i < N * N; i++) blocked[i] = blocked[i] || b[i]
      }
      let depN = null
      if (slide) {
        if (g._depU8) {
          depN = rasterizeMask(g, g._depU8, W, H)
          for (let i = 0; i < N * N; i++) blocked[i] = blocked[i] || depN[i]
        }
        if (g._sourceU8) {
          const sb = rasterizeMask(g, g._sourceU8, W, H)
          for (let i = 0; i < N * N; i++) blocked[i] = blocked[i] || sb[i]
        }
      }
      const centroid = maskCentroid(blocked, N)
      if (!centroid) {
        setNote('No hazardous cells in this scenario — nothing to route around.')
        return
      }
      // Start from the nearest passable ground — never inside the hazard.
      const start = nearestFree(blocked, N, centroid[0], centroid[1])
      if (!start) {
        setNote('No passable ground adjacent to the hazard — the area is fully enclosed.')
        return
      }
      const { h } = downsampleGrid(dsm.heights, W, H, N)
      const slope = slopeGrid(h, N)
      // Unknown terrain is impassable — routes only cross measured ground.
      for (let i = 0; i < N * N; i++) {
        if (!Number.isFinite(h[i])) blocked[i] = 1
      }
      // Safe zones must themselves be passable — never inside the hazard.
      const goals = pickGoals(h, slope, N, 5).filter((gl) => !blocked[gl.y * N + gl.x]).slice(0, 3)
      console.info('[evac-debug]', JSON.stringify({
        blocked: blocked.reduce((a, b) => a + b, 0),
        start,
        goals: goals.map((g) => [g.x, g.y]),
        nanH: [...h].filter((v) => !Number.isFinite(v)).length,
        nanS: [...slope].filter((v) => !Number.isFinite(v)).length,
      }))
      if (!goals.length) {
        setNote('No reachable high ground outside the hazard on this terrain.')
        return
      }
      const toWorld = (x, y) => {
        const c = ((x + 0.5) / N) * W
        const r = ((y + 0.5) / N) * H
        const elev = h[y * N + x]
        return {
          x: (c - (W - 1) / 2) * sx,
          z: (r - (H - 1) / 2) * sy,
          elev,
          y: (elev - minH) * zScale + 1.2,
        }
      }
      const withHz = (extra) => {
        const c = Float32Array.from(slope)
        c.hz = extra
        return c
      }
      const hzBase = new Float32Array(N * N)
      const pathStats = (cells) => {
        let len = 0
        let maxSlope = 0
        const pts = cells.map(([x, y]) => ({ ...toWorld(x, y), dep: depN ? depN[y * N + x] : 0 }))
        for (let i = 1; i < cells.length; i++) {
          const [x0, y0] = cells[i - 1]
          const [x1, y1] = cells[i]
          const runPx = Math.hypot(x1 - x0, y1 - y0) * ((W / N + H / N) / 2)
          const run = runPx * ((sx + sy) / 2)
          len += Math.max(run, 1e-6)
          const dh = Math.abs(h[y1 * N + x1] - h[y0 * N + x0])
          maxSlope = Math.max(maxSlope, (Math.atan(dh / Math.max(run, 1e-6)) * 180) / Math.PI)
        }
        return { pts, len, maxSlope }
      }
      const primary = (() => {
        // Try safe zones in order — the nearest high ground can sit in a
        // sealed pocket while a farther zone stays reachable.
        for (let gi = 0; gi < goals.length; gi++) {
          const p = astar(N, start, [goals[gi].x, goals[gi].y], blocked, withHz(hzBase))
          if (p) return { cells: p, goalIdx: gi }
        }
        return null
      })()
      if (!primary) {
        setNote('No passable corridor to high ground — the hazard seals every exit.')
        return
      }
      // Alternative: penalize the primary corridor and route again to a
      // different safe zone.
      const hz2 = Float32Array.from(hzBase)
      for (const [x, y] of primary.cells) hz2[y * N + x] = 1
      const altGoals = goals.filter((_, gi) => gi !== primary.goalIdx)
      let altCells = null
      for (const gl of altGoals) {
        altCells = astar(N, start, [gl.x, gl.y], blocked, withHz(hz2))
        if (altCells && altCells.join() !== primary.cells.join()) break
        altCells = null
      }
      const alternative = altCells ? pathStats(altCells) : null
      const zones = goals.map((gl) => {
        const w = toWorld(gl.x, gl.y)
        return { x: w.x, z: w.z, y: w.y - 0.6, r: Math.max(sx, sy) * ((W / N + H / N) / 2) * 0.8 }
      })
      const sw = toWorld(start[0], start[1])
      setHazard({
        evac: {
          simulated: true,
          metric,
          start: { x: sw.x, y: sw.y - 0.4, z: sw.z },
          primary: pathStats(primary.cells),
          alternative,
          zones,
          waterLevel: coastal ? Number(result.statistics?.water_level_m) : null,
          depAt: slide ? 0.45 : null,
          baseMinH: minH,
        },
      })
    } finally {
      setComputing(false)
    }
  }

  const status = useMemo(() => {
    if (!evac) return null
    if (evac.waterLevel != null && Number.isFinite(evac.waterLevel)) {
      const wl = evac.baseMinH + animT * (evac.waterLevel - evac.baseMinH)
      const flooded = evac.primary.pts.filter((p) => p.elev < wl).length
      const frac = evac.primary.pts.length ? flooded / evac.primary.pts.length : 0
      if (frac >= 1) return { tone: 'bad', text: 'PRIMARY ROUTE CUT at current water level — use the alternative' }
      if (frac > 0) return { tone: 'warn', text: `PRIMARY PARTIALLY FLOODED (${Math.round(frac * 100)}% of length) at current water level` }
      return { tone: 'ok', text: 'PRIMARY ROUTE CLEAR at current water level' }
    }
    if (evac.depAt != null) {
      const crossed = evac.primary.pts.some((p) => p.dep)
      if (!crossed) return { tone: 'ok', text: 'ROUTE AVOIDS MAPPED DEBRIS — clear at every timeline position' }
      if (animT < evac.depAt) return { tone: 'ok', text: 'ROUTE CLEAR FOR NOW — debris arrives later on the timeline' }
      return { tone: 'warn', text: 'ROUTE CROSSES FRESH DEBRIS — consider the alternative' }
    }
    return null
  }, [evac, animT])

  if (!result || (!coastal && !slide)) {
    return (
      <div>
        <p className="hz-note">
          Evacuation routing needs an active flood or landslide simulation — run one first, then plan
          routes over the real terrain with hazard cells avoided.
        </p>
      </div>
    )
  }

  const routeBlock = (label, r, color) =>
    r && (
      <div className="hz-row">
        <span className="hz-label">{label} <span className="hz-dim">simulated</span></span>
        <span style={{ color }}>{fmtLen(r.len, evac.metric)} · max {r.maxSlope.toFixed(0)}°</span>
      </div>
    )

  return (
    <div>
      <button type="button" className="hz-run" disabled={computing} onClick={compute}>
        {computing ? 'Routing…' : evac ? 'Recalculate routes' : 'Compute evacuation routes'}
      </button>
      {note && <div className="hz-error" role="alert">{note}</div>}
      {evac && (
        <>
          {routeBlock('Route A', evac.primary, '#4fd9c4')}
          {routeBlock('Route B', evac.alternative, '#e8a33d')}
          {!evac.alternative && <p className="hz-note">No distinct alternative corridor exists on this terrain.</p>}
          {status && (
            <div className={`hz-route-status ${status.tone}`} role="status">
              {status.text}
            </div>
          )}
          {quake && (
            <div className="hz-route-status warn" role="alert">
              Shaking occurred after these routes were computed — recalculate before relying on them.
            </div>
          )}
          <p className="hz-note">
            A* over the measured {N}×{N} DSM grid. Slope raises traversal cost; hazard cells are
            impassable. Simulated planning aid — not a verified safe path.
          </p>
        </>
      )}
    </div>
  )
}
