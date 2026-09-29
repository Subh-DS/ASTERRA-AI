import { useEffect, useState } from 'react'
import { dsmModeInfo } from '../utils/dsmMode'

function fmtRange(v, unit) {
  if (!Number.isFinite(v)) return '—'
  const abs = Math.abs(v)
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2
  return `${v.toFixed(digits)} ${unit}`
}

/**
 * Phase 19: compact scientific metadata HUD. Every value is real —
 * dataset, CRS, vertical reference, GSD, and elevation range come from the
 * loaded DSM; camera range is measured live from the scene. Missing values
 * render as em-dashes, never invented.
 */
export default function MetaHud({ dsm, viewerApi }) {
  const [, setTick] = useState(0)

  useEffect(() => {
    if (!viewerApi) return
    const id = setInterval(() => setTick((t) => t + 1), 1500)
    return () => clearInterval(id)
  }, [viewerApi])

  if (!dsm) return null
  const mode = dsmModeInfo(dsm)
  const unit = mode.isMetric ? 'm' : 'rel'

  let gsd = '—'
  const ps = dsm.pixelSizeM
  const r2 = (v) => (Number.isFinite(v) && v > 0 ? (v >= 100 ? Math.round(v) : Math.round(v * 100) / 100) : null)
  const g0 = r2(typeof ps === 'number' ? ps : Array.isArray(ps) ? ps[0] : NaN)
  const g1 = r2(Array.isArray(ps) ? ps[1] : NaN)
  if (g0 != null && g1 != null && g0 !== g1) gsd = `${g0} × ${g1} m`
  else if (g0 != null) gsd = `${g0} m`

  let elev = viewerApi ? 'Loading…' : '—'
  try {
    const hm = viewerApi?.getHeightMapping?.()
    if (hm && Number.isFinite(hm.minH) && Number.isFinite(hm.maxH)) {
      elev = `${fmtRange(hm.minH, unit)} → ${fmtRange(hm.maxH, unit)}`
    }
  } catch {
    /* mapping unavailable before first load */
  }

  let cam = viewerApi ? 'Loading…' : '—'
  try {
    cam = viewerApi?.getCamRange?.()?.label || 'Loading…'
  } catch {
    /* camera not ready */
  }

  return (
    <div className="meta-hud mono" aria-label="Dataset metadata">
      <div className="meta-row">
        <span>DATASET</span><b>{dsm.id || '—'}</b>
      </div>
      <div className="meta-row">
        <span>CRS</span><b>{mode.crs || 'LOCAL GRID'}</b>
      </div>
      <div className="meta-row">
        <span>VERT REF</span><b>{mode.verticalRef || '—'}</b>
      </div>
      <div className="meta-row">
        <span>GSD</span><b>{gsd}</b>
      </div>
      <div className="meta-row">
        <span>ELEV RANGE</span><b>{elev}</b>
      </div>
      <div className="meta-row">
        <span>CAM RANGE</span><b>{cam}</b>
      </div>
      <div className="meta-mode">{dsmModeLabel(mode, dsm)}</div>
      {!mode.isMetric && (
        <div className="meta-warn">
          {dsm.offline
            ? 'Demo terrain — heuristic surface, NOT measured elevation'
            : 'Metric calibration unavailable — values are NOT physical meters'}
        </div>
      )}
    </div>
  )
}

function dsmModeLabel(mode, dsm) {
  if (mode.offline) return 'DEMO SURFACE · NOT METRIC'
  if (mode.isMetric) return 'METRIC DSM · METERS'
  return 'RELATIVE SURFACE · 0–100'
}
