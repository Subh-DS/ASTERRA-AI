import { useMemo } from 'react'
import { useApp } from '../store/useAppStore'
import { dsmModeInfo } from '../utils/dsmMode'

function histogram(heights, bins = 36) {
  let mn = Infinity
  let mx = -Infinity
  const stride = Math.max(1, Math.floor(heights.length / 120000))
  for (let i = 0; i < heights.length; i += stride) {
    if (heights[i] < mn) mn = heights[i]
    if (heights[i] > mx) mx = heights[i]
  }
  const span = mx - mn || 1
  const h = new Array(bins).fill(0)
  for (let i = 0; i < heights.length; i += stride) {
    h[Math.min(bins - 1, Math.floor(((heights[i] - mn) / span) * bins))]++
  }
  return { h, mn, mx }
}

function slopeHistogram(heights, w, h, bins = 30) {
  const n = Math.min(220, w)
  const step = Math.max(1, Math.floor(w / n))
  const deg = new Array(bins).fill(0)
  for (let y = step; y < h - step; y += step) {
    for (let x = step; x < w - step; x += step) {
      const dx = (heights[y * w + x + step] - heights[y * w + x - step]) / (2 * step)
      const dy = (heights[(y + step) * w + x] - heights[(y - step) * w + x]) / (2 * step)
      const slope = (Math.atan(Math.hypot(dx, dy)) * 180) / Math.PI
      deg[Math.min(bins - 1, Math.floor((slope / 90) * bins))]++
    }
  }
  return deg
}

export default function AnalysisPanel() {
  const dsm = useApp((s) => s.dsm)

  const heightHist = useMemo(() => (dsm ? histogram(dsm.heights) : null), [dsm])
  const slopeHist = useMemo(() => (dsm ? slopeHistogram(dsm.heights, dsm.width, dsm.height) : null), [dsm])

  if (!dsm || !heightHist) {
    return <p className="empty-note">Terrain analysis appears here once a DSM is computed.</p>
  }

  const maxH = Math.max(...heightHist.h)
  const hotBin = heightHist.h.indexOf(maxH)
  const maxS = Math.max(...slopeHist)
  const mode = dsmModeInfo(dsm)
  const isMetric = mode.isMetric
  const unit = mode.unitLabel
  const modeLabel = mode.modeLabel
  const rangeLabel = mode.offline ? 'Demo range' : isMetric ? 'Elevation' : 'Relative depth'
  const scaleLabel = mode.offline ? 'demo units · not physical' : isMetric ? 'meters (m)' : 'Scale: 0–100'

  return (
    <>
      <div className="panel-head">
        <button className="panel-close" onClick={() => useApp.getState().setViewer({ panelOpen: null })} aria-label="Close analysis panel">
          ← back
        </button>
        <h2>Analysis</h2>
      </div>

      <div className="panel-section">
        <div className="badge-row">
          {['urban', 'sparse', 'hilly', 'forest'].map((t) => (
            <span key={t} className={`landscape-badge${dsm.landscape === t ? ' on' : ''}`}>
              {t}
            </span>
          ))}
        </div>
        <dl className="meta-list">
          <dt>Mode</dt>
          <dd>{modeLabel}</dd>
          <dt>Grid</dt>
          <dd>
            {dsm.width} × {dsm.height}
          </dd>
          {dsm.reconstruction && (
            <>
              <dt>Reconstruction</dt>
              <dd>{dsm.reconstruction.mode === 'structure-aware' ? `structure-aware · ${dsm.reconstruction.buildings_detected} buildings` : 'DSM-only mesh'}</dd>
            </>
          )}
          {dsm.source?.type === 'map' && dsm.source.imagery && (
            <>
              <dt>Source</dt>
              <dd>Map · {dsm.source.imagery.provider}{dsm.source.imagery.synthetic ? ' (test pattern)' : ''}</dd>
              <dt>AOI</dt>
              <dd>{dsm.source.aoi.center_lat?.toFixed(4)}, {dsm.source.aoi.center_lon?.toFixed(4)} · {dsm.source.aoi.width_m}×{dsm.source.aoi.height_m} m</dd>
              <dt>Imagery</dt>
              <dd>{dsm.source.imagery.acquisition_datetime ? new Date(dsm.source.imagery.acquisition_datetime).toLocaleDateString() : 'date unknown'} · {dsm.source.imagery.resolution_m} m · {(dsm.source.imagery.bands || []).join('/')}</dd>
            </>
          )}
          <dt>{rangeLabel}</dt>
          <dd>
            {heightHist.mn.toFixed(1)} – {heightHist.mx.toFixed(1)} {unit}
          </dd>
          <dt>Scale</dt>
          <dd>{scaleLabel}</dd>
          {mode.verticalRef && (
            <>
              <dt>Vertical reference</dt>
              <dd>{mode.verticalRef}</dd>
            </>
          )}
        </dl>
        {!isMetric && (
          <p className="mono" style={{ fontSize: 'var(--t-12)', color: 'var(--paper-dim)', marginTop: 6 }}>
            {mode.offline
              ? 'Demo units — heuristic preview surface, not measured elevation.'
              : 'No CRS on the source image — values are relative depth (0–100), not metric meters. Metric calibration unavailable.'}
          </p>
        )}
      </div>

      <div className="panel-section">
        <h4>Height distribution</h4>
        <div className="hist">
          <div className="bars">
            {heightHist.h.map((v, i) => (
              <div key={i} className={`bar${i === hotBin ? ' hot' : ''}`} style={{ height: `${(v / maxH) * 100}%` }} />
            ))}
          </div>
          <div className="axis">
            <span>{heightHist.mn.toFixed(0)} {unit}</span>
            <span>{heightHist.mx.toFixed(0)} {unit}</span>
          </div>
        </div>
      </div>

      <div className="panel-section">
        <h4>Slope distribution</h4>
        <div className="hist">
          <div className="bars">
            {slopeHist.map((v, i) => (
              <div key={i} className={`bar${v === maxS ? ' hot' : ''}`} style={{ height: `${(v / maxS) * 100}%` }} />
            ))}
          </div>
          <div className="axis">
            <span>0°</span>
            <span>45°</span>
            <span>90°</span>
          </div>
        </div>
      </div>

      <div className="panel-section">
        <h4>In-scene measurement</h4>
        <p className="empty-note" style={{ padding: '0 0 10px' }}>
          In Orbit mode, click any two terrain points: distance, bearing and elevation
          difference appear directly on the terrain in {unit}. Esc clears the measurement.
        </p>
        <button className="btn" onClick={() => useApp.getState().viewerApi?.clearMeasure()}>
          Clear measurement
        </button>
      </div>

      <div className="panel-section">
        <h4>Methodology</h4>
        <dl className="meta-list">
          <dt>Depth model</dt>
          <dd>{dsm.model || 'unknown'}</dd>
          <dt>Depth output</dt>
          <dd>Relative depth (monocular, single view)</dd>
          <dt>Metric calibration</dt>
          <dd>{mode.calibrationLabel}</dd>
          {dsm.metadata?.calibration?.polarity && (
            <>
              <dt>Depth polarity</dt>
              <dd>{dsm.metadata.calibration.polarity}</dd>
            </>
          )}
          {dsm.metadata?.confidence?.available && (
            <>
              <dt>Confidence (mean)</dt>
              <dd>{dsm.metadata.confidence.mean} — heuristic score, not a probability</dd>
            </>
          )}
          <dt>Reference</dt>
          <dd>{mode.offline ? 'none (synthetic/heuristic)' : 'AWS Terrarium DEM tiles where CRS is present'}</dd>
          <dt>Vertical reference</dt>
          <dd>
            {mode.verticalRef
              ? `${mode.verticalRef} label as used by the implementation; no geoid conversion is applied`
              : 'n/a — not a metric product'}
          </dd>
        </dl>
      </div>
    </>
  )
}
