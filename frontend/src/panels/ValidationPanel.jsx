import { useRef, useState } from 'react'
import { useApp } from '../store/useAppStore'
import { dsmModeInfo } from '../utils/dsmMode'
import { api } from '../api/client'

function Scatter({ points }) {
  const S = 240
  let mn = Infinity
  let mx = -Infinity
  for (const [a, b] of points) {
    if (a < mn) mn = a
    if (a > mx) mx = a
    if (b < mn) mn = b
    if (b > mx) mx = b
  }
  const span = mx - mn || 1
  const px = (v) => 12 + ((v - mn) / span) * (S - 24)
  const step = Math.max(1, Math.floor(points.length / 500))
  return (
    <svg className="scatter" viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Predicted vs reference scatter">
      {[0.25, 0.5, 0.75].map((f) => (
        <g key={f}>
          <line x1={12 + f * (S - 24)} y1="12" x2={12 + f * (S - 24)} y2={S - 12} stroke="var(--contour)" strokeWidth="0.5" />
          <line x1="12" y1={12 + f * (S - 24)} x2={S - 12} y2={12 + f * (S - 24)} stroke="var(--contour)" strokeWidth="0.5" />
        </g>
      ))}
      <line x1="12" y1={S - 12} x2={S - 12} y2="12" stroke="var(--signal)" strokeWidth="1" strokeDasharray="4 3" opacity="0.7" />
      {points.map(([p, r], i) =>
        i % step === 0 ? (
          <circle key={i} cx={px(p)} cy={S - px(r)} r="1.6" fill="var(--elevation)" opacity="0.55" />
        ) : null,
      )}
      <text x="16" y="20" fill="var(--paper-dim)" fontSize="9" fontFamily="IBM Plex Mono">
        ref ↑ pred →
      </text>
    </svg>
  )
}

export default function ValidationPanel() {
  const dsm = useApp((s) => s.dsm)
  const validation = useApp((s) => s.validation)
  const setValidation = useApp((s) => s.setValidation)
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const handleRef = async (file) => {
    if (!file || !dsm) return
    setError(null)
    try {
      const real = await api.validate(dsm.id, file)
      setValidation(real)
    } catch (err) {
      // Never fabricate metrics: a failed validation must surface as an
      // explicit unavailable state, not synthetic RMSE/MAE/correlation.
      setValidation(null)
      setError(
        'Validation unavailable — the reference DEM could not be processed ' +
          `(${err?.message || 'request failed'}). No accuracy metrics were generated.`,
      )
    }
  }

  if (!dsm) {
    return <p className="empty-note">Validation runs against a completed DSM.</p>
  }

  const v = validation
  const isMetric = dsmModeInfo(dsm).isMetric
  const errUnit = isMetric ? 'm' : 'relative'
  const verdictPass = v
    ? `Within tolerance for this terrain type — structural heights track the reference (${errUnit}).`
    : ''
  const verdictFail = v
    ? `RMSE above the 6 ${errUnit} threshold — expect degraded fidelity on fine structures for ${dsm.landscape} terrain.`
    : ''

  return (
    <>
      <div className="panel-head">
        <button className="panel-close" onClick={() => useApp.getState().setViewer({ panelOpen: null })} aria-label="Close validation panel">
          ← back
        </button>
        <h2>Validation</h2>
      </div>
      <div className="panel-section">
        <h4>Reference DEM / LiDAR</h4>
        {!validation && (
          <div
            className={`mini-drop${drag ? ' drag' : ''}`}
            role="button"
            tabIndex={0}
            aria-label="Reference elevation drop zone"
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setDrag(true)
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDrag(false)
              handleRef(e.dataTransfer.files?.[0])
            }}
          >
            Drop reference DEM here — GeoTIFF or ASCII grid
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".tif,.tiff,.asc,.png"
          className="sr-only"
          onChange={(e) => handleRef(e.target.files?.[0])}
        />
      </div>

      {error && !v && (
        <div className="panel-section">
          <div className="verdict fail" role="alert">
            {error}
          </div>
        </div>
      )}

      {v && (
        <>
          <div className="panel-section">
            <h4>Error readout</h4>
            <div className="readout-trio">
              <Readout name="RMSE" value={`${v.rmse.toFixed(2)} ${errUnit}`} warn={!v.pass}>
                average elevation error, weighted toward outliers
              </Readout>
              <Readout name="MAE" value={`${v.mae.toFixed(2)} ${errUnit}`} warn={!v.pass}>
                mean absolute deviation from reference
              </Readout>
              <Readout name="PEARSON r" value={v.corr.toFixed(3)}>
                structural agreement between surfaces
              </Readout>
            </div>
            <div className={`verdict${v.pass ? '' : ' fail'}`}>
              {v.pass ? verdictPass : verdictFail}
            </div>
          </div>

          <div className="panel-section">
            <h4>Predicted vs reference</h4>
            <Scatter points={v.scatter} />
          </div>

          <div className="panel-section">
            <h4>Absolute error histogram</h4>
            <div className="hist">
              <div className="bars">
                {v.hist.map((c, i) => {
                  const mx = Math.max(...v.hist)
                  return <div key={i} className={`bar${i >= v.hist.length * 0.75 ? ' hot' : ''}`} style={{ height: `${(c / mx) * 100}%` }} />
                })}
              </div>
              <div className="axis">
                <span>0 {errUnit}</span>
                <span>max err</span>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  )
}

function Readout({ name, value, warn, children }) {
  return (
    <div className={`readout${warn ? ' warn' : ''}`}>
      <div>
        <span className="name mono">{name}</span>
        <span className="big">{value}</span>
      </div>
      <span className="desc">{children}</span>
    </div>
  )
}
