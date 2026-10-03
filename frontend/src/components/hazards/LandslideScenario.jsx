import { useState } from 'react'
import { useApp } from '../../store/useAppStore'
import { hazardApi } from '../../utils/hazard'

export default function LandslideScenario() {
  const dsm = useApp((s) => s.dsm)
  const status = useApp((s) => s.hazard.status)
  const setHazard = useApp((s) => s.setHazard)
  const [rainfall, setRainfall] = useState(50)
  const busy = status === 'running'

  async function runSusceptibility() {
    setHazard({ status: 'running', error: null })
    try {
      const res = await hazardApi.simulate(dsm.id, 'landslide_susceptibility', {
        rainfall_mm: rainfall,
      })
      setHazard({ status: 'complete', result: res, animT: 1, showSimulated: true, error: null })
    } catch (err) {
      setHazard({ status: 'error', error: err.message })
    }
  }

  return (
    <div>
      <div className="hz-row"><span className="hz-label">Method</span><span>Physical susceptibility model (per-pixel)</span></div>
      <label className="hz-field">Rainfall trigger (mm)
        <input
          type="range" min={0} max={200} step={5}
          value={rainfall}
          onChange={(e) => setRainfall(Number(e.target.value))}
          aria-label="Rainfall in millimeters"
        />
        <span className="hz-value">{rainfall} mm</span>
      </label>
      <div className="hz-row"><span className="hz-label">Factors</span><span>Slope, relief, drainage, TWI, curvature, land cover</span></div>
      <button type="button" className="hz-run" disabled={busy} onClick={runSusceptibility}>
        {busy ? 'Computing…' : '▶ Compute susceptibility map'}
      </button>
      <p className="hz-note">
        Evaluates every pixel using terrain factors. Flat areas are suppressed.
        This is a susceptibility map — not a prediction of where a landslide will occur.
      </p>
    </div>
  )
}
