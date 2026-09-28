import { useState } from 'react'
import { useApp } from '../../store/useAppStore'
import { hazardApi } from '../../utils/hazard'

const SEVS = ['small', 'medium', 'large']
const DEPTHS = { small: '3.0 m', medium: '5.0 m', large: '8.0 m' }

export default function LandslideScenario() {
  const dsm = useApp((s) => s.dsm)
  const status = useApp((s) => s.hazard.status)
  const setHazard = useApp((s) => s.setHazard)
  const [sev, setSev] = useState('medium')
  const busy = status === 'running'

  async function run() {
    setHazard({ status: 'running', error: null })
    try {
      const res = await hazardApi.simulate(dsm.id, 'landslide', { severity: sev })
      setHazard({ status: 'complete', result: res, animT: 0, showSimulated: true, error: null })
    } catch (err) {
      setHazard({ status: 'error', error: err.message })
    }
  }

  return (
    <div>
      <div className="hz-row"><span className="hz-label">Release zone</span><span>Auto — steepest high terrain</span></div>
      <div className="hz-sev" role="group" aria-label="Landslide severity">
        {SEVS.map((s) => (
          <button key={s} type="button" className={sev === s ? 'on' : ''} disabled={busy}
            onClick={() => setSev(s)}>{s[0].toUpperCase() + s.slice(1)}</button>
        ))}
      </div>
      <div className="hz-row"><span className="hz-label">Release depth</span><span>{DEPTHS[sev]} (scenario parameter)</span></div>
      <button type="button" className="hz-run" disabled={busy} onClick={run}>
        {busy ? 'Simulating…' : '▶ Run landslide'}
      </button>
      <p className="hz-note">Simplified terrain-aware mass-movement visualization. Not a geotechnical prediction.</p>
    </div>
  )
}
