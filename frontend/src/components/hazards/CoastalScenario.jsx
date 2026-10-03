import { useState } from 'react'
import { useApp } from '../../store/useAppStore'
import { clampWaterLevel, hazardApi } from '../../utils/hazard'

const PRESETS = [1, 2, 3, 5]

export default function CoastalScenario({ onDone }) {
  const dsm = useApp((s) => s.dsm)
  const status = useApp((s) => s.hazard.status)
  const storm = useApp((s) => s.hazard.storm)
  const setHazard = useApp((s) => s.setHazard)
  const setScreen = useApp((s) => s.setScreen)
  const setMap = useApp((s) => s.setMap)
  const maxElev = dsm?.stats?.max
  const [wl, setWl] = useState(3.0)
  const busy = status === 'running'

  async function run(level) {
    const clamped = clampWaterLevel(level ?? wl, maxElev)
    if (clamped == null) {
      setHazard({ status: 'error', error: 'Enter a water level between 0.1 and 10 m.' })
      return
    }
    setHazard({ status: 'running', error: null })
    try {
      const res = await hazardApi.simulate(dsm.id, 'coastal_inundation', {
        water_level_m: clamped,
        // Hydrological connectivity is computed in the backend:
        // water can only reach cells connected to the coast through
        // a continuous path of cells below water level
      })
      setHazard({ status: 'complete', result: res, animT: 1, showSimulated: true, error: null })
      onDone?.()
    } catch (err) {
      setHazard({ status: 'error', error: err.message })
    }
  }

  function useMumbai() {
    setMap({ center: [72.8777, 18.922], zoom: 12, basemap: 'satellite' })
    setScreen('map')
  }

  return (
    <div>
      <div className="hz-row">
        <span className="hz-label">Location</span>
        <span>Mumbai <button type="button" className="hz-link" onClick={useMumbai}>open demo area on map</button></span>
      </div>
      <label className="hz-field">Water level (+m)
        <input
          type="range" min={0.1} max={Math.min(10, Math.max(0.1, maxElev ?? 10))} step={0.1}
          value={Math.min(wl, Math.min(10, Math.max(0.1, maxElev ?? 10)))}
          onChange={(e) => setWl(Number(e.target.value))}
          aria-label="Water level in meters"
        />
        <span className="hz-value">+{Number(wl).toFixed(1)} m</span>
      </label>
      <div className="hz-presets" role="group" aria-label="Water level presets">
        {PRESETS.map((p) => (
          <button key={p} type="button" disabled={busy} onClick={() => { setWl(p); run(p) }}>+{p}m</button>
        ))}
      </div>
      <button type="button" className="hz-run" disabled={busy} onClick={() => run()}>
        {busy ? 'Simulating…' : 'Run simulation'}
      </button>
      <p className="hz-note">Hypothetical water-elevation scenario. Vertical datum: {dsm?.metadata?.vertical_reference ?? 'assumed / unknown'}.</p>
      {storm && (
        <p className="hz-note" role="note">
          Storm cover is active — field conditions would keep pushing water levels up, but this
          scenario stays fixed at the level above unless you re-run it.
        </p>
      )}
    </div>
  )
}
