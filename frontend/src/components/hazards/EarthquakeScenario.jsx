import { useEffect, useState } from 'react'
import { useApp } from '../../store/useAppStore'

/**
 * Earthquake is a restrained procedural visualization driven entirely in
 * the frontend — there is no seismic forecast model behind it, and the UI
 * says so. No backend call is made.
 */
export default function EarthquakeScenario() {
  const quake = useApp((s) => s.hazard.quake)
  const setHazard = useApp((s) => s.setHazard)
  const [intensity, setIntensity] = useState(0.6)
  const [aftershocks, setAftershocks] = useState(true)

  const active = !!quake

  // The visualization ends on its own; release the trigger state then.
  useEffect(() => {
    if (!quake) return
    const ms = quake.aftershocks !== false ? 13500 : 6500
    const id = setTimeout(() => {
      const cur = useApp.getState().hazard.quake
      if (cur && cur.startedAt === quake.startedAt) useApp.getState().setHazard({ quake: null })
    }, ms)
    return () => clearTimeout(id)
  }, [quake])

  return (
    <div>
      <label className="hz-field">Shaking intensity
        <input
          type="range" min={0.2} max={1} step={0.05}
          value={intensity}
          disabled={active}
          onChange={(e) => setIntensity(Number(e.target.value))}
          aria-label="Shaking intensity"
        />
        <span className="hz-value">{Math.round(intensity * 100)}%</span>
      </label>
      <label className="hz-layers" style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
        <input
          type="checkbox"
          checked={aftershocks}
          disabled={active}
          onChange={(e) => setAftershocks(e.target.checked)}
          aria-label="Include aftershock sequence"
        />
        <span>Aftershock sequence <span className="hz-dim">two decaying events</span></span>
      </label>
      <button
        type="button"
        className="hz-run"
        disabled={active}
        onClick={() => setHazard({ quake: { intensity, aftershocks, startedAt: Date.now() } })}
      >
        {active ? 'Shaking…' : 'Trigger earthquake'}
      </button>
      {active && (
        <button type="button" className="hz-ghost" onClick={() => setHazard({ quake: null })}>
          Stop shaking
        </button>
      )}
      <p className="hz-note">
        Procedural camera/terrain vibration with dust — a visualization aid, not a seismic
        forecast. Peak motion stays small so the scene remains readable.
      </p>
    </div>
  )
}
