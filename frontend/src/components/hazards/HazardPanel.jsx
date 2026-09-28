import { useApp } from '../../store/useAppStore'
import { DISCLAIMER_SHORT } from '../../utils/hazard'
import CoastalScenario from './CoastalScenario'
import EarthquakeScenario from './EarthquakeScenario'
import EvacuationScenario from './EvacuationScenario'
import HazardLayerControls from './HazardLayerControls'
import HazardLegend from './HazardLegend'
import HazardSelector from './HazardSelector'
import HazardStats from './HazardStats'
import MissionTimeline from './MissionTimeline'
import LandslideScenario from './LandslideScenario'

export default function HazardPanel() {
  const hazard = useApp((s) => s.hazard)
  const setHazard = useApp((s) => s.setHazard)
  const resetHazard = useApp((s) => s.resetHazard)
  if (!hazard.panelOpen) return null
  const coastal = hazard.type === 'coastal_inundation'
  const quakeMode = hazard.type === 'earthquake'
  const evacMode = hazard.type === 'evacuation'

  return (
    <aside className="hz-panel" aria-label="Hazard simulation">
      <header>
        <h3>Hazard simulation</h3>
        <button type="button" className="hz-x" onClick={() => setHazard({ panelOpen: false })} aria-label="Close hazard panel">✕</button>
      </header>
      <HazardSelector />
      {evacMode ? (
        <EvacuationScenario />
      ) : quakeMode ? (
        <EarthquakeScenario />
      ) : coastal ? (
        <CoastalScenario />
      ) : (
        <LandslideScenario />
      )}
      <label className="hz-layers" style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
        <input
          type="checkbox"
          checked={!!hazard.storm}
          onChange={(e) => setHazard({ storm: e.target.checked })}
          aria-label="Toggle storm visualization"
        />
        <span>Storm <span className="hz-dim">visualization — rain, haze, dimmed light</span></span>
      </label>
      {hazard.error && <div className="hz-error" role="alert">{hazard.error}</div>}
      {hazard.result && (
        <>
          <div className="hz-viewmode" role="group" aria-label="Before and after comparison">
            <button type="button" className={!hazard.showSimulated ? 'on' : ''} onClick={() => setHazard({ showSimulated: false })}>Before</button>
            <button type="button" className={hazard.showSimulated ? 'on' : ''} onClick={() => setHazard({ showSimulated: true })}>Simulated</button>
          </div>
          <HazardStats />
          <HazardLayerControls />
          <HazardLegend />
        </>
      )}
      <MissionTimeline />
      <button type="button" className="hz-ghost" onClick={resetHazard}>Reset simulation</button>
      <p className="hz-disc">{DISCLAIMER_SHORT}</p>
      <p className="hz-disc">{evacMode
        ? 'Simulated A* routes over the measured DSM. Planning aid — not a verified safe path.'
        : quakeMode
          ? 'Procedural shaking visualization. Not a seismic forecast or engineering prediction.'
          : coastal
            ? 'Hypothetical water-elevation inundation scenario. Local vertical datum and hydrodynamic effects may limit interpretation.'
            : 'Simplified terrain-aware mass-movement visualization. Not a geotechnical prediction.'}</p>
    </aside>
  )
}
