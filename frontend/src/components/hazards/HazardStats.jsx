import { useApp } from '../../store/useAppStore'

export default function HazardStats() {
  const result = useApp((s) => s.hazard.result)
  if (!result) return null
  const st = result.statistics
  if (result.simulation_type === 'coastal_inundation') {
    return (
      <dl className="hz-stats" aria-live="polite">
        <div><dt>Water level</dt><dd>+{st.water_level_m.toFixed(1)} m</dd></div>
        <div><dt>Inundated area</dt><dd>{st.inundated.display}</dd></div>
        <div><dt>Max depth</dt><dd>{st.max_depth_m.toFixed(1)} m</dd></div>
        <div><dt>Potentially affected buildings</dt><dd>{st.buildings.potentially_affected} <span className="hz-dim">of {st.buildings.total}</span></dd></div>
      </dl>
    )
  }
  return (
    <dl className="hz-stats" aria-live="polite">
      <div><dt>Severity</dt><dd>{st.severity[0].toUpperCase() + st.severity.slice(1)}</dd></div>
      <div><dt>Modeled affected area</dt><dd>{st.affected.display}</dd></div>
      <div><dt>Deposition area</dt><dd>{st.deposition.display}</dd></div>
      <div><dt>Potentially affected buildings</dt><dd>{st.buildings.potentially_affected} <span className="hz-dim">of {st.buildings.total}</span></dd></div>
      <div><dt>Source slope</dt><dd>{st.source_slope_deg.toFixed(1)}°</dd></div>
    </dl>
  )
}
