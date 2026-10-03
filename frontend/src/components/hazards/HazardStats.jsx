import { useApp } from '../../store/useAppStore'

export default function HazardStats() {
  const result = useApp((s) => s.hazard.result)
  if (!result) return null
  const st = result.statistics

  if (result.simulation_type === 'coastal_inundation') {
    return (
      <dl className="hz-stats" aria-live="polite">
        <div><dt>Water level</dt><dd>+{st.water_level_m.toFixed(1)} m</dd></div>
        <div><dt>Inundated cells</dt><dd>{st.inundated_cells?.toLocaleString()}</dd></div>
        <div><dt>Inundated area</dt><dd>{st.inundated_percentage?.toFixed(1)}%</dd></div>
        <div><dt>Max depth</dt><dd>{st.max_depth_m?.toFixed(1)} m</dd></div>
        <div><dt>Mean depth</dt><dd>{st.mean_depth_m?.toFixed(1)} m</dd></div>
      </dl>
    )
  }

  if (result.simulation_type === 'landslide_susceptibility') {
    return (
      <dl className="hz-stats" aria-live="polite">
        <div><dt>Mean susceptibility</dt><dd>{(st.mean_susceptibility * 100).toFixed(1)}%</dd></div>
        <div><dt>Max susceptibility</dt><dd>{(st.max_susceptibility * 100).toFixed(1)}%</dd></div>
        <div><dt>High susceptibility area</dt><dd>{st.high_susceptibility_percentage?.toFixed(1)}%</dd></div>
        <div><dt>Low susceptibility cells</dt><dd>{st.low_susceptibility_cells?.toLocaleString()}</dd></div>
        <div><dt>Moderate cells</dt><dd>{st.moderate_susceptibility_cells?.toLocaleString()}</dd></div>
        <div><dt>High cells</dt><dd>{st.high_susceptibility_cells?.toLocaleString()}</dd></div>
        <div><dt>Very high cells</dt><dd>{st.very_high_susceptibility_cells?.toLocaleString()}</dd></div>
        {st.factors && (
          <>
            <div><dt>Mean slope</dt><dd>{st.factors.slope_mean_deg}°</dd></div>
            <div><dt>Max slope</dt><dd>{st.factors.slope_max_deg}°</dd></div>
            <div><dt>Local relief</dt><dd>{st.factors.local_relief_mean_m} m</dd></div>
          </>
        )}
      </dl>
    )
  }

  return (
    <dl className="hz-stats" aria-live="polite">
      <div><dt>Source slope</dt><dd>{st.source_slope_deg?.toFixed(1)}°</dd></div>
      <div><dt>Elevation drop</dt><dd>{st.elevation_drop_m} m</dd></div>
      <div><dt>Path length</dt><dd>{st.path_length_cells} cells</dd></div>
      <div><dt>Affected cells</dt><dd>{st.affected_cells?.toLocaleString()}</dd></div>
    </dl>
  )
}
