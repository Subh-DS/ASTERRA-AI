import { useApp } from '../../store/useAppStore'

// Must match WATER_COLORS in three/hazardLayers.js.
const DEPTH_COLORS = ['#9ed9f2', '#59b3e6', '#3385d1', '#1f5cb3', '#123d8c']
const DEPTH_LABELS = ['0–0.5 m', '0.5–1 m', '1–2 m', '2–5 m', '>5 m']

export default function HazardLegend() {
  const result = useApp((s) => s.hazard.result)
  if (!result) return null
  if (result.simulation_type === 'coastal_inundation') {
    return (
      <div className="hz-legend">
        <h4>Water depth</h4>
        {DEPTH_LABELS.map((l, i) => (
          <div key={l}><span className="hz-chip" style={{ background: DEPTH_COLORS[i] }} />{l}</div>
        ))}
      </div>
    )
  }
  return (
    <div className="hz-legend">
      <h4>Landslide zones</h4>
      <div><span className="hz-chip" style={{ background: '#5c422b' }} />Source / scar</div>
      <div><span className="hz-chip" style={{ background: '#735637' }} />Debris in motion</div>
      <div><span className="hz-chip" style={{ background: '#8a6f4d' }} />Deposition</div>
    </div>
  )
}
