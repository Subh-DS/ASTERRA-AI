import { useApp } from '../../store/useAppStore'

export default function HazardSelector() {
  const type = useApp((s) => s.hazard.type)
  const setHazard = useApp((s) => s.setHazard)
  return (
    <label className="hz-field">Scenario
      <select value={type} onChange={(e) => setHazard({ type: e.target.value })} aria-label="Hazard scenario">
        <option value="landslide_susceptibility">Landslide Susceptibility Map</option>
        <option value="coastal_inundation">Coastal Inundation</option>
        <option value="landslide">Landslide (single event)</option>
        <option value="earthquake">Earthquake (procedural)</option>
        <option value="evacuation">Evacuation routes</option>
      </select>
    </label>
  )
}
