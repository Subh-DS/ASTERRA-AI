import { useApp } from '../../store/useAppStore'

const LAYER_LABELS = { water: 'Water', shoreline: 'Shoreline', scar: 'Scar', debris: 'Debris mass', deposition: 'Deposition', particles: 'Debris particles' }

export default function HazardLayerControls() {
  const result = useApp((s) => s.hazard.result)
  const layers = useApp((s) => s.hazard.layers)
  const setHazardLayers = useApp((s) => s.setHazardLayers)
  if (!result) return null
  const coastal = result.simulation_type === 'coastal_inundation'
  const keys = coastal ? ['water', 'shoreline'] : ['scar', 'debris', 'deposition', 'particles']
  return (
    <div className="hz-layers">
      <h4>Layers</h4>
      {keys.map((k) => (
        <label key={k}>
          <input type="checkbox" checked={!!layers[k]} onChange={(e) => setHazardLayers({ [k]: e.target.checked })} />
          {LAYER_LABELS[k]}
        </label>
      ))}
    </div>
  )
}
