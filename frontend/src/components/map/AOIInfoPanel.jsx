import { AOI_LIMITS, aoiMetrics, formatArea, validateAoiClient } from '../../utils/aoi'
import { useApp } from '../../store/useAppStore'

export default function AOIInfoPanel() {
  const aoi = useApp((s) => s.map.aoi)
  const imagery = useApp((s) => s.map.imagery)
  const clearMapAoi = useApp((s) => s.clearMapAoi)

  if (!aoi) {
    return (
      <div className="map-panel-block aoi-empty">
        <h3>Selected area</h3>
        <p className="muted">No area selected. Click “Draw AOI” and drag a rectangle on the map.</p>
      </div>
    )
  }
  const err = validateAoiClient(aoi)
  const m = aoiMetrics(aoi.north, aoi.south, aoi.east, aoi.west)
  const tooBig =
    m.widthM > AOI_LIMITS.maxSideM ||
    m.heightM > AOI_LIMITS.maxSideM ||
    m.areaKm2 > AOI_LIMITS.maxAreaKm2
  const tooSmall = m.widthM < AOI_LIMITS.minSideM || m.heightM < AOI_LIMITS.minSideM
  const sel = imagery?.items?.find((i) => i.item_id === imagery.selectedId) || null
  const gsd = sel?.resolution_m ?? imagery?.resolution ?? null

  return (
    <div className="map-panel-block aoi-selected">
      <h3><span className="asterra-dot" aria-hidden="true" /> Area selected</h3>
      <p className="aoi-promise">This exact rectangle is what DepthWizard will reconstruct.</p>
      <dl className="map-facts">
        <div><dt>Center</dt><dd>{m.centerLat.toFixed(6)}, {m.centerLon.toFixed(6)}</dd></div>
        <div><dt>Width</dt><dd>{m.widthM.toLocaleString('en-US')} m</dd></div>
        <div><dt>Height</dt><dd>{m.heightM.toLocaleString('en-US')} m</dd></div>
        <div><dt>Area</dt><dd>{formatArea(m.areaKm2)}</dd></div>
        <div>
          <dt>Source GSD</dt>
          <dd>{gsd != null ? `≈ ${gsd} m/px` : 'resolving…'}</dd>
        </div>
      </dl>
      {err && <div className="map-error">{err}</div>}
      {tooBig && <div className="map-error">Selected area is too large. Reduce the area before reconstruction.</div>}
      {tooSmall && !tooBig && <div className="map-error">Selected area is too small — zoom in or draw a larger rectangle.</div>}
      <button type="button" className="map-ghost" onClick={clearMapAoi}>Clear selection</button>
    </div>
  )
}
