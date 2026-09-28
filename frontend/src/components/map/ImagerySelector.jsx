import { useEffect, useState } from 'react'
import { api } from '../../api/client'
import { useApp } from '../../store/useAppStore'

function fmtDate(iso) {
  if (!iso) return 'date unknown'
  const d = new Date(iso)
  return isNaN(d) ? 'date unknown' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

export default function ImagerySelector() {
  const aoi = useApp((s) => s.map.aoi)
  const imagery = useApp((s) => s.map.imagery)
  const setMap = useApp((s) => s.setMap)
  const [providers, setProviders] = useState([])
  const [provider, setProvider] = useState('auto')
  const [maxCloud, setMaxCloud] = useState(20)
  const [quality, setQuality] = useState('medium')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.imageryProviders().then((r) => setProviders(r.providers || [])).catch(() => setProviders([]))
  }, [])

  useEffect(() => {
    if (!aoi) { setMap({ imagery: null }); return }
    let live = true
    setBusy(true)
    setError(null)
    api.imagerySearch(aoi, { provider, maxCloud })
      .then((r) => {
        if (!live) return
        const items = r.items || []
        const best = items.reduce((b, it) => (!b || (it.cloud_cover ?? 101) < (b.cloud_cover ?? 101) ? it : b), null)
        setMap({ imagery: {
          provider: r.provider, attribution: r.attribution, resolution: r.resolution_m,
          items, selectedId: best ? best.item_id : null, maxCloud, quality,
        } })
      })
      .catch((e) => { if (live) { setError(`Imagery search failed — ${e.message}`); setMap({ imagery: null }) } })
      .finally(() => { if (live) setBusy(false) })
    return () => { live = false }
  }, [aoi, provider, maxCloud]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (imagery) setMap({ imagery: { ...imagery, quality } })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [quality])

  if (!aoi) return null
  const items = imagery?.items || []
  const sel = items.find((i) => i.item_id === imagery.selectedId)
  // Lowest-cloud candidate, computed from the actual search results — the
  // same item the auto-pick chose. Labels the recommendation honestly.
  const bestCloud = items.reduce((m, i) => Math.min(m, i.cloud_cover ?? 101), 101)

  return (
    <div className="map-panel-block">
      <h3>Reconstruction imagery</h3>
      <p className="muted small">Display map ≠ reconstruction image. DepthWizard runs on the source below.</p>
      <label className="map-field">Source
        <select value={provider} onChange={(e) => setProvider(e.target.value)}>
          <option value="auto">Auto (best available)</option>
          {providers.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}{p.synthetic ? ' (test pattern)' : ''} · {p.resolution_m} m
            </option>
          ))}
        </select>
      </label>
      <label className="map-field">Max cloud cover
        <select value={maxCloud} onChange={(e) => setMaxCloud(Number(e.target.value))}>
          <option value={10}>10%</option>
          <option value={20}>20%</option>
          <option value={50}>50%</option>
          <option value={100}>Any</option>
        </select>
      </label>
      <label className="map-field">Quality
        <select value={quality} onChange={(e) => setQuality(e.target.value)}>
          <option value="low">Low (faster)</option>
          <option value="medium">Medium</option>
          <option value="high">High (slower)</option>
        </select>
      </label>
      {busy && <p className="muted">Finding imagery…</p>}
      {error && <div className="map-error">{error}</div>}
      {imagery && !busy && !error && (
        <>
          {items.length > 1 && (
            <p className="muted small">{items.length} scenes cover this area — pick the source to reconstruct from.</p>
          )}
          <div className="scene-list" role="radiogroup" aria-label="Candidate imagery scenes">
            {items.map((it) => {
              const active = it.item_id === imagery.selectedId
              const cloud = it.cloud_cover
              const isBest = items.length > 1 && (cloud ?? 101) <= bestCloud && bestCloud <= 100
              return (
                <button
                  key={it.item_id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  className={`scene-card${active ? ' active' : ''}`}
                  onClick={() => setMap({ imagery: { ...imagery, selectedId: it.item_id } })}
                >
                  <span className="scene-top">
                    <span className="scene-date">{fmtDate(it.acquisition_datetime)}</span>
                    <span className="scene-res">{it.resolution_m ?? imagery.resolution ?? '?'} m</span>
                  </span>
                  <span className="scene-prov">
                    {imagery.provider}{it.synthetic ? ' · test pattern' : ''} · {(it.bands || []).join(', ')}
                  </span>
                  <span className="scene-cloud">
                    <span className="cloud-meter" aria-hidden="true">
                      <span style={{ width: cloud != null ? `${Math.min(100, Math.max(0, cloud))}%` : '0%' }} />
                    </span>
                    <span className="mono tiny">{cloud != null ? `${Number(cloud).toFixed(1)}% cloud` : 'cloud n/a'}</span>
                    {isBest && <span className="scene-best">Lowest cloud</span>}
                    {it.synthetic && <span className="scene-best synth">Synthetic</span>}
                  </span>
                  <span className="scene-id mono tiny">{it.item_id}</span>
                </button>
              )
            })}
          </div>
          {sel ? (
            <p className="muted small">
              Selected scene covers the AOI — the rectangle on the map turns teal when a scene is locked in.
            </p>
          ) : (
            <div className="map-error">No suitable imagery was found for this area.</div>
          )}
          {(sel?.resolution_m ?? imagery.resolution) > 5 && sel && (
            <div className="map-warn">This source has coarse spatial resolution. Fine building geometry may not be recoverable.</div>
          )}
          {imagery.attribution && <p className="muted tiny">{imagery.attribution}</p>}
        </>
      )}
    </div>
  )
}
