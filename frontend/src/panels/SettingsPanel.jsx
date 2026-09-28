import { useApp } from '../store/useAppStore'
import { TIER_CAPS } from '../quality/detectTier'
import { buildGLB, buildOBJ, downloadBlob, heightmapPNG } from '../utils/exporters'
import { api } from '../api/client'

const LEVELS = ['auto', 'high', 'medium', 'low']

export default function SettingsPanel() {
  const tier = useApp((s) => s.tier)
  const setTier = useApp((s) => s.setTier)
  const reduced = useApp((s) => s.reducedMotion)
  const dsm = useApp((s) => s.dsm)
  const theme = useApp((s) => s.theme)
  const setThemeStore = useApp((s) => s.setTheme)

  const current = tier.override || 'auto'

  const exportObj = () => {
    if (!dsm) return
    const obj = buildOBJ(dsm.heights, dsm.width, dsm.height, useApp.getState().viewer.zScale, Math.max(2, Math.floor(Math.max(dsm.width, dsm.height) / 220)))
    downloadBlob(`depthwizard_${dsm.id}.obj`, new Blob([obj], { type: 'text/plain' }))
  }

  const exportGlbLocal = async () => {
    if (!dsm) return
    try {
      const blob = await buildGLB(
        dsm.heights,
        dsm.width,
        dsm.height,
        useApp.getState().viewer.zScale,
        dsm.textureCanvas || dsm.textureSrc || null,
      )
      downloadBlob(`depthwizard_${dsm.id}.glb`, blob)
    } catch (err) {
      useApp.getState().setBanner(`Local 3D export failed — ${err?.message || 'unknown error'}`)
      setTimeout(() => useApp.getState().setBanner(null), 6000)
    }
  }

  const exportHeightmap = async () => {
    if (!dsm) return
    const blob = await heightmapPNG(dsm.heights, dsm.width, dsm.height)
    downloadBlob(`depthwizard_${dsm.id}_heightmap.png`, blob)
  }

  const exportStats = () => {
    if (!dsm) return
    downloadBlob(
      `depthwizard_${dsm.id}_stats.json`,
      new Blob([JSON.stringify({ job: dsm.id, crs: dsm.crs, landscape: dsm.landscape, ...dsm.stats }, null, 2)], {
        type: 'application/json',
      }),
    )
  }

  const dsmMeta = dsm?.metadata || null
  const isMetric = dsm ? (dsmMeta ? !!dsmMeta.is_metric : !!dsm.crs) : false
  const elevKind = isMetric ? 'Metric DSM' : 'Relative Surface Model'
  const elevSubLabel = isMetric
    ? `DSM · ${dsm.crs || dsmMeta?.crs || ''}`
    : 'rDSM · relative depth (0–100) · not metric'

  return (
    <>
      <div className="panel-head">
        <h2>Settings</h2>
      </div>

      <div className="panel-section">
        <h4>Appearance</h4>
        <div className="seg" role="radiogroup" aria-label="Color theme">
          {['dark', 'light'].map((t) => (
            <button key={t} className={theme === t ? 'on' : ''} onClick={() => setThemeStore(t)}>
              {t.toUpperCase()}
            </button>
          ))}
        </div>
        <p className="mono" style={{ fontSize: 'var(--t-12)', color: 'var(--paper-dim)', marginTop: 10 }}>
          daylight topo-paper vs surveyed-green night · viewer atmosphere follows
        </p>
      </div>

      <div className="panel-section">
        <h4>Rendering quality</h4>
        <div className="seg" role="radiogroup" aria-label="Rendering quality">
          {LEVELS.map((lv) => (
            <button
              key={lv}
              className={current === lv ? 'on' : ''}
              onClick={() =>
                setTier({
                  ...tier,
                  override: lv === 'auto' ? null : lv,
                  resolved: lv === 'auto' ? tier.auto : lv,
                })
              }
            >
              {lv.toUpperCase()}
            </button>
          ))}
        </div>
        <p className="mono" style={{ fontSize: 'var(--t-12)', color: 'var(--paper-dim)', marginTop: 10 }}>
          detected: {tier.auto} · {tier.reason}
          <br />
          active: {tier.resolved} · shadows {TIER_CAPS[tier.resolved].shadows ? 'on' : 'off'} ·{' '}
          {TIER_CAPS[tier.resolved].meshSegments}px mesh
        </p>
        <p className="mono" style={{ fontSize: 'var(--t-12)', color: 'var(--paper-dim)', marginTop: 6 }}>
          Auto-demotion is disabled for stability. The viewer does not dispose on transient frame stalls.
        </p>
      </div>

      <div className="panel-section">
        <h4>Motion</h4>
        <p style={{ fontSize: 'var(--t-14)', color: 'var(--paper-dim)' }}>
          {reduced
            ? 'Reduced motion is active (system preference) — cinematic sequences render at their settled state.'
            : 'Full motion enabled. Cinematic reveals and camera transitions are active.'}
        </p>
      </div>

      <div className="panel-section">
        <h4>Export</h4>
        {!dsm && <p className="empty-note">Export unlocks once a DSM is computed.</p>}
        {dsm && (
          <>
            <div className="export-row">
              <div>
                <div className="what">{isMetric ? 'Elevation raster (metric)' : 'Elevation raster (relative)'}</div>
                <div className="detail mono">{elevSubLabel}</div>
              </div>
              {isMetric && dsm.crs && !String(dsm.id).startsWith('DEMO') ? (
                <a className="btn" href={api.exportUrl(dsm.id, 'dsm.tif', dsm.fileToken)} download>
                  Download .tif
                </a>
              ) : (
                <button className="btn" onClick={exportHeightmap}>
                  Heightmap .png
                </button>
              )}
            </div>
            <div className="export-row">
              <div>
                <div className="what">3D Model (GLB)</div>
                <div className="detail mono">Textured 3D binary glTF</div>
              </div>
              {dsm.offline || String(dsm.id).startsWith('OFFLINE') ? (
                <button className="btn primary" onClick={exportGlbLocal}>
                  Build .glb locally
                </button>
              ) : (
                <a className="btn primary" href={api.exportUrl(dsm.id, 'model.glb', dsm.fileToken)} download>
                  Download .glb
                </a>
              )}
            </div>
            <div className="export-row">
              <div>
                <div className="what">Mesh</div>
                <div className="detail mono">OBJ · 3D geometry</div>
              </div>
              <button className="btn" onClick={exportObj}>
                Export .obj
              </button>
            </div>
            {!dsm.offline && !String(dsm.id).startsWith('OFFLINE') && (
              <div className="export-row">
                <div>
                  <div className="what">Point cloud</div>
                  <div className="detail mono">PLY · XYZ + RGB</div>
                </div>
                <a className="btn" href={api.exportUrl(dsm.id, 'model.ply', dsm.fileToken)} download>
                  Download .ply
                </a>
              </div>
            )}
            {dsm.reconstruction?.has_geojson && (
              <div className="export-row">
                <div>
                  <div className="what">Building footprints</div>
                  <div className="detail mono">GeoJSON · CRS polygons + heights</div>
                </div>
                <a className="btn" href={api.exportUrl(dsm.id, 'buildings.geojson', dsm.fileToken)} download>
                  Download .geojson
                </a>
              </div>
            )}
            {dsm.reconstruction?.has_mask && (
              <div className="export-row">
                <div>
                  <div className="what">Semantic mask</div>
                  <div className="detail mono">PNG · land-cover classes</div>
                </div>
                <a className="btn" href={api.exportUrl(dsm.id, 'mask.png', dsm.fileToken)} download>
                  Download mask
                </a>
              </div>
            )}
            <div className="export-row">
              <div>
                <div className="what">Statistics</div>
                <div className="detail mono">JSON · min/max/mean</div>
              </div>
              <button className="btn" onClick={exportStats}>
                Export .json
              </button>
            </div>
            <p className="mono" style={{ fontSize: 'var(--t-12)', color: 'var(--paper-dim)', marginTop: 8 }}>
              {elevKind}{dsmMeta?.vertical_reference ? ` · Vertical reference: ${dsmMeta.vertical_reference}` : ''}
            </p>
          </>
        )}
      </div>
    </>
  )
}
