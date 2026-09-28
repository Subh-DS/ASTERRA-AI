import { useMemo, useRef, useState } from 'react'
import { useApp } from '../store/useAppStore'
import { inspectFile, fmtBytes } from '../utils/inspect'
import { startJob } from '../api/startJob'

function gcpIsValid(g) {
  if (!g) return false
  const lat = parseFloat(g.lat)
  const lon = parseFloat(g.lon)
  const elev = parseFloat(g.elev)
  if (!isFinite(lat) || !isFinite(lon) || !isFinite(elev)) return false
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return false
  return true
}

export default function Uploader() {
  const upload = useApp((s) => s.upload)
  const setUploaderFile = useApp((s) => s.setUploaderFile)
  const gcps = useApp((s) => s.gcps)
  const addGcp = useApp((s) => s.addGcp)
  const updateGcp = useApp((s) => s.updateGcp)
  const removeGcp = useApp((s) => s.removeGcp)
  const setScreen = useApp((s) => s.setScreen)
  const [drag, setDrag] = useState(false)
  const [showGcp, setShowGcp] = useState(false)
  const [dismissed, setDismissed] = useState(false)
  const inputRef = useRef(null)

  const info = upload.fileInfo
  const hasFile = !!info
  const geo = info?.hasCRS

  const validGcpCount = useMemo(() => gcps.filter(gcpIsValid).length, [gcps])
  // GEO inputs are always ready (GCPs optional refinement); non-georeferenced
  // inputs need the explicit Relative Surface decision — backend can only map
  // GCPs onto CRS-tagged rasters.
  const gcpReady = geo || dismissed

  const accept = async (file, cx, cy) => {
    if (!file) return
    setDismissed(false)
    setShowGcp(false)
    useApp.getState().fieldRef?.triggerLift(cx ?? window.innerWidth / 3, cy ?? window.innerHeight / 2)
    const parsed = await inspectFile(file)
    setUploaderFile(parsed, parsed.kind === 'TIFF' ? null : URL.createObjectURL(file), file)
  }

  const begin = () => {
    if (!hasFile) return
    if (!gcpReady) return
    startJob({ fileInfo: info, previewUrl: upload.previewUrl })
  }

  return (
    <section className="screen">
      <div className="screen-head">
        <span className="step">01</span>
        <h2>Ingest</h2>
        <button className="back-link" onClick={() => setScreen('hero')}>
          ← back
        </button>
      </div>
      <div className="upload-grid">
        <div>
          <div
            className={`dropzone${drag ? ' drag' : ''}${hasFile && upload.previewUrl && upload.previewUrl !== 'sample' ? ' has-file' : ''}`}
            role="button"
            tabIndex={0}
            aria-label="Image drop zone. Drop or click to browse."
            onClick={() => !upload.previewUrl && inputRef.current?.click()}
            onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setDrag(true)
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDrag(false)
              accept(e.dataTransfer.files?.[0], e.clientX, e.clientY)
            }}
          >
            {upload.previewUrl && upload.previewUrl !== 'sample' && (
              <img src={upload.previewUrl} alt="Uploaded aerial image preview" className="preview" />
            )}
            {(!upload.previewUrl || upload.previewUrl === 'sample') && (
              <>
                <span className="dz-label">{!hasFile ? 'Drop an overhead image here' : upload.previewUrl === 'sample' ? 'Sample scene loaded' : 'TIFF staged — browsers cannot preview TIFF; backend view appears during processing'}</span>
                <span className="dz-formats">PNG · JPG · TIFF · GEOTIFF</span>
              </>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,image/tiff,.tif,.tiff"
            className="sr-only"
            onChange={(e) => accept(e.target.files?.[0])}
          />
        </div>

        <div>
          {!hasFile && (
            <p className="empty-note">File metadata will appear here the moment an image is dropped — dimensions,
            format and georeferencing are parsed locally before anything is sent.</p>
          )}
          {hasFile && (
            <>
              <dl className="meta-list">
                <dt>File</dt>
                <dd>{info.name}</dd>
                <dt>Format</dt>
                <dd>{info.kind}</dd>
                <dt>Size</dt>
                <dd>{fmtBytes(info.size)}</dd>
                <dt>Dimensions</dt>
                <dd>{info.width ? `${info.width} × ${info.height} px` : 'unavailable'}</dd>
                <dt>CRS</dt>
                <dd style={{ color: geo ? 'var(--signal)' : 'var(--elevation)' }}>
                  {geo ? 'georeferenced' : 'none detected'}
                </dd>
              </dl>

              <div className={`branch-copy${geo ? '' : ' rel'}`}>
                <span className="tag mono">{geo ? 'GEO' : 'REL'}</span>
                {geo ? (
                  <div>
                    <p>CRS detected — metric DSM will be attempted, scale-anchored to a SRTM-derived Terrarium reference. Vertical reference: EGM96. If no reference DEM is reachable the job honestly falls back to a relative surface.</p>
                    <div style={{ marginTop: 10, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                      <button
                        className={`btn${showGcp ? ' primary' : ''}`}
                        onClick={() => setShowGcp((v) => !v)}
                      >
                        {showGcp ? 'Hide GCP form' : 'Add GCPs to refine calibration'}
                      </button>
                    </div>
                    <p className="mono" style={{ fontSize: 'var(--t-12)', color: 'var(--paper-dim)', marginTop: 8 }}>
                      Optional: ≥3 valid GCPs override the DEM fit with a GCP affine calibration.
                    </p>
                  </div>
                ) : (
                  <div>
                    <p>No CRS found — this input can only produce a Relative Surface.</p>
                    <div style={{ marginTop: 10, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                      <button
                        className="btn ghost"
                        onClick={() => {
                          setShowGcp(false)
                          setDismissed(true)
                        }}
                      >
                        Generate Relative Surface
                      </button>
                    </div>
                    {dismissed && !showGcp && (
                      <p className="mono" style={{ fontSize: 'var(--t-12)', color: 'var(--paper-dim)', marginTop: 8 }}>
                        Relative surface mode — output values are NOT metric meters. Metric calibration needs a georeferenced GeoTIFF (ground control points can only be mapped onto CRS-tagged rasters).
                      </p>
                    )}
                  </div>
                )}
              </div>

              {showGcp && geo && (
                <div className="gcp-block">
                  <div className="gcp-head">
                    <span>Ground control points</span>
                    <span className="count mono">
                      {validGcpCount} valid / {gcps.length} entered
                    </span>
                    <span className="mono" style={{ color: validGcpCount >= 3 ? 'var(--signal)' : 'var(--elevation)' }}>
                      {validGcpCount >= 3
                        ? '≥3 valid GCPs — GCP calibration will override the DEM fit'
                        : `Need ${Math.max(0, 3 - validGcpCount)} more valid GCP(s) for GCP refinement`}
                    </span>
                  </div>
                  <table className="gcp-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Latitude</th>
                        <th>Longitude</th>
                        <th>Elevation (m)</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {gcps.map((g) => (
                        <tr key={g.id}>
                          <td>{String(g.id).padStart(2, '0')}</td>
                          {['lat', 'lon', 'elev'].map((k) => (
                            <td key={k}>
                              <input
                                value={g[k]}
                                inputMode="decimal"
                                onChange={(e) => updateGcp(g.id, k, e.target.value)}
                                aria-label={`${k} for point ${g.id}`}
                              />
                            </td>
                          ))}
                          <td>
                            <button className="btn ghost" onClick={() => removeGcp(g.id)} aria-label={`Remove point ${g.id}`}>
                              ✕
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="gcp-actions">
                    <button className="btn" onClick={addGcp}>
                      + Add point
                    </button>
                  </div>
                  <p className="mono" style={{ fontSize: 'var(--t-12)', color: 'var(--paper-dim)', marginTop: 8 }}>
                    Each GCP must have a valid latitude (±90), longitude (±180), and elevation in meters. Fewer than 3 valid
                    GCPs means the DEM-anchored calibration is used instead.
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>
      <div className="upload-footer">
        <button
          className="btn primary"
          disabled={!hasFile || !gcpReady}
          onClick={begin}
          title={!gcpReady ? 'Choose Generate Relative Surface above — metric calibration needs a georeferenced image' : undefined}
        >
          Begin processing →
        </button>
        <span className="mono" style={{ fontSize: 'var(--t-12)', color: 'var(--paper-dim)', letterSpacing: '0.08em' }}>
          depth inference runs on the connected backend when available, otherwise fully offline in demo mode
        </span>
      </div>
    </section>
  )
}
