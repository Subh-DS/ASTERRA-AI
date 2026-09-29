import { useEffect, useState } from 'react'
import { useApp } from '../store/useAppStore'
import { API_URL } from '../api/client'
import { retryJob } from '../api/startJob'
import { aoiMetrics, formatArea } from '../utils/aoi'

const STATIONS = [
  { key: 'ingest', label: 'INGEST', idx: '01', desc: 'Validates the upload — parses raster CRS, transform and nodata.' },
  { key: 'depth', label: 'DEPTH', idx: '02', desc: 'Depth Anything V2 inference over feather-stitched tiles.' },
  { key: 'calibrate', label: 'CALIBRATE', idx: '03', desc: 'Affine Z = a·d + b fit against Terrarium DEM or GCP control.' },
  { key: 'mesh', label: 'MESH', idx: '04', desc: 'Composes the DSM grid, texture, normals and flyable mesh.' },
]

const STATUS_WORD = { queued: 'Queued', active: 'Active', done: 'Done' }

// Elapsed is always currentTime - startTime with a validated stamp. A
// missing stamp renders as "Initializing"; a clock that ever runs backward
// renders as "—". A negative duration is never displayed.
function elapsedText(startedAt, elapsed) {
  if (startedAt == null || typeof startedAt !== 'number' || !Number.isFinite(startedAt) || startedAt <= 0) {
    return 'Initializing'
  }
  if (!Number.isFinite(elapsed) || elapsed < 0) return '—'
  return `${elapsed.toFixed(1)}s`
}

export default function JobProgress() {
  const job = useApp((s) => s.job)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    // Anchor the clock to a validated stamp; fall back to mount time so the
    // first paint never derives from an unknown origin.
    const t0 =
      typeof job.startedAt === 'number' && Number.isFinite(job.startedAt) && job.startedAt > 0
        ? job.startedAt
        : performance.now()
    setElapsed(0)
    const int = setInterval(() => setElapsed(Math.max(0, (performance.now() - t0) / 1000)), 100)
    return () => clearInterval(int)
  }, [job.startedAt, job.id])

  const fileInfo = useApp.getState().upload.fileInfo
  const mapJob = useApp((s) => s.mapJob)
  const imagery = useApp((s) => s.map.imagery)
  const hasCRS = !!fileInfo?.hasCRS
  const scaleLabel = hasCRS ? 'ABSOLUTE (DSM)' : 'RELATIVE (rDSM)'
  // Continuity: the back target follows the path that started this job, and
  // the subject line keeps WHAT is being reconstructed visible.
  const backTarget = mapJob ? 'map' : 'upload'
  const sel = imagery?.items?.find((i) => i.item_id === imagery.selectedId) || null
  let subject = null
  if (mapJob?.aoi) {
    const m = aoiMetrics(mapJob.aoi.north, mapJob.aoi.south, mapJob.aoi.east, mapJob.aoi.west)
    subject = `${m.centerLat.toFixed(4)}, ${m.centerLon.toFixed(4)} · ${formatArea(m.areaKm2)}${sel?.acquisition_datetime ? ` · ${new Date(sel.acquisition_datetime).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })}` : ''}`
  } else if (fileInfo?.name) {
    subject = `${fileInfo.name} · ${hasCRS ? 'georeferenced' : 'relative surface'}`
  }
  const engine = useApp((s) => s.backendEngine)
  const showBackendTexture =
    !job.demo && job.id && (job.stages.mesh.state === 'active' || job.stages.mesh.state === 'done')
  const textureUrl = showBackendTexture
    ? `${API_URL}/files/${job.id}/texture.jpg${job.fileToken ? `?token=${encodeURIComponent(job.fileToken)}` : ''}`
    : null
  const modeLabel = job.demo
    ? 'DEMO · OFFLINE'
    : engine?.device === 'cuda'
      ? 'BACKEND · CUDA'
      : engine?.device === 'cpu'
        ? 'BACKEND · CPU'
        : 'BACKEND'

  return (
    <section className="screen">
      <div className="screen-head">
        <span className="step">02</span>
        <h2>Pipeline</h2>
        <button className="back-link" onClick={() => useApp.getState().setScreen(backTarget)}>
          ← back
        </button>
      </div>
      {subject && (
        <p className="mono" style={{ fontSize: 'var(--t-12)', color: 'var(--paper-dim)', letterSpacing: '0.06em', marginBottom: 16 }}>
          Reconstructing {subject} → Twin next
        </p>
      )}
      <div className="progress-stage">
        <div className="rail" role="status" aria-label="Reconstruction pipeline stages">
          {STATIONS.map((st) => {
            const s = job.stages[st.key]
            const state = s.state === 'active' || s.state === 'done' ? s.state : 'queued'
            return (
              <div key={st.key} className={`station ${state}`} data-state={state}>
                <span className="idx mono">{st.idx}</span>
                <span className="dot" aria-hidden="true" />
                <h3>{st.label}</h3>
                <div className="status mono">{STATUS_WORD[state]}</div>
                <p className="desc">{st.desc}</p>
                <div className="bar" aria-hidden="true">
                  <i className={state === 'active' ? 'flow' : ''} />
                </div>
                <div className="sub">{state === 'queued' ? '' : s.sub}</div>
              </div>
            )
          })}
        </div>
        <div className="progress-meta">
          <span>
            ELAPSED <b className="mono">{elapsedText(job.startedAt, elapsed)}</b>
          </span>
          <span>
            MODE{' '}
            <b>{modeLabel}</b>
          </span>
          <span>
            SCALE <b>{scaleLabel}</b>
          </span>
        </div>
        {textureUrl && (
          <div className="progress-preview">
            <img
              key={job.stages.mesh.state}
              src={textureUrl}
              alt="Processing input as seen by the backend"
              className="preview"
              onLoad={(e) => { e.currentTarget.style.opacity = '1' }}
              onError={(e) => {
                e.currentTarget.style.display = 'none'
              }}
              style={{ opacity: 0, transition: 'opacity 300ms ease' }}
            />
            <span className="mono">backend view · texture staged for 3D mesh</span>
          </div>
        )}
        {job.error && (
          <div className="job-error">
            <strong className="mono" style={{ color: 'var(--alert)' }}>
              ERROR ·{' '}
            </strong>
            {job.error}
            {!job.demo && job.id && (
              <div style={{ marginTop: 8 }}>
                <button className="btn" onClick={() => retryJob(job.id)}>
                  Retry without re-upload →
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  )
}
