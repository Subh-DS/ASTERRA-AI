import { useEffect, useState } from 'react'
import { useApp } from '../store/useAppStore'

/**
 * Compact scientific processing indicator. The pipeline rail remains the
 * primary structure; this layer mirrors the live operation. Every value is
 * backend-true: stage states + sub-status from the job socket, engine
 * model/device from /api/health, elapsed from the job clock. No GPU
 * utilization, throughput, ETA, or accuracy is shown — the backend does
 * not report any of those.
 */
const STAGE_META = {
  ingest: { label: 'INGESTION', transform: 'SOURCE → VALIDATED RASTER' },
  depth: { label: 'DEPTH ESTIMATION', transform: 'RGB → RELATIVE DEPTH' },
  calibrate: { label: 'METRIC CALIBRATION', transform: 'RELATIVE → METRIC ELEVATION' },
  mesh: { label: 'MESH GENERATION', transform: 'ELEVATION → DSM + MESH' },
}

function engineLine(job, engine) {
  if (job.demo) return 'OFFLINE DEMO · CLIENT-SIDE'
  const model = engine?.model ? String(engine.model).split('/').pop() : 'DEPTH BACKBONE'
  const device = engine?.device === 'fallback'
    ? 'PSEUDO FALLBACK'
    : (engine?.device || 'BACKEND').toUpperCase()
  return `${model} · ${device}`
}

export default function LoadingOverlay() {
  const screen = useApp((s) => s.screen)
  const job = useApp((s) => s.job)
  const engine = useApp((s) => s.backendEngine)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (screen !== 'progress') return
    const t0 =
      typeof job.startedAt === 'number' && Number.isFinite(job.startedAt) && job.startedAt > 0
        ? job.startedAt
        : performance.now()
    setElapsed(0)
    const id = setInterval(() => setElapsed(Math.max(0, (performance.now() - t0) / 1000)), 200)
    return () => clearInterval(id)
  }, [screen, job.startedAt, job.id])

  if (screen !== 'progress') return null
  const stages = Object.entries(job.stages || {})
  const total = stages.length || 4
  const doneCount = stages.filter(([, s]) => s.state === 'done').length
  // The rail owns the complete state — the indicator steps aside.
  if (doneCount >= total) return null
  // Show a brief error state in the overlay before the rail takes over.
  if (job.error) {
    return (
      <div className="loading-overlay" role="alert">
        <div className="loading-content live">
          <div className="loading-top">
            <span className="loading-stage">ERROR</span>
          </div>
          <div className="loading-row">
            <span className="loading-sub">{job.error}</span>
          </div>
        </div>
      </div>
    )
  }

  const current = stages.find(([, s]) => s.state === 'active')
  const live = !!current
  const meta = (current && STAGE_META[current[0]]) || { label: 'QUEUED', transform: 'AWAITING BACKEND' }
  const sub = current?.[1]?.sub || 'Waiting for the backend…'
  const clockValid =
    typeof job.startedAt === 'number' && Number.isFinite(job.startedAt) && job.startedAt > 0
  const clock = !clockValid ? '—' : !Number.isFinite(elapsed) || elapsed < 0 ? '—' : `${elapsed.toFixed(1)}s`

  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      <div className={`loading-content${live ? ' live' : ''}`}>
        <div className="loading-top">
          <span className="loading-stage">{meta.label}</span>
          <span className="loading-elapsed mono">{clock}</span>
        </div>
        <div className="loading-engine mono">{engineLine(job, engine)}</div>
        <div className="loading-transform mono">{meta.transform}</div>
        <div className="loading-row">
          <span className="loading-key">Current stage</span>
          <span className="loading-sub">{sub}</span>
        </div>
        <div className="loading-row">
          <span className="loading-key">Progress</span>
          <span className="loading-bar" aria-hidden="true">
            <span className="loading-fill" style={{ width: `${(doneCount / total) * 100}%` }} />
          </span>
          <span className="loading-count mono">{doneCount}/{total}</span>
        </div>
      </div>
    </div>
  )
}
