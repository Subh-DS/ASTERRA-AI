import { useApp } from '../store/useAppStore'
import { aoiMetrics, formatArea } from '../utils/aoi'
import { dsmModeInfo } from '../utils/dsmMode'

function fmtDate(iso) {
  if (!iso) return null
  const d = new Date(iso)
  return isNaN(d) ? null : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

const f4 = (n) => (Number.isFinite(n) ? n.toFixed(4) : '—')

/**
 * Persistent mission strip: WHERE you are, WHAT you selected, WHAT is
 * happening, WHAT comes next. Rendered from live store state only —
 * every value traces to the AOI, the chosen imagery scene, the job, or
 * the loaded DSM. Never fabricated.
 */
export default function MissionBar() {
  const screen = useApp((s) => s.screen)
  const aoi = useApp((s) => s.map.aoi)
  const imagery = useApp((s) => s.map.imagery)
  const mapJob = useApp((s) => s.mapJob)
  const job = useApp((s) => s.job)
  const dsm = useApp((s) => s.dsm)
  const fileInfo = useApp((s) => s.upload.fileInfo)

  const sel = imagery?.items?.find((i) => i.item_id === imagery.selectedId) || null
  // The trail follows WHERE the user is; completion follows WHAT exists.
  const fromMap = screen === 'map'
    ? true
    : screen === 'upload'
      ? false
      : !!(aoi || mapJob || (dsm && dsm.source?.type === 'map'))
  const originLabel = fromMap ? 'Map' : 'Ingest'

  const originDone = fromMap
    ? !!((aoi && sel) || mapJob || (dsm && dsm.source?.type === 'map'))
    : !!((fileInfo && (fileInfo.hasCRS || fileInfo.name)) || job.id || job.startedAt || dsm)
  const pipeActive = !!(job.id || job.startedAt) && !dsm
  const pipeDone = !!dsm || Object.values(job.stages || {}).every((s) => s.state === 'done')
  const twinOn = !!dsm

  const stageIdx = screen === 'viewer' ? 2 : screen === 'progress' ? 1 : 0
  const states = [
    originDone || stageIdx > 0 ? 'done' : stageIdx === 0 ? 'now' : 'todo',
    pipeDone || stageIdx > 1 ? 'done' : stageIdx === 1 ? 'now' : 'todo',
    twinOn ? (stageIdx === 2 ? 'now' : 'done') : 'todo',
  ]
  const trail = [originLabel, 'Pipeline', 'Twin']

  let context = ''
  if (screen === 'map') {
    if (aoi) {
      const m = aoiMetrics(aoi.north, aoi.south, aoi.east, aoi.west)
      const where = `${f4(m.centerLat)}, ${f4(m.centerLon)} · ${formatArea(m.areaKm2)}`
      context = sel
        ? `${where} · ${imagery.provider || 'auto'} · ${sel.resolution_m ?? imagery.resolution ?? '?'} m${fmtDate(sel.acquisition_datetime) ? ` · ${fmtDate(sel.acquisition_datetime)}` : ''} → ready for 3D`
        : `${where} · finding imagery…`
    } else {
      context = 'Draw an area on the map to begin → imagery → 3D'
    }
  } else if (screen === 'upload') {
    context = fileInfo?.name
      ? `${fileInfo.name} · ${fileInfo.hasCRS ? 'georeferenced' : 'relative surface'} → pipeline → twin`
      : 'Drop an overhead image to begin → pipeline → twin'
  } else if (screen === 'progress') {
    const active = Object.entries(job.stages || {}).find(([, s]) => s.state === 'active')
    const doing = active ? `${active[0].toUpperCase()}${active[1].sub ? ` · ${active[1].sub}` : ''}` : 'QUEUED'
    if (job.error) {
      context = `Pipeline halted — ${job.error}`
    } else if (mapJob?.aoi) {
      const m = aoiMetrics(mapJob.aoi.north, mapJob.aoi.south, mapJob.aoi.east, mapJob.aoi.west)
      context = `Reconstructing ${f4(m.centerLat)}, ${f4(m.centerLon)} · ${formatArea(m.areaKm2)} · ${doing}`
    } else if (fileInfo?.name) {
      context = `Reconstructing ${fileInfo.name} · ${doing}`
    } else {
      context = `Reconstructing · ${doing}`
    }
  } else if (screen === 'viewer' && dsm) {
    const mode = dsmModeInfo(dsm)
    const what = dsm.source?.type === 'map' && dsm.source?.aoi
      ? (() => {
          const a = dsm.source.aoi
          const m = aoiMetrics(a.north, a.south, a.east, a.west)
          return `${f4(m.centerLat)}, ${f4(m.centerLon)} · ${formatArea(m.areaKm2)}`
        })()
      : 'uploaded image'
    context = `Twin · ${what} · ${mode.isMetric ? 'metric DSM' : 'relative surface'} · ${dsm.id || ''}`.trim()
  }

  return (
    <div className="mission-bar" role="status" aria-live="polite" aria-label={`Mission: ${trail[stageIdx]}. ${context}`}>
      <span className="ms-trail" aria-hidden="true">
        {trail.map((t, i) => (
          <span key={t} className="ms-seg">
            {i > 0 && <span className="ms-sep">→</span>}
            <span className={`ms ms-${states[i]}`}>{t}</span>
          </span>
        ))}
      </span>
      <span className="ms-context" title={context}>{context}</span>
    </div>
  )
}
