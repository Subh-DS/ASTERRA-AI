import { useState } from 'react'
import { startMapJob } from '../../api/startJob'
import { useApp } from '../../store/useAppStore'
import { AOI_LIMITS, aoiMetrics, validateAoiClient } from '../../utils/aoi'

const DEPARTURE_MS = 1500

export default function ReconstructionButton() {
  const aoi = useApp((s) => s.map.aoi)
  const imagery = useApp((s) => s.map.imagery)
  const jobError = useApp((s) => s.job.error)
  const jobId = useApp((s) => s.job.id)
  const leaving = useApp((s) => s.mapLeaving)
  const [submitting, setSubmitting] = useState(false)

  if (!aoi) return null
  const clientErr = validateAoiClient(aoi)
  const m = aoiMetrics(aoi.north, aoi.south, aoi.east, aoi.west)
  const sizeOk =
    !clientErr &&
    m.widthM >= AOI_LIMITS.minSideM && m.heightM >= AOI_LIMITS.minSideM &&
    m.widthM <= AOI_LIMITS.maxSideM && m.heightM <= AOI_LIMITS.maxSideM &&
    m.areaKm2 <= AOI_LIMITS.maxAreaKm2
  const hasImagery = !!imagery?.items?.length && !!imagery?.selectedId
  const ready = sizeOk && hasImagery && !submitting && !leaving

  async function onClick() {
    if (!ready) return
    const st = useApp.getState()
    const reduced = st.reducedMotion
    // The departure cinematic and the backend request run in parallel: the
    // backend is never delayed, and the route commits only when both the
    // animation has played and the job is accepted.
    st.setMapLeaving(true)
    setSubmitting(true)
    try {
      const [res] = await Promise.all([
        startMapJob({
          aoi,
          provider: imagery.provider || 'auto',
          itemId: imagery.selectedId,
          maxCloud: imagery.maxCloud ?? null,
          quality: imagery.quality || 'medium',
        }),
        new Promise((r) => setTimeout(r, reduced ? 250 : DEPARTURE_MS)),
      ])
      if (res.ok) {
        // A fast (e.g. reused) job may already have finished and routed to
        // the Twin while the cinematic played — only commit to Pipeline if
        // we are still on the map.
        if (useApp.getState().screen === 'map') st.setScreen('progress')
      } else {
        st.setMapLeaving(false)
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="map-panel-block">
      <button
        type="button"
        className="map-cta"
        disabled={!ready}
        onClick={onClick}
        title={!sizeOk ? 'Fix the selected area first' : !hasImagery ? 'Waiting for imagery' : 'Start 3D reconstruction'}
      >
        {leaving ? 'Locking AOI…' : submitting ? 'Starting…' : 'Use this scene → Reconstruct'}
      </button>
      {!hasImagery && sizeOk && <p className="muted small">Select an imagery scene above to continue.</p>}
      {ready && !leaving && (
        <p className="muted small">
          Locks this area and scene → runs depth → DSM → mesh. Watch it in Pipeline, then explore the Twin.
        </p>
      )}
      {jobError && !jobId && !leaving && <div className="map-error">{jobError}</div>}
    </div>
  )
}
