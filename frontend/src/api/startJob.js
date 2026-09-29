import { useApp } from '../store/useAppStore'
import { api, API_URL, JobSocket } from './client'
import { runDemoPipeline } from './demoEngine'

const STAGE_ORDER = ['ingest', 'depth', 'calibrate', 'mesh']

export async function startJob({ fileInfo, previewUrl, isSample = false }) {
  const store = useApp.getState()
  const { resetJob, setScreen } = store
  resetJob()
  setScreen('progress')

  const backendUp = await api.health()
  api.engineInfo().then((info) => useApp.getState().setBackendEngine(info.up ? info : null))
  if (!backendUp || isSample) {
    const dsm = await runDemoPipeline({
      file: isSample ? null : store.upload.blob,
      previewUrl: isSample ? 'sample' : previewUrl,
      gcps: [],
      hasCRS: !!fileInfo?.hasCRS,
    })
    finish(dsm)
    return
  }

  const { setJobMeta } = useApp.getState()
  try {
    // Forensic fix: truthiness dropped valid 0 values (equator lon 0,
    // sea-level elev 0). Mirror the Uploader's explicit validity instead.
    const gcpOk = (g) => {
      const lat = parseFloat(g?.lat)
      const lon = parseFloat(g?.lon)
      const elev = parseFloat(g?.elev)
      return Number.isFinite(lat) && Number.isFinite(lon) && Number.isFinite(elev) &&
        Math.abs(lat) <= 90 && Math.abs(lon) <= 180
    }
    const gcps = store.gcps.filter(gcpOk).map((g) => ({ lat: +g.lat, lon: +g.lon, elev: +g.elev }))
    const res = await api.uploadImage(store.upload.blob, gcps)
    setJobMeta({ id: res.job_id, demo: false, fileToken: res.file_token || null, startedAt: performance.now() })
    useApp.getState().setStage('ingest', 'active', 'uploaded')
    watchJob(res.job_id)
  } catch (err) {
    useApp.getState().setJobMeta({ error: `Upload failed — ${err.message}. Falling back is not available in connected mode.` })
  }
}

export async function startMapJob({ aoi, provider = "auto", itemId = null, maxCloud = null, quality = "medium" }) {
  // Navigation is owned by the caller: it plays the map departure cinematic
  // first and routes to Pipeline only once the backend has accepted the job.
  const { resetJob, setJobMeta, setMapJob } = useApp.getState()
  resetJob()
  const backendUp = await api.health()
  api.engineInfo().then((info) => useApp.getState().setBackendEngine(info.up ? info : null))
  if (!backendUp) {
    const error = 'Backend unavailable — map reconstruction needs the processing service.'
    useApp.getState().setJobMeta({ error })
    return { ok: false, error }
  }
  try {
    const res = await api.createMapJob({ aoi, provider, itemId, maxCloud, quality })
    setJobMeta({ id: res.job_id, demo: false, fileToken: null, startedAt: performance.now() })
    setMapJob({ jobId: res.job_id, aoi, status: res.reused ? 'complete' : 'running', reused: !!res.reused })
    useApp.getState().setStage('ingest', 'active', res.reused ? 'opening existing reconstruction' : 'finding imagery')
    watchJob(res.job_id)
    return { ok: true, jobId: res.job_id }
  } catch (err) {
    const error = `Map reconstruction failed — ${err.message}`
    useApp.getState().setJobMeta({ error })
    return { ok: false, error }
  }
}

export function watchJob(jobId) {
  const { setStage } = useApp.getState()
  const socket = new JobSocket(jobId, {
    onEvent: (evt) => {
      if (STAGE_ORDER.includes(evt.stage)) {
        setStage(evt.stage, evt.status || 'active', evt.sub || '')
      }
      if (evt.type === 'job_complete') {
        socket.close()
        finishFromReal(jobId)
      }
      if (evt.type === 'job_error') {
        socket.close()
        useApp.getState().setJobMeta({ error: evt.message })
      }
    },
    onDead: () => useApp.getState().setBanner('Connection to processing service lost — results may be stale.'),
  })
  return socket
}

export async function retryJob(jobId) {
  const { setStage, setJobMeta, setScreen } = useApp.getState()
  setJobMeta({ id: jobId, demo: false, error: null, startedAt: performance.now() })
  for (const st of STAGE_ORDER) setStage(st, 'queued', '')
  setScreen('progress')
  try {
    await api.retry(jobId)
    watchJob(jobId)
  } catch (err) {
    setJobMeta({ error: `Retry failed — ${err.message}` })
  }
}

async function finishFromReal(jobId) {
  try {
    const status = await pollJob(jobId)
    const meta = status.result || {}
    const fileToken = meta.file_token || useApp.getState().job.fileToken || null
    const withToken = (path) =>
      !path ? path : `${API_URL}${path}${fileToken ? `?token=${encodeURIComponent(fileToken)}` : ''}`
    let heights = null
    try {
      heights = await api.getDsmBinary(jobId)
    } catch {
      /* binary not provided */
    }
    finish({
      id: jobId,
      heights: heights || new Float32Array(256 * 256),
      width: meta.width || 256,
      height: meta.height || 256,
      textureSrc: withToken(meta.texture_url),
      normalSrc: withToken(meta.normal_url) || null,
      landscape: meta.landscape || 'mixed',
      crs: meta.crs || null,
      stats: meta.stats || {},
      model: meta.model || null,
      metadata: meta.metadata || null,
      reconstruction: meta.reconstruction || null,
      buildings: meta.buildings || [],
      environment: meta.environment || { roads: [], water: [], landcover: [], trees: [], semantic_regions: [] },
      environmentSource: meta.environment_source || null,
      source: meta.source || null,
      georeference: meta.georeference || null,
      pixelSizeM: meta.metadata?.pixel_size_m || null,
      fileToken,
    })  } catch (e) {
    const message = `Result retrieval failed — ${e.message}`
    useApp.getState().setJobMeta({ error: message })
    throw new Error(message)
  }
}

async function pollJob(jobId, tries = 120) {
  for (let i = 0; i < tries; i++) {
    try {
      const s = await api.getJob(jobId)
      if (s.status === 'complete') return s
      if (s.status === 'error') throw new Error(s.message || 'backend error')
      await new Promise((r) => setTimeout(r, 1000))
    } catch (err) {
      const msg = String(err?.message || '')
      if (msg.includes('404') || msg.includes('unknown job') || msg.includes('Processing session expired')) {
        const expired = 'Processing session expired. Please start a new reconstruction.'
        useApp.getState().setJobMeta({ error: expired })
        useApp.getState().setScreen('upload')
        throw new Error(expired)
      }
      throw err
    }
  }
  throw new Error('timed out waiting for result')
}

function finish(dsm) {
  const field = useApp.getState().fieldRef
  const n = downsampleSize(Math.max(dsm.width, dsm.height))
  const heightsForField = dsm.heightMap || dsm.heights
  field?.setRealGrid(downsampleForField(heightsForField, dsm.width, dsm.height, n), n, n, 1)
  
  // Pass additional offline data to the store
  const dsmData = {
    ...dsm,
    textureCanvas: dsm.textureCanvas,
    masks: dsm.masks,
    offline: dsm.offline || false
  }
  useApp.getState().setDsm(dsmData)
  // Use requestAnimationFrame to ensure the viewer is ready before transitioning
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      useApp.getState().setScreen('viewer')
    })
  })
}

function downsampleSize(n) {
  return Math.min(160, n)
}

function downsampleForField(heights, w, h, n) {
  const out = new Float32Array(n * n)
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      out[y * n + x] = heights[Math.floor((y / n) * h) * w + Math.floor((x / n) * w)]
    }
  }
  return out
}
