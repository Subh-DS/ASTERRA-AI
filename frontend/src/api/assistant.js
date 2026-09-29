import { useApp } from '../store/useAppStore'
import { api, API_URL } from './client'

export async function remoteReply(messages) {
  const ctrl = new AbortController()
  const timeout = setTimeout(() => ctrl.abort(), 30000)
  try {
    const res = await fetch(`${API_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: messages.slice(-12) }),
      signal: ctrl.signal,
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    return data.reply
  } finally {
    clearTimeout(timeout)
  }
}

function fmt(n, d = 2) {
  return Number(n).toFixed(d)
}

function describeState() {
  const s = useApp.getState()
  const dsm = s.dsm
  const job = s.job
  if (s.screen === 'viewer' && dsm) {
    const meta = dsm.metadata || null
    const isMetric = meta ? !!meta.is_metric : !!dsm.crs
    const range = dsm.stats?.min !== undefined && dsm.stats?.max !== undefined ? `${fmt(dsm.stats.min, 1)}–${fmt(dsm.stats.max, 1)} ${isMetric ? 'm' : 'relative'}` : 'full grid'
    const modeLabel = isMetric ? `Metric DSM (${dsm.crs || meta?.crs || ''})` : 'Relative Surface Model (rDSM)'
    return `You are flying the reconstructed surface for job ${dsm.id} — ${dsm.width}×${dsm.height} samples, ${range}, classified as ${dsm.landscape} terrain. Scale mode: ${modeLabel}.`
  }
  if (s.screen === 'progress') {
    const active = Object.entries(job.stages).find(([, v]) => v.state === 'active')
    return active ? `Pipeline is running — ${active[0].toUpperCase()} station: ${active[1].sub || 'in progress'}.` : 'Pipeline stations are warming up.'
  }
  if (s.screen === 'upload') {
    const info = s.upload.fileInfo
    return info
      ? `Ingest screen — ${info.name} parsed locally: ${info.kind}${info.width ? `, ${info.width}×${info.height}px` : ''}, ${info.hasCRS ? 'georeferenced so a metric DSM will be produced' : 'no CRS — pick Relative Surface or add at least 3 GCPs for metric calibration'}.`
      : 'Ingest screen — drop an overhead PNG, JPG or TIFF and metadata appears before anything is sent.'
  }
  return 'Landing screen. Drop any single overhead image on the amber button and DepthWizard turns it into a flyable terrain: monocular depth, scale calibration, textured mesh.'
}

function runCommand(q) {
  const s = useApp.getState()
  const api = s.viewerApi
  const num = q.match(/(-?\d+(?:\.\d+)?)/)
  const say = (text) => text

  if (/exaggerat|vertical.*(scale|height)|z.?scale/.test(q)) {
    let v = num ? parseFloat(num[1]) : NaN
    if (/^\d+(\.\d+)?\s*x\b/.test(q) && !num) v = NaN
    if (!isNaN(v)) {
      v = Math.min(4, Math.max(0.2, v > 10 ? v / 100 : v))
      s.setViewer({ zScale: v })
      api?.setZScale(v)
      return say(`Vertical exaggeration set to ${v.toFixed(2)}×.`)
    }
    return say(`Current exaggeration is ${s.viewer.zScale.toFixed(2)}×. Say e.g. "set exaggeration to 2.5".`)
  }
  if (/\b(fly|first.?person|cockpit)\b/.test(q)) {
    s.setViewer({ mode: 'fly' })
    api?.setMode('fly')
    return say('Fly mode engaging — click the viewport to capture the mouse. WASD to move, Q/E altitude, Shift sprint.')
  }
  if (/\borbit\b|\breview\b/.test(q)) {
    s.setViewer({ mode: 'orbit' })
    api?.setMode('orbit')
    return say('Back to orbit.')
  }
  if (/isolines?|contour line/.test(q)) {
    const next = !s.viewer.isolines
    s.setViewer({ isolines: next })
    return say(next ? 'Real-data isolines overlaid on the terrain.' : 'Isolines hidden.')
  }
  if (/slope|ramp/.test(q)) {
    const next = !s.viewer.slopeView
    s.setViewer({ slopeView: next })
    api?.setSlopeView(next)
    return say(next ? 'Slope color ramp on — teal flats through amber to red faces.' : 'Back to optical texture.')
  }
  if (/hillshade|light(ing)?|shadow/.test(q)) {
    const next = !s.viewer.hillshade
    s.setViewer({ hillshade: next })
    api?.setHillshade(next)
    return say(next ? 'Hillshade lighting restored.' : 'Lighting flattened for even reading.')
  }
  if (/reset (view|camera)|recenter|home view/.test(q)) {
    api?.resetView()
    return say('Camera returned to the default aerial framing.')
  }
  if (/screenshot|capture|snapshot/.test(q)) {
    const url = api?.screenshot()
    if (!url) return say('Screenshot needs the viewer running.')
    const a = document.createElement('a')
    a.href = url
    a.download = `depthwizard_capture_${Date.now()}.png`
    a.click()
    return say('Frame captured — check your downloads.')
  }
  return null
}

function answerQuery(q) {
  const s = useApp.getState()
  const dsm = s.dsm

  if (/rmse|mae|correlation|pearson|accuracy|how (good|accurate)/.test(q)) {
    const v = s.validation
    if (v) {
      return `Against your reference DEM: RMSE ${v.rmse.toFixed(2)} m, MAE ${v.mae.toFixed(2)} m, Pearson r ${v.corr.toFixed(3)}${v.pass ? ' — within tolerance for this terrain.' : ' — above the 6 m threshold; expect soft detail on fine structures.'}`
    }
    return dsm
      ? 'No reference comparison yet. Open Validation and drop a LiDAR DEM or ASCII grid — I read RMSE, MAE and correlation straight off the co-registration.'
      : 'Accuracy numbers appear once a DSM exists and you provide a reference DEM in the Validation panel.'
  }
  if (/elevation|range|highest|lowest|how (high|tall)|stats/.test(q)) {
    if (!dsm) return 'Compute a terrain first, then ask me about its statistics.'
    const { min = 0, max = 0 } = dsm.stats || {}
    const meta = dsm.metadata || null
    const isMetric = meta ? !!meta.is_metric : !!dsm.crs
    const unit = isMetric ? 'm' : 'relative'
    return `${isMetric ? 'Elevation' : 'Relative depth'} spans ${fmt(min, 1)} to ${fmt(max, 1)} ${unit} across ${dsm.width}×${dsm.height} samples (${((dsm.width * dsm.height) / 1e6).toFixed(2)} M measurements). Histograms live in the Analysis panel.`
  }
  if (/gcp|ground control|calibrat/.test(q)) {
    return s.upload.fileInfo?.hasCRS
      ? 'This scene is georeferenced, so scale comes from a SRTM-derived Terrarium DEM reference automatically — no ground control points needed.'
      : 'Without CRS metadata the surface is relative. Add at least 3 valid ground control points (lat, lon, elevation in meters) on the ingest screen and I fit an affine scale Z = a·d + b through them instead.'
  }
  if (/pipeline|how does|what does this|under the hood|works?/.test(q)) {
    return 'Four stations: INGEST parses your image and its georeferencing. DEPTH runs a monocular depth backbone (Depth Anything V2 Large) over overlapping tiles and stitches them. CALIBRATE fits relative depth to absolute meters using a SRTM-derived Terrarium DEM reference, your ground control points, or both, under a RANSAC affine fit. MESH extrudes the calibrated heights into a textured, flyable surface. Every number you see on screen comes off that chain.'
  }
  if (/stage|status|progress|how long/.test(q)) {
    return describeState()
  }
  if (/quality|tier|lag|slow|fps|performance/.test(q)) {
    const t = s.tier
    return `Rendering tier: ${t.resolved} (${t.reason}). Shadows ${t.resolved === 'high' ? 'on' : 'off'}, fog ${t.resolved === 'high' ? 'full' : t.resolved === 'medium' ? 'flat' : 'off'}, mesh capped at ${t.resolved === 'high' ? '512' : t.resolved === 'medium' ? '288' : '160'} segments per side. Override it any time in Settings → Rendering quality.`
  }
  if (/new (image|scene)|start over|upload another|back/.test(q)) {
    s.setScreen(s.screen === 'viewer' ? 'upload' : 'hero')
    return 'Ready for the next scene.'
  }
  if (/help|what can you/.test(q)) {
    return 'I can drive the instrument ("set exaggeration to 2.5", "switch to fly", "toggle isolines", "reset view", "screenshot"), report what you are looking at, read accuracy numbers after validation, and explain the pipeline. Ask plainly.'
  }
  if (/^(hi|hello|hey|yo)\b/.test(q)) {
    return describeState()
  }
  return null
}

export async function getAssistantReply(query) {
  const cmd = runCommand(query)
  if (cmd) return cmd
  const local = answerQuery(query)
  const state = useApp.getState()
  if (state.backendUp) {
    try {
      return await remoteReply(state.transcript.map((m) => ({ role: m.role === 'you' ? 'user' : 'assistant', content: m.text })).concat([{ role: 'user', content: query }]))
    } catch (err) {
      console.warn('[assistant] remote reply failed:', err?.message || err)
      /* fall through to local */
    }
  }
  if (local) return local
  return 'I can only answer geospatial questions. Try asking about elevation models (DSM/DEM/DTM), remote sensing, terrain analysis, photogrammetry, GIS, LiDAR, SAR, building extraction, or the ASTERRA pipeline.'
}
