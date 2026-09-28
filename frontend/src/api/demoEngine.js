import { useApp } from '../store/useAppStore'
import { makeSyntheticCity, computeStats } from '../utils/sampleScene'
import { fmtInt } from '../utils/inspect'
import { ClientImageAnalyzer } from '../utils/clientImageAnalysis'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

function stageClock() {
  const t0 = performance.now()
  return () => ((performance.now() - t0) / 1000).toFixed(1)
}

const analyzer = new ClientImageAnalyzer({
  downscale: 256,
  buildingBoost: 1.8,
  edgeThreshold: 25,
  roadSuppression: 0.2,
  vegetationFactor: 0.8
})

export async function runDemoPipeline({ file, previewUrl, gcps, hasCRS }) {
  const { setStage, setJobMeta } = useApp.getState()
  const field = useApp.getState().fieldRef
  setJobMeta({ startedAt: performance.now(), demo: true })

  setStage('ingest', 'active', 'reading image · analyzing features')
  await sleep(400)
  setStage('ingest', 'done', hasCRS ? 'geo keys found' : 'rDSM mode · client-side analysis')

  let scene
  let analysisResult = null

  if (file) {
    try {
      const bmp = await createImageBitmap(file)
      setStage('depth', 'active', 'analyzing image features · client-side')
      await sleep(300)
      
      // Client-side image analysis - no backend needed
      analysisResult = await analyzer.analyze(bmp)
      scene = {
        heights: analysisResult.heightMap,
        width: analysisResult.width,
        height: analysisResult.height,
        textureSrc: (previewUrl && previewUrl !== 'sample') ? previewUrl : analysisResult.textureCanvas.toDataURL('image/jpeg', 0.92),
        pyramids: analysisResult.pyramids,
        landscape: 'auto-detected',
        masks: analysisResult.masks
      }
      setStage('depth', 'active', 'extracting features · edges · colors')
      await sleep(200)
      
      // Stream pyramids to contour field for live preview
      for (let i = 0; i < scene.pyramids.length; i++) {
        const pyr = scene.pyramids[i]
        field?.setRealGrid(pyr.data, pyr.w, pyr.h, (i + 1) / scene.pyramids.length * 0.9)
        await sleep(150)
      }
      
    } catch (err) {
      console.warn('Client-side analysis fallback:', err)
      try {
        const bmp = await createImageBitmap(file)
        const { fromImageBitmap } = await import('../utils/sampleScene')
        scene = fromImageBitmap(bmp)
        if (previewUrl && previewUrl !== 'sample') scene.textureSrc = previewUrl
      } catch {
        scene = makeSyntheticCity()
      }
    }
  } else {
    scene = makeSyntheticCity()
  }

  const tiles = scene.pyramids.length || 3
  for (let i = 0; i < Math.min(tiles, 4); i++) {
    setStage('depth', 'active', `building mesh ${i + 1}/${Math.min(tiles, 4)} · client-side`)
    const pyr = scene.pyramids[Math.min(i, scene.pyramids.length - 1)]
    field?.setRealGrid(pyr.data, pyr.w, pyr.h, (i + 1) / Math.min(tiles, 4) * 0.9)
    await sleep(200)
  }
  setStage('depth', 'done', `mesh built · ${fmtInt(scene.heights.length)} vertices`)

  setStage('calibrate', 'active', 'demo scale only · not metric')
  await sleep(300)

  // Offline demo NEVER produces metric elevations: no depth backbone runs
  // here, so any affine constants would be fabricated. Label honestly.
  setStage('calibrate', 'done', hasCRS ? 'demo mode · no metric fit applied' : 'relative scale · no CRS · rDSM mode')
  const calibNote = 'demo · not metric'
  field?.setRealGrid(scene.heightMap || scene.heights, scene.width, scene.height, 1)

  setStage('mesh', 'active', 'finalizing mesh · texture mapping')
  await sleep(400)
  setStage('mesh', 'done', '3D mesh ready · offline mode')

  const stats = computeStats(scene.heights)
  return {
    id: `OFFLINE-${Math.floor(Math.random() * 90000 + 10000)}`,
    heights: scene.heights,
    width: scene.width,
    height: scene.height,
    textureSrc: scene.textureSrc,
    pyramids: scene.pyramids,
    landscape: scene.landscape,
    masks: scene.masks,
    // No CRS is ever claimed for demo output: heuristic heights have no
    // georeferencing, so crs stays null and metadata forces non-metric.
    crs: null,
    stats,
    offline: true,
    model: 'client heuristic (no depth model)',
    metadata: {
      mode: 'demo',
      is_metric: false,
      metric_valid: false,
      unit: 'relative',
      crs: null,
      vertical_reference: null,
      calibration: { available: false, method: 'demo' },
    },
    textureCanvas: scene.textureCanvas
  }
}