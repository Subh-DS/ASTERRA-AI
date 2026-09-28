import { useEffect, useRef } from 'react'
import { ContourField } from '../three/contourField'
import { useApp } from '../store/useAppStore'

export default function IsolineOverlay() {
  const ref = useRef(null)
  const dsm = useApp((s) => s.dsm)
  const dsmId = dsm?.id ?? dsm?.jobId ?? null
  const dsmWidth = dsm?.width
  const dsmHeight = dsm?.height
  const tier = dsm ? 'medium' : 'low'

  useEffect(() => {
    if (!ref.current || !dsm || !dsmId) return
    const f = new ContourField(ref.current, {
      tier,
      interactive: false,
    })
    const n = Math.min(384, Math.max(dsmWidth, dsmHeight))
    const small = new Float32Array(n * n)
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        small[y * n + x] = dsm.heights[Math.floor((y / n) * dsmHeight) * dsmWidth + Math.floor((x / n) * dsmWidth)]
      }
    }
    f.setRealGrid(small, n, n, 1)
    const ro = new ResizeObserver(() => f.resize())
    ro.observe(ref.current)
    return () => {
      ro.disconnect()
      f.dispose()
    }
  }, [dsmId, dsmWidth, dsmHeight, tier])

  if (!dsm) return null
  return <canvas ref={ref} className="isoline-overlay" aria-hidden="true" />
}
