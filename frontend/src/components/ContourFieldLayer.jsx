import { useEffect, useRef, useState } from 'react'
import { ContourField } from '../three/contourField'
import { useApp } from '../store/useAppStore'

const DIM_BY_SCREEN = { hero: 1, upload: 0.4, progress: 0.3, viewer: 0 }

export default function ContourFieldLayer() {
  const ref = useRef(null)
  const screen = useApp((s) => s.screen)
  const tier = useApp((s) => s.tier.resolved)
  const reduced = useApp((s) => s.reducedMotion)
  const theme = useApp((s) => s.theme)
  const setFieldRef = useApp((s) => s.setFieldRef)
  // Ambient decorative layer only: if WebGL2/shader init fails (seen as a
  // flaky driver hiccup in some browsers), hide it instead of crashing React
  // into a blank page. All fieldRef callers are null-safe.
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (!ref.current || failed) return
    let f = null
    try {
      f = new ContourField(ref.current, { tier, reducedMotion: reduced })
    } catch (err) {
      console.warn('[contour-field] disabled after init failure:', err?.message || err)
      setFailed(true)
      return
    }
    f.setTheme(theme)
    setFieldRef(f)
    const ro = new ResizeObserver(() => f.resize())
    ro.observe(ref.current)
    return () => {
      ro.disconnect()
      f.dispose()
      setFieldRef(null)
    }
  }, [tier, failed])

  useEffect(() => {
    useApp.getState().fieldRef?.setTheme(theme)
  }, [theme])

  useEffect(() => {
    const f = useApp.getState().fieldRef
    f?.setDim(DIM_BY_SCREEN[screen] ?? 0.4)
    if (ref.current) {
      ref.current.style.opacity = screen === 'viewer' ? '0' : String(DIM_BY_SCREEN[screen] ?? 0.4)
    }
  }, [screen])

  if (failed) return null
  return <canvas ref={ref} className="field-canvas" aria-hidden="true" />
}
