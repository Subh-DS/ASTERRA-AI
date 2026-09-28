import { useEffect, useState } from 'react'
import { useApp } from '../store/useAppStore'

export default function HudOverlays({ viewerApi }) {
  const [probe, setProbe] = useState(null)
  const [measure, setMeasure] = useState(null)
  const [flyLocked, setFlyLocked] = useState(false)
  const [orient, setOrient] = useState({ heading: 0, speed: 0 })
  const [scale, setScale] = useState(null)
  const [revealCount, setRevealCount] = useState(null)
  const dsm = useApp((s) => s.dsm)
  const mode = useApp((s) => s.viewer.mode)

  useEffect(() => {
    if (!viewerApi) return
    viewerApi.onProbe = (p) => setProbe(p)
    viewerApi.onMeasure = (m) => setMeasure(m)
    viewerApi.onFlyLock = (locked) => setFlyLocked(locked)
    viewerApi.onRevealProgress = (count, done) => {
      if (done) {
        setTimeout(() => setRevealCount(null), 900)
        setRevealCount(count)
      } else {
        setRevealCount(count)
      }
    }
  }, [viewerApi])

  useEffect(() => {
    let raf
    const loop = () => {
      raf = requestAnimationFrame(loop)
      if (!viewerApi) return
      const o = viewerApi.getOrientation()
      setOrient((prev) => (Math.abs(prev.heading - o.heading) > 0.4 || Math.abs(prev.speed - o.speed) > 0.3 ? o : prev))
      if (typeof viewerApi.getScaleBar === 'function') {
        const sb = viewerApi.getScaleBar()
        setScale((prev) => (sb?.label !== prev?.label || Math.abs((sb?.px || 0) - (prev?.px || 0)) > 2 ? sb : prev))
      }
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [viewerApi])

  const dsmMeta = dsm?.metadata || null
  const isMetric = dsmMeta ? !!dsmMeta.is_metric : !!dsm?.crs
  const heightUnit = isMetric ? 'm' : 'relative'
  const card = (h) => {
    const c = Math.round(h)
    return `+${c} ${heightUnit}`
  }
  const distUnit = isMetric ? 'm' : 'relative'
  const dhUnit = isMetric ? 'm' : 'relative'

  return (
    <>
      {probe && mode === 'orbit' && (
        <div className="hud-probe mono" style={{ position: 'fixed', left: probe.x, top: probe.y }}>
          {card(probe.height)}
        </div>
      )}
      {measure && (
        <div className="hud-measure mono" style={{ position: 'fixed', left: measure.x, top: measure.y }}>
          Δh {measure.dh.toFixed(1)} {dhUnit}
          <br />
          dist {measure.dist.toFixed(1)} {distUnit}
        </div>
      )}
      <div className="hud-compass" aria-hidden="true">
        <div className="compass-dial">
          <svg className="compass-needle" style={{ transform: `rotate(${-orient.heading}deg)` }} viewBox="0 0 44 44">
            <circle cx="22" cy="22" r="20" fill="none" stroke="var(--contour)" />
            <path d="M22 5 L26 20 L22 17 L18 20 Z" fill="var(--elevation)" />
            <path d="M22 39 L26 24 L22 27 L18 24 Z" fill="var(--contour)" />
          </svg>
        </div>
        <div className="readouts">
          <span>
            HDG <b>{String(Math.round(orient.heading)).padStart(3, '0')}°</b>
          </span>
          {mode === 'fly' && (
            <span>
              SPD <b>{orient.speed.toFixed(1)} m/s</b>
            </span>
          )}
        </div>
      </div>
      {mode === 'fly' && !flyLocked && (
        <div className="fly-hint">Click to engage mouse-look · WASD move · Q/E altitude · Shift sprint · Esc exits</div>
      )}
      {scale && mode === 'orbit' && (
        <div className="hud-scale mono" aria-hidden="true">
          <span>{scale.label}</span>
          <div className="scale-rule" style={{ width: `${Math.round(scale.px)}px` }} />
        </div>
      )}
      {revealCount !== null && (
        <div className="reveal-counter">
          <span className="eyebrow">Vertices</span>
          <div className="num">{revealCount.toLocaleString('en-US')}</div>
        </div>
      )}
    </>
  )
}
