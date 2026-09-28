import { useEffect, useRef } from 'react'
import { useApp } from '../store/useAppStore'
import { initHeroTerrain } from '../three/heroTerrain'

export default function Hero() {
  const setScreen = useApp((s) => s.setScreen)
  const tier = useApp((s) => s.tier.resolved)
  const reduced = useApp((s) => s.reducedMotion)
  const backendUp = useApp((s) => s.backendUp)
  const theme = useApp((s) => s.theme)

  const terrainWrapRef = useRef(null)
  const terrainCanvasRef = useRef(null)
  const terrainApiRef = useRef(null)

  useEffect(() => {
    if (!terrainCanvasRef.current) return
    if (tier === 'low' || reduced) return
    const t = initHeroTerrain(terrainCanvasRef.current, { reducedMotion: reduced })
    terrainApiRef.current = t
    t.setTheme(theme)
    return () => {
      terrainApiRef.current = null
      t.dispose()
    }
  }, [tier, reduced])

  useEffect(() => {
    terrainApiRef.current?.setTheme(theme)
  }, [theme])

  return (
    <section className="asterra-hero" aria-label="ASTERRA entry">
      <div className="asterra-hero-canvas" ref={terrainWrapRef}>
        {tier !== 'low' && !reduced && <canvas ref={terrainCanvasRef} aria-hidden="true" />}
      </div>
      <div className="asterra-hero-veil" />
      <div className="asterra-hero-vignette" />

      <div className="asterra-corner tl"><span className="asterra-cross" /> ASTERRA // SIH-26</div>
      <div className="asterra-corner bl">DEPTH ANYTHING V2 · DSM PIPELINE</div>
      <div className="asterra-corner br">SYS {backendUp === null ? 'PROBING' : backendUp ? 'NOMINAL' : 'OFFLINE'}</div>

      <div className="asterra-hero-main">
        <div className="asterra-kicker asterra-rise">Geospatial intelligence · Digital twin</div>
        <h1 className="asterra-title asterra-rise d1">
          <span className="thin">One overhead</span><br />
          image. A <span className="accent">flyable</span><br />
          terrain.
        </h1>
        <p className="asterra-sub asterra-rise d2">
          <span className="mono">ASTERRA / DEPTHWIZARD</span> transforms a single overhead RGB image into a{' '}
          <span className="mono">metric 3D terrain</span> representation using{' '}
          <span className="mono">monocular depth estimation</span>, terrain calibration and geospatial
          reconstruction.
        </p>

        <p className="asterra-pipeline asterra-rise d2" aria-label="Processing pipeline">
          <span className="pl-label">RGB image</span>
          <span className="pl-arrow" aria-hidden="true">→</span>
          <span className="pl-label">Depth</span>
          <span className="pl-arrow" aria-hidden="true">→</span>
          <span className="pl-label">Metric height</span>
          <span className="pl-arrow" aria-hidden="true">→</span>
          <span className="pl-label">DSM</span>
          <span className="pl-arrow" aria-hidden="true">→</span>
          <span className="pl-final">3D digital twin</span>
        </p>

        <div className="asterra-cta-row asterra-rise d3">
          <button className="asterra-cta primary" onClick={() => setScreen('workflow')}>
            CHOOSE YOUR WORKFLOW
            <span className="mono">MAP · IMAGE → 3D</span>
            <span aria-hidden="true">→</span>
          </button>
        </div>
      </div>
    </section>
  )
}
