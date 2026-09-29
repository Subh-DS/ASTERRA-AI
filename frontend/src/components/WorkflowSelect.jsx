import { useEffect, useRef, useState } from 'react'
import { useApp } from '../store/useAppStore'
import mapVisual from '../assets/workflow-map.svg'
import reconstructionVisual from '../assets/workflow-reconstruction.svg'

const COMMIT_MS = 480

export default function WorkflowSelect() {
  const setScreen = useApp((s) => s.setScreen)
  const reduced = useApp((s) => s.reducedMotion)
  const [active, setActive] = useState(null)
  const timer = useRef(null)

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current)
  }, [])

  // Card visually activates (zoom, accent, lift, arrow) before the route
  // commits, so the map feels like the next stage — not a new page.
  const choose = (key, target) => {
    if (active) return
    if (reduced) {
      setScreen(target)
      return
    }
    setActive(key)
    timer.current = setTimeout(() => setScreen(target), COMMIT_MS)
  }

  return (
    <section className="wf" aria-label="Choose your workflow">
      <div className="wf-inner asterra-rise">
        <button type="button" className="wf-back" onClick={() => setScreen('hero')}>
          ← Entry
        </button>
        <p className="wf-eyebrow">Begin reconstruction</p>
        <h1 className="wf-title">How do you want to start?</h1>
        <p className="wf-sub">
          Two paths into the same pipeline. Pick the one that matches the data you have —
          both converge on a calibrated DSM and a flyable 3D twin.
        </p>

        <div className="wf-grid">
          <button
            type="button"
            className={`wf-card wf-teal${active === 'map' ? ' active' : ''}${active && active !== 'map' ? ' idle' : ''}`}
            aria-pressed={active === 'map'}
            onClick={() => choose('map', 'map')}
          >
            <span className="wf-bar" aria-hidden="true" />
            <span className="wf-visual">
              <img src={mapVisual} alt="Satellite map with a selected area extruded into 3D terrain" loading="eager" />
            </span>
            <span className="wf-body">
              <span className="wf-num">01</span>
              <span className="wf-name">Geospatial Workspace</span>
              <span className="wf-desc">
                Start from the map. Search any location, draw an area of interest,
                and reconstruct that exact place as an explorable 3D digital twin.
              </span>
              <span className="wf-go">Open map <span aria-hidden="true">→</span></span>
            </span>
          </button>

          <button
            type="button"
            className={`wf-card wf-amber${active === 'upload' ? ' active' : ''}${active && active !== 'upload' ? ' idle' : ''}`}
            aria-pressed={active === 'upload'}
            onClick={() => choose('upload', 'upload')}
          >
            <span className="wf-bar" aria-hidden="true" />
            <span className="wf-visual">
              <img src={reconstructionVisual} alt="Aerial image progressing through depth and elevation into 3D terrain" loading="eager" />
            </span>
            <span className="wf-body">
              <span className="wf-num">02</span>
              <span className="wf-name">Image → 3D Reconstruction</span>
              <span className="wf-desc">
                Start from an overhead image. Upload it, validate georeferencing, and run
                depth estimation into a metric DSM and flyable mesh.
              </span>
              <span className="wf-go">Ingest image <span aria-hidden="true">→</span></span>
            </span>
          </button>
        </div>
      </div>
    </section>
  )
}
