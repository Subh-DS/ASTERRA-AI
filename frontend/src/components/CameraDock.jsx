import { useEffect, useState } from 'react'
import { useApp } from '../store/useAppStore'

const SPEEDS = [0.5, 1, 2, 4]

/**
 * Unified bottom-center camera dock: ORBIT | FLY mode tabs plus the
 * flythrough transport in a single bar. Replaces the two competing
 * floating clusters (and the overlapping speed dropdown) with one
 * hierarchy. All viewerApi calls are unchanged.
 */
export default function CameraDock({ viewerApi }) {
  const mode = useApp((s) => s.viewer.mode)
  const setViewer = useApp((s) => s.setViewer)
  const [touring, setTouring] = useState(false)
  const [paused, setPaused] = useState(false)
  const [speed, setSpeed] = useState(1)

  useEffect(() => {
    if (!viewerApi) return
    viewerApi.onTourChange = (on) => {
      setTouring(on)
      if (!on) setPaused(false)
      else setPaused(!!viewerApi.tour?.paused)
    }
    return () => {
      if (viewerApi) viewerApi.onTourChange = () => {}
    }
  }, [viewerApi])

  useEffect(() => {
    viewerApi?.setTourSpeed(speed)
  }, [viewerApi, speed])

  const setMode = (m) => {
    setViewer({ mode: m })
    viewerApi?.setMode(m)
  }

  return (
    <div className="cameradock" role="toolbar" aria-label="Camera controls">
      <div className="dock-seg" role="tablist" aria-label="Camera mode">
        {['orbit', 'fly'].map((m) => (
          <button
            key={m}
            role="tab"
            aria-selected={mode === m}
            className={mode === m ? 'active' : ''}
            onClick={() => setMode(m)}
          >
            {m.toUpperCase()}
          </button>
        ))}
      </div>

      <span className="dock-div" aria-hidden="true" />

      {!viewerApi || !touring ? (
        <button
          className="dock-action"
          disabled={!viewerApi}
          onClick={() => viewerApi?.startTour(speed)}
          title="Cinematic flythrough of the terrain"
        >
          ▶ Flythrough
        </button>
      ) : (
        <div className="dock-tour">
          <button
            className="dock-action"
            aria-label={paused ? 'Resume flythrough' : 'Pause flythrough'}
            onClick={() => {
              if (paused) viewerApi.resumeTour()
              else viewerApi.pauseTour()
              setPaused(!paused)
            }}
          >
            {paused ? '▶ Resume' : '⏸ Pause'}
          </button>
          <button
            className="dock-icon"
            aria-label="Restart flythrough"
            title="Restart flythrough"
            onClick={() => {
              viewerApi.startTour(speed)
              setPaused(false)
            }}
          >
            ↻
          </button>
          <div className="dock-speeds" role="radiogroup" aria-label="Flythrough speed">
            {SPEEDS.map((s) => (
              <button
                key={s}
                role="radio"
                aria-checked={speed === s}
                className={speed === s ? 'active' : ''}
                onClick={() => setSpeed(s)}
              >
                {s}×
              </button>
            ))}
          </div>
          <button
            className="dock-action"
            onClick={() => {
              viewerApi.stopTour()
              setTouring(false)
              setPaused(false)
            }}
          >
            Exit
          </button>
        </div>
      )}
    </div>
  )
}
