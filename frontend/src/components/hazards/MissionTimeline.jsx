import { useEffect, useRef } from 'react'
import { useApp } from '../../store/useAppStore'

const END = 80
const SPEEDS = [1, 2, 4]

function fmtClock(s) {
  const m = Math.floor(s / 60)
  const r = Math.floor(s % 60)
  return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`
}

/**
 * Unified mission timeline (Phases 37–38). One clock drives every hazard
 * system as a causal chain instead of isolated demos:
 *
 *   RAIN → SCENARIO PLAY → ROUTE CHECK → EARTHQUAKE → SETTLE
 *
 * Storm, scenario animation, evacuation recompute and the quake are all
 * driven from this clock. Scrubbing moves continuous states (storm cover,
 * scenario progress); one-shot events (quake, route compute) fire only
 * while playing forward, never on scrub.
 */
export default function MissionTimeline() {
  const result = useApp((s) => s.hazard.result)
  const mission = useApp((s) => s.hazard.mission)
  const setHazard = useApp((s) => s.setHazard)
  const raf = useRef(0)
  const last = useRef(0)
  const fired = useRef(new Set())
  const { t, playing, speed } = mission

  const simType = result?.simulation_type
  const hasScenario = simType === 'coastal_inundation' || simType === 'landslide'
  const scenarioLabel = simType === 'coastal_inundation' ? 'Flood' : simType === 'landslide' ? 'Landslide' : 'Scenario'

  const EVENTS = [
    { t: 0, label: 'Normal', detail: 'twin at rest' },
    { t: 10, label: 'Rain start', detail: 'storm cover moves in' },
    ...(hasScenario ? [{ t: 20, label: `${scenarioLabel} play`, detail: 'simulation runs to full extent' }] : []),
    ...(hasScenario ? [{ t: 45, label: 'Route check', detail: 'evacuation recomputed against the hazard' }] : []),
    { t: 55, label: 'Earthquake', detail: 'main shock + aftershocks' },
    { t: 80, label: 'Settle', detail: 'systems at rest, routes hold' },
  ]

  function applyContinuous(nt) {
    const st = useApp.getState().hazard
    if ((nt >= 10) !== !!st.storm) setHazard({ storm: nt >= 10 })
    if (hasScenario) {
      const at = Math.min(1, Math.max(0, (nt - 20) / 20))
      if (Math.abs(at - st.animT) > 0.001) setHazard({ animT: Math.round(at * 1000) / 1000 })
    }
  }

  function fire(nt) {
    const st = useApp.getState().hazard
    if (nt >= 45 && !fired.current.has('evac') && hasScenario) {
      fired.current.add('evac')
      setHazard({ evacRequest: (st.evacRequest || 0) + 1 })
    }
    if (nt >= 55 && !fired.current.has('quake')) {
      fired.current.add('quake')
      if (!st.quake) setHazard({ quake: { intensity: 0.6, aftershocks: true, startedAt: Date.now() } })
    }
  }

  function stop() {
    setHazard({ mission: { ...useApp.getState().hazard.mission, playing: false } })
  }

  function play() {
    const st = useApp.getState().hazard.mission
    if (st.playing) return
    if (st.t >= END) {
      fired.current = new Set()
      setHazard({ mission: { ...st, t: 0, playing: true } })
    } else {
      setHazard({ mission: { ...st, playing: true } })
    }
  }

  useEffect(() => {
    if (!playing) return
    last.current = performance.now()
    const step = (now) => {
      const ms = useApp.getState().hazard.mission
      if (!ms.playing) return
      const dt = Math.min(0.1, (now - last.current) / 1000) * ms.speed
      last.current = now
      const nt = Math.min(END, ms.t + dt)
      applyContinuous(nt)
      fire(nt)
      if (nt >= END) {
        setHazard({ mission: { ...ms, t: END, playing: false } })
        return
      }
      setHazard({ mission: { ...ms, t: Math.round(nt * 10) / 10 } })
      raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing])

  useEffect(() => () => cancelAnimationFrame(raf.current), [])

  function scrub(nt) {
    stop()
    const st = useApp.getState().hazard
    if (nt < 55 && st.quake) setHazard({ quake: null })
    applyContinuous(nt)
    setHazard({ mission: { ...useApp.getState().hazard.mission, t: nt } })
  }

  function reset() {
    stop()
    fired.current = new Set()
    setHazard({
      storm: false,
      quake: null,
      animT: 0,
      showSimulated: true,
      mission: { playing: false, t: 0, speed },
    })
  }

  const current = [...EVENTS].reverse().find((e) => t >= e.t) || EVENTS[0]

  return (
    <div className="msn" aria-label="Mission timeline">
      <div className="msn-head">
        <span>Mission timeline</span>
        <span className="mono">{fmtClock(t)} / {fmtClock(END)}</span>
      </div>
      <div className="hz-timeline">
        <div className="hz-transport" role="group" aria-label="Mission transport">
          <button type="button" onClick={play} aria-label="Play mission">▶</button>
          <button type="button" onClick={stop} aria-label="Pause mission">⏸</button>
          <button type="button" onClick={reset} aria-label="Reset mission">↺</button>
        </div>
        <input
          type="range" min={0} max={END} step={0.5} value={t}
          onChange={(e) => scrub(Number(e.target.value))}
          aria-label="Mission time scrub"
        />
        <div className="msn-speeds" role="radiogroup" aria-label="Mission speed">
          {SPEEDS.map((s) => (
            <button
              key={s}
              role="radio"
              aria-checked={speed === s}
              className={speed === s ? 'active' : ''}
              onClick={() => setHazard({ mission: { ...useApp.getState().hazard.mission, speed: s } })}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>
      <div className="msn-now" role="status">
        ● {current.label} <span className="hz-dim">— {current.detail}</span>
      </div>
      <ul className="msn-events">
        {EVENTS.map((e) => (
          <li key={e.t}>
            <button
              type="button"
              className={e.t === current.t ? 'on' : ''}
              onClick={() => scrub(e.t)}
              title={e.detail}
            >
              <span className="mono">{fmtClock(e.t)}</span> {e.label}
            </button>
          </li>
        ))}
      </ul>
      {!hasScenario && (
        <p className="hz-note">
          Run a flood or landslide scenario to drive the full chain — storm and earthquake run standalone.
        </p>
      )}
    </div>
  )
}
