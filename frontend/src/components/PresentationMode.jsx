import { useEffect, useMemo, useRef, useState } from 'react'
import { useApp } from '../store/useAppStore'

const HOLD_MS = 6000

function buildSteps({ hasFlood, hasSlide, hasEvac, modeLabel }) {
  const steps = [
    {
      id: 'overview',
      title: 'Area overview',
      desc: `Reconstructed digital twin — ${modeLabel}. Orbit freely once the briefing ends.`,
      run: (api) => {
        api.resetView()
        api.setColorMode?.('rgb')
        api.setSlopeView?.(false)
      },
    },
    {
      id: 'terrain',
      title: 'Terrain',
      desc: 'Hypsometric tint exposes ridges, valleys and elevation structure.',
      run: (api) => {
        api.setColorMode?.('elevation')
        api.setSlopeView?.(false)
      },
    },
    {
      id: 'slope',
      title: 'Slope',
      desc: 'Gradient steepness — the basis for stability and routing analysis.',
      run: (api) => {
        api.setColorMode?.('rgb')
        api.setSlopeView?.(true)
      },
    },
  ]
  if (hasFlood) {
    steps.push({
      id: 'flood',
      title: 'Water',
      desc: 'Simulated inundation rising over the measured terrain.',
      animate: 'scenario',
      run: (api) => {
        api.setColorMode?.('rgb')
        api.setSlopeView?.(false)
      },
    })
  }
  if (hasSlide) {
    steps.push({
      id: 'slide',
      title: 'Landslide',
      desc: 'Simulated mass movement following the slope downhill.',
      animate: 'scenario',
      run: (api) => {
        api.setColorMode?.('rgb')
        api.setSlopeView?.(false)
      },
    })
  }
  if (hasEvac) {
    steps.push({
      id: 'evac',
      title: 'Evacuation',
      desc: 'Simulated egress corridors to high-ground safe zones.',
      run: () => {},
    })
  }
  steps.push({
    id: 'final',
    title: 'Final analysis',
    desc: 'Full interactive control restored — orbit, measure, simulate.',
    run: (api) => {
      api.resetView()
      api.setColorMode?.('rgb')
      api.setSlopeView?.(false)
    },
  })
  return steps
}

/**
 * Presentation / mission mode (Phase 39): scripted cinematic briefing over
 * the live twin. Hides all chrome, steps through visualization states with
 * captions, then restores full control. Everything shown is the real scene
 * and real simulation state.
 */
export default function PresentationMode() {
  const viewerApi = useApp((s) => s.viewerApi)
  const dsm = useApp((s) => s.dsm)
  const result = useApp((s) => s.hazard.result)
  const evac = useApp((s) => s.hazard.evac)
  const setHazard = useApp((s) => s.setHazard)
  const setPresentation = useApp((s) => s.setPresentation)
  const setViewer = useApp((s) => s.setViewer)
  const [idx, setIdx] = useState(0)
  const timer = useRef(0)
  const animRaf = useRef(0)

  const steps = useMemo(() => {
    const sim = result?.simulation_type
    return buildSteps({
      hasFlood: sim === 'coastal_inundation',
      hasSlide: sim === 'landslide',
      hasEvac: !!evac,
      modeLabel: dsm?.id ? `job ${dsm.id}` : 'live reconstruction',
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result?.simulation_type, !!evac, dsm?.id])

  const step = steps[Math.min(idx, steps.length - 1)]

  // Body-level chrome hiding (assistant, mission bar live outside the viewer).
  // Panels close for a clean stage; briefing restores nothing (user reopens).
  useEffect(() => {
    document.body.classList.add('presenting')
    setViewer({ panelOpen: null })
    setHazard({ panelOpen: false })
    return () => document.body.classList.remove('presenting')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Run the step state + auto-advance.
  useEffect(() => {
    if (!viewerApi || !step) return
    try {
      step.run(viewerApi)
      setViewer(
        step.id === 'slope'
          ? { slopeView: true }
          : { slopeView: false },
      )
      if (step.id === 'terrain') setViewer({ colorMode: 'elevation' })
      else setViewer({ colorMode: 'rgb' })
    } catch { /* briefing must never break the scene */ }
    if (step.animate === 'scenario') {
      const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
      if (reduced) {
        setHazard({ animT: 1 })
      } else {
        const t0 = performance.now()
        const dur = 4500
        const tick = (now) => {
          const k = Math.min(1, (now - t0) / dur)
          setHazard({ animT: Math.round(k * 1000) / 1000 })
          if (k < 1) animRaf.current = requestAnimationFrame(tick)
        }
        animRaf.current = requestAnimationFrame(tick)
      }
    }
    timer.current = setTimeout(() => {
      setIdx((i) => (i + 1 < steps.length ? i + 1 : i))
    }, HOLD_MS)
    return () => {
      clearTimeout(timer.current)
      cancelAnimationFrame(animRaf.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx, viewerApi, steps.length])

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') setPresentation(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setPresentation])

  if (!viewerApi || !step) return null

  const go = (d) => {
    cancelAnimationFrame(animRaf.current)
    clearTimeout(timer.current)
    setIdx((i) => Math.min(steps.length - 1, Math.max(0, i + d)))
  }

  return (
    <>
      <div className="present-mark" aria-hidden="true">
        <span>ASTERRA · MISSION BRIEFING</span>
      </div>
      <div className="present-bar" role="dialog" aria-label="Mission briefing">
        <div className="present-cap">
          <span className="present-step mono">
            {String(idx + 1).padStart(2, '0')} / {String(steps.length).padStart(2, '0')}
          </span>
          <strong>{step.title}</strong>
          <span>{step.desc}</span>
        </div>
        <div className="present-dots" aria-hidden="true">
          {steps.map((s, i) => (
            <span key={s.id} className={i === idx ? 'on' : i < idx ? 'done' : ''} />
          ))}
        </div>
        <div className="present-ctrl">
          <button type="button" onClick={() => go(-1)} disabled={idx === 0} aria-label="Previous briefing step">
            ←
          </button>
          <button type="button" onClick={() => go(1)} disabled={idx >= steps.length - 1} aria-label="Next briefing step">
            →
          </button>
          <button type="button" className="present-exit" onClick={() => setPresentation(false)}>
            Exit · Esc
          </button>
        </div>
      </div>
    </>
  )
}
