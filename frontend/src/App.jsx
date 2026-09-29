import { useEffect, useState } from 'react'
import { useApp } from './store/useAppStore'
import { detectQualityTier, resolveTier, TIER_CAPS } from './quality/detectTier'
import { api } from './api/client'
import ContourFieldLayer from './components/ContourFieldLayer'
import ErrorBoundary from './components/ErrorBoundary'
import AppShell from './components/AppShell'
import Hero from './components/Hero'
import WorkflowSelect from './components/WorkflowSelect'
import Uploader from './components/Uploader'
import MapScreen from './components/map/MapScreen'
import JobProgress from './components/JobProgress'
import ViewerScreen from './components/ViewerScreen'
import AssistantDock from './components/AssistantDock'
import LoadingOverlay from './components/LoadingOverlay'

function useSmallScreen() {
  const [small, setSmall] = useState(window.innerWidth < 768)
  useEffect(() => {
    const fn = () => setSmall(window.innerWidth < 768)
    window.addEventListener('resize', fn)
    return () => window.removeEventListener('resize', fn)
  }, [])
  return small
}

export default function App() {
  const screen = useApp((s) => s.screen)
  const banner = useApp((s) => s.banner)
  const setTier = useApp((s) => s.setTier)
  const setReduced = useApp((s) => s.setReducedMotion)
  const small = useSmallScreen()
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let live = true
    detectQualityTier().then((t) => {
      if (!live) return
      const override = localStorage.getItem('dw-tier-override')
      const resolved = resolveTier(t.level, override)
      setTier({ auto: t.level, override: override || null, resolved, reason: t.reason })
      document.body.classList.toggle('no-idle-motion', resolved === 'low')
      setReady(true)
    })
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    setReduced(mq.matches)
    const mqFn = (e) => setReduced(e.matches)
    mq.addEventListener?.('change', mqFn)
    return () => {
      live = false
      mq.removeEventListener?.('change', mqFn)
    }
  }, [])

  useEffect(() => {
    api.health().then((up) => useApp.setState({ backendUp: up }))
    api.engineInfo().then((info) => useApp.setState({ backendEngine: info.up ? info : null }))
  }, [])

  // SPA navigation: each screen starts at the top instead of inheriting the
  // previous screen's scroll position.
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [screen])

  return (
    <>
      <ContourFieldLayer />
      {banner && (
        <div className="banner" role="alert">
          <span className="mono">!</span>
          {banner}
        </div>
      )}
      {!ready ? null : small ? (
        <div className="size-notice">
          <div className="box">
            <span className="eyebrow">DepthWizard</span>
            <h2>Larger screen required</h2>
            <p>
              The flythrough viewer needs at least 768 px of width for its flight controls. Open DepthWizard on a
              laptop or desktop to explore terrain.
            </p>
          </div>
        </div>
      ) : (
        <>
          <LoadingOverlay />
          <ErrorBoundary>
          <AppShell>
          <div className="app-content">
            {screen === 'hero' && <Hero />}
            {screen === 'workflow' && <WorkflowSelect />}
            {screen === 'upload' && (
              <div className="asterra-screen">
                <div className="asterra-crumb"><span className="live">● MISSION 01</span><span>/</span><span>SOURCE INGEST</span><span>/</span><span>GEOTIFF · GCP · RELATIVE</span></div>
                <Uploader />
              </div>
            )}
            {screen === 'map' && <MapScreen />}
            {screen === 'progress' && (
              <div className="asterra-screen">
                <div className="asterra-crumb"><span className="live">● MISSION 02</span><span>/</span><span>RECONSTRUCTION</span><span>/</span><span>DEPTH · CALIBRATE · MESH</span></div>
                <JobProgress />
              </div>
            )}
            {screen === 'viewer' && <ViewerScreen />}
          </div>
          </AppShell>
          </ErrorBoundary>
          <AssistantDock />
        </>
      )}
    </>
  )
}
