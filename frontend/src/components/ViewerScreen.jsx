import { useEffect, useRef, useState } from 'react'
import { useApp } from '../store/useAppStore'
import { dsmModeInfo } from '../utils/dsmMode'
import { API_URL } from '../api/client'
import { TerrainViewer } from '../three/TerrainViewer'
import { StructureViewer } from '../three/StructureViewer'
import { TIER_CAPS } from '../quality/detectTier'
import LeftRail from './LeftRail'
import CameraDock from './CameraDock'
import MetaHud from './MetaHud'
import PresentationMode from './PresentationMode'
import HudOverlays from './HudOverlays'
import IsolineOverlay from './IsolineOverlay'
import AnalysisPanel from '../panels/AnalysisPanel'
import ValidationPanel from '../panels/ValidationPanel'
import SettingsPanel from '../panels/SettingsPanel'
import HazardPanel from './hazards/HazardPanel'
import { useHazardScene } from './hazards/useHazardScene'

function HazardSceneBridge() {
  useHazardScene()
  return null
}

const TIER_ORDER = ['low', 'medium', 'high']

const SHORTCUTS = {
  'KeyO': () => ({ mode: 'orbit' }),
  'KeyF': () => ({ mode: 'fly' }),
  'KeyH': () => ({ hillshade: true }),
  'KeyS': () => ({ slopeView: true }),
  'KeyC': () => 'colorMode',
  'KeyI': () => ({ isolines: true }),
  'KeyR': () => 'reset',
  'KeyE': () => 'export',
  'KeyA': () => 'assistant',
  'Escape': 'clearMeasure',
  'Slash': () => 'shortcuts', // ? key
}

const SHORTCUT_LABELS = {
  'KeyO': 'O - Orbit mode',
  'KeyF': 'F - Fly mode',
  'KeyH': 'H - Hillshade',
  'KeyS': 'S - Slope view',
  'KeyI': 'I - Isolines',
  'KeyR': 'R - Reset view',
  'KeyC': 'C - Cycle visualization',
  'KeyE': 'E - Export panel',
  'KeyA': 'A - Assistant',
  'Escape': 'Esc - Clear measurement',
  'Slash': '? - Show shortcuts',
}

export default function ViewerScreen() {
  const containerRef = useRef(null)
  const viewerRef = useRef(null)
  const [api, setApi] = useState(null)
  const [showShortcuts, setShowShortcuts] = useState(false)
  const tier = useApp((s) => s.tier)
  const reduced = useApp((s) => s.reducedMotion)
  const themeArg = useApp((s) => s.theme)
  const setThemeStore = useApp((s) => s.setTheme)
  const dsm = useApp((s) => s.dsm)
  const dsmLoadKey = dsm ? `${dsm.id ?? 'offline'}:${dsm.width ?? 0}x${dsm.height ?? 0}:${dsm.heights?.length ?? 0}` : null
  const viewer = useApp((s) => s.viewer)
  const hazardOpen = useApp((s) => s.hazard.panelOpen)
  const presentation = useApp((s) => s.presentation)
  const setPresentation = useApp((s) => s.setPresentation)
  const setViewer = useApp((s) => s.setViewer)
  const setBanner = useApp((s) => s.setBanner)
  const [zVal, setZVal] = useState(1.75)
  // Twin-birth sequence: played once per reconstruction when the mesh first
  // opens — splash, darkness emergence (engine), staged HUD reveal.
  const [twinBorn, setTwinBorn] = useState(true)
  const [birthSplash, setBirthSplash] = useState(false)
  const birthShownRef = useRef(new Set())
  // A completed GLB is the canonical structure-aware scene. Older jobs used
  // the dsm-terrain mode label even though their GLB was already valid, so
  // availability is based on the artifact contract instead of that label.
  const structAvail = !!(dsm && !dsm.offline && dsm.reconstruction?.has_glb && (dsm.reconstruction?.glb_url || dsm.glb_url))
  const [structMode, setStructMode] = useState(false)
  const [structLayers, setStructLayers] = useState([])
  const [layersOpen, setLayersOpen] = useState(true)
  const [structError, setStructError] = useState(null)
  const [pickedId, setPickedId] = useState(null)
  const structContainerRef = useRef(null)
  const structApiRef = useRef(null)
  const [structApi, setStructApi] = useState(null)
  const activeApi = structMode && structApi ? structApi : api

  useEffect(() => {
    setStructMode(!!structAvail)
    setPickedId(null)
    setStructError(null)
  }, [dsmLoadKey, structAvail])

  useEffect(() => {
    if (!dsmLoadKey) return
    const fresh = !birthShownRef.current.has(dsmLoadKey)
    if (fresh) birthShownRef.current.add(dsmLoadKey)
    if (reduced || !fresh) {
      setTwinBorn(true)
      setBirthSplash(false)
      return
    }
    setTwinBorn(false)
    setBirthSplash(true)
    const t1 = setTimeout(() => setTwinBorn(true), 500)
    const t2 = setTimeout(() => setBirthSplash(false), 1700)
    return () => {
      clearTimeout(t1)
      clearTimeout(t2)
    }
  }, [dsmLoadKey, reduced])

  useEffect(() => {
    if (!structMode || !structAvail || !structContainerRef.current || !dsm) return
    setStructError(null)
    const v = new StructureViewer(structContainerRef.current, {
      legacyZUp: dsm.reconstruction?.coordinate_system !== 'Local Y-up mesh coordinates',
      gridW: dsm.width,
      gridH: dsm.height,
      pixelSizeM: dsm.pixelSizeM,
      sceneAlignment: dsm.reconstruction?.scene_alignment,
      verticalOrigin: dsm.reconstruction?.vertical_origin_m ?? dsm.metadata?.vertical_origin_m ?? dsm.stats?.min ?? 0,
      maxH: dsm.stats?.max ?? dsm.reconstruction?.vertical_origin_m ?? 1,
      onPick: (id) => setPickedId(id),
      onLoad: (info) => {
        if (info.ok) setStructLayers(info.groups)
        else {
          setStructError(info.error || '3D scene failed to load')
          setStructMode(false)
        }
      },
    })
    structApiRef.current = v
    setStructApi(v)
    const glbPath = dsm.reconstruction?.glb_url || dsm.glb_url || `/api/jobs/${dsm.id}/export/model.glb`
    const token = dsm.fileToken ? `?token=${encodeURIComponent(dsm.fileToken)}` : ''
    v.load(`${API_URL}${glbPath}${glbPath.includes('?') ? '&' : token ? '?' : ''}${token ? `token=${encodeURIComponent(dsm.fileToken)}` : ''}`)
    v.setDisplayScale?.(zVal)
    return () => {
      v.dispose()
      structApiRef.current = null
      setStructApi(null)
    }
  }, [structMode, structAvail, dsmLoadKey])

  useEffect(() => {
    useApp.getState().setViewerApi(activeApi)
    return () => {
      if (useApp.getState().viewerApi === activeApi) useApp.getState().setViewerApi(null)
    }
  }, [activeApi])

  useEffect(() => {
    if (!containerRef.current) return
    const caps = TIER_CAPS[tier.resolved] || TIER_CAPS.high
    const initialZScale = 1.75
    const v = new TerrainViewer(containerRef.current, {
      tier: tier.resolved,
      caps,
      reducedMotion: reduced,
      revealedForJobId: useApp.getState().dsm?.id ?? null,
      zScale: initialZScale,
      onFpsDrop: (avg) => {
        // Stability-first: do NOT auto-demote tier by disposing the viewer.
        // A transient render stall is reported as a banner only; if the user
        // wants a lower tier they can pick it explicitly in Settings.
        setBanner(`Frame time ${avg.toFixed(0)} ms — viewer held at current tier. Choose a lower quality in Settings to reduce GPU load.`)
        setTimeout(() => setBanner(null), 5000)
      },
    })
    viewerRef.current = v
    setApi(v)
    useApp.getState().setViewerApi(v)
    return () => {
      v.dispose()
      viewerRef.current = null
      useApp.getState().setViewerApi(null)
    }
  // We deliberately key this ONLY on tier.resolved. Z-scale, DSM, theme, etc.
  // are applied through the methods exposed by the API; recreating the viewer
  // is reserved for explicit tier changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tier.resolved])

  useEffect(() => {
    if (!api || !dsmLoadKey) return
    api.load(dsm)
  }, [api, dsmLoadKey, dsm])

  useEffect(() => {
    if (!api) return
    api.setZScale(zVal)
    structApiRef.current?.setDisplayScale?.(zVal)
  }, [api, zVal])

  useEffect(() => {
    if (!api || !dsm) return
    api.setSlopeView(viewer.slopeView)
    api.setHillshade(viewer.hillshade)
    api.setColorMode?.(viewer.colorMode || 'rgb')
  }, [api, viewer.slopeView, viewer.hillshade, viewer.colorMode, dsm])

  useEffect(() => {
    api?.setTheme(themeArg)
    structApi?.setTheme?.(themeArg)
  }, [api, structApi, themeArg])

  const onZChange = (v) => {
    setZVal(v)
    setViewer({ zScale: v })
    api?.setZScale(v)
    structApiRef.current?.setDisplayScale?.(v)
  }

  useEffect(() => {
    const onKeyDown = (e) => {
      const activeElement = document.activeElement
      if (activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA')) return

      const action = SHORTCUTS[e.code]
      if (!action) return

      if (action === 'shortcuts') {
        setShowShortcuts(!showShortcuts)
        return
      }

      if (action === 'reset') {
        activeApi?.resetView()
      } else if (action === 'export') {
        setViewer({ panelOpen: viewer.panelOpen === 'settings' ? null : 'settings' })
      } else if (action === 'assistant') {
        useApp.getState().setAssistantOpen(!useApp.getState().assistantOpen)
      } else if (action === 'clearMeasure') {
        activeApi?.clearMeasure()
      } else if (action === 'colorMode') {
        const order = ['rgb', 'elevation', 'hybrid', 'wire']
        const next = order[(order.indexOf(viewer.colorMode) + 1) % order.length]
        setViewer({ colorMode: next })
        activeApi?.setColorMode(next)
      } else if (typeof action === 'function') {
        const patch = action()
        if (patch === 'reset') {
          activeApi?.resetView()
        } else if (typeof patch === 'object') {
          setViewer(patch)
          Object.entries(patch).forEach(([k, v]) => {
            if (k === 'mode') activeApi?.setMode(v)
            else if (k === 'hillshade') activeApi?.setHillshade(v)
            else if (k === 'slopeView') activeApi?.setSlopeView(v)
            else if (k === 'isolines') activeApi?.setIsolines?.(v)
          })
        }
      } else if (typeof action === 'object') {
        setViewer(action)
        Object.entries(action).forEach(([k, v]) => {
          if (k === 'mode') activeApi?.setMode(v)
          else if (k === 'hillshade') activeApi?.setHillshade(v)
          else if (k === 'slopeView') activeApi?.setSlopeView(v)
          else if (k === 'isolines') activeApi?.setIsolines?.(v)
        })
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [activeApi, viewer, showShortcuts])

  const openPanel = (name) => {
    if (name && useApp.getState().assistantOpen) useApp.getState().setAssistantOpen(false)
    setViewer({ panelOpen: viewer.panelOpen === name ? null : name })
  }

  // Decide which semantic mode the current DSM is in, via the shared helper
  // (utils/dsmMode.js) so all panels agree on REAL vs DEMO labeling.
  const mode = dsmModeInfo(dsm)
  const isMetric = mode.isMetric
  const dsmModeLabel = mode.modeLabel
  const backendEngine = useApp((s) => s.backendEngine)
  // Basename on both POSIX and Windows separators: a Windows absolute path
  // (C:\...\dav2-large-hf) has no '/', so split('/') alone leaks the full
  // path into the chip and pushes Validation/Export off the topbar.
  const engineModelShort = backendEngine?.model
    ? String(backendEngine.model).split(/[/\\]/).pop()
    : null

  return (
    <div className={`viewer-root${twinBorn ? ' twin-born' : ''}${presentation ? ' presenting' : ''}`}>
      <div
        ref={containerRef}
        style={{ position: 'absolute', inset: 0, display: structMode ? 'none' : 'block' }}
      />
      <div
        ref={structContainerRef}
        style={{ position: 'absolute', inset: 0, display: structMode ? 'block' : 'none' }}
      />
      {viewer.isolines && !structMode && <IsolineOverlay />}
      {!reduced && dsmLoadKey && <div key={`twin-veil-${dsmLoadKey}`} className="twin-birth-veil" aria-hidden="true" />}
      {birthSplash && (
        <div className="twin-birth-splash" aria-hidden="true">
          <div className="twin-birth-kicker">Mesh complete</div>
          <div className="twin-birth-title">Digital twin ready</div>
          <div className="twin-birth-meta mono">
            {dsmModeLabel}
            {dsm?.heights?.length ? ` · ${dsm.heights.length.toLocaleString('en-US')} vertices` : ''}
          </div>
        </div>
      )}

      <div className="topbar">
        <span className="wordmark">
          Depth<span>Wizard</span>
        </span>
        {dsm && (
          <>
            <span className="job-chip mono" title="Project job and surface mode">
              JOB {dsm.id} · {mode.offline ? 'DEMO · NOT METRIC' : isMetric ? 'ABS' : 'REL (rDSM)'}
            </span>
            <span className="hud-chip mono" title="Coordinate reference system">
              {mode.crs || 'LOCAL GRID'}
            </span>
            {dsm.offline && (
              <span className="offline-badge mono" title="Running in offline mode — client-side analysis">
                ● OFFLINE
              </span>
            )}
          </>
        )}
        <span className="hud-chip mono view-chip" title="Current camera and scene view">
          {(viewer.mode || 'orbit').toUpperCase()} · {structMode ? 'STRUCTURES' : 'TERRAIN'}
        </span>
        <button className="back-link" onClick={() => useApp.getState().setScreen('progress')}>
          ← back
        </button>
        {dsm?.source?.type === 'map' && (
          <button
            className="back-link"
            title="Center the 2D map on this reconstruction's AOI"
            onClick={() => {
              const st = useApp.getState()
              const a = dsm.source.aoi
              if (a) st.setMapAoi({ north: a.north, south: a.south, east: a.east, west: a.west })
              st.setScreen('map')
            }}
          >
            Show on map
          </button>
        )}
        <div className="topbar-spacer" />
        {backendEngine?.up !== false && engineModelShort && (
          <span
            className="model-chip mono"
            title={backendEngine.device === 'fallback' ? 'Backend in pseudo-depth fallback — not neural inference' : `Depth backbone ready on ${backendEngine.device || 'unknown device'}`}
          >
            <span className={`model-dot${backendEngine.device === 'fallback' ? ' warn' : ''}`} aria-hidden="true" />
            {engineModelShort} · {backendEngine.device === 'fallback' ? 'fallback' : backendEngine.device || '?'}
          </span>
        )}
        <button
          className="btn ghost"
          onClick={() => setThemeStore(themeArg === 'light' ? 'dark' : 'light')}
          aria-label={themeArg === 'light' ? 'Switch to dark theme' : 'Switch to light theme'}
          title={themeArg === 'light' ? 'Dark theme' : 'Light theme'}
        >
          {themeArg === 'light' ? (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3">
              <path d="M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7Z" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3">
              <circle cx="8" cy="8" r="3.2" />
              <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.4 1.4M11.6 11.6 13 13M13 3l-1.4 1.4M4.4 11.6 3 13" />
            </svg>
          )}
        </button>
        <button className={`btn ghost${viewer.panelOpen === 'analysis' ? ' active' : ''}`} onClick={() => openPanel('analysis')}>
          Analysis
        </button>
        <button
          className={`btn ghost${hazardOpen ? ' active' : ''}`}
          onClick={() => useApp.getState().setHazard({ panelOpen: true })}
        >
          Hazards
        </button>
        <button className={`btn ghost${viewer.panelOpen === 'validation' ? ' active' : ''}`} onClick={() => openPanel('validation')}>
          Validation
        </button>
        <button className={`btn${viewer.panelOpen === 'settings' ? ' active' : ''}`} onClick={() => openPanel('settings')}>
          Export · Settings
        </button>
        <button className="btn ghost" onClick={() => setPresentation(true)} title="Cinematic mission briefing">
          Present
        </button>
      </div>

      <LeftRail viewerApi={activeApi} />

      <CameraDock viewerApi={activeApi} />

      <div className="display-stack" aria-label="Display controls">
        <div className="viewswitch" role="tablist" aria-label="Scene view">
          <button
            role="tab"
            aria-selected={!structMode}
            className={!structMode ? 'active' : ''}
            onClick={() => setStructMode(false)}
          >
            TERRAIN
          </button>
          <button
            role="tab"
            aria-selected={structMode}
            className={structMode ? 'active' : ''}
            disabled={!structAvail}
            title={
              structAvail
                ? 'Structure-aware 3D scene (metric geometry)'
                : 'Structures appear when the backend detects buildings in this job'
            }
            onClick={() => structAvail && setStructMode(true)}
          >
            STRUCTURES
          </button>
        </div>

        <div
          className="zslider"
          title="Visualization-only elevation exaggeration. Does not modify source DSM values."
        >
          <label htmlFor="zscale">Vertical exaggeration</label>
          <input
            id="zscale"
            type="range"
            min="0.2"
            max="10"
            step="0.05"
            value={zVal}
            onChange={(e) => onZChange(parseFloat(e.target.value))}
          />
          <span className="val">{zVal.toFixed(1)}×</span>
        </div>
      </div>

      {structMode && (
        <div className={`layerbox${layersOpen ? '' : ' collapsed'}`} aria-label="Scene layers">
          <button
            type="button"
            className="layer-toggle"
            aria-expanded={layersOpen}
            onClick={() => setLayersOpen((open) => !open)}
          >
            <span>Metric geometry</span>
            <span aria-hidden="true">{layersOpen ? '−' : '+'}</span>
          </button>
          {layersOpen && (
            <>
          {structLayers.filter((l) => l.id !== 'SEGMENTATION').map((l) => (
            <label key={l.id} className="layer-row">
              <input
                type="checkbox"
                defaultChecked
                onChange={(e) => structApiRef.current?.setLayerVisible(l.id, e.target.checked)}
              />
              {l.label} <span className="mono">×{l.count}</span>
            </label>
          ))}
          <span className="mono layer-note">metric geometry · click a building to inspect</span>
            </>
          )}
        </div>
      )}

      {structMode && pickedId != null && (
        <BuildingCard
          building={(dsm?.buildings || []).find((b) => b.id === pickedId)}
          onClose={() => setPickedId(null)}
        />
      )}
      {structMode && structError && (
        <div className="job-error struct-error" role="alert">
          3D scene unavailable — {structError}. Terrain view still works.
        </div>
      )}

      {dsm && (
        <MetaHud dsm={dsm} viewerApi={activeApi} />
      )}

      <HudOverlays viewerApi={activeApi} />

      {showShortcuts && (
        <div className="shortcuts-panel" role="dialog" aria-label="Keyboard shortcuts">
          <div className="shortcuts-content">
            <div className="shortcuts-header">
              <h3>Keyboard Shortcuts</h3>
              <button className="shortcuts-close" onClick={() => setShowShortcuts(false)} aria-label="Close shortcuts">
                ✕
              </button>
            </div>
            <div className="shortcuts-list">
              {Object.entries(SHORTCUT_LABELS).map(([key, label]) => (
                <div key={key} className="shortcut-item">
                  <kbd className="shortcut-key">{label.split(' - ')[0]}</kbd>
                  <span className="shortcut-desc">{label.split(' - ')[1]}</span>
                </div>
              ))}
            </div>
            <p className="shortcuts-hint">Press <kbd>?</kbd> again to close</p>
          </div>
        </div>
      )}

      <div className={`panel-host${viewer.panelOpen ? ' open' : ''}`} aria-hidden={!viewer.panelOpen}>
        <div className="panel-inner">
          {viewer.panelOpen === 'analysis' && <AnalysisPanel />}
          {viewer.panelOpen === 'validation' && <ValidationPanel />}
          {viewer.panelOpen === 'settings' && <SettingsPanel />}
        </div>
      </div>
      <HazardPanel />
      <HazardSceneBridge />
      {presentation && <PresentationMode />}
    </div>
  )
}

function BuildingCard({ building, onClose }) {
  if (!building) return null
  const row = (k, v) => (
    <div className="bld-row" key={k}>
      <span className="mono">{k}</span>
      <b>{v}</b>
    </div>
  )
  return (
    <div className="building-card" role="dialog" aria-label={`Building ${building.id}`}>
      <div className="bld-head">
        <h3>Building {building.id}</h3>
        <button className="panel-close" onClick={onClose} aria-label="Close building card">
          ✕
        </button>
      </div>
      {row('Height', building.height != null ? `${building.height.toFixed(1)} m` : 'withheld — unstable estimate')}
      {row('Roof', building.roof_elevation != null ? `${building.roof_elevation.toFixed(1)} m (${building.roof_type})` : 'unknown')}
      {row('Ground', building.ground_elevation != null ? `${building.ground_elevation.toFixed(1)} m` : 'unknown')}
      {row('Area', building.area_m2 != null ? `${Math.round(building.area_m2)} m²` : `${building.area_px} px`)}
      {row('Confidence', `${Math.round((building.confidence || 0) * 100)}%`)}
    </div>
  )
}
