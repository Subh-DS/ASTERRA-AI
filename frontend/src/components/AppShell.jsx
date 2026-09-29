import { useApp } from '../store/useAppStore'
import MissionBar from './MissionBar'

const SECTION_LABEL = {
  workflow: 'Workflow',
  upload: 'Ingest',
  map: 'Map',
  progress: 'Pipeline',
  viewer: 'Twin',
}

function Mark({ size }) {
  return (
    <span className="asterra-mark" aria-hidden="true">
      <svg width={size} height={size} viewBox="0 0 18 18" fill="none">
        <path d="M2 13.5c2.4-1.2 3.6-6 7-6s4.6 4.8 7 4.5" stroke="#4fd9c4" strokeWidth="1.2" />
        <path d="M2 15.5c3-1 4.4-7.2 7.4-7.2" stroke="#e8a33d" strokeWidth="0.9" />
        <circle cx="9" cy="9" r="7.4" stroke="rgba(148,196,180,0.5)" strokeWidth="0.8" />
      </svg>
    </span>
  )
}

function ThemeToggle({ theme, setTheme }) {
  return (
    <button
      className="asterra-theme"
      onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
      aria-label={theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'}
      title={theme === 'light' ? 'Dark theme' : 'Light theme'}
    >
      {theme === 'light' ? (
        <svg width="18" height="18" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
          <path d="M13.5 9.5A6 6 0 0 1 6.5 2.5a6 6 0 1 0 7 7Z" />
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
          <circle cx="8" cy="8" r="3.2" />
          <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.4 1.4M11.6 11.6 13 13M13 3l-1.4 1.4M4.4 11.6 3 13" />
        </svg>
      )}
    </button>
  )
}

export default function AppShell({ children }) {
  const screen = useApp((s) => s.screen)
  const backendUp = useApp((s) => s.backendUp)
  const backendEngine = useApp((s) => s.backendEngine)
  const theme = useApp((s) => s.theme)
  const setTheme = useApp((s) => s.setTheme)

  const engineLabel = !backendEngine
    ? 'Engine…'
    : backendEngine.up === false
      ? 'Engine offline'
      : backendEngine.device === 'fallback'
        ? 'Pseudo-depth fallback'
        : `${String(backendEngine.model || 'Depth backbone').split('/').pop()} · ${String(backendEngine.device || '').toUpperCase()}`

  // Landing keeps the full brand header; every screen inside the application
  // gets a slim workspace bar instead.
  if (screen !== 'hero') {
    const showMission = screen !== 'workflow'
    return (
      <div className={`asterra-shell internal${showMission ? ' mission' : ''}`}>
        <header className="asterra-topbar internal" aria-label="Workspace">
          <Mark size={18} />
          <span className="asterra-section">
            Asterra <span aria-hidden="true">·</span> {SECTION_LABEL[screen] || 'Workspace'}
          </span>
          <div className="asterra-status">
            <span
              className="asterra-backend"
              title={backendUp ? `Backend connected — ${engineLabel}` : 'Backend status'}
            >
              <span className={`asterra-dot${backendUp === null ? ' warn' : backendUp ? '' : ' off'}`} aria-hidden="true" />
              {backendUp === null ? 'Connecting…' : backendUp ? 'Backend online' : 'Backend offline'}
            </span>
            <ThemeToggle theme={theme} setTheme={setTheme} />
          </div>
        </header>
        {showMission && <MissionBar />}

        <main style={{ position: 'relative', zIndex: 1 }}>{children}</main>
      </div>
    )
  }

  return (
    <div className="asterra-shell">
      <header className="asterra-topbar">
        <div className="asterra-brand" aria-label="ASTERRA DepthWizard">
          <Mark size={24} />
          <span className="asterra-word">
            Asterra
            <small>DepthWizard</small>
          </span>
        </div>

        <div className="asterra-status">
          <span
            className="asterra-backend"
            title={backendUp ? `Backend connected — ${engineLabel}` : 'Backend status'}
          >
            <span className={`asterra-dot${backendUp === null ? ' warn' : backendUp ? '' : ' off'}`} aria-hidden="true" />
            {backendUp === null ? 'Connecting…' : backendUp ? 'Backend online' : 'Backend offline'}
          </span>
          <ThemeToggle theme={theme} setTheme={setTheme} />
        </div>
      </header>

      <main style={{ position: 'relative', zIndex: 1 }}>{children}</main>
    </div>
  )
}
