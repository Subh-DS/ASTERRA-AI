import { useApp } from '../store/useAppStore'

const I = {
  slope: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M2 15 L16 3 M2 15 h14" />
      <path d="M6 12 l4 -4 4 4" strokeWidth="1" opacity="0.6" />
    </svg>
  ),
  hillshade: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4">
      <circle cx="9" cy="9" r="6.5" />
      <path d="M9 2.5 A6.5 6.5 0 0 1 9 15.5 Z" fill="currentColor" stroke="none" />
    </svg>
  ),
  isolines: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.3">
      <path d="M9 3c4 0 6 2.5 6 6s-2 6-6 6-6-2.5-6-6 2-6 6-6Z" />
      <path d="M9 5.5c2.5 0 3.8 1.5 3.8 3.5S11.5 12.5 9 12.5 5.2 11 5.2 9 6.5 5.5 9 5.5Z" />
      <circle cx="9" cy="9" r="1" fill="currentColor" stroke="none" />
    </svg>
  ),
  reset: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M3 8a6 6 0 1 1 1.7 5.4" />
      <path d="M3 4v4h4" />
    </svg>
  ),
  camera: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="2" y="5" width="14" height="10" rx="1" />
      <circle cx="9" cy="10" r="3" />
      <path d="M6 5l1.2-2h3.6L12 5" />
    </svg>
  ),
  rgb: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="3" y="3" width="12" height="12" rx="1" />
      <path d="M3 12l3.5-4 2.5 2.5L12 7l3 4" />
    </svg>
  ),
  elev: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M2 14h14" />
      <path d="M3 14V9h3.5V6H11V3h4v11" />
    </svg>
  ),
  hybrid: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4">
      <rect x="3" y="3" width="12" height="12" rx="1" />
      <path d="M3 11h12" strokeDasharray="2 1.6" />
    </svg>
  ),
  wire: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.1">
      <path d="M3 13 9 4l6 9M3 13h12M6.5 8.5h5M4.8 10.8h8.4" />
    </svg>
  ),
  top: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4">
      <circle cx="9" cy="9" r="6" />
      <circle cx="9" cy="9" r="1.4" fill="currentColor" stroke="none" />
      <path d="M9 3v-1.5M9 16.5V15" />
    </svg>
  ),
  north: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M9 14V4M9 4L5.5 7.5M9 4l3.5 3.5" />
    </svg>
  ),
  iso: (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.4">
      <path d="M9 2.5 15 6v6l-6 3.5L3 12V6Z" />
      <path d="M9 2.5V9m0 0 6-3M9 9 3 6" strokeWidth="1" opacity="0.7" />
    </svg>
  ),
}

function RailBtn({ icon, label, desc, shortcut, active, onClick, disabled }) {
  return (
    <button
      className={`rail-btn${active ? ' active' : ''}${disabled ? ' is-disabled' : ''}`}
      aria-label={shortcut ? `${label} (shortcut ${shortcut})` : label}
      aria-disabled={disabled || undefined}
      onClick={onClick}
    >
      {icon}
      <span className="rail-tip" aria-hidden="true">
        <span className="t-name">{label}</span>
        {desc && <span className="t-desc">{desc}</span>}
        {shortcut && <kbd>{shortcut}</kbd>}
      </span>
    </button>
  )
}

export default function LeftRail({ viewerApi }) {
  const v = useApp((s) => s.viewer)
  const setViewer = useApp((s) => s.setViewer)
  const dsm = useApp((s) => s.dsm)
  const setMode = (colorMode) => {
    setViewer({ colorMode })
    viewerApi?.setColorMode(colorMode)
  }

  return (
    <div className="leftrail" role="toolbar" aria-label="Terrain layers">
      <RailBtn
        icon={I.slope}
        label="Slope"
        desc="Color terrain by gradient steepness"
        shortcut="S"
        active={v.slopeView}
        onClick={() => {
          setViewer({ slopeView: !v.slopeView })
          viewerApi?.setSlopeView(!v.slopeView)
        }}
      />
      <RailBtn
        icon={I.hillshade}
        label="Hillshade"
        desc="Sun-shaded relief lighting"
        shortcut="H"
        active={v.hillshade}
        onClick={() => {
          setViewer({ hillshade: !v.hillshade })
          viewerApi?.setHillshade(!v.hillshade)
        }}
      />
      <RailBtn
        icon={I.isolines}
        label="Contours"
        desc="Elevation isolines overlay"
        shortcut="I"
        active={v.isolines}
        onClick={() => setViewer({ isolines: !v.isolines })}
      />
      <div className="rail-sep" />
      <RailBtn icon={I.rgb} label="RGB" desc="Satellite imagery drape" shortcut="C" active={v.colorMode === 'rgb'} onClick={() => setMode('rgb')} />
      <RailBtn icon={I.elev} label="Elevation" desc="Hypsometric tint by height" shortcut="C" active={v.colorMode === 'elevation'} onClick={() => setMode('elevation')} />
      <RailBtn icon={I.hybrid} label="Hybrid" desc="Imagery fused with elevation" shortcut="C" active={v.colorMode === 'hybrid'} onClick={() => setMode('hybrid')} />
      <RailBtn icon={I.wire} label="Wireframe" desc="Bare mesh structure" shortcut="C" active={v.colorMode === 'wire'} onClick={() => setMode('wire')} />
      <div className="rail-sep" />
      <RailBtn icon={I.top} label="Top-down" desc="Nadir overview camera" onClick={() => viewerApi?.setPreset('top')} />
      <RailBtn icon={I.north} label="North" desc="North-facing oblique camera" onClick={() => viewerApi?.setPreset('north')} />
      <RailBtn icon={I.iso} label="Isometric" desc="Classic iso overview camera" onClick={() => viewerApi?.setPreset('iso')} />
      <div className="rail-sep" />
      <RailBtn icon={I.reset} label="Reset view" desc="Return to overview camera" shortcut="R" onClick={() => viewerApi?.resetView()} />
      <RailBtn
        icon={I.camera}
        label="Snapshot"
        desc={dsm ? 'Export a PNG of this view' : 'Needs a loaded reconstruction'}
        disabled={!dsm}
        onClick={() => {
          const url = viewerApi?.screenshot()
          if (!url) return
          const a = document.createElement('a')
          a.href = url
          a.download = `depthwizard_${dsm.id || 'scene'}.png`
          a.click()
        }}
      />
    </div>
  )
}
