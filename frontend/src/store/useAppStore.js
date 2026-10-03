import { create } from 'zustand'

export const useApp = create((set, get) => ({
  theme:
    typeof document !== 'undefined' && document.documentElement.dataset.theme === 'light' ? 'light' : 'dark',
  setTheme: (t) => {
    const theme = t === 'light' ? 'light' : 'dark'
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('dw-theme', theme)
    } catch {
      /* storage unavailable */
    }
    const meta = document.querySelector('meta[name="theme-color"]')
    meta?.setAttribute('content', theme === 'light' ? '#eae5d6' : '#05090c')
    set({ theme })
  },

  tier: { auto: 'high', override: null, resolved: 'high', reason: '' },
  setTier: (t) => set({ tier: t }),
  reducedMotion: false,
  setReducedMotion: (v) => set({ reducedMotion: v }),

  screen: 'hero',
  setScreen: (s) => set({ screen: s }),

  banner: null,
  setBanner: (b) => set({ banner: b }),

  backendUp: null,
  backendEngine: null,
  setBackendEngine: (e) => set({ backendEngine: e }),

  fieldRef: null,
  setFieldRef: (f) => set({ fieldRef: f }),

  upload: { fileInfo: null, previewUrl: null, blob: null },
  setUploaderFile: (fileInfo, previewUrl, blob = null) =>
    set({ upload: { fileInfo, previewUrl, blob }, job: freshJob(), validation: null, dsm: null }),

  gcps: [],
  addGcp: () => set((s) => ({ gcps: [...s.gcps, { id: s.gcps.length + 1, lat: '', lon: '', elev: '' }] })),
  updateGcp: (id, key, val) =>
    set((s) => ({ gcps: s.gcps.map((g) => (g.id === id ? { ...g, [key]: val } : g)) })),
  removeGcp: (id) => set((s) => ({ gcps: s.gcps.filter((g) => g.id !== id).map((g, i) => ({ ...g, id: i + 1 })) })),

  job: freshJob(),
  setStage: (stage, state, sub) =>
    set((s) => ({ job: { ...s.job, stages: { ...s.job.stages, [stage]: { state, sub } } } })),
  setJobMeta: (meta) => set((s) => ({ job: { ...s.job, ...meta } })),
  resetJob: () => set({ job: freshJob() }),

  dsm: null,
  setDsm: (dsm) => set({ dsm }),

  viewer: {
    mode: 'orbit',
    zScale: 1.0,
    hillshade: true,
    slopeView: false,
    colorMode: 'rgb',
    isolines: false,
    revealedFor: null,
    panelOpen: null,
  },
  setViewer: (patch) => set((s) => ({ viewer: { ...s.viewer, ...patch } })),

  viewerApi: null,
  setViewerApi: (api) => set({ viewerApi: api }),

  presentation: false,
  setPresentation: (v) => set({ presentation: !!v }),

  assistantOpen: false,
  setAssistantOpen: (v) =>
    set((s) => ({
      assistantOpen: v,
      // Opening the assistant closes the viewer panel to prevent overlap.
      // This is intentional — the assistant dock and viewer panel share screen space.
      ...(v && s.viewer.panelOpen ? { viewer: { ...s.viewer, panelOpen: null } } : {})
    })),

  transcript: [],
  pushMessage: (msg) => set((s) => ({ transcript: [...s.transcript, msg] })),
  updateLastMessage: (text) =>
    set((s) => {
      if (!s.transcript.length) return {}
      const next = s.transcript.slice()
      next[next.length - 1] = { ...next[next.length - 1], text }
      return { transcript: next }
    }),
  clearTranscript: () => set({ transcript: [] }),

  validation: null,
  setValidation: (v) => set({ validation: v }),

  map: {
    center: [85.8245, 20.2961],
    zoom: 12,
    basemap: 'satellite',
    aoi: null,
    imagery: null,
  },
  setMap: (patch) => set((s) => ({ map: { ...s.map, ...patch } })),
  setMapAoi: (aoi) => set((s) => ({ map: { ...s.map, aoi } })),
  clearMapAoi: () => set((s) => ({ map: { ...s.map, aoi: null, imagery: null } })),

  mapJob: null,
  setMapJob: (mj) => set({ mapJob: mj }),

  // Map → Pipeline departure cinematic (AOI lock, highlight, dive, handoff).
  mapLeaving: false,
  setMapLeaving: (v) => set({ mapLeaving: !!v }),

  hazard: {
    panelOpen: false,
    type: 'landslide_susceptibility',
    status: 'idle',
    error: null,
    result: null,
    layers: { water: true, shoreline: true, scar: true, debris: true, deposition: true, particles: true, susceptibility: true },
    animT: 0,
    showSimulated: true,
    storm: false,
    rainfallIntensity: 0.5,
    quake: null,
    evac: null,
    evacRequest: 0,
    mission: { playing: false, t: 0, speed: 1 },
  },
  setHazard: (patch) => set((s) => ({ hazard: { ...s.hazard, ...patch } })),
  setHazardLayers: (patch) => set((s) => ({ hazard: { ...s.hazard, layers: { ...s.hazard.layers, ...patch } } })),
  resetHazard: () => set((s) => ({ hazard: { ...s.hazard, status: 'idle', error: null, result: null, animT: 0, showSimulated: true, storm: false, quake: null, evac: null, mission: { playing: false, t: 0, speed: 1 }, layers: { water: true, shoreline: true, scar: true, debris: true, deposition: true, particles: true, susceptibility: true } } })),
}))

function freshJob() {
  return {
    id: null,
    demo: false,
    startedAt: null,
    stages: {
      ingest: { state: 'queued', sub: '' },
      depth: { state: 'queued', sub: '' },
      calibrate: { state: 'queued', sub: '' },
      mesh: { state: 'queued', sub: '' },
    },
    error: null,
  }
}
